import os
import json
import pickle
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_prepro import DataPreprocessor

with open('config.json') as f:
    config_dict = json.load(f)

print('Asignando Parametros')
path = '../data/raw/'
info_pagos_file = config_dict["model"]["data_files"]["info_pagos_file"]
info_var_rpta_file = config_dict["model"]["data_files"]["info_var_rpta_file"]
info_mod_cbza_file = config_dict["model"]["data_files"]["info_mod_cbza_file"]
cols_select = config_dict["model"]["cols_select"]
cols_ohe = config_dict["model"]["cols_ohe"]
cols_std = config_dict["model"]["cols_std"]
cols_model = config_dict["model"]["cols_model"]
target = config_dict["model"]["target"]

#Leer archivos
df_pagos = pd.read_csv(os.path.join(path,info_pagos_file ))
df_cbza = pd.read_csv(os.path.join(path,info_mod_cbza_file ))
df_y = pd.read_csv(os.path.join(path,info_var_rpta_file ))
#Preprocesamiento y junte de archivos
prepro = DataPreprocessor(cols_ohe, cols_std, cols_select, fecha_t)
df_p = prepro.fit_transform(df_pagos, df_cbza)
df_p.columns = prepro.get_features_names()
#Filtros de informacion posiblemente ruido
list_oblig_error = df_p[
    (df_p['marca_pago']=='PAGO_MENOS') & 
    (df_p['pago_total'].astype(int) > df_p['valor_cuota_mes'].astype(int))]['num_oblig_enmascarado'].unique().tolist()
df_p_filt = df_p[~df_p['num_oblig_enmascarado'].isin(list_oblig_error)]
df_p_filt = pd.merge(
    df_p_filt,
    df_y[['num_oblig_enmascarado', 'fecha_var_rpta_alt', target]],
    how = 'left',
    left_on = ['num_oblig_enmascarado', 'fecha_corte_rep'],
    right_on = ['num_oblig_enmascarado', 'fecha_var_rpta_alt']
)
max_dates = df_p_filt.groupby('num_oblig_enmascarado')['fecha_var_rpta_alt'].max().reset_index()
df_p_filt = df_p_filt.merge(max_dates, on='num_oblig_enmascarado', suffixes=('', '_max'))
df_p_filt = df_p_filt[df_p_filt['fecha_corte_rep'] <= df_p_filt['fecha_var_rpta_alt_max']]

df_p_filt = df_p_filt.dropna(subset=target)
x = df_p_filt[cols_model]
y = df_p_filt[target]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)
best_param = {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
model = RandomForestClassifier(**best_param)
model = model.fit(X_train, y_train)

dump(model, 'models/model_op_compres.pkl', compress=('xz', 3))
