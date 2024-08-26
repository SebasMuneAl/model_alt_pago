import pickle
import os
import datetime
import json
from joblib import load
import pandas as pd
from src.data_preprocessing import DataPreprocessor

with open('config.json') as f:
    config_dict = json.load(f)

print('Asignando Parametros')
path = '../data/raw/'
info_pagos_file = config_dict["eval"]["data_files"]["info_pagos_file"]
info_eval_file = config_dict["eval"]["data_files"]["info_eval_file"]
info_mod_cbza_file = config_dict["eval"]["data_files"]["info_mod_cbza_file"]
cols_select = config_dict["eval"]["cols_select"]
cols_ohe = config_dict["eval"]["cols_ohe"]
cols_std = config_dict["eval"]["cols_std"]
cols_model = config_dict["eval"]["cols_model"]
fecha_ft = config_dict["eval"]["fecha_ft"]

#Leer archivos
try:
    print('Leyendo datos')
    df_pagos = pd.read_csv(os.path.join(path,info_pagos_file ))
    df_cbza = pd.read_csv(os.path.join(path,info_mod_cbza_file ))
    df_eval = pd.read_csv(os.path.join(path,info_var_rpta_file ))
except as e:
    print(f'Error en lectura datos \nError: {e}')

prepro = DataPreprocessor()
prepro.load('models/preprocessador.pkl')
try:
    print('Creacion de features y filtro de la fecha para pronostico')
    df_p = prepro.features(df_pagos, df_cbza)
    df_p = prepro.transform(df_p)
    df_p = df_p[df_p['fecha_corte_rep'] ==202312]
    #Añadir info a la base eval
    print('Aplicando features a la base de evaluacion y filtrando el periodo')
    df_eval = pd.merge(
        df_y_oot[['nit_enmascarado','num_oblig_enmascarado', 'num_oblig_orig_enmascarado']],
        df_a,
        how='left',
        left_on = 'num_oblig_enmascarado',
        right_on = 'num_oblig_enmascarado'
    )

print('Seleccionando variables y aplicando modelo')
#Cargue modelo
model = joblib.load('models/model_op_compres.pkl')

x_eval = df_eval[cols_model]
y_pred = model.predict(x_eval)
y_pred_prob = model.predict_proba(x_eval)[:,1]
df_eval['ID'] = df_eval[['nit_enmascarado_x', 'num_oblig_orig_enmascarado', 'num_oblig_enmascarado']].astype('str').agg('#'.join, axis=1)
file_prediction = df_eval[['ID', 'var_rpta_alt', 'prob_uno']]
name_f = f"predict_{datetime.datetime.now().strftime('%Y%m%d_%H:%m:%S')}.csv"
print(f'Guardando predicciones en carpeta predict: {name_f}')
file_prediction.to_csv(os.path.join(path,'predict',name_f), encoding='utf-8', index=False)