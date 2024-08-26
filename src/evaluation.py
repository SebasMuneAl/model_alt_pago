import pickle
import os
import datetime
import json
import joblib
import sys
import logging
import pandas as pd
from data_prepro import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # Envía los logs a la consola
        logging.FileHandler('log.log')   # Envía los logs a un archivo
    ]
)
logger = logging.getLogger(__name__)

path_file = os.path.dirname(__file__)
path_config = os.path.join(path_file, 'config.json')
path = os.path.join(path_file, '../data')
with open(path_config) as f:
    config_dict = json.load(f)

logger.info('Asignando Parametros')
#path = '../data/raw/'
info_pagos_file = config_dict["evaluation"]["data_files"]["info_pagos_file"]
info_eval_file = config_dict["evaluation"]["data_files"]["info_eval_file"]
info_mod_cbza_file = config_dict["evaluation"]["data_files"]["info_mod_cbza_file"]
cols_select = config_dict["evaluation"]["cols_select"]
cols_ohe = config_dict["evaluation"]["cols_ohe"]
cols_std = config_dict["evaluation"]["cols_std"]
cols_model = config_dict["evaluation"]["cols_model"]
fecha_ft = config_dict["evaluation"]["fecha_ft"]

#Leer archivos
try:
    logger.info('Leyendo datos')
    df_pagos = pd.read_csv(os.path.join(path,'raw',info_pagos_file ))
    df_cbza = pd.read_csv(os.path.join(path,'raw',info_mod_cbza_file ))
    df_eval = pd.read_csv(os.path.join(path,'raw',info_eval_file ))
except Exception as e:
    logger.info(f'Error en lectura datos \nError: {e}')
    sys.exit(1)

prepro = DataPreprocessor(cols_ohe, cols_std, cols_select, fecha_ft)
prepro.load('models/preprocessador.pkl')
try:
    logger.info('Creacion de features y filtro de la fecha para pronostico')
    df_p = prepro.features(df_pagos, df_cbza)
    logger.info('Features OK')
    df_p = prepro.transform(df_p)
    logger.info('OHE y Scaler OK')
    df_p = df_p[df_p['fecha_corte_rep'] == int(fecha_ft)]
    logger.info('Filtro Fecha OK')
    #Añadir info a la base eval
    logger.info('Aplicando features a la base de evaluacion y filtrando el periodo')
    df_eval = pd.merge(
        df_eval[['nit_enmascarado','num_oblig_enmascarado', 'num_oblig_orig_enmascarado']],
        df_p,
        how='left',
        left_on = 'num_oblig_enmascarado',
        right_on = 'num_oblig_enmascarado'
    )
except Exception as e:
    logger.info(f'ERROR: {e}')
    #sys.exit(1)

logger.info('Seleccionando variables y aplicando modelo')
#Cargue modelo
model = joblib.load('models/model_op_compres.pkl')

x_eval = df_eval[cols_model]
y_pred = model.predict(x_eval)
y_pred_prob = model.predict_proba(x_eval)[:,1]
df_eval['ID'] = df_eval[['nit_enmascarado_x', 'num_oblig_orig_enmascarado', 'num_oblig_enmascarado']].astype('str').agg('#'.join, axis=1)
df_eval['var_rpta_alt'] = y_pred
df_eval['prob_uno'] = y_pred_prob
file_prediction = df_eval[['ID', 'var_rpta_alt', 'prob_uno']]
name_f = f"predict_{datetime.datetime.now().strftime('%Y%m%d_%H%m%S')}.csv"
logger.info(f"Guardando predicciones en carpeta: {os.path.join(path,'predict',name_f)}")
file_prediction.to_csv(os.path.join(path,'predict',name_f), encoding='utf-8', index=False)