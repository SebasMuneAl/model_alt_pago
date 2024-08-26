from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import pickle

class DataPreprocessor:
    def __init__(self, cols_ohe, cols_std, cols_select, fecha_t):
        self.cols_ohe = cols_ohe #columnas para one hot
        self.cols_std = cols_std #columnas para scaler
        self.cols_select = cols_select #columnas para scaler
        self.fecha = fecha_t #Fecha anterior a cuando pronosticaremos (si pronosticamos para febrero, la fecha es enero)
        # Tansformer y params
        self.transformer = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output = False), self.cols_ohe),
                ('scaler', StandardScaler(), self.cols_std),
            ],
            remainder='passthrough'  # Deja las columnas no transformadas tal como están
        )
    
    def fit(self, X: pd.DataFrame):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        #Transforma el OHE y Scaler y utiliza el features names previamente cargado
        return pd.DataFrame(self.transformer.transform(X), columns = self.get_features_names())
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = pd.DataFrame(self.transformer.fit_transform(X))
        return data
    
    def features(self, X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
        df_pagos = X
        df_cbza = Y
        #Fecha Aniomes
        df_pagos['fecha_corte_rep'] = df_pagos['fecha_corte'].astype('str').str.slice(0, 6).astype('int64')
        df_merge = pd.merge(
            df_pagos,
            df_cbza[['num_oblig_enmascarado','fecha_corte','prob_propension','prob_alrt_temprana','prob_auto_cura']],
            how='left',
            left_on = ['num_oblig_enmascarado', 'fecha_corte_rep'],
            right_on = ['num_oblig_enmascarado', 'fecha_corte']
        )
        #Reset index para lags y windows
        df_merge = df_merge.sort_values(by=['num_oblig_enmascarado', 'fecha_corte_rep']).reset_index(drop=True)
        #creacion de variables lags
        cols_lag = ['pago_total','prob_propension','prob_alrt_temprana','prob_auto_cura']
        df_merge = self.lag_info(df_merge, cols_lag, 1)
        df_merge = self.lag_info(df_merge, cols_lag, 2)
        #Variables flag y window
        df_merge['prob_propension_bajo'] = (df_merge['prob_propension_lag_1'] < df_merge['prob_propension_lag_2']).astype(int)
        df_merge['prob_alrt_temprana_bajo'] = (df_merge['prob_alrt_temprana_lag_1'] > df_merge['prob_alrt_temprana_lag_2']).astype(int)
        df_merge['prob_auto_cura_bajo'] = (df_merge['prob_auto_cura_lag_1'] < df_merge['prob_auto_cura_lag_2']).astype(int)
        df_merge = self.rolling_info(df_merge, 'pago_total_lag_1', 3)
        #Filtramos DF a la fecha para pronostico y eliminamos duplicados
        #df_merge = df_merge[df_merge['fecha_corte_rep'] == self.fecha]
        df_merge= df_merge.drop_duplicates(subset=['num_oblig_enmascarado', 'fecha_corte_rep'], keep='first')
        #Seleccionamos las columnas del DF final y el transformer
        df_final = df_merge[self.cols_select]
        return df_final
    
    def get_features_names(self):
        # Sacar nombres
        names = self.transformer.get_feature_names_out()
        names_clean = [name.split('__',1)[-1] for name in names]
        return names_clean
    
    def save(self, filepath: str):
        # guardar el pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.transformer, f)
    
    def load(self, filepath: str):
        # Cargar el pickle
        with open(filepath, 'rb') as f:
            self.transformer = pickle.load(f)

    def lag_info(self, df, col, n_lag):
        if isinstance(col, list):
            for c in col:
                df[f'{c}_lag_{n_lag}'] = df.groupby('num_oblig_enmascarado')[c].shift(n_lag)
        else:
            df[f'{col}_lag_{n_lag}'] = df.groupby('num_oblig_enmascarado')[col].shift(n_lag)
        return df
        
    def rolling_info(self, df, col, window):
        df[f'{col}_sum_{window}'] = df.groupby('num_oblig_enmascarado')[col].transform(lambda x: x.rolling(window = window).sum())
        df[f'{col}_mean_{window}'] = df.groupby('num_oblig_enmascarado')[col].transform(lambda x: x.rolling(window = window).mean())
        return df