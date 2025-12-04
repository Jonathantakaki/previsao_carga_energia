# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 20:32:03 2025

@author: Usuario
"""


import pandas as pd
import re, unicodedata, os

# def _rm_acc(s):  # remover acentos
#     return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

# def _clean_val(x):
#     if x is None: return None
#     x = str(x).strip()
#     x = re.sub(r'^[;:\s]+','', x)     # remove ; : e espaços do início
#     x = re.sub(r'[;:\s]+$','', x)     # remove ; : e espaços do fim
#     return x.strip()

# #Mapear chaves dos metadados para nomes padronizados
# KEYMAP = {
#     'uf':'UF',
#     'estacao':'ESTACAO','estação':'ESTACAO',
#     'codigo (wmo)':'CODIGO_WMO','codigo wmo':'CODIGO_WMO',
#     'latitude':'LATITUDE','longitude':'LONGITUDE','altitude':'ALTITUDE',
#     'data de fundacao':'DATA_FUNDACAO','data de fundação':'DATA_FUNDACAO'
# }

# def read_inmet_file(path, top_lines=60):
#     """Lê um arquivo INMET e adiciona metadados nas colunas"""
#     with open(path, 'r', encoding='latin1', errors='ignore') as f:
#         top = [next(f) for _ in range(top_lines)]

#     # Encontra a linha do cabeçalho da tabela
#     header_idx = None
#     for i, ln in enumerate(top):
#         s = ln.strip().lower()
#         if 'data' in s and 'hora' in s and 'utc' in s and ';' in s:
#             header_idx = i
#             break
#     if header_idx is None:
#         raise ValueError(f'Cabeçalho não encontrado em {os.path.basename(path)}')

#     # Extrai metadados linhas acima do cabeçalho)
#     meta = {}
#     for ln in top[:header_idx]:
#         m = re.match(r'^\s*([^:;]+)\s*[:;]\s*(.+?)\s*$', ln)
#         if not m:
#             continue
#         k_raw = _rm_acc(m.group(1)).lower().strip()
#         k_std = KEYMAP.get(k_raw)
#         if k_std:
#             meta[k_std] = _clean_val(m.group(2))

#     # Normaliza metadados numéricos e datas
#     for c in ['LATITUDE','LONGITUDE','ALTITUDE']:
#         if c in meta:
#             meta[c] = _clean_val(meta[c]).replace(',', '.')
#     if 'DATA_FUNDACAO' in meta:
#         meta['DATA_FUNDACAO'] = pd.to_datetime(meta['DATA_FUNDACAO'], dayfirst=True, errors='coerce')

#     # Lê o CSV a partir do cabeçalho real
#     df = pd.read_csv(path, sep=';', encoding='latin1', skiprows=header_idx,
#                      low_memory=False, decimal=',')
#     df.columns = [re.sub(r'\s+', ' ', c.strip()) for c in df.columns]
    
#     # Remove colunas "Unnamed" 
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


#     # Adiciona metadados como colunas
#     for k, v in meta.items():
#         df[k] = v

#     df['_arquivo_origem'] = os.path.basename(path)

#     # Converte metadados numéricos
#     for c in ['LATITUDE','LONGITUDE','ALTITUDE']:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors='coerce')

#     return df



# PATH_DIR = r"C:\Users\Usuario\OneDrive\Área de Trabalho\projeto\projeto\2024_2025"
# arquivos = [os.path.join(PATH_DIR, f) for f in os.listdir(PATH_DIR) if f.lower().endswith('.csv')]
# dfs = []

# for i, fp in enumerate(arquivos, 1):
#     try:
#         dfs.append(read_inmet_file(fp))
#         if i % 20 == 0:
#             print(f" {i} arquivos lidos...")
#     except Exception as e:
#         print(f" Erro em {os.path.basename(fp)}: {e}")

# inmet = pd.concat(dfs, ignore_index=True)
# print("✔️ Linhas INMET:", len(inmet))
# print(inmet[['UF','ESTACAO','LATITUDE','LONGITUDE','ALTITUDE','DATA_FUNDACAO']].head())


# map_uf_regiao = {
#     'AC':'NORTE','AL':'NORDESTE','AM':'NORTE','AP':'NORTE','BA':'NORDESTE','CE':'NORDESTE',
#     'DF':'CENTRO-OESTE','ES':'SUDESTE','GO':'CENTRO-OESTE','MA':'NORDESTE','MG':'SUDESTE',
#     'MS':'CENTRO-OESTE','MT':'CENTRO-OESTE','PA':'NORTE','PB':'NORDESTE','PE':'NORDESTE',
#     'PI':'NORDESTE','PR':'SUL','RJ':'SUDESTE','RN':'NORDESTE','RO':'NORTE','RR':'NORTE',
#     'RS':'SUL','SC':'SUL','SE':'NORDESTE','SP':'SUDESTE','TO':'NORTE'
# }

# inmet['Regiao'] = inmet['UF'].map(map_uf_regiao)
#%%
#limpar Data
# inmet["Data"] = inmet["Data"].astype(str).str.strip()

# # limpar Hora UTC
# inmet["Hora UTC"] = (
#     inmet["Hora UTC"]
#         .astype(str)
#         .str.replace(" UTC", "", regex=False)
#         .str.extract(r"(\d{1,4})")[0]
#         .str.zfill(4)
# )

# # montar a string completa
# dt_str = (
#     inmet["Data"] + " " +
#     inmet["Hora UTC"].str.slice(0,2) + ":" +
#     inmet["Hora UTC"].str.slice(2,4)
# )

# # converter para datetime (não forçar format!)
# inmet["data_hora"] = pd.to_datetime(
#     dt_str,
#     errors="coerce"
# )


#%% Estrutura e tipos INMET

# print(inmet.info())
# print(inmet.describe(include='all').T)



#%% Filtrar colunas inmet

# inmet = inmet.drop(columns=['Unnamed: 19', 'Data', 'Hora UTC', 'UF','_arquivo_origem','CODIGO_WMO', 'DATA_FUNDACAO'], errors="ignore")



#%%

# #ler arquivo curva carga CURVA_CARGA
# curva_carga_horario_2024 = pd.read_parquet(
#     "CURVA_CARGA_2024.parquet",
#     engine="fastparquet"
# )
# curva_carga_horario_2025 = pd.read_parquet(
#     "CURVA_CARGA_2025.parquet",
#     engine="fastparquet"
# )

# curva_carga = pd.concat(
#     [curva_carga_horario_2024, curva_carga_horario_2025],
#     ignore_index=True
# )

# curva_carga.rename(columns={
#     'id_subsistema': 'Sigla_Região',
#     'nom_subsistema': 'Regiao',
#     'din_instante': 'data_hora',
#     'val_cargaenergiahomwmed': 'Carga_ener_mw_medio'
# }, inplace=True)




#%%

# #Converte Carga_ener_mw_medio para float (substitui vírgula por ponto, se existir)
# curva_carga["Carga_ener_mw_medio"] = (
#     curva_carga["Carga_ener_mw_medio"]
#     .astype(str)                   # garante que seja string
#     .str.replace(",", ".", regex=False)  # troca vírgula por ponto decimal
# )
# curva_carga["Carga_ener_mw_medio"] = pd.to_numeric(
#     curva_carga["Carga_ener_mw_medio"], errors="coerce"
# )

# # Converte Regiao e Sigla_Região para string 
# curva_carga["Regiao"] = curva_carga["Regiao"].astype(str)
# curva_carga["Sigla_Região"] = curva_carga["Sigla_Região"].astype(str)


# # Verifica resultado
# print(curva_carga.dtypes.head(5))

#%% Estrutura e tipos CURVA_CARGA

# print(curva_carga.info())
# print(curva_carga.describe(include='all').T)

#%%

# parquet_path = r"C:\Users\Usuario\OneDrive\Área de Trabalho\projeto\projeto\CURVA_CARGA.parquet"

# curva_carga.to_parquet(parquet_path, index=False, engine="pyarrow")
# print(f"\n Arquivo Parquet salvo em:\n{parquet_path}")


#%% Fazer o merge

# df_merged = pd.merge(
#     curva_carga,
#     inmet,
#     on=['Regiao', 'data_hora'],
#     how='outer'
# )

#%%

# print("\n===== df_merged =====")
# print(df_merged.info())
# print(df_merged.describe(include='all').T)



# # Criar tabela de NaN
# missing = pd.DataFrame({
#     "Total Linhas": len(df_merged),
#     "NaN": df_merged.isna().sum(),
#     "% NaN": (df_merged.isna().sum() / len(df_merged)) * 100,
#     "Tipo": df_merged.dtypes,
#     "Valores Únicos": df_merged.nunique()
# })

# print("\n===== Porcentagem de NaN por coluna =====")
# print(missing)

#%%
# converter alvo para numérico
# df_merged["Carga_ener_mw_medio"] = pd.to_numeric(df_merged["Carga_ener_mw_medio"], errors="coerce")

# # converter strings curtas para category
# for col in ["Regiao", "Sigla_Região"]:
#     if col in df_merged.columns:
#         df_merged[col] = df_merged[col].astype("category")


#%%

# df_merged.to_parquet(
#     r"C:\Users\Usuario\OneDrive\Área de Trabalho\projeto\merged_carga_inmet.parquet",
#     index=False
# )


#%% Ler parquet

import pandas as pd
import re, unicodedata, os
import pyarrow.dataset as ds

caminho = r"C:\Users\Usuario\OneDrive\Área de Trabalho\projeto\projeto"

# Leitura do parquet
df_merged = pd.read_parquet(r"C:\Users\Usuario\OneDrive\Área de Trabalho\projeto\projeto\merged_carga_inmet.parquet")

df_merged = df_merged.sort_values("data_hora")

# Verifica estrutura
print(df_merged.shape)
print(df_merged.info())

df_merged["ano"] = df_merged["data_hora"].dt.year

df_2024 = df_merged[df_merged["ano"] == 2024].copy()
df_2025 = df_merged[df_merged["ano"] == 2025].copy()


df_2024.to_parquet("merged_2024.parquet", index=False)
df_2025.to_parquet("merged_2025.parquet", index=False)




#%% EDA

from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np

# descartar colunas irrelevantes
colunas_descartar = [
    "Sigla_Região"]

df_eda = df_merged.drop(columns=[c for c in colunas_descartar if c in df_merged.columns])

# Tipagem — importante para o cálculo correto
# - alvo numérico
if df_eda["Carga_ener_mw_medio"].dtype == "O":
    df_eda["Carga_ener_mw_medio"] = pd.to_numeric(df_eda["Carga_ener_mw_medio"], errors="coerce")

# - variáveis categóricas (para não entrarem na correlação)
for c in ["Regiao", "Sigla_Região"]:
    if c in df_eda.columns:
        df_eda[c] = df_eda[c].astype("category")

# Amostra leve
amostra = df_eda

# Seleciona apenas colunas numéricas para cálculo de correlação
num_cols = amostra.select_dtypes(include=["number"]).columns
amostra_numerica = amostra[num_cols].copy()

# Gera o relatório
profile = ProfileReport(
    amostra,
    title="EDA (leve + Pearson numérico) - df_merged",
    minimal=True,
    explorative=False,
    correlations={
        "pearson": {"calculate": True},    # ativa apenas Pearson
        "spearman": {"calculate": False},
        "kendall": {"calculate": False},
        "phi_k": {"calculate": False},
        "cramers": {"calculate": False},
    },
    interactions={"continuous": False},
    missing_diagrams={
        "bar": True,
        "matrix": False,
        "heatmap": False,
        "dendrogram": False,
    },
    duplicates={"calculate": False},
    samples=None,
)

profile.to_file("EDA_df_merged_Pearson.html")
print("✅ Relatório salvo: EDA.html")

#%%
def make_features(df):
    df = df.drop(columns=['Sigla_Região', 'ano'], errors="ignore").copy()
    df = df.dropna(subset=["Carga_ener_mw_medio"])
    df = df.sort_values("data_hora")

    # features de tempo
    df["hora"] = df["data_hora"].dt.hour
    df["dia_da_semana"] = df["data_hora"].dt.dayofweek
    df["mes"] = df["data_hora"].dt.month
    df["dia_do_ano"] = df["data_hora"].dt.dayofyear
    df["fim_de_semana"] = (df["dia_da_semana"] >= 5).astype(int)

    #lags
    df["lag_1"]   = df["Carga_ener_mw_medio"].shift(1)
    df["lag_2"]   = df["Carga_ener_mw_medio"].shift(2)
    df["lag_3"]   = df["Carga_ener_mw_medio"].shift(3)
    df["lag_24"]  = df["Carga_ener_mw_medio"].shift(24)
    df["lag_168"] = df["Carga_ener_mw_medio"].shift(168)

    # tira linhas com NaN geradas pelos lags
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "lag_24", "lag_168"])

    # remove localização antes de montar X
    cols_drop = [
        "Sigla_Região", "Regiao", "UF", "LATITUDE", "LONGITUDE",
        "ALTITUDE", "CODIGO_WMO", "_arquivo_origem", "ESTACAO", "DATA_FUNDACAO"
    ]
    cols_drop = [c for c in cols_drop if c in df.columns]
    df = df.drop(columns=cols_drop)

    return df

#%%
df_2024_feat = make_features(df_2024)
df_2025_feat = make_features(df_2025)

#%%

df_train = df_2024_feat.tail(200_000).copy()


#%%

df_val = df_2025_feat.copy()


#%%

X_train = df_train.drop(columns=["Carga_ener_mw_medio", "data_hora"], errors="ignore")
y_train = df_train["Carga_ener_mw_medio"]

X_val   = df_val.drop(columns=["Carga_ener_mw_medio", "data_hora"], errors="ignore")
y_val   = df_val["Carga_ener_mw_medio"]

print("Tamanho treino:", X_train.shape)
print("Tamanho validação:", X_val.shape)


#%% Modelo 1  — Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"Validação 2025 - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

#%%

# === MÉTRICAS RANDOM FOREST (Treino e Teste) ===

# Previsões de treino
y_pred_train_rf = rf.predict(X_train)

# Métricas de treino
mae_train_rf  = mean_absolute_error(y_train, y_pred_train_rf)
rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
r2_train_rf   = r2_score(y_train, y_pred_train_rf)

# Métricas de teste (você já calculou y_pred)
mae_test_rf  = mae
rmse_test_rf = rmse
r2_test_rf   = r2

print("\n=== RANDOM FOREST ===")
print(f"Treino → MAE: {mae_train_rf:.2f} | RMSE: {rmse_train_rf:.2f} | R²: {r2_train_rf:.3f}")
print(f"Teste  → MAE: {mae_test_rf:.2f} | RMSE: {rmse_test_rf:.2f} | R²: {r2_test_rf:.3f}")

#%% Modelo 2 — XGBoost
# ============================================================
# 1. Treino 2024 (somente últimas 200k linhas)
# ============================================================

from xgboost import XGBRegressor

df_train = df_2024_feat.tail(200_000).copy()

X_train = df_train.drop(columns=["Carga_ener_mw_medio", "data_hora"], errors="ignore")
y_train = df_train["Carga_ener_mw_medio"]

# ============================================================
# 2. Validação 2025
# ============================================================

df_val = df_2025_feat.copy()

X_val = df_val.drop(columns=["Carga_ener_mw_medio", "data_hora"], errors="ignore")
y_val = df_val["Carga_ener_mw_medio"]

print("Treino XGBoost:", X_train.shape)
print("Validação XGBoost:", X_val.shape)

# ============================================================
# 3. Modelo XGBoost
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",   # MUITO mais rápido
    n_jobs=-1
)

# treino
xgb_model.fit(X_train, y_train)

# validação 2025
y_pred_xgb = xgb_model.predict(X_val)

# métricas
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
r2_xgb = r2_score(y_val, y_pred_xgb)

print(f"\nValidação 2025 - XGBoost")
print(f"MAE:  {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²:   {r2_xgb:.3f}")


#%% Resultados
resultados = pd.DataFrame({
    "Modelo": ["RandomForestRegressor", "XGBoost"],
    "MAE":  [mae,     mae_xgb],     
    "RMSE": [rmse,    rmse_xgb],
    "R²":   [r2,      r2_xgb]
})

print(resultados)


#%%
# === MÉTRICAS XGBOOST (Treino e Teste) ===

# Previsões de treino
y_pred_train_xgb = xgb_model.predict(X_train)

# Métricas de treino
mae_train_xgb  = mean_absolute_error(y_train, y_pred_train_xgb)
rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
r2_train_xgb   = r2_score(y_train, y_pred_train_xgb)

# Métricas de teste (você já calculou y_pred_xgb)
mae_test_xgb  = mae_xgb
rmse_test_xgb = rmse_xgb
r2_test_xgb   = r2_xgb

print("\n=== XGBOOST ===")
print(f"Treino → MAE: {mae_train_xgb:.2f} | RMSE: {rmse_train_xgb:.2f} | R²: {r2_train_xgb:.3f}")
print(f"Teste  → MAE: {mae_test_xgb:.2f} | RMSE: {rmse_test_xgb:.2f} | R²: {r2_test_xgb:.3f}")

#%% Importância das Features
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost
xgb_importances = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=xgb_importances.iloc[:15], y=xgb_importances.index[:15])
plt.title("XGBoost - Importância das Features (treino 2024)")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

# RandomForest
rf_importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top_n = min(15, len(rf_importances))
plt.figure(figsize=(10,6))
sns.barplot(x=rf_importances.iloc[:top_n], y=rf_importances.index[:top_n])
plt.title("Random Forest - Importância das Features (treino 2024)")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

#%% 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(12,6))
plt.plot(df_val["data_hora"].iloc[:500], y_val.iloc[:500], label="Real", linewidth=2, color="black")
plt.plot(df_val["data_hora"].iloc[:500], y_pred_xgb[:500], label="Previsto (XGBoost)", linestyle="--", color="orange", linewidth=2)
plt.title("Previsão da Carga - Real vs Previsto (Primeiras 500 horas de 2025)")
plt.xlabel("Tempo")
plt.ylabel("Carga (MW)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()





