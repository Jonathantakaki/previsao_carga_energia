Previsão de Carga de Energia (2024–2025)

Objetivo
Construir um modelo para prever a carga horária de energia (MW médio) por região, combinando histórico da carga com variáveis meteorológicas do INMET.

Dados

Curva de carga (parquet 2024 e 2025)

INMET (CSV com metadados de estação)

Metodologia

Ingestão e limpeza (padronização de data/hora)

Merge por Regiao e data_hora

Feature engineering:

Calendário: hora, dia da semana, mês, dia do ano, fim de semana

Lags: 1,2,3,24 e 168 horas

Modelagem: RandomForest e XGBoost

Validação temporal: treino em 2024 e teste em 2025

Métricas: MAE, RMSE e R² + gráficos Real vs Previsto
