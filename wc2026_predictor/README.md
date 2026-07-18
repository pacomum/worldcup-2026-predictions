# ⚽ Predicción de Resultados — Copa Mundial FIFA 2026

Sistema de machine learning end-to-end para predecir resultados de partidos de la Copa Mundial FIFA 2026, desde la extracción de datos históricos hasta la generación de reportes de predicción por ronda.

## 📋 Descripción

Este proyecto construye un pipeline completo de ciencia de datos para pronosticar resultados de partidos de fútbol internacional, aplicado específicamente al Mundial 2026. Combina ingeniería de características basada en fútbol (ratings Elo, forma reciente, head-to-head), validación temporal rigurosa para evitar fuga de datos, y comparación sistemática de modelos de machine learning.

El proyecto se actualiza de forma iterativa conforme avanza el torneo: cada ronda (Fase de Grupos, Dieciseisavos, Octavos, Cuartos, Semifinales) se evalúa contra resultados reales, y los aprendizajes retroalimentan la siguiente versión del modelo.

## 🎯 Objetivo

Predecir el resultado de partidos de fútbol internacional (victoria local / empate / victoria visitante) con un enfoque reproducible, documentado y basado en datos históricos verificables — no en opinión ni en estadísticas de "hype" mediático.

## 📊 Resultados finales por ronda

| Ronda | Modelo | Aciertos | Accuracy | Nota |
|-------|--------|----------|----------|------|
| Fase de grupos J1-J2 | v1 | 35/52 | 67.3% | Modelo base |
| Fase de grupos J3 | v2 | 18/28 | 64.3% | J3 estratégica, más empates |
| 16avos de final | v3 | 12/16 | 75.0% | 3 fallos = partidos en penales |
| Octavos de final | v4 | 6/8 | 75.0% | Noruega-Brasil indetectable |
| Cuartos de final | v5 | **4/4** | **100.0% 🏆** | Primera ronda perfecta |
| Semifinales | v6 | **2/2** | **100.0% 🏆** | España + Argentina confirmados |
| **Final + 3er lugar** | **v7** | **TBD** | **TBD** | **49.9% España / 50.1% Argentina** |

**Accuracy KO acumulado: 24/38 (63.2%)** — los fallos son casi todos penales o remontadas en tiempo añadido, estructuralmente no predecibles.

## 🏗️ Estructura del proyecto

```
wc2026_predictor/
├── data/
│   ├── fixtures/              # Fixture por ronda (CSV)
│   │   ├── semis.csv
│   │   ├── cuartos.csv
│   │   ├── octavos.csv
│   │   └── final.csv
│   ├── results/               # Comparaciones predicción vs realidad
│   │   ├── comparacion_16avos.csv
│   │   ├── comparacion_octavos.csv
│   │   ├── comparacion_cuartos.csv
│   │   ├── comparacion_semis.csv
│   │   └── predicciones_final_v7.csv
│   ├── xg/
│   │   └── xg_stats_wc2026.csv   # xG acumulado del torneo
│   ├── feature_importance.csv     # Permutation Importance v7
│   └── model_comparison.csv       # Walk-forward validation v7
├── src/
│   ├── elo.py                 # Cálculo de rating Elo desde cero
│   ├── features.py            # Ingeniería de las 14 variables seleccionadas
│   ├── train.py               # Entrenamiento y validación walk-forward
│   └── predict.py             # Predicciones + exportación Excel
├── reports/                   # Reportes Excel por ronda
│   ├── v4_Octavos.xlsx
│   ├── v5_Cuartos.xlsx
│   ├── v6_Semifinales.xlsx
│   └── v7_Final_y_3erLugar.xlsx
├── notebooks/
│   └── 01_EDA.ipynb           # Exploración y análisis
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔬 Metodología

### Datos
- **Historial de partidos**: [martj42/international_results](https://github.com/martj42/international_results) — 49,000+ partidos (1872–2026)
- **xG del torneo**: xGscore.io / RealGM (cada partido del Mundial 2026)
- **Período de entrenamiento**: últimos 10 años (2016–2026), ~9,600 partidos
- **Validación**: walk-forward temporal estricta — nunca se entrena con datos del futuro

### Rating Elo
Implementación propia replicando [eloratings.net](https://www.eloratings.net):
- Factor K por importancia del torneo (Mundial=60, Clasificatorias=40, Amistosos=20)
- Ajuste por margen de gol (factor G)
- Ventaja de local (+100 puntos)

### Features — 14 variables seleccionadas por Permutation Importance

Tras 7 versiones y selección rigurosa, el modelo converge a **14 variables estables**. Las variables xG del torneo y variables de racha fueron descartadas sistemáticamente (importancia ≤0 en todos los folds).

| # | Feature | Descripción |
|---|---------|-------------|
| 1 | `we_h` | Probabilidad Elo esperada del local — **la más predictiva** |
| 2 | `elo_diff` | Diferencia de Elo entre equipos |
| 3 | `elo_ratio` | Cociente de Elo (captura no-linealidad) |
| 4 | `gf_a` | Goles promedio del visitante |
| 5 | `conf_diff` | Diferencial de confederación (UEFA/CONMEBOL > resto) |
| 6 | `elo_a` | Elo absoluto del visitante |
| 7 | `wc_a` | Rendimiento histórico visitante en Mundiales |
| 8 | `form_diff` | Diferencia de forma reciente (5 partidos) |
| 9 | `conf_prod` | Producto de fortalezas de confederación |
| 10 | `conf_a` | Confederación del visitante |
| 11 | `h2h` | Head-to-head ponderado por recencia |
| 12 | `form_h3` | Forma ultrareciente local (últimos 3 partidos) |
| 13 | `draw_risk` | Índice compuesto de riesgo de empate |
| 14 | `wc_diff` | Diferencial de rendimiento en Mundiales histórico |

### Modelo
- **Tipo**: Ensemble soft-voting de los 3 mejores modelos (walk-forward)
- **Composición v7**: Logística + Random Forest + Logística Calibrada
- **Accuracy walk-forward**: ~60.5% (log-loss ~0.864)

### Hallazgos clave del proyecto

1. **Redes neuronales = peor resultado**: MLP (2-4 capas) obtiene log-loss 3.3–4.1 vs 0.86 de Logística. Con ~9,600 muestras y alto ruido del fútbol, los modelos lineales calibrados ganan siempre.

2. **Variables xG del torneo = ruido**: Con 4-8 partidos por equipo de muestra, el xG individual es descartado sistemáticamente por Permutation Importance en todas las versiones del modelo.

3. **El modelo converge**: De 60 variables candidatas en v4, el modelo estabiliza en las mismas 14 en v5, v6 y v7 — señal de que se encontró el óptimo.

4. **Penales son azar puro**: Los 5 fallos "evitables" del proyecto fueron todos partidos decididos en penales o en los últimos minutos — ningún modelo histórico puede predecirlos.

## 🚀 Uso rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos históricos
python src/elo.py --download

# 3. Construir features de entrenamiento
python src/features.py

# 4. Entrenar modelo (walk-forward + selección de features)
python src/train.py

# 5. Generar predicciones para un fixture
python src/predict.py --fixture data/fixtures/final.csv --round "Final" --knockout
```

## 🔄 Workflow por ronda

```
Nueva ronda disponible
        ↓
git pull  (actualiza results.csv desde martj42)
        ↓
python src/elo.py          → Recalcula ratings Elo post-resultados
        ↓
python src/features.py     → Construye features del fixture nuevo
        ↓
python src/train.py        → Re-entrena + Permutation Importance
        ↓
python src/predict.py      → Predicciones + Excel para la ronda
        ↓
Guardar reporte en reports/ + commit al repo
```

## 📦 Dependencias

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.0
openpyxl>=3.1
requests>=2.31
matplotlib>=3.7
seaborn>=0.12
jupyterlab>=4.0
```

## 📁 Datos históricos

El archivo `data/results.csv` no se incluye en el repositorio (tamaño). Se descarga automáticamente:

```bash
python src/elo.py --download
```

Fuente: https://github.com/martj42/international_results

## 📄 Licencia

MIT — datos de partidos bajo licencia de [martj42/international_results](https://github.com/martj42/international_results).



## 👤 Autor

**José Francisco Muñoz Martínez**
Analista de Datos Sr | Ciencia de Datos & Machine Learning
[LinkedIn](https://www.linkedin.com/in/franciscomu%C3%B1ozm/) · franciscomumz@gmail.com

---
*Proyecto personal desarrollado con fines de aprendizaje y portafolio. No afiliado a la FIFA ni a ninguna casa de apuestas.*