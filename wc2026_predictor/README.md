# ⚽ Predicción de Resultados — Copa Mundial FIFA 2026

Sistema de machine learning end-to-end para predecir resultados de partidos de la Copa Mundial FIFA 2026, desde la extracción de datos históricos hasta la generación de reportes de predicción por ronda.

## 📋 Descripción

Este proyecto construye un pipeline completo de ciencia de datos para pronosticar resultados de partidos de fútbol internacional, aplicado específicamente al Mundial 2026. Combina ingeniería de características basada en fútbol (ratings Elo, forma reciente, head-to-head), validación temporal rigurosa para evitar fuga de datos, y comparación sistemática de modelos de machine learning.

El proyecto se actualiza de forma iterativa conforme avanza el torneo: cada ronda (Fase de Grupos, Dieciseisavos, Octavos, Cuartos, Semifinales) se evalúa contra resultados reales, y los aprendizajes retroalimentan la siguiente versión del modelo.

## 🎯 Objetivo

Predecir el resultado de partidos de fútbol internacional (victoria local / empate / victoria visitante) con un enfoque reproducible, documentado y basado en datos históricos verificables — no en opinión ni en estadísticas de "hype" mediático.

## 📊 Resultados por ronda

| Ronda | Modelo | Aciertos | Accuracy |
|-------|--------|----------|----------|
| Fase de grupos J1 | v1 | 17/24 | 70.8% |
| Fase de grupos J2 | v1 | 17/24 | 70.8% |
| Fase de grupos J3 | v2 | 18/28 | 64.3% |
| 16avos de final   | v3 | 12/16 | 75.0% |
| Octavos de final  | v4 | 6/8   | 75.0% |
| Cuartos de final  | v5 | 4/4   | **100.0%** |

## 🏗️ Estructura del proyecto

```
wc2026_predictor/
├── data/
│   ├── results.csv              # Dataset histórico de partidos (martj42/international_results)
│   ├── elo_current.csv          # Rating Elo actual post-torneo
│   └── xg_wc2026.csv           # Estadísticas xG del Mundial 2026
├── src/
│   ├── elo.py                   # Cálculo de rating Elo desde cero (metodología eloratings.net)
│   ├── features.py              # Ingeniería de características (20 variables seleccionadas)
│   ├── train.py                 # Entrenamiento y validación walk-forward temporal
│   └── predict.py               # Generación de predicciones por ronda
├── reports/                     # Reportes Excel de predicción por ronda
├── notebooks/
│   └── 01_EDA.ipynb             # Exploración y análisis de datos
└── README.md
```

## 🔬 Metodología

### Datos
- **Fuente principal:** [martj42/international_results](https://github.com/martj42/international_results) — 49,000+ partidos internacionales (1872–2026)
- **xG del torneo:** xGscore.io / RealGM (partidos del Mundial 2026)
- **Período de entrenamiento:** últimos 10 años (2016–2026), ~9,600 partidos

### Rating Elo
Implementación propia replicando la metodología de [eloratings.net](https://www.eloratings.net):
- Factor K por importancia del torneo (Mundial=60, Clasificatorias=40, Amistosos=20)
- Ajuste por margen de gol (factor G)
- Ventaja de local (+100 puntos)

### Features (20 variables — seleccionadas por Permutation Importance)
| # | Feature | Descripción |
|---|---------|-------------|
| 1 | `we_h` | Probabilidad Elo esperada del local |
| 2 | `elo_ratio` | Cociente de Elo entre equipos |
| 3 | `elo_diff` | Diferencia de Elo |
| 4 | `conf_diff` | Diferencial de confederación |
| 5 | `conf_a` | Confederación del visitante |
| 6 | `elo_a` | Elo absoluto del visitante |
| 7 | `form_h` | Forma reciente local (últimos 5 partidos) |
| 8 | `gf_a` | Goles promedio del visitante |
| 9 | `h2h` | Head-to-head ponderado por recencia |
| 10 | `underdog` | Flag: visitante tiene Elo mayor |
| 11 | `conf_prod` | Producto de confederaciones |
| 12 | `streak_h` | Racha local (victorias/derrotas consecutivas) |
| 13 | `streak_diff` | Diferencia de rachas |
| 14 | `gd_a` | Diferencial de goles del visitante |
| 15 | `form_diff` | Diferencia de forma reciente |
| 16 | `wc_diff` | Diferencial de rendimiento en Mundiales |
| 17 | `form_a` | Forma reciente del visitante (últimos 5 partidos) |
| 18 | `form_h3` | Forma ultrareciente local (últimos 3 partidos) |
| 19 | `draw_a` | Tasa histórica de empates del visitante |
| 20 | `wc_a` | Rendimiento histórico del visitante en Mundiales |

### Modelo
- **Tipo:** Ensemble soft-voting de los 3 mejores modelos (validación walk-forward)
- **Validación:** Walk-forward temporal (5 folds) — sin fuga de datos del futuro
- **Modelos comparados:** Logística, Random Forest, XGBoost, LightGBM, GBM, MLP (redes neuronales)
- **Ganador consistente:** Regresión Logística calibrada (menor log-loss en todos los folds)

### Hallazgos clave
- Las **redes neuronales (MLP)** obtienen el peor rendimiento (log-loss 3.3–4.1 vs 0.87 de Logística). Con ~9,600 partidos y alto ruido del fútbol, los modelos lineales bien calibrados superan a arquitecturas complejas.
- Las **variables xG del torneo actual** son descartadas sistemáticamente por Permutation Importance. Con 4-8 partidos de muestra por equipo, no generalizan al modelo histórico.
- El **Elo y la forma histórica** son las señales más robustas y consistentes en todos los experimentos.

## 🚀 Uso rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos históricos
python src/elo.py --download

# 3. Entrenar modelo
python src/train.py

# 4. Generar predicciones para una ronda
python src/predict.py --fixture data/fixtures/semis.csv --output reports/
```

## 📦 Dependencias

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
openpyxl>=3.1
requests>=2.31
```

## 📁 Datos

El archivo `data/results.csv` no se incluye en el repositorio por tamaño. Se descarga automáticamente al ejecutar:

```bash
python src/elo.py --download
```

Fuente: https://github.com/martj42/international_results

## 🔄 Workflow por ronda

```
Nueva ronda disponible
        ↓
git pull (datos actualizados)
        ↓
python src/elo.py          → Recalcula ratings Elo
        ↓
python src/features.py     → Construye features del fixture
        ↓
python src/train.py        → Re-entrena y compara modelos
        ↓
python src/predict.py      → Genera predicciones + Excel
        ↓
Guardar reporte en reports/
```

## 📄 Licencia

MIT License — datos de partidos históricos bajo licencia de [martj42/international_results](https://github.com/martj42/international_results).


## 👤 Autor

**José Francisco Muñoz Martínez**
Analista de Datos Sr | Ciencia de Datos & Machine Learning
[LinkedIn](https://www.linkedin.com/in/franciscomu%C3%B1ozm/) · franciscomumz@gmail.com

---
*Proyecto personal desarrollado con fines de aprendizaje y portafolio. No afiliado a la FIFA ni a ninguna casa de apuestas.*