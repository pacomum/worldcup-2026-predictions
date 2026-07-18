# CHANGELOG — WC2026 Predictor

Historial de versiones del modelo, iterado ronda a ronda durante el torneo.

---

## v7 — Predicciones Final + Tercer lugar (julio 2026)

**Resultados semifinales (v6): 2/2 (100%)**
- España 2-0 Francia ✓ (predijo España 52.7%)
- Argentina 2-1 Inglaterra ✓ (predijo Argentina 60.5%, remontada 87'+90+2')

**Cambios técnicos:**
- Feature selection v7: convergencia definitiva a **14 variables** (de 22 candidatas)
- Variables descartadas definitivamente: `draw_a`, `gd_a`, `form_a`, `streak_diff`, `streak_h`, `form_h`, `draw_h`, `underdog` — importancia ≤0 en todos los folds
- Hiperparámetros ajustados: C=0.18 (Logística), n_estimators=800 (RF), lr=0.035 (GBM)
- Accuracy walk-forward: 60.5% | Log-Loss: 0.864

**Predicciones:**
- 🏆 Final: España vs Argentina → **Coinflip perfecto** (49.9% España / 50.1% Argentina)
- 🥉 3er lugar: Francia vs Inglaterra → **Francia** (57.1%)

**Archivos nuevos:**
- `data/fixtures/final.csv`, `data/results/comparacion_semis.csv`
- `data/results/predicciones_final_v7.csv`
- `reports/v7_Final_y_3erLugar.xlsx`

---

## v6 — Predicciones Semifinales (julio 2026)

**Resultados cuartos (v5): 4/4 (100%) 🏆**
- Francia 2-0 Marruecos ✓, España 2-1 Bélgica ✓
- Inglaterra 2-1 Noruega (TE) ✓, Argentina 3-1 Suiza (TE) ✓

**Cambios técnicos:**
- xG ponderado por ronda (grupos=1.0, 16avos=1.5, octavos=2.0, cuartos=2.5)
- Nueva feature: `elo_trend` (cambio de Elo durante el torneo)
- Nueva feature: `ko_record` (% victorias en KO del torneo actual)
- Feature selection: 20 variables seleccionadas
- Accuracy walk-forward: 60.7% | Log-Loss: 0.869

**Predicciones:**
- Francia vs España → **España** (52.7%)
- Argentina vs Inglaterra → **Argentina** (60.5%)

---

## v5 — Predicciones Cuartos de final (julio 2026)

**Resultados octavos (v4): 6/8 (75.0%)**
- Fallos: Brasil-Noruega (sorpresa total, xG Brasil 2.36 vs 1.16)
- Colombia-Suiza (0-0 → penales, Suiza avanzó 4-3)

**Cambios técnicos:**
- Permutation Importance formal sobre set de test temporal
- Primera selección rigurosa de features: de 60 a 29 variables
- Descubrimiento clave: variables xG del torneo actual son sistemáticamente descartadas
- Las mismas 20 features base de v5 persisten en v6 y v7
- Accuracy walk-forward: 60.6% | Log-Loss: 0.867

---

## v4 — Predicciones Octavos de final (julio 2026)

**Resultados 16avos (v3): 12/16 (75.0%)**
- Fallos: Alemania-Paraguay penales, Países Bajos-Marruecos penales, Australia-Egipto penales
- México sorprendió a Ecuador (Ecuador venía de ganar a Alemania)

**Cambios técnicos:**
- Redes neuronales (MLP 2/3/4 capas) probadas y **descartadas** (log-loss 3.3-4.1)
- Nuevas features: `streak` (racha), `form_h3` (forma 3 partidos), términos cuadráticos de Elo
- xG del torneo incorporado como features (luego descartado en v5 por Permutation Importance)
- 60 features totales candidatas
- Accuracy walk-forward: 60.6% | Log-Loss: 0.868

---

## v3 — Predicciones 16avos de final (junio 2026)

**Resultados J3 Grupos (v2): 18/28 (64.3%)**
- 7 empates no predichos (J3 estratégica: equipos clasificados rotan)

**Cambios técnicos:**
- xG del Mundial 2026 incorporado por primera vez (grupos + 16avos)
- Nueva feature: `ko_flag` (eliminatoria directa)
- Calibración isotónica (CalibratedClassifierCV)
- 40 features totales
- Ensemble: Logística + RF + Logística Calibrada

---

## v2 — Predicciones Jornada 3 de Grupos (junio 2026)

**Resultados J2 Grupos (v1): 17/24 (70.8%)**
- 5 de 7 fallos fueron empates no predichos
- Patrón: modelo sobreestimaba al favorito con Elo moderadamente superior

**Cambios técnicos:**
- `draw_risk_index`: índice compuesto de riesgo de empate
- `draw_rate_home/away`: tasa histórica de empates por equipo
- `form_std`: consistencia de la forma reciente
- `wc_form`: rendimiento específico en Copas del Mundo
- `h2h_weighted`: H2H ponderado por recencia
- De 8 a 30 features
- Ensemble: XGBoost añadido al ensemble

---

## v1 — Predicciones Fase de Grupos J1-J2 (junio 2026)

**Modelo inicial:**
- Dataset: 9,528 partidos (últimos 10 años, 2016-2026)
- Fuente de datos: martj42/international_results (GitHub)
- Rating Elo calculado desde cero (metodología eloratings.net)
- 8 features base: `elo_diff`, `elo_ratio`, `we_home`, `form_diff`, `gd_avg_diff`, `h2h`, `neutral`, `is_world_cup`
- Validación: walk-forward 5 folds
- Mejor modelo: Regresión Logística (log-loss 0.857, accuracy 60.7%)
- Accuracy J1: 17/24 (70.8%), J2: 17/24 (70.8%)
