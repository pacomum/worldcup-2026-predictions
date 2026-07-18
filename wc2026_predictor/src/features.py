"""
features.py — Ingeniería de características para predicción de partidos
Construye las 20 variables seleccionadas por Permutation Importance
a partir del historial de partidos y el estado Elo calculado.

Todas las features se calculan con información ANTERIOR al partido
para evitar fuga de datos (data leakage).
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from elo import EloEngine, INITIAL_ELO, HOME_ADV, tournament_k, goal_diff_factor

DATA_DIR  = Path(__file__).parent.parent / "data"
STATE_PKL = DATA_DIR / "elo_state.pkl"

# ── Confederaciones y sus fortalezas relativas ────────────────────────────────
CONFEDERATIONS: dict[str, list[str]] = {
    "conmebol": [
        "Argentina", "Brazil", "Uruguay", "Colombia", "Chile",
        "Peru", "Ecuador", "Venezuela", "Paraguay", "Bolivia",
    ],
    "uefa": [
        "France", "Spain", "England", "Germany", "Portugal", "Netherlands",
        "Belgium", "Italy", "Croatia", "Switzerland", "Austria", "Scotland",
        "Norway", "Sweden", "Denmark", "Czech Republic", "Poland", "Turkey",
        "Serbia", "Bosnia and Herzegovina", "Albania", "Iceland", "Romania",
        "Ukraine", "Georgia", "Slovenia",
    ],
    "caf": [
        "Morocco", "Senegal", "Ivory Coast", "Ghana", "Algeria", "Tunisia",
        "Egypt", "Nigeria", "Cameroon", "DR Congo", "South Africa", "Cape Verde",
    ],
    "concacaf": [
        "United States", "Mexico", "Canada", "Jamaica", "Costa Rica",
        "Panama", "Honduras", "El Salvador", "Haiti", "Curaçao",
    ],
    "afc": [
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
        "Qatar", "Uzbekistan", "Iraq", "Jordan",
    ],
    "ofc": ["New Zealand"],
}
TEAM_CONF: dict[str, str] = {
    team: conf for conf, teams in CONFEDERATIONS.items() for team in teams
}
CONF_STRENGTH: dict[str, float] = {
    "conmebol": 1.00,
    "uefa":     0.98,
    "caf":      0.80,
    "afc":      0.75,
    "concacaf": 0.72,
    "ofc":      0.55,
}

# Features finales — 14 variables seleccionadas por Permutation Importance (v7, convergencia definitiva)
# Tras 7 versiones del modelo, estas 14 variables son estables y consistentes.
# Variables descartadas en todos los experimentos (importancia ≤0):
#   draw_a, gd_a, form_a, streak_diff, streak_h, form_h, draw_h, underdog
SELECTED_FEATURES = [
    "we_h",       # #1 — probabilidad Elo esperada, la más informativa
    "elo_diff",   # diferencia de Elo
    "elo_ratio",  # cociente de Elo (captura no-linealidad)
    "gf_a",       # goles promedio del visitante
    "conf_diff",  # diferencial de confederación
    "elo_a",      # Elo absoluto del visitante
    "wc_a",       # rendimiento histórico visitante en Mundiales
    "form_diff",  # diferencia de forma reciente (5 partidos)
    "conf_prod",  # producto de fortalezas de confederación
    "conf_a",     # confederación del visitante
    "h2h",        # head-to-head ponderado por recencia
    "form_h3",    # forma ultrareciente local (últimos 3 partidos)
    "draw_risk",  # índice compuesto de riesgo de empate
    "wc_diff",    # diferencial de rendimiento en Mundiales histórico
]


# ── Estado mutable (se actualiza partido a partido) ───────────────────────────
class FeatureState:
    """
    Mantiene el estado actualizado de las estadísticas por equipo
    para construir features en tiempo real (sin fuga de datos futuros).
    """

    def __init__(self):
        self.form:     dict[str, list[int]]   = {}  # puntos por partido
        self.gfor:     dict[str, list[int]]   = {}  # goles anotados
        self.gagainst: dict[str, list[int]]   = {}  # goles encajados
        self.drawh:    dict[str, list[int]]   = {}  # 1 si empate, 0 si no
        self.wcr:      dict[str, list[int]]   = {}  # puntos en partidos de WC
        self.streak:   dict[str, int]         = {}  # racha actual
        self.h2h:      dict[tuple, list]      = {}  # historial h2h

    # ── Getters ───────────────────────────────────────────────────────────────
    def get_form(self, team: str, n: int = 5) -> float:
        f = self.form.get(team, [])
        return float(np.mean(f[-n:])) if f else 1.0

    def get_form_std(self, team: str, n: int = 5) -> float:
        f = self.form.get(team, [])
        return float(np.std(f[-n:])) if len(f) >= 2 else 1.0

    def get_goals_for(self, team: str, n: int = 8) -> float:
        g = self.gfor.get(team, [])[-n:]
        return float(np.mean(g)) if g else 1.0

    def get_goals_against(self, team: str, n: int = 8) -> float:
        g = self.gagainst.get(team, [])[-n:]
        return float(np.mean(g)) if g else 1.0

    def get_draw_rate(self, team: str, n: int = 10) -> float:
        d = self.drawh.get(team, [])[-n:]
        return float(np.mean(d)) if d else 0.22

    def get_wc_form(self, team: str, n: int = 6) -> float:
        w = self.wcr.get(team, [])
        return float(np.mean(w[-n:])) if w else 1.0

    def get_streak(self, team: str) -> int:
        return self.streak.get(team, 0)

    def get_h2h(self, home: str, away: str, n: int = 8) -> float:
        key = tuple(sorted([home, away]))
        lst = self.h2h.get(key, [])
        if not lst:
            return 0.5
        pts = []
        for i, (h_, a_, hs_, as_) in enumerate(lst[-n:]):
            w = 0.5 + 0.5 * (i / max(len(lst[-n:]) - 1, 1))
            won = (h_ == home and hs_ > as_) or (a_ == home and as_ > hs_)
            draw = hs_ == as_
            pts.append(w * (3 if won else (1 if draw else 0)))
        return float(np.sum(pts) / (3 * len(pts))) if pts else 0.5

    # ── Actualizar estado tras un partido ─────────────────────────────────────
    def update(self, home: str, away: str, hs: int, as_: int, is_wc: bool) -> None:
        ph = 3 if hs > as_ else (1 if hs == as_ else 0)
        pa = 3 - ph if hs != as_ else 1

        self.form.setdefault(home, []).append(ph)
        self.form.setdefault(away, []).append(pa)
        self.gfor.setdefault(home, []).append(hs)
        self.gfor.setdefault(away, []).append(as_)
        self.gagainst.setdefault(home, []).append(as_)
        self.gagainst.setdefault(away, []).append(hs)
        self.drawh.setdefault(home, []).append(1 if hs == as_ else 0)
        self.drawh.setdefault(away, []).append(1 if hs == as_ else 0)

        prev_h = self.streak.get(home, 0)
        prev_a = self.streak.get(away, 0)
        self.streak[home] = prev_h + 1 if ph == 3 else (-1 if ph == 0 else 0)
        self.streak[away] = prev_a + 1 if pa == 3 else (-1 if pa == 0 else 0)

        if is_wc:
            self.wcr.setdefault(home, []).append(ph)
            self.wcr.setdefault(away, []).append(pa)

        key = tuple(sorted([home, away]))
        self.h2h.setdefault(key, []).append((home, away, hs, as_))


# ── Construcción de features para un partido ─────────────────────────────────
def build_row(
    home:    str,
    away:    str,
    elo_h:   float,
    elo_a:   float,
    neutral: bool,
    is_wc:   bool,
    state:   FeatureState,
) -> dict:
    """
    Construye el vector de features PRE-partido para un enfrentamiento dado.
    Usa el estado actual del FeatureState (no incluye el resultado de este partido).
    """
    home_adv = 0 if neutral else HOME_ADV
    dr  = elo_h - elo_a
    we  = 1 / (10 ** (-((elo_h + home_adv) - elo_a) / 400) + 1)
    eabs = abs(dr)

    ch = CONF_STRENGTH.get(TEAM_CONF.get(home, "unknown"), 0.65)
    ca = CONF_STRENGTH.get(TEAM_CONF.get(away, "unknown"), 0.65)

    gf_a = state.get_goals_for(away)
    gc_a = state.get_goals_against(away)

    return {
        # Elo
        "we_h":       we,
        "elo_diff":   dr,
        "elo_ratio":  elo_h / max(elo_a, 1),
        "elo_h":      elo_h,
        "elo_a":      elo_a,
        # Confederación
        "conf_h":     ch,
        "conf_a":     ca,
        "conf_diff":  ch - ca,
        "conf_prod":  ch * ca,
        # Forma local
        "form_h":     state.get_form(home, 5),
        "form_h3":    state.get_form(home, 3),
        "form_std_h": state.get_form_std(home),
        "streak_h":   state.get_streak(home),
        # Forma visitante
        "form_a":     state.get_form(away, 5),
        "form_a3":    state.get_form(away, 3),
        "form_std_a": state.get_form_std(away),
        "streak_a":   state.get_streak(away),
        # Difs forma y racha
        "form_diff":   state.get_form(home, 5) - state.get_form(away, 5),
        "streak_diff": state.get_streak(home) - state.get_streak(away),
        # Goles
        "gf_h":   state.get_goals_for(home),
        "gf_a":   gf_a,
        "gc_h":   state.get_goals_against(home),
        "gc_a":   gc_a,
        "gd_h":   state.get_goals_for(home) - state.get_goals_against(home),
        "gd_a":   gf_a - gc_a,
        "gd_diff": (state.get_goals_for(home) - state.get_goals_against(home))
                   - (gf_a - gc_a),
        # Empates
        "draw_h":    state.get_draw_rate(home),
        "draw_a":    state.get_draw_rate(away),
        "draw_risk": (state.get_draw_rate(home) + state.get_draw_rate(away)) / 2
                     + 1 / (1 + eabs / 200),
        # H2H
        "h2h": state.get_h2h(home, away),
        # Copa del Mundo
        "wc_h":    state.get_wc_form(home),
        "wc_a":    state.get_wc_form(away),
        "wc_diff": state.get_wc_form(home) - state.get_wc_form(away),
        # Contexto
        "neutral":  1 if neutral else 0,
        "is_wc":    1 if is_wc else 0,
        "underdog": 1 if elo_a > elo_h else 0,
    }


# ── Construcción del dataset de entrenamiento completo ────────────────────────
def build_training_dataset(
    results_path: Path,
    cutoff_date:  str = "2016-06-17",
) -> pd.DataFrame:
    """
    Procesa todos los partidos históricos en orden cronológico y construye
    el dataset de entrenamiento con features calculadas PRE-partido.

    Solo incluye partidos desde `cutoff_date` en el dataset de entrenamiento,
    pero el estado Elo y estadísticas se construyen desde el inicio del historial.
    """
    df = pd.read_csv(results_path)
    df["date"] = pd.to_datetime(df["date"])
    played = (
        df.dropna(subset=["home_score", "away_score"])
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    cutoff = pd.Timestamp(cutoff_date)
    engine = EloEngine()
    state  = FeatureState()
    rows   = []

    for _, row in played.iterrows():
        home, away    = row["home_team"], row["away_team"]
        hs, as_       = row["home_score"], row["away_score"]
        neutral       = bool(row["neutral"])
        date          = row["date"]
        tournament    = row["tournament"]
        is_wc         = "World Cup" in tournament and "qualification" not in tournament

        elo_h = engine.get(home)
        elo_a = engine.get(away)

        # Construir features ANTES de actualizar el estado
        if date >= cutoff:
            feat = build_row(home, away, elo_h, elo_a, neutral, is_wc, state)
            feat.update({
                "date":       date,
                "home_team":  home,
                "away_team":  away,
                "tournament": tournament,
                "home_score": hs,
                "away_score": as_,
            })
            rows.append(feat)

        # Actualizar estado DESPUÉS de registrar las features
        engine.update(row)
        state.update(home, away, hs, as_, is_wc)

    df_out = pd.DataFrame(rows)

    # Target: H=gana local, D=empate, A=gana visitante
    df_out["result"] = df_out.apply(
        lambda r: "H" if r["home_score"] > r["away_score"]
        else ("D" if r["home_score"] == r["away_score"] else "A"),
        axis=1,
    )

    print(f"✅ Dataset construido: {df_out.shape}")
    print(f"   Distribución: {df_out['result'].value_counts(normalize=True).round(3).to_dict()}")
    return df_out, engine, state


# ── Features para un fixture pendiente ────────────────────────────────────────
def build_fixture_features(
    fixture:  pd.DataFrame,
    engine:   EloEngine,
    state:    FeatureState,
) -> pd.DataFrame:
    """
    Construye el vector de features para un fixture de partidos pendientes.

    Args:
        fixture: DataFrame con columnas [home_team, away_team, date, neutral]
        engine:  EloEngine con los ratings actualizados
        state:   FeatureState con el estado actualizado

    Returns:
        DataFrame con las features de cada partido
    """
    rows = []
    for _, row in fixture.iterrows():
        home    = row["home_team"]
        away    = row["away_team"]
        neutral = bool(row.get("neutral", True))
        is_wc   = bool(row.get("is_world_cup", True))
        elo_h   = engine.get(home)
        elo_a   = engine.get(away)

        feat = build_row(home, away, elo_h, elo_a, neutral, is_wc, state)
        feat.update({
            "date":       row.get("date", ""),
            "home_team":  home,
            "away_team":  away,
            "city":       row.get("city", ""),
        })
        rows.append(feat)

    return pd.DataFrame(rows)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_path = DATA_DIR / "results.csv"
    df_train, engine, state = build_training_dataset(results_path)
    df_train.to_csv(DATA_DIR / "train_features.csv", index=False)
    print(f"💾 Features guardadas en: {DATA_DIR / 'train_features.csv'}")
    print(df_train[SELECTED_FEATURES + ["result"]].head(3))
