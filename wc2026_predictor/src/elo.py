"""
elo.py — Cálculo de Rating Elo para selecciones nacionales
Replica la metodología de eloratings.net:
  - Factor K por importancia del torneo
  - Ajuste por margen de gol (factor G)
  - Ventaja de local (+100 puntos en Elo esperado)
  - Rating inicial: 1500 para cualquier selección sin historial
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Rutas por defecto ──────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"
RESULTS_CSV = DATA_DIR / "results.csv"
ELO_CSV     = DATA_DIR / "elo_current.csv"
STATE_PKL   = DATA_DIR / "elo_state.pkl"

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# ── Constantes ─────────────────────────────────────────────────────────────────
INITIAL_ELO = 1500
HOME_ADV    = 100   # ventaja de local en puntos Elo


# ── Peso K por tipo de torneo ──────────────────────────────────────────────────
def tournament_k(tournament: str) -> int:
    """Devuelve el factor K según la importancia del torneo."""
    t = tournament.lower()
    if "world cup" in t and "qualification" not in t:
        return 60
    if any(
        x in t
        for x in [
            "euro", "copa am", "african cup", "afc asian",
            "gold cup", "concacaf nations", "confederations",
        ]
    ) and "qualification" not in t:
        return 50
    if "nations league" in t and "qualification" not in t:
        return 35
    if "qualification" in t:
        return 40
    if "friendly" in t:
        return 20
    return 30


# ── Factor G (margen de gol) ───────────────────────────────────────────────────
def goal_diff_factor(goal_diff: int) -> float:
    """Factor multiplicador por margen de gol (metodología eloratings.net)."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


# ── Probabilidad Elo esperada ──────────────────────────────────────────────────
def expected_win_prob(elo_home: float, elo_away: float, neutral: bool) -> float:
    """
    Probabilidad esperada de victoria del local dado el Elo de ambos equipos.
    En cancha neutral no se aplica la ventaja de local.
    """
    home_adv = 0 if neutral else HOME_ADV
    dr = (elo_home + home_adv) - elo_away
    return 1 / (10 ** (-dr / 400) + 1)


# ── Motor de cálculo Elo ───────────────────────────────────────────────────────
class EloEngine:
    """
    Calcula y actualiza ratings Elo para todos los equipos
    procesando el historial de partidos en orden cronológico.
    """

    def __init__(self, initial_elo: float = INITIAL_ELO):
        self.initial_elo = initial_elo
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.initial_elo)

    def update(self, row: pd.Series) -> dict:
        """
        Procesa un partido y actualiza los ratings de ambos equipos.
        Devuelve un dict con los ratings PRE-partido (usados como features).
        """
        home, away    = row["home_team"], row["away_team"]
        hs, as_       = int(row["home_score"]), int(row["away_score"])
        neutral       = bool(row["neutral"])
        tournament    = row["tournament"]
        date          = row["date"]

        r_home = self.get(home)
        r_away = self.get(away)

        we_home = expected_win_prob(r_home, r_away, neutral)

        # Resultado real desde perspectiva del local
        if hs > as_:
            w_home = 1.0
        elif hs == as_:
            w_home = 0.5
        else:
            w_home = 0.0

        gd    = abs(hs - as_)
        g     = goal_diff_factor(gd)
        k     = tournament_k(tournament)
        delta = k * g * (w_home - we_home)

        # Snapshot PRE-actualización (lo que el modelo ve antes del partido)
        snapshot = {
            "date":       date,
            "home_team":  home,
            "away_team":  away,
            "elo_home":   r_home,
            "elo_away":   r_away,
            "we_home":    we_home,
            "neutral":    1 if neutral else 0,
            "tournament": tournament,
        }

        # Actualizar ratings
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta
        self.history.append({
            "date": date, "team": home, "elo": self.ratings[home],
        })
        self.history.append({
            "date": date, "team": away, "elo": self.ratings[away],
        })

        return snapshot

    def current_table(self) -> pd.DataFrame:
        """Tabla de ratings actuales ordenada de mayor a menor."""
        df = pd.DataFrame(
            [{"team": t, "elo": r} for t, r in self.ratings.items()]
        ).sort_values("elo", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


# ── Función principal ──────────────────────────────────────────────────────────
def compute_elo(
    results_path: Path = RESULTS_CSV,
    save_current: Path = ELO_CSV,
    save_state:   Path = STATE_PKL,
) -> tuple[EloEngine, pd.DataFrame]:
    """
    Lee el CSV de resultados, calcula Elo acumulado y guarda:
      - elo_current.csv  → tabla de ratings al día de hoy
      - elo_state.pkl    → estado completo del engine (para reutilizar en features)
    Devuelve (engine, snapshots_df).
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

    engine    = EloEngine()
    snapshots = []
    for _, row in played.iterrows():
        snap = engine.update(row)
        snapshots.append(snap)

    current = engine.current_table()
    current.to_csv(save_current, index=False)
    print(f"✅ Elo calculado para {len(engine.ratings)} selecciones.")
    print(f"   → Guardado en: {save_current}")
    print(current.head(20).to_string(index=False))

    with open(save_state, "wb") as f:
        pickle.dump(engine, f)
    print(f"   → Estado guardado en: {save_state}")

    return engine, pd.DataFrame(snapshots)


# ── Descarga del dataset ───────────────────────────────────────────────────────
def download_results(dest: Path = RESULTS_CSV) -> None:
    """Descarga el CSV histórico de martj42/international_results desde GitHub."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Descargando datos históricos desde GitHub...")
    r = requests.get(RESULTS_URL, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    lines = dest.read_text().count("\n")
    print(f"✅ Descargado: {lines:,} partidos → {dest}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cálculo de Rating Elo — WC2026 Predictor")
    parser.add_argument("--download", action="store_true", help="Descargar datos históricos desde GitHub")
    parser.add_argument("--results",  default=str(RESULTS_CSV), help="Ruta al CSV de resultados")
    args = parser.parse_args()

    if args.download:
        download_results(Path(args.results))

    compute_elo(Path(args.results))
