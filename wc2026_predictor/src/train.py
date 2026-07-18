"""
train.py — Entrenamiento y validación walk-forward temporal
Compara múltiples modelos de ML usando validación temporal estricta
(entrena en pasado, evalúa en futuro) para evitar data leakage.

Modelos comparados:
  - Regresión Logística (ganador consistente)
  - Random Forest
  - XGBoost
  - LightGBM
  - GBM Calibrado
  - Random Forest Calibrado

Hallazgos clave del proyecto:
  - Las redes neuronales (MLP) obtienen el PEOR rendimiento con estos datos.
    Con ~9,600 partidos y alto ruido del fútbol, los modelos lineales calibrados
    superan consistentemente a arquitecturas complejas.
  - Las variables xG del torneo actual son descartadas sistemáticamente por
    Permutation Importance (muestra insuficiente: 4-8 partidos por equipo).
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from features import SELECTED_FEATURES

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_PKL = DATA_DIR / "trained_models.pkl"
SCALER_PKL = DATA_DIR / "scaler.pkl"
FEATS_TXT  = DATA_DIR / "selected_features.txt"


# ── Catálogo de modelos ────────────────────────────────────────────────────────
def get_model_catalog() -> dict:
    # Hiperparámetros optimizados en v7 (convergencia definitiva tras 7 rondas del torneo).
    # El ensemble Logistica + RF + Log_Calibrada es consistentemente el mejor en walk-forward.
    # Redes neuronales (MLP) fueron probadas en v4 y descartadas: log-loss 3.3-4.1 vs 0.86 de Logística.
    catalog = {
        "Logistica":     lambda: LogisticRegression(max_iter=2000, C=0.18, solver="lbfgs"),
        "RF":            lambda: RandomForestClassifier(n_estimators=800, max_depth=8,
                                                        min_samples_leaf=10, random_state=42),
        "Log_Calibrada": lambda: CalibratedClassifierCV(
                                     LogisticRegression(max_iter=2000, C=0.18), cv=3),
        "RF_Calibrado":  lambda: CalibratedClassifierCV(
                                     RandomForestClassifier(n_estimators=600, max_depth=7,
                                                            min_samples_leaf=12, random_state=42), cv=3),
        "GBM_Calibrado": lambda: CalibratedClassifierCV(
                                     GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                                learning_rate=0.035, random_state=42), cv=3),
    }
    if HAS_XGB:
        catalog["XGBoost"] = lambda: XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.018,
            subsample=0.8, colsample_bytree=0.7,
            eval_metric="mlogloss", random_state=42,
        )
    if HAS_LGB:
        catalog["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.018,
            subsample=0.8, random_state=42, verbose=-1,
        )
    return catalog


def _needs_scaling(name: str) -> bool:
    return any(k in name for k in ["Log", "Cal", "GBM"])


def _needs_label_encoding(name: str) -> bool:
    return any(k in name for k in ["XGBoost", "LightGBM"])


# ── Permutation Importance — selección de features ────────────────────────────
def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_ratio: float = 0.2,
    n_repeats: int = 10,
) -> list[str]:
    """
    Calcula Permutation Importance con RF sobre el set de test temporal
    y devuelve solo las features con importancia positiva.
    """
    split = int(len(X) * (1 - test_ratio))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    rf = RandomForestClassifier(n_estimators=400, max_depth=7,
                                min_samples_leaf=10, random_state=42)
    rf.fit(Xtr, ytr)

    perm = permutation_importance(rf, Xte, yte, n_repeats=n_repeats,
                                  random_state=42, scoring="accuracy")

    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    good = imp_df[imp_df["importance"] > 0]["feature"].tolist()
    bad  = imp_df[imp_df["importance"] <= 0]["feature"].tolist()

    print(f"\n📊 Permutation Importance:")
    print(imp_df.head(20).to_string(index=False))
    print(f"\n✅ Seleccionadas: {len(good)}  |  ❌ Descartadas: {len(bad)}")
    if bad:
        print(f"   Descartadas: {bad}")

    return good, imp_df


# ── Walk-forward validation ───────────────────────────────────────────────────
def walk_forward_validation(
    X:       np.ndarray,
    y:       np.ndarray,
    y_enc:   np.ndarray,
    le:      LabelEncoder,
    catalog: dict,
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Validación walk-forward temporal con `n_folds` particiones.
    En cada fold se entrena con partidos anteriores y se evalúa en partidos posteriores.
    """
    fold_size = len(X) // (n_folds + 1)
    scores = {name: {"acc": [], "ll": []} for name in catalog}

    print(f"\n🔄 Walk-forward validation ({n_folds} folds)...")
    for fold in range(1, n_folds + 1):
        Xt = X[:fold_size * fold];    yt = y[:fold_size * fold];    yte = y_enc[:fold_size * fold]
        Xv = X[fold_size * fold:fold_size * (fold + 1)]
        yv = y[fold_size * fold:fold_size * (fold + 1)]
        yve = y_enc[fold_size * fold:fold_size * (fold + 1)]

        sc = StandardScaler().fit(Xt)
        Xts, Xvs = sc.transform(Xt), sc.transform(Xv)

        for name, builder in catalog.items():
            m = builder()
            ns = _needs_scaling(name)
            ue = _needs_label_encoding(name)
            m.fit(Xts if ns else Xt, yte if ue else yt)
            p  = m.predict_proba(Xvs if ns else Xv)
            pr = m.predict(Xvs if ns else Xv)

            if ue:
                scores[name]["acc"].append(accuracy_score(yv, le.inverse_transform(pr)))
                scores[name]["ll"].append(log_loss(yve, p))
            else:
                scores[name]["acc"].append(accuracy_score(yv, pr))
                scores[name]["ll"].append(log_loss(yv, p, labels=m.classes_))

        print(f"   Fold {fold}/{n_folds} ✓")

    summary = sorted(
        [
            {
                "Modelo":    name,
                "Acc_mean":  np.mean(s["acc"]),
                "Acc_std":   np.std(s["acc"]),
                "LL_mean":   np.mean(s["ll"]),
                "LL_std":    np.std(s["ll"]),
            }
            for name, s in scores.items()
        ],
        key=lambda x: x["LL_mean"],
    )

    print("\n=== RESULTADOS WALK-FORWARD ===")
    for r in summary:
        print(f"  {r['Modelo']:20s}  Acc={r['Acc_mean']:.4f}±{r['Acc_std']:.4f}"
              f"  LL={r['LL_mean']:.4f}±{r['LL_std']:.4f}")

    return pd.DataFrame(summary)


# ── Entrenamiento del ensemble final ─────────────────────────────────────────
def train_final_ensemble(
    X:        np.ndarray,
    y:        np.ndarray,
    y_enc:    np.ndarray,
    le:       LabelEncoder,
    catalog:  dict,
    top_n:    int = 3,
    summary:  pd.DataFrame = None,
) -> tuple[dict, StandardScaler]:
    """
    Re-entrena los `top_n` mejores modelos con TODOS los datos disponibles
    para generar el ensemble final de predicción.
    """
    top_names = summary.head(top_n)["Modelo"].tolist()
    print(f"\n🏆 Ensemble top-{top_n}: {top_names}")

    scaler = StandardScaler().fit(X)
    fitted = {}

    for name in top_names:
        m  = catalog[name]()
        ns = _needs_scaling(name)
        ue = _needs_label_encoding(name)
        m.fit(scaler.transform(X) if ns else X, y_enc if ue else y)
        fitted[name] = (m, ns, ue)
        n_params = sum(
            p.size for p in [getattr(m, a, np.array([])) for a in
                              ["coef_", "feature_importances_", "estimators_"]]
        )
        print(f"  ✅ Entrenado: {name}")

    return fitted, scaler


# ── Función principal ─────────────────────────────────────────────────────────
def train(
    features_path: Path = DATA_DIR / "train_features.csv",
    output_dir:    Path = DATA_DIR,
    n_folds:       int  = 5,
    top_n:         int  = 3,
    cutoff:        str  = "2016-06-17",
) -> dict:
    """
    Pipeline completo de entrenamiento:
    1. Carga el dataset de features
    2. Aplica Permutation Importance para selección de variables
    3. Walk-forward validation con todos los modelos
    4. Re-entrena ensemble final con todos los datos
    5. Guarda modelos, scaler y features seleccionadas

    Returns:
        dict con modelos entrenados, scaler, features y resumen de validación
    """
    print(f"📂 Cargando features desde: {features_path}")
    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= cutoff].sort_values("date").reset_index(drop=True)

    # Usar SELECTED_FEATURES si están disponibles, sino usar todas las numéricas
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    if not available:
        available = [c for c in df.columns
                     if c not in ["date", "home_team", "away_team",
                                  "tournament", "result", "home_score", "away_score"]]

    print(f"📐 Features candidatas: {len(available)}")

    X = df[available].values
    y = df["result"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    catalog = get_model_catalog()

    # Feature selection
    good_feats, imp_df = select_features(X, y, available)
    X_sel = df[good_feats].values
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Walk-forward
    summary = walk_forward_validation(X_sel, y, le.transform(y), le, catalog, n_folds)
    summary.to_csv(output_dir / "model_comparison.csv", index=False)

    # Ensemble final
    fitted, scaler = train_final_ensemble(X_sel, y, le.transform(y), le, catalog, top_n, summary)

    # Persistir
    artifact = {
        "models":   fitted,
        "scaler":   scaler,
        "features": good_feats,
        "le":       le,
        "summary":  summary,
        "all_features": available,
    }
    with open(output_dir / "trained_models.pkl", "wb") as f:
        pickle.dump(artifact, f)

    with open(output_dir / "selected_features.txt", "w") as f:
        f.write("\n".join(good_feats))

    print(f"\n💾 Modelos guardados en: {output_dir / 'trained_models.pkl'}")
    print(f"   Features seleccionadas ({len(good_feats)}): {good_feats}")
    return artifact


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento WC2026 Predictor")
    parser.add_argument("--features", default=str(DATA_DIR / "train_features.csv"))
    parser.add_argument("--folds",    type=int, default=5)
    parser.add_argument("--top",      type=int, default=3,
                        help="Número de modelos en el ensemble")
    parser.add_argument("--cutoff",   default="2016-06-17",
                        help="Fecha mínima de partidos para entrenamiento")
    args = parser.parse_args()

    train(
        features_path=Path(args.features),
        n_folds=args.folds,
        top_n=args.top,
        cutoff=args.cutoff,
    )
