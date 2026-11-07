"""
entrenamiento.py - v14
- Predice log(costo/kg_total) y reconstruye costo = exp(pred) × kg_total
- Features basadas en la lógica del Excel (kg_total + n_componentes + tipo)
- 5-fold CV repetido 3×
"""
from pathlib import Path
import joblib, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA       = BASE_DIR / "data" / "dataset_limpio.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAS = ["familia", "cerradura", "base", "tipo_puerta"]


def mape(y, yp):
    return float(np.mean(np.abs((y - yp) / np.clip(y, 1, None))) * 100)


def pipe(modelo, num_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ])
    return Pipeline([("prep", pre), ("model", modelo)])


def modelos():
    return {
        "Ridge":  Ridge(alpha=5.0),
        "RF":     RandomForestRegressor(
                     n_estimators=800, max_depth=10, min_samples_leaf=2,
                     max_features=0.6, random_state=42, n_jobs=-1),
        "XGB":    XGBRegressor(
                     n_estimators=600, learning_rate=0.03, max_depth=3,
                     subsample=0.8, colsample_bytree=0.7,
                     reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3,
                     objective="reg:squarederror", random_state=42),
        "HistGB": HistGradientBoostingRegressor(
                     max_iter=400, learning_rate=0.04, max_depth=3,
                     min_samples_leaf=3, l2_regularization=2.0,
                     loss="squared_error", random_state=42),
    }


def cv_predict(X, y_log, kg_total, modelo, num_cols, kf):
    """Predice log(costo_kg) OOF y reconstruye costo total."""
    ylog = np.zeros(len(y_log))
    for tr, te in kf.split(X):
        p = pipe(modelo, num_cols); p.fit(X.iloc[tr], y_log[tr])
        yp = p.predict(X.iloc[te])
        lo, hi = np.percentile(y_log[tr], [2, 98])
        ylog[te] = np.clip(yp, lo, hi)
    return np.expm1(ylog) * kg_total


def entrenar():
    df = pd.read_csv(DATA)
    print(f"[INFO] Dataset: {len(df)} filas\n")

    DROP = ["costo_kg", "costo_kg_log", "costo_log"]
    X = df.drop(columns=DROP)
    num_cols = [c for c in X.columns if c not in CATEGORICAS]

    y_log     = df["costo_kg_log"].values
    kg_total  = df["kg_total"].values
    y_real    = np.expm1(y_log) * kg_total

    kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    resultados = []

    print("="*72)
    print("TARGET: log(costo/kg_total) → reconstruido × kg_total")
    print("="*72)
    for nombre, m in modelos().items():
        try:
            yp = cv_predict(X, y_log, kg_total, m, num_cols, kf)
            errs = np.abs((y_real - yp) / y_real) * 100
            r = dict(
                nombre=nombre, modelo=m, yp=yp,
                mape=float(errs.mean()),
                mediana=float(np.median(errs)),
                trim90=float(np.percentile(errs, 90)),
                mae=float(mean_absolute_error(y_real, yp)),
                r2=float(r2_score(y_real, yp)),
            )
            print(f"  [{nombre:8s}] MAPE {r['mape']:5.2f}% | Med {r['mediana']:5.2f}% "
                  f"| trim90 {r['trim90']:5.2f}% | R² {r['r2']:.3f}")
            resultados.append(r)
        except Exception as e:
            print(f"  [{nombre}] ERROR: {e}")

    # Ensemble ponderado
    pesos = np.array([1.0 / (r["mape"]**2) for r in resultados]); pesos /= pesos.sum()
    yp_ens = np.zeros(len(y_real))
    for r, w in zip(resultados, pesos):
        yp_ens += w * r["yp"]
    errs = np.abs((y_real - yp_ens) / y_real) * 100
    r_ens = dict(
        nombre="ENS", modelo=None, yp=yp_ens,
        mape=float(errs.mean()), mediana=float(np.median(errs)),
        trim90=float(np.percentile(errs, 90)),
        mae=float(mean_absolute_error(y_real, yp_ens)),
        r2=float(r2_score(y_real, yp_ens)),
    )
    print(f"  [ENS_pond] MAPE {r_ens['mape']:5.2f}% | Med {r_ens['mediana']:5.2f}% "
          f"| trim90 {r_ens['trim90']:5.2f}% | R² {r_ens['r2']:.3f}")
    resultados.append(r_ens)

    mejor = min(resultados, key=lambda r: r["mape"])
    print(f"\n[✓] GANADOR: {mejor['nombre']}")
    print(f"    MAPE {mejor['mape']:.2f}% | Mediana {mejor['mediana']:.2f}% "
          f"| R² {mejor['r2']:.3f}")
    print("="*72)

    tmp = X.copy()
    tmp["real"], tmp["pred"] = y_real, mejor["yp"]
    tmp["err"] = np.abs((tmp.real - tmp.pred) / tmp.real) * 100
    print("\nTop-5 peores:")
    for idx, row in tmp.nlargest(5, "err").iterrows():
        print(f"  idx={idx:3d}  kg={row.kg_total:7.1f}  "
              f"real=${row.real:>9,.0f}  pred=${row.pred:>9,.0f}  err={row.err:5.1f}%")

    # Reentrenar ganador sobre todo el dataset
    y_log_full = df["costo_kg_log"].values
    clip_lo, clip_hi = np.percentile(y_log_full, [2, 98])

    if mejor["modelo"] is None:
        pipes = []
        for r, w in zip(resultados[:-1], pesos):
            p = pipe(r["modelo"], num_cols); p.fit(X, y_log_full)
            pipes.append((p, float(w)))
        artefacto = {"ensemble_weighted": pipes}
    else:
        p = pipe(mejor["modelo"], num_cols); p.fit(X, y_log_full)
        artefacto = p

    objetivo = mejor["mape"] <= 15 and mejor["mediana"] <= 15

    joblib.dump({
        "model": artefacto,
        "nombre": mejor["nombre"],
        "target": "costo_kg_log",
        "es_ensemble": mejor["modelo"] is None,
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "mape_global": mejor["mape"],
        "mape_mediana": mejor["mediana"],
        "mape_trimmed_90": mejor["trim90"],
        "r2": mejor["r2"],
        "objetivo_cumplido": bool(objetivo),
        "features": list(X.columns),
        "categoricas": CATEGORICAS,
    }, MODELS_DIR / "modelo_costo_fabricacion.joblib")

    print(f"\n[{'✓' if objetivo else '!'}] Objetivo MAPE ≤15%: {'CUMPLIDO' if objetivo else 'NO'}")
    print(f"[OK] → {MODELS_DIR}")


if __name__ == "__main__":
    entrenar()