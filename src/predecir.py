"""predecir.py - v14"""
from pathlib import Path
import joblib, numpy as np, pandas as pd, re

BASE = Path(__file__).resolve().parent.parent
ARTEFACTO = joblib.load(BASE / "models" / "modelo_costo_fabricacion.joblib")

FAMILIA_MAP   = {"Locker":"Lockers","LK":"Lockers","Storage":"Storage",
                 "Archivo":"Lockers","Ropero":"Ropero","Lockers Oficina":"Lockers Oficina"}
CERRADURA_MAP = {"Cerradura estándar":"Cerradura portacandado",
                 "Cerradura guantera":"Cerradura portacandado",
                 "Cerradura españoleta":"Españoleta"}
PUERTA_MAP    = {"Puerta lisa":"Puerta Lisa","Puerta malla":"Puerta Malla",
                 "Puerta CNC láser":"Puerta Lisa"}


def _predict_log(X):
    model = ARTEFACTO["model"]
    if ARTEFACTO.get("es_ensemble"):
        y = np.zeros(len(X))
        for p, w in model["ensemble_weighted"]:
            y += w * np.asarray(p.predict(X))
    else:
        y = np.asarray(model.predict(X))
    return np.clip(y, ARTEFACTO["clip_lo"], ARTEFACTO["clip_hi"])


def predecir(d: dict) -> float:
    f = FAMILIA_MAP.get(d["familia"], "Lockers")
    c = CERRADURA_MAP.get(d["cerradura"], d["cerradura"])
    t = PUERTA_MAP.get(d["tipo_puerta"], d["tipo_puerta"])

    txt = str(d.get("producto_txt", "")).lower()
    mc = re.search(r'(\d+)\s*cuerpo', txt)
    mp = re.search(r'(\d+)\s*puerta', txt)
    n_cuerpos = int(mc.group(1)) if mc else int(d.get("n_cuerpos", 1))
    n_puertas = int(mp.group(1)) if mp else int(d.get("n_puertas", 1))
    n_cuerpos = max(1, n_cuerpos); n_puertas = max(1, n_puertas)
    n_puertas_total = n_puertas * n_cuerpos

    bandejas   = int(d.get("bandejas", 0))
    techo      = int(d.get("techo_inclinado", 0))
    divisiones = int(d.get("divisiones", 1))

    # kg_total: el dato más importante. Si el usuario lo provee, usarlo.
    if "kg_total" in d and d["kg_total"]:
        kg_total = float(d["kg_total"])
    else:
        # Estimación por componentes (misma fórmula del Excel: 7.85e-6 kg/mm³)
        alto = float(d["alto"]); ancho = float(d["ancho"]); fondo = float(d["fondo"])
        esp  = float(d["espesor"])
        # Aproximación: kg = superficie_efectiva × espesor × 7.85e-6
        sup = 2 * (alto*ancho + alto*fondo + ancho*fondo)  # cm² aprox
        kg_total = (sup * (esp/10) * 7.85e-6 * 100 +  # conversión cm²→mm²
                    bandejas * 2.5 + n_puertas_total * 3.5 +
                    techo * 2.5 + divisiones * 2.0 + 4)

    row = {
        "familia": f, "cerradura": c, "base": d.get("base", "Zócalo"),
        "tipo_puerta": t,
        "espesor": float(d["espesor"]),
        "kg_total": kg_total,
        "log_kg_total": float(np.log1p(kg_total)),
        "n_componentes": float(d.get("n_componentes", 10)),
        "n_cuerpos_txt": n_cuerpos,
        "n_puertas": n_puertas,
        "n_puertas_total": n_puertas_total,
        "bandejas": bandejas,
        "divisiones": divisiones,
        "techo_inclinado": techo,
        "tiene_ruedas": int("rueda" in txt),
        "tiene_ventilacion": int("ventil" in txt),
        "tiene_malla_txt": int("malla" in txt),
        "sin_puerta_txt": int("sin puerta" in txt),
    }
    for col in ARTEFACTO["features"]:
        if col not in row:
            row[col] = 0

    X = pd.DataFrame([row])[ARTEFACTO["features"]]
    costo_kg = np.expm1(_predict_log(X)[0])
    return float(costo_kg * kg_total)


if __name__ == "__main__":
    print(f"[INFO] Modelo: {ARTEFACTO['nombre']}")
    print(f"[INFO] MAPE:  {ARTEFACTO['mape_global']:.2f}%  "
          f"| Mediana: {ARTEFACTO['mape_mediana']:.2f}%  "
          f"| R²: {ARTEFACTO['r2']:.3f}\n")

    pruebas = [
        ("Ropero 210×100×60", 86992,
         {"familia":"Locker","cantidad":39,"alto":210,"ancho":100,"fondo":60,
          "espesor":0.6,"cerradura":"Cerradura estándar","base":"Zócalo",
          "techo_inclinado":1,"n_puertas":1,"tipo_puerta":"Puerta lisa",
          "bandejas":3,"divisiones":1,"producto_txt":"Lockers Ropero",
          "kg_total":53.39}),
        ("Locker 1c 2p", 36755,
         {"familia":"Locker","cantidad":300,"alto":176,"ancho":40,"fondo":60,
          "espesor":0.6,"cerradura":"Cerradura estándar","base":"Patas",
          "techo_inclinado":0,"n_puertas":2,"tipo_puerta":"Puerta lisa",
          "bandejas":2,"divisiones":1,"producto_txt":"Locker 1 cuerpo 2 puertas",
          "kg_total":20.26}),
        ("Locker 5c 15p", 153829,
         {"familia":"Locker","cantidad":15,"alto":176,"ancho":150,"fondo":60,
          "espesor":0.6,"cerradura":"Cerradura estándar","base":"Patas",
          "techo_inclinado":0,"n_puertas":15,"tipo_puerta":"Puerta lisa",
          "bandejas":2,"divisiones":1,"producto_txt":"Lockers 5 cuerpos 15 puertas",
          "kg_total":75.7}),
    ]
    errs = []
    for n, r, d in pruebas:
        p = predecir(d); e = abs(p-r)/r*100; errs.append(e)
        print(f"{n}: real ${r:,} | pred ${p:,.0f} | err {e:.1f}%")
    print(f"\nMAPE muestra: {sum(errs)/len(errs):.2f}%")