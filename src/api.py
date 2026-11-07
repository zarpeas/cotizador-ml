"""
api.py - v14
API del cotizador ML.

Modelo: predice log(costo/kg_total) y reconstruye costo × kg_total.
- Campo crítico: kg_total (peso real del mueble en kg).
  Si no se envía, se estima con la fórmula física de componentes.
- Soporta ensemble ponderado.
- Aplica clip_lo/clip_hi del artefacto.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# ARTEFACTO
# =========================================================
BASE_DIR  = Path(__file__).resolve().parent.parent
ARTEFACTO = joblib.load(BASE_DIR / "models" / "modelo_costo_fabricacion.joblib")


# =========================================================
# APP
# =========================================================
app = FastAPI(title="Cotizador ML API v14", version="14.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MAPAS DE NORMALIZACIÓN
# =========================================================
FAMILIA_MAP = {
    "Locker":          "Lockers",
    "LK":              "Lockers",
    "Storage":         "Storage",
    "Archivo":         "Lockers",
    "Ropero":          "Ropero",
    "Lockers Oficina": "Lockers Oficina",
}
CERRADURA_MAP = {
    "Cerradura estándar":  "Cerradura portacandado",
    "Cerradura guantera":  "Cerradura portacandado",
    "Cerradura españoleta": "Españoleta",
}
PUERTA_MAP = {
    "Puerta lisa":      "Puerta Lisa",
    "Puerta malla":     "Puerta Malla",
    "Puerta CNC láser": "Puerta Lisa",
}


# =========================================================
# SCHEMAS
# =========================================================
class Mueble(BaseModel):
    familia:         str
    # Dimensiones
    alto:            float = Field(..., gt=0)
    ancho:           float = Field(..., gt=0)
    fondo:           float = Field(..., gt=0)
    espesor:         float = Field(..., gt=0)
    # Peso real del mueble (kg). Si no se envía, se estima.
    kg_total:        float | None = None
    # Configuración
    cerradura:       str
    base:            str = "Zócalo"
    techo_inclinado: int = Field(0, ge=0, le=1)
    n_puertas:       int = Field(..., ge=1, le=50)
    tipo_puerta:     str
    bandejas:        int = Field(0, ge=0)
    divisiones:      int = Field(1, ge=0, le=10)
    # Metadata opcional
    producto_txt:    str = ""
    n_componentes:   float = 10.0
    n_cuerpos:       int = Field(1, ge=1, le=20)
    # Compatibilidad con clientes anteriores
    cantidad:        int = Field(1, ge=1)
    color:           str = "Gris Eco"


class Prediccion(BaseModel):
    costo_fabricacion_unitario: float
    mape_esperado_pct:          float
    mape_mediana_pct:           float
    kg_total_usado:             float
    kg_total_fuente:            str    # "usuario" | "estimado"
    modelo:                     str
    confianza_r2:               float


# =========================================================
# HELPERS
# =========================================================
def _predict_log_ensemble(X: pd.DataFrame) -> np.ndarray:
    """Predice en log-espacio (target: log(costo/kg)) con clip del artefacto."""
    model = ARTEFACTO["model"]
    if ARTEFACTO.get("es_ensemble"):
        y = np.zeros(len(X))
        for p, w in model["ensemble_weighted"]:
            y += w * np.asarray(p.predict(X))
    else:
        y = np.asarray(model.predict(X))

    lo, hi = ARTEFACTO["clip_lo"], ARTEFACTO["clip_hi"]
    return np.clip(y, lo, hi)


def _estimar_kg_total(alto: float, ancho: float, fondo: float,
                      espesor: float, bandejas: int, n_puertas_total: int,
                      techo_inclinado: int, divisiones: int,
                      n_cuerpos: int) -> float:
    """
    Estimación física del peso (kg) si el usuario no provee kg_total.
    Aproximación por superficie efectiva del mueble + componentes.

    Basada en la fórmula del Excel de costos:
        kg_acero = Σ(cantidad × largo × ancho × espesor × 7.85e-6)
    Se aproxima la superficie total del cerramiento y se añade peso de
    componentes internos (bandejas, puertas, divisiones, techo).
    """
    # Superficie del cerramiento (cm²)
    area_cerramiento = 2 * (alto * ancho + alto * fondo + ancho * fondo)

    # kg de acero del cerramiento: area × espesor(mm/10 para cm) × densidad
    # 7.85 g/cm³ × 1e-3 kg/g = 7.85e-3 kg/cm³
    kg_cerramiento = area_cerramiento * (espesor / 10) * 7.85e-3

    # Componentes adicionales (kg aproximados)
    kg_bandejas   = bandejas * 2.5 * n_cuerpos
    kg_puertas    = n_puertas_total * 3.5
    kg_divisiones = divisiones * 2.0 * n_cuerpos
    kg_techo      = techo_inclinado * 4.0
    kg_estructura = n_cuerpos * 2.0

    kg = kg_cerramiento + kg_bandejas + kg_puertas + kg_divisiones + kg_techo + kg_estructura
    return max(kg, 5.0)


def _normalizar_familia(f: str) -> str:
    return FAMILIA_MAP.get(f, f if f in FAMILIA_MAP.values() else "Lockers")


# =========================================================
# ENDPOINTS
# =========================================================
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "version":          "14.0.0",
        "modelo":           ARTEFACTO["nombre"],
        "target":           ARTEFACTO["target"],
        "es_ensemble":      ARTEFACTO.get("es_ensemble", False),
        "mape_global":      round(ARTEFACTO["mape_global"], 2),
        "mape_mediana":     round(ARTEFACTO["mape_mediana"], 2),
        "mape_trimmed_90":  round(ARTEFACTO.get("mape_trimmed_90", 0), 2),
        "r2":               round(ARTEFACTO.get("r2", 0), 4),
        "objetivo_cumplido": ARTEFACTO.get("objetivo_cumplido", False),
        "features":         ARTEFACTO["features"],
    }


@app.post("/predecir", response_model=Prediccion)
def predecir(m: Mueble):
    try:
        # ---------- Normalizar categóricas ----------
        fam = _normalizar_familia(m.familia)
        cer = CERRADURA_MAP.get(m.cerradura, m.cerradura)
        tp  = PUERTA_MAP.get(m.tipo_puerta, m.tipo_puerta)

        # ---------- Parsear texto del producto ----------
        txt = (m.producto_txt or "").lower()
        mc = re.search(r'(\d+)\s*cuerpo', txt)
        mp = re.search(r'(\d+)\s*puerta', txt)
        n_cuerpos_txt = int(mc.group(1)) if mc else m.n_cuerpos
        n_puertas_txt = int(mp.group(1)) if mp else m.n_puertas
        n_cuerpos_txt = max(1, n_cuerpos_txt)
        n_puertas_txt = max(1, n_puertas_txt)

        n_puertas_total = n_puertas_txt * n_cuerpos_txt

        # ---------- kg_total ----------
        if m.kg_total is not None and m.kg_total > 0:
            kg_total = float(m.kg_total)
            kg_fuente = "usuario"
        else:
            kg_total = _estimar_kg_total(
                alto=m.alto, ancho=m.ancho, fondo=m.fondo,
                espesor=m.espesor, bandejas=m.bandejas,
                n_puertas_total=n_puertas_total,
                techo_inclinado=m.techo_inclinado,
                divisiones=m.divisiones,
                n_cuerpos=n_cuerpos_txt,
            )
            kg_fuente = "estimado"

        # ---------- Construir fila de features ----------
        row = {
            "familia":            fam,
            "cerradura":          cer,
            "base":               m.base,
            "tipo_puerta":        tp,
            "espesor":            m.espesor,
            "kg_total":           kg_total,
            "log_kg_total":       float(np.log1p(kg_total)),
            "n_componentes":      float(m.n_componentes),
            "n_cuerpos_txt":      n_cuerpos_txt,
            "n_puertas":          n_puertas_txt,
            "n_puertas_total":    n_puertas_total,
            "bandejas":           m.bandejas,
            "divisiones":         m.divisiones,
            "techo_inclinado":    m.techo_inclinado,
            "tiene_ruedas":       int("rueda"  in txt),
            "tiene_ventilacion":  int("ventil" in txt),
            "tiene_malla_txt":    int("malla"  in txt),
            "sin_puerta_txt":     int("sin puerta" in txt),
        }

        # Asegurar que existan todas las features esperadas por el modelo
        for col in ARTEFACTO["features"]:
            if col not in row:
                row[col] = 0

        X = pd.DataFrame([row])[ARTEFACTO["features"]]

        # ---------- Predicción ----------
        pred_log = _predict_log_ensemble(X)[0]
        costo_kg = float(np.expm1(pred_log))
        costo    = costo_kg * kg_total

        # ---------- Respuesta ----------
        return Prediccion(
            costo_fabricacion_unitario = round(costo, 2),
            mape_esperado_pct          = round(ARTEFACTO["mape_global"], 2),
            mape_mediana_pct           = round(ARTEFACTO["mape_mediana"], 2),
            kg_total_usado             = round(kg_total, 2),
            kg_total_fuente            = kg_fuente,
            modelo                     = ARTEFACTO["nombre"],
            confianza_r2               = round(ARTEFACTO.get("r2", 0), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# EJECUCIÓN DIRECTA (para test rápido sin uvicorn)
# =========================================================
if __name__ == "__main__":
    print(f"[INFO] Modelo cargado: {ARTEFACTO['nombre']}")
    print(f"[INFO] MAPE:  {ARTEFACTO['mape_global']:.2f}%  "
          f"| Mediana: {ARTEFACTO['mape_mediana']:.2f}%  "
          f"| R²: {ARTEFACTO.get('r2', 0):.3f}\n")

    pruebas = [
        # (nombre, real, mueble)
        ("Ropero 210×100×60 (kg=53.39)", 86992,
         Mueble(familia="Locker", alto=210, ancho=100, fondo=60,
                espesor=0.6, kg_total=53.39, cerradura="Cerradura estándar",
                base="Zócalo", techo_inclinado=1, n_puertas=1,
                tipo_puerta="Puerta lisa", bandejas=3, divisiones=1,
                producto_txt="Lockers Ropero")),
        ("Locker 1c 2p (kg=20.26)", 36755,
         Mueble(familia="Locker", alto=176, ancho=40, fondo=60,
                espesor=0.6, kg_total=20.26, cerradura="Cerradura estándar",
                base="Patas", techo_inclinado=0, n_puertas=2,
                tipo_puerta="Puerta lisa", bandejas=2, divisiones=1,
                producto_txt="Locker 1 cuerpo 2 puertas")),
        ("Locker 5c 15p (kg=75.7)", 153829,
         Mueble(familia="Locker", alto=176, ancho=150, fondo=60,
                espesor=0.6, kg_total=75.7, cerradura="Cerradura estándar",
                base="Patas", techo_inclinado=0, n_puertas=15,
                tipo_puerta="Puerta lisa", bandejas=2, divisiones=1,
                producto_txt="Lockers 5 cuerpos 15 puertas")),
        # Sin kg_total: cae a estimación física
        ("Locker SIN kg (estimado)", 36755,
         Mueble(familia="Locker", alto=176, ancho=40, fondo=60,
                espesor=0.6, cerradura="Cerradura estándar",
                base="Patas", techo_inclinado=0, n_puertas=2,
                tipo_puerta="Puerta lisa", bandejas=2, divisiones=1,
                producto_txt="Locker 1 cuerpo 2 puertas")),
    ]

    errs = []
    for nombre, real, mueble in pruebas:
        try:
            p = predecir(mueble)
            err = abs(p.costo_fabricacion_unitario - real) / real * 100
            errs.append(err)
            print(f"{nombre}")
            print(f"  → pred ${p.costo_fabricacion_unitario:,.0f}  "
                  f"(real ${real:,})  err {err:.1f}%  "
                  f"[kg={p.kg_total_usado} ({p.kg_total_fuente})]")
        except Exception as e:
            print(f"{nombre}: ERROR {e}")

    if errs:
        print(f"\nMAPE muestra: {sum(errs)/len(errs):.2f}%")