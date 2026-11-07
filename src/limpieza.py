"""
limpieza.py 
"""
from pathlib import Path
import unicodedata, re
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT    = BASE_DIR / "data" / "DataSetVentasHistoricasv2.xlsx"
OUTPUT   = BASE_DIR / "data" / "dataset_limpio.csv"

DROP_COLOR = True


def normalizar_columna(col):
    col = str(col).strip().lower()
    col = ''.join(c for c in unicodedata.normalize('NFD', col)
                  if unicodedata.category(c) != 'Mn')
    for ch in [' ', '(', ')', '.', '/', '-']:
        col = col.replace(ch, '_')
    while '__' in col:
        col = col.replace('__', '_')
    return col.strip('_')


def parse_money(v):
    if pd.isna(v): return np.nan
    if isinstance(v, (int, float, np.number)): return float(v)
    s = str(v).strip().replace('$', '').replace(' ', '')
    if not s: return np.nan
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        s = s.replace(',', '.')
    try: return float(s)
    except ValueError: return np.nan


def extraer_texto(txt):
    s = str(txt).lower()
    mc = re.search(r'(\d+)\s*cuerpo', s)
    mp = re.search(r'(\d+)\s*puerta', s)
    n_cuerpos = int(mc.group(1)) if mc else np.nan
    n_puertas = int(mp.group(1)) if mp else np.nan
    if pd.isna(n_cuerpos) and 'doble' in s:
        n_cuerpos = 2
    return {
        "n_cuerpos_txt":     n_cuerpos,
        "n_puertas_txt":     n_puertas,
        "tiene_ruedas":      int("rueda"      in s),
        "tiene_ventilacion": int("ventil"     in s),
        "tiene_malla_txt":   int("malla"      in s),
        "sin_puerta_txt":    int("sin puerta" in s),
    }


def limpiar():
    df = pd.read_excel(INPUT)
    print(f"[INFO] Filas originales: {len(df)}")
    df.columns = [normalizar_columna(c) for c in df.columns]
    df["producto_txt"] = df["productos"].astype(str)

    df = df.rename(columns={
        "categoria": "familia", "alto_cm": "alto", "ancho_cm": "ancho",
        "fondo_cm": "fondo", "tipo_componente": "base",
        "tipo_puerta": "tipo_puerta", "cantidad_puertas": "n_puertas_col",
        "pintura_color": "color", "tipo_cerradura": "cerradura",
        "bandeja": "bandejas", "division": "divisiones",
        "cantidad": "cantidad", "costo_fabricacion": "costo_fabricacion",
        "kg_total": "kg_total", "espesor": "espesor",
        "techo_inclinado": "techo_inclinado",
        "n_componentes": "n_componentes",
    })

    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        print(f"[WARN] Duplicadas: {sorted(set(dups))} → keep first")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    for c in ["alto","ancho","fondo","espesor","kg_total","cantidad","n_componentes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["n_puertas_col","bandejas","techo_inclinado","divisiones"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["costo_fabricacion"] = df["costo_fabricacion"].apply(parse_money)

    df = df.dropna(subset=["alto","ancho","fondo","espesor","kg_total",
                            "costo_fabricacion","cantidad","familia"])
    print(f"[INFO] Datos completos: {len(df)}")

    # Filtros geométricos + costo/kg razonable (según Excel real: 1500-2500 $/kg)
    df = df[(df.alto  >= 30)  & (df.alto  <= 300)]
    df = df[(df.ancho >= 10)  & (df.ancho <= 250)]
    df = df[(df.fondo >= 20)  & (df.fondo <= 120)]
    df = df[(df.costo_fabricacion > 0) & (df.kg_total > 5) & (df.cantidad > 0)]
    print(f"[INFO] Tras geométricos: {len(df)}")

    df["costo_kg"] = df["costo_fabricacion"] / df["kg_total"]
    df.loc[df["costo_kg"] < 100, "costo_fabricacion"] *= 1000
    df["costo_kg"] = df["costo_fabricacion"] / df["kg_total"]

    antes = len(df)
    df = df[(df.costo_kg >= 800) & (df.costo_kg <= 5000)]
    print(f"[INFO] Tras filtro costo/kg [800-5000]: {antes} → {len(df)}")

    for c in ["familia","color","cerradura","base","tipo_puerta"]:
        df[c] = df[c].astype(str).str.strip() if c in df.columns else "Desconocido"

    feats = df["producto_txt"].apply(extraer_texto).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    df["n_puertas"]       = df["n_puertas_txt"].fillna(df["n_puertas_col"]).astype(int).clip(lower=1)
    df["n_cuerpos_txt"]   = df["n_cuerpos_txt"].fillna(1).astype(int).clip(lower=1)
    df["n_puertas_total"] = df["n_puertas"] * df["n_cuerpos_txt"]

    if "n_componentes" not in df.columns:
        df["n_componentes"] = 10
    df["n_componentes"] = df["n_componentes"].fillna(10).astype(float)

    df["log_kg_total"] = np.log1p(df["kg_total"])
    df["costo_kg_log"] = np.log1p(df["costo_kg"])
    df["costo_log"]    = np.log1p(df["costo_fabricacion"])  # referencia

    cols = [
        "familia", "cerradura", "base", "tipo_puerta",
        "espesor", "kg_total", "log_kg_total", "n_componentes",
        "n_cuerpos_txt", "n_puertas", "n_puertas_total",
        "bandejas", "divisiones", "techo_inclinado",
        "tiene_ruedas", "tiene_ventilacion", "tiene_malla_txt", "sin_puerta_txt",
        "costo_kg", "costo_kg_log", "costo_log",
    ]
    if not DROP_COLOR:
        cols.insert(1, "color")

    df[cols].drop_duplicates().to_csv(OUTPUT, index=False)
    print(f"[INFO] Filas finales: {len(df[cols].drop_duplicates())}")
    print(f"[INFO] costo/kg: mediana={df['costo_kg'].median():.0f} "
          f"p5={df['costo_kg'].quantile(0.05):.0f} "
          f"p95={df['costo_kg'].quantile(0.95):.0f}")
    print(f"[OK] → {OUTPUT}")


if __name__ == "__main__":
    limpiar()