"""
=============================================================
DICO S.A. — Módulo 9: Seguimiento de Accuracy del Modelo
=============================================================
Responde: ¿Puedo confiar en lo que me dice el modelo?

Métricas calculadas:
  1. Resumen global de precisión del portafolio (MAPE_cv ponderado)
  2. Distribución de MAPE por referencia (histograma)
  3. Deriva del modelo: precio predicho próximo mes vs último precio real
  4. Ranking de referencias más/menos confiables
  5. Análisis por método predictivo

Entradas:
  df_predicciones → predecir_todas_las_referencias()
    cols: referencia, descripcion, fecha_pred, yhat, yhat_lower, yhat_upper,
          metodo, mape, n_puntos
  df_historico    → histórico ETL
    cols: referencia, fecha, precio_cop, descripcion, moneda

Autor: Data Science DICO S.A.
Versión: 1.0 — Marzo 2026
=============================================================
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Umbrales de confiabilidad (MAPE %)
MAPE_ALTA   = 12.0   # ≤ 12%  → Alta confiabilidad
MAPE_MEDIA  = 25.0   # ≤ 25%  → Media
# > 25% → Baja confiabilidad


def _nivel_confiabilidad(mape: float) -> str:
    if pd.isna(mape) or mape > MAPE_MEDIA:
        return "Baja"
    if mape > MAPE_ALTA:
        return "Media"
    return "Alta"


def _color_nivel(nivel: str) -> str:
    return {"Alta": "#2a9d8f", "Media": "#e9c46a", "Baja": "#d60000"}.get(nivel, "#b3bbc1")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RESUMEN GLOBAL DE PRECISIÓN
# ══════════════════════════════════════════════════════════════════════════════

def calcular_resumen_precision(
    df_predicciones: pd.DataFrame,
    df_historico: pd.DataFrame,
) -> dict:
    """
    Métricas globales de calidad del modelo sobre el portafolio completo.

    Returns
    -------
    dict:
      mape_ponderado      : MAPE promedio ponderado por n_puntos (más datos = más peso)
      mape_mediano        : percentil 50 del MAPE por referencia
      n_refs_total        : total de referencias modeladas
      n_alta / n_media / n_baja : count por nivel de confiabilidad
      pct_alta / pct_media / pct_baja : porcentaje
      mape_por_metodo     : dict {metodo: mape_promedio}
      costo_cubierto_alta : COP de compras del portafolio con confiabilidad Alta
      costo_total_modelado: COP total modelado
    """
    if df_predicciones is None or df_predicciones.empty:
        return {}

    # Una fila por referencia (tomar primer registro = todos tienen el mismo mape)
    resumen_ref = (
        df_predicciones
        .sort_values("fecha_pred")
        .groupby("referencia")
        .agg(
            mape     =("mape",     "first"),
            metodo   =("metodo",   "first"),
            n_puntos =("n_puntos", "first"),
            yhat_sum =("yhat",     "sum"),    # proxy de costo total proyectado
        )
        .reset_index()
    )
    resumen_ref["nivel"]  = resumen_ref["mape"].apply(_nivel_confiabilidad)

    # MAPE ponderado por n_puntos (más historial = más confiable el dato)
    pesos = resumen_ref["n_puntos"].fillna(1).clip(lower=1)
    mape_ponderado = float((resumen_ref["mape"].fillna(99) * pesos).sum() / pesos.sum())
    mape_mediano   = float(resumen_ref["mape"].dropna().median()) if not resumen_ref["mape"].dropna().empty else 99.0

    n_total = len(resumen_ref)
    counts  = resumen_ref["nivel"].value_counts()
    n_alta  = int(counts.get("Alta",  0))
    n_media = int(counts.get("Media", 0))
    n_baja  = int(counts.get("Baja",  0))

    mape_por_metodo = (
        resumen_ref.groupby("metodo")["mape"].mean().round(1).to_dict()
    )

    costo_alta    = float(resumen_ref.loc[resumen_ref["nivel"] == "Alta",  "yhat_sum"].sum())
    costo_total   = float(resumen_ref["yhat_sum"].sum())

    return {
        "mape_ponderado":        round(mape_ponderado, 1),
        "mape_mediano":          round(mape_mediano, 1),
        "n_refs_total":          n_total,
        "n_alta":                n_alta,
        "n_media":               n_media,
        "n_baja":                n_baja,
        "pct_alta":              round(n_alta  / max(n_total, 1) * 100, 1),
        "pct_media":             round(n_media / max(n_total, 1) * 100, 1),
        "pct_baja":              round(n_baja  / max(n_total, 1) * 100, 1),
        "mape_por_metodo":       mape_por_metodo,
        "costo_cubierto_alta":   costo_alta,
        "costo_total_modelado":  costo_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUCIÓN DE MAPE
# ══════════════════════════════════════════════════════════════════════════════

def calcular_distribucion_mape(df_predicciones: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa referencias por rango de MAPE para histograma ejecutivo.

    Returns
    -------
    pd.DataFrame [rango, n_refs, color, nivel]
    """
    if df_predicciones is None or df_predicciones.empty:
        return pd.DataFrame()

    mapes = (
        df_predicciones.groupby("referencia")["mape"].first().dropna()
    )

    bins   = [0, 5, 10, 15, 20, 25, 35, 50, 9_999]
    labels = ["0–5%", "5–10%", "10–15%", "15–20%", "20–25%", "25–35%", "35–50%", "> 50%"]
    colores = ["#2a9d8f", "#2a9d8f", "#2a9d8f", "#e9c46a", "#e9c46a", "#f4a261", "#d60000", "#7a0000"]
    niveles = ["Alta",    "Alta",    "Alta",    "Media",   "Media",   "Baja",    "Baja",    "Baja"]

    dist = (
        pd.cut(mapes, bins=bins, labels=labels, right=True)
        .value_counts()
        .reindex(labels)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    dist.columns = ["rango", "n_refs"]
    dist["color"] = colores
    dist["nivel"] = niveles
    return dist


# ══════════════════════════════════════════════════════════════════════════════
# 3. DERIVA DEL MODELO (precio predicho próx. mes vs último precio real)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_deriva_modelo(
    df_predicciones: pd.DataFrame,
    df_historico: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara el precio predicho para el próximo mes vs el último precio real
    registrado. Detecta referencias donde el modelo se ha alejado de la realidad.

    Interpretación:
      - Error pequeño (|desvio_pct| ≤ 15%)  → modelo alineado con mercado actual
      - Error medio (15–30%)                → revisar tendencia
      - Error alto (> 30%)                  → posible deriva — reentrenar o revisar

    Returns
    -------
    pd.DataFrame [referencia, descripcion, ultimo_precio_real, fecha_ultimo,
                  precio_predicho_1m, desvio_cop, desvio_pct, nivel_deriva, mape]
    """
    if df_predicciones is None or df_predicciones.empty:
        return pd.DataFrame()

    hoy = pd.Timestamp(datetime.today().date())

    # Último precio real por referencia
    ultimo_real = (
        df_historico.sort_values("fecha")
        .groupby("referencia")
        .agg(
            ultimo_precio_real=("precio_cop", "last"),
            fecha_ultimo       =("fecha",       "last"),
            descripcion        =("descripcion", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    # Primer mes de predicción futura por referencia
    futuro = df_predicciones[df_predicciones["fecha_pred"] > hoy].copy()
    primer_mes = (
        futuro.sort_values("fecha_pred")
        .groupby("referencia")
        .first()
        [["fecha_pred", "yhat", "mape"]]
        .reset_index()
        .rename(columns={"yhat": "precio_predicho_1m", "fecha_pred": "fecha_pred_1m"})
    )

    df = ultimo_real.merge(primer_mes, on="referencia", how="inner")
    df["desvio_cop"] = df["precio_predicho_1m"] - df["ultimo_precio_real"]
    df["desvio_pct"] = (
        df["desvio_cop"] / df["ultimo_precio_real"].clip(lower=1) * 100
    ).round(1)

    def _nivel(pct):
        if abs(pct) <= 15:  return "Alineado ✅"
        if abs(pct) <= 30:  return "Revisar ⚠️"
        return "Deriva alta 🚨"

    df["nivel_deriva"] = df["desvio_pct"].apply(_nivel)

    return df[[
        "referencia", "descripcion", "ultimo_precio_real", "fecha_ultimo",
        "precio_predicho_1m", "fecha_pred_1m", "desvio_cop", "desvio_pct",
        "nivel_deriva", "mape",
    ]].sort_values("desvio_pct", key=abs, ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RANKING POR PRECISIÓN
# ══════════════════════════════════════════════════════════════════════════════

def top_refs_por_precision(
    df_predicciones: pd.DataFrame,
    df_historico: pd.DataFrame,
    n: int = 10,
) -> tuple:
    """
    Returns (df_mejores, df_peores) — top N por MAPE más bajo y más alto.

    Cada df tiene: referencia, descripcion, mape, nivel, metodo, n_puntos,
                   ultimo_precio, n_compras_12m
    """
    if df_predicciones is None or df_predicciones.empty:
        return pd.DataFrame(), pd.DataFrame()

    hoy     = pd.Timestamp(datetime.today().date())
    hace12m = hoy - pd.DateOffset(months=12)

    compras_12m = (
        df_historico[df_historico["fecha"] >= hace12m]
        .groupby("referencia")
        .size()
        .rename("n_compras_12m")
    )
    ultimo_precio = (
        df_historico.sort_values("fecha")
        .groupby("referencia")["precio_cop"]
        .last()
        .rename("ultimo_precio")
    )

    base = (
        df_predicciones.groupby("referencia")
        .agg(
            descripcion=("descripcion", "first"),
            mape       =("mape",        "first"),
            metodo     =("metodo",      "first"),
            n_puntos   =("n_puntos",    "first"),
        )
        .reset_index()
    )
    base = base.join(compras_12m, on="referencia").join(ultimo_precio, on="referencia")
    base["nivel"] = base["mape"].apply(_nivel_confiabilidad)
    base["n_compras_12m"] = base["n_compras_12m"].fillna(0).astype(int)

    mejores = base.dropna(subset=["mape"]).nsmallest(n, "mape").reset_index(drop=True)
    peores  = base.dropna(subset=["mape"]).nlargest(n,  "mape").reset_index(drop=True)

    return mejores, peores


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANÁLISIS POR MÉTODO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_stats_por_metodo(df_predicciones: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen estadístico de precisión agrupado por método predictivo.

    Returns
    -------
    pd.DataFrame [metodo, n_refs, mape_promedio, mape_mediano, mape_min, mape_max,
                  pct_alta, pct_media, pct_baja]
    """
    if df_predicciones is None or df_predicciones.empty:
        return pd.DataFrame()

    base = (
        df_predicciones.groupby("referencia")
        .agg(mape=("mape", "first"), metodo=("metodo", "first"))
        .reset_index()
    )
    base["nivel"] = base["mape"].apply(_nivel_confiabilidad)

    def _stats(grp):
        n = len(grp)
        return pd.Series({
            "n_refs":       n,
            "mape_promedio": round(grp["mape"].mean(), 1),
            "mape_mediano":  round(grp["mape"].median(), 1),
            "mape_min":      round(grp["mape"].min(), 1),
            "mape_max":      round(grp["mape"].max(), 1),
            "pct_alta":      round((grp["nivel"] == "Alta").sum()  / n * 100, 1),
            "pct_media":     round((grp["nivel"] == "Media").sum() / n * 100, 1),
            "pct_baja":      round((grp["nivel"] == "Baja").sum()  / n * 100, 1),
        })

    return base.groupby("metodo").apply(_stats).reset_index().sort_values("mape_promedio")
