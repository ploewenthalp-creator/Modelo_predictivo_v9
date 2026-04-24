"""
=============================================================
DICO S.A. — Módulo 7: Presupuesto y Priorización de Compras
=============================================================
Responde la pregunta estratégica que el modelo no respondía antes:

  "Dado un presupuesto disponible de $X, ¿QUÉ compro primero?"

Lógica:
  1. Calcula un score de prioridad 0–100 por referencia combinando:
       40% urgencia (cobertura real vs umbral por criticidad)
       30% criticidad (Alta/Media/Baja)
       20% señal de precio (Compra ya / Monitorea / Espera)
       10% ahorro estimado en COP (normalizado)

  2. Ordena de mayor a menor prioridad y llena el presupuesto disponible
     con un algoritmo greedy (compra primero lo más urgente/crítico).

  3. Distribuye el excedente (lo que no cabe en el presupuesto actual)
     en los meses siguientes respetando un presupuesto mensual configurable.

  4. Proyecta el cash flow mes a mes para los próximos N meses.

  5. Identifica referencias que SE DEBEN comprar aunque no haya presupuesto
     (cobertura real = 0 o alerta crítica) como "compras ineludibles".

Autor: Data Science DICO S.A.
Versión: 1.0 — Marzo 2026
=============================================================
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

# ─── Constantes ───────────────────────────────────────────────────────────────

# Pesos del score de prioridad
_W_URGENCIA    = 0.40
_W_CRITICIDAD  = 0.30
_W_PRECIO      = 0.20
_W_AHORRO      = 0.10

# Meses de cobertura en que se considera "sin cobertura urgente" por nivel
_UMBRAL_URGENTE = {"Alta": 2.0, "Media": 1.0, "Baja": 0.5}

# Score de criticidad
_SCORE_CRITICIDAD = {"Alta": 1.0, "Media": 0.6, "Baja": 0.3}

# Score de señal de precio
_SCORE_SENAL = {
    "🟢 Compra ya": 1.0,
    "🟡 Monitorea": 0.5,
    "🔴 Espera":    0.2,
    "—":            0.5,
}


# ─── 1. Score de prioridad ────────────────────────────────────────────────────

def calcular_prioridad_compra(
    df_compras: pd.DataFrame,
    df_op: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Asigna un score de prioridad 0–100 a cada referencia con sugerencia de compra.

    Parameters
    ----------
    df_compras : Salida de calcular_compras_sugeridas() — debe tener:
                 referencia, meses_cobertura_real, criticidad, cantidad_sugerida, costo_total
    df_op      : Salida de calcular_oportunidades_compra() — opcional, aporta señal de precio
                 y ahorro_cop_estimado.

    Returns
    -------
    DataFrame con columnas adicionales:
      score_urgencia, score_criticidad_val, score_precio, score_ahorro,
      prioridad_score (0–100), prioridad_nivel (Crítica/Alta/Media/Baja)
    """
    if df_compras.empty:
        return df_compras.copy()

    df = df_compras[df_compras["cantidad_sugerida"] > 0].copy()
    if df.empty:
        return df

    # ── Score de urgencia (0–1) ──
    # Cuánto le falta de cobertura respecto al umbral crítico de su nivel
    def _urgencia(row):
        umbral = _UMBRAL_URGENTE.get(row.get("criticidad", "Media"), 1.0)
        cob    = row.get("meses_cobertura_real", row.get("cobertura_meses", 99))
        if cob == float("inf"):
            return 0.0
        if cob <= 0:
            return 1.0
        return max(0.0, min(1.0, 1.0 - cob / (umbral * 2)))

    df["score_urgencia"] = df.apply(_urgencia, axis=1)

    # ── Score de criticidad (0–1) ──
    df["score_criticidad_val"] = (
        df.get("criticidad", pd.Series("Media", index=df.index))
        .map(_SCORE_CRITICIDAD)
        .fillna(0.5)
    )

    # ── Score señal precio (0–1) — desde df_op si existe ──
    senal_map = {}
    ahorro_map = {}
    if df_op is not None and not df_op.empty:
        if "senal" in df_op.columns:
            senal_map  = dict(zip(df_op["referencia"], df_op["senal"]))
        if "ahorro_cop_estimado" in df_op.columns:
            ahorro_map = dict(zip(df_op["referencia"], df_op["ahorro_cop_estimado"]))

    df["senal_precio"]  = df["referencia"].map(senal_map).fillna("—")
    df["score_precio"]  = df["senal_precio"].map(_SCORE_SENAL).fillna(0.5)

    # ── Score ahorro (0–1 normalizado) ──
    df["ahorro_cop"] = df["referencia"].map(ahorro_map).fillna(
        df.get("ahorro_estimado_cop", 0)
    ).fillna(0)
    max_ahorro = df["ahorro_cop"].max()
    df["score_ahorro"] = (df["ahorro_cop"] / max_ahorro).clip(0, 1) if max_ahorro > 0 else 0.0

    # ── Score final ponderado (0–100) ──
    df["prioridad_score"] = (
        _W_URGENCIA   * df["score_urgencia"]
        + _W_CRITICIDAD * df["score_criticidad_val"]
        + _W_PRECIO     * df["score_precio"]
        + _W_AHORRO     * df["score_ahorro"]
    ).round(4) * 100

    # ── Nivel de prioridad legible ──
    def _nivel(s: float) -> str:
        if s >= 70: return "🔴 Crítica"
        if s >= 45: return "🟠 Alta"
        if s >= 25: return "🟡 Media"
        return "🟢 Baja"

    df["prioridad_nivel"] = df["prioridad_score"].apply(_nivel)

    # ── Flag ineludible: sin cobertura real (< 0.25 meses) o alerta crítica ──
    df["ineludible"] = (
        (df.get("meses_cobertura_real", df.get("cobertura_meses", 99)) < 0.25) |
        (df.get("alerta_stock", False) & (df.get("criticidad", "Media") == "Alta"))
    )

    return df.sort_values("prioridad_score", ascending=False).reset_index(drop=True)


# ─── 2. Selección dentro del presupuesto ─────────────────────────────────────

def priorizar_dentro_presupuesto(
    df_priorizado: pd.DataFrame,
    presupuesto_total: float,
) -> tuple:
    """
    Selecciona qué comprar dado un presupuesto total, respetando el orden de prioridad.

    Regla: los ítems marcados como 'ineludible=True' SIEMPRE se incluyen,
    incluso si superan el presupuesto (se muestran con alerta de sobregiro).

    Returns
    -------
    (df_dentro, df_fuera, resumen)
      df_dentro  : Referencias que entran en presupuesto + ineludibles
      df_fuera   : Referencias diferidas por falta de presupuesto
      resumen    : dict con KPIs del resultado
    """
    if df_priorizado.empty:
        empty = pd.DataFrame()
        return empty, empty, {}

    df = df_priorizado.copy()
    if "costo_total" not in df.columns:
        df["costo_total"] = np.nan

    # Separar ineludibles (van siempre)
    ineludibles = df[df.get("ineludible", pd.Series(False, index=df.index)) == True].copy()
    opcionales  = df[~df.index.isin(ineludibles.index)].copy()

    costo_ineludible = ineludibles["costo_total"].sum(skipna=True)
    presupuesto_restante = presupuesto_total - costo_ineludible

    # Greedy: agregar opcionales por orden de prioridad mientras haya presupuesto
    dentro_idx = []
    fuera_idx  = []
    acumulado  = 0.0

    for idx, row in opcionales.iterrows():
        costo = row["costo_total"] if pd.notna(row["costo_total"]) else 0
        if acumulado + costo <= presupuesto_restante:
            dentro_idx.append(idx)
            acumulado += costo
        else:
            fuera_idx.append(idx)

    df_opcionales_dentro = opcionales.loc[dentro_idx]
    df_fuera             = opcionales.loc[fuera_idx]

    df_dentro = pd.concat([ineludibles, df_opcionales_dentro], ignore_index=True)
    df_dentro = df_dentro.sort_values("prioridad_score", ascending=False).reset_index(drop=True)
    df_fuera  = df_fuera.reset_index(drop=True)

    costo_total_dentro = df_dentro["costo_total"].sum(skipna=True)
    sobregiro          = max(0, costo_ineludible - presupuesto_total)

    resumen = {
        "presupuesto_total":   presupuesto_total,
        "costo_dentro":        round(costo_total_dentro),
        "costo_ineludible":    round(costo_ineludible),
        "costo_diferido":      round(df_fuera["costo_total"].sum(skipna=True)),
        "sobregiro":           round(sobregiro),
        "n_dentro":            len(df_dentro),
        "n_ineludibles":       len(ineludibles),
        "n_diferidos":         len(df_fuera),
        "pct_presupuesto_uso": round(min(costo_total_dentro / presupuesto_total, 1) * 100, 1)
                               if presupuesto_total > 0 else 0,
        "hay_sobregiro":       sobregiro > 0,
    }

    return df_dentro, df_fuera, resumen


# ─── 3. Proyección de cash flow mensual ──────────────────────────────────────

def proyectar_cashflow(
    df_priorizado: pd.DataFrame,
    presupuesto_mensual: float,
    n_meses: int = 4,
) -> pd.DataFrame:
    """
    Distribuye las compras en los próximos N meses respetando el presupuesto mensual.

    Reglas:
      - Mes 1: ineludibles + lo que quepa del presupuesto mensual (orden de prioridad)
      - Meses 2–N: el remanente se distribuye en orden de prioridad
      - Si algo no cabe en N meses: queda como "Sin presupuesto en horizonte"

    Returns
    -------
    DataFrame [mes, referencia, descripcion, criticidad, prioridad_nivel,
               costo_total, cantidad_sugerida, tipo_compra]
    """
    if df_priorizado.empty or presupuesto_mensual <= 0:
        return pd.DataFrame()

    df = df_priorizado.copy()
    if "costo_total" not in df.columns:
        df["costo_total"] = 0.0
    df["costo_total"] = df["costo_total"].fillna(0)

    hoy       = pd.Timestamp(datetime.today().replace(day=1))
    meses     = [hoy + pd.DateOffset(months=i) for i in range(n_meses)]
    pendientes = df.to_dict("records")
    resultado  = []

    for i, mes in enumerate(meses):
        budget_mes  = presupuesto_mensual
        mes_label   = mes.strftime("%B %Y").capitalize()
        asignados   = []
        sin_asignar = []

        for item in pendientes:
            costo = item.get("costo_total", 0) or 0
            es_ineludible = item.get("ineludible", False)

            if es_ineludible and i == 0:
                # Ineludibles siempre en mes 1, aunque supere budget
                item["mes"]       = mes_label
                item["tipo_compra"] = "🚨 Ineludible"
                resultado.append(item)
                asignados.append(item)
                budget_mes -= costo
            elif costo <= budget_mes:
                item["mes"]       = mes_label
                item["tipo_compra"] = "✅ Programada"
                resultado.append(item)
                asignados.append(item)
                budget_mes -= costo
            else:
                sin_asignar.append(item)

        pendientes = sin_asignar

    # Lo que no entró en ningún mes
    for item in pendientes:
        item["mes"]       = "Sin presupuesto en horizonte"
        item["tipo_compra"] = "⏸️ Diferida"
        resultado.append(item)

    if not resultado:
        return pd.DataFrame()

    cols = [
        "mes", "tipo_compra", "prioridad_nivel", "referencia",
        "criticidad", "estado_stock", "cantidad_sugerida", "costo_total",
        "meses_cobertura_real", "fuente_demanda",
    ]
    df_res = pd.DataFrame(resultado)
    cols_disponibles = [c for c in cols if c in df_res.columns]
    return df_res[cols_disponibles].reset_index(drop=True)


# ─── 4. Resumen por mes ───────────────────────────────────────────────────────

def resumen_cashflow_por_mes(df_cashflow: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega el cash flow proyectado por mes.

    Returns
    -------
    DataFrame [mes, n_referencias, costo_total, n_ineludibles, n_programadas, n_diferidas]
    """
    if df_cashflow.empty:
        return pd.DataFrame()

    resumen = (
        df_cashflow.groupby("mes")
        .agg(
            n_referencias  = ("referencia", "count"),
            costo_total    = ("costo_total", "sum"),
            n_ineludibles  = ("tipo_compra", lambda x: (x == "🚨 Ineludible").sum()),
            n_programadas  = ("tipo_compra", lambda x: (x == "✅ Programada").sum()),
            n_diferidas    = ("tipo_compra", lambda x: (x == "⏸️ Diferida").sum()),
        )
        .reset_index()
    )
    resumen["costo_total"] = resumen["costo_total"].round(0)

    # Ordenar: meses reales primero, "Sin presupuesto" al final
    resumen["_orden"] = resumen["mes"].apply(
        lambda m: 99 if "Sin presupuesto" in m
        else list(resumen["mes"]).index(m)
    )
    resumen = resumen.sort_values("_orden").drop(columns="_orden").reset_index(drop=True)
    return resumen
