"""
=============================================================
DICO S.A. — Módulo 8: Panel Ejecutivo de Compras
=============================================================
Calcula métricas de alto nivel para el tab "📋 Resumen Ejecutivo".

Diseñado para funcionar SIN predicciones masivas:
  - Usa generar_reporte_inventario() (stock + demanda + criticidad)
  - Fallback de costo = último precio registrado × cantidad_sugerida
  - Cuando df_predicciones está disponible, mejora precisión y agrega
    señales de precio y cuantificación del error en COP.

Entradas:
  df_compras   → generar_reporte_inventario()["compras"]
  df_historico → histórico ETL (cols: referencia, fecha, precio_cop, moneda, proveedor, descripcion)
  df_trm       → TRM histórica (cols: fecha, trm)
  df_gap_req   → generar_reporte_inventario()["gap_req"]  (opcional)
  df_op        → calcular_oportunidades_compra()           (opcional)
  df_pred      → predecir_todas_las_referencias()          (opcional)

Autor: Data Science DICO S.A.
Versión: 1.0 — Marzo 2026
=============================================================
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNOS
# ══════════════════════════════════════════════════════════════════════════════

def _ultimo_precio(df_historico: pd.DataFrame) -> pd.Series:
    """Último precio COP registrado por referencia (del histórico de compras)."""
    return (
        df_historico.sort_values("fecha")
        .groupby("referencia")["precio_cop"]
        .last()
    )


def _safe_cob(v) -> float:
    """Normaliza cobertura: inf → 999, NaN/None → 99."""
    try:
        f = float(v)
        if np.isnan(f):
            return 99.0
        if f == float("inf"):
            return 999.0
        return f
    except Exception:
        return 99.0


def fmt_cop(v: float) -> str:
    """Formatea valor COP con sufijo B/M para valores grandes."""
    if pd.isna(v) or v == 0:
        return "$ 0"
    if v >= 1_000_000_000:
        return f"$ {v / 1_000_000_000:.1f} B"
    if v >= 1_000_000:
        return f"$ {v / 1_000_000:.1f} M"
    return f"$ {v:,.0f}"


def _enriquecer_costo(df_compras: pd.DataFrame, df_historico: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena costo_total NaN/0 con último_precio × cantidad_sugerida.
    Agrega columna booleana 'costo_con_prediccion' para advertir al UI.
    """
    df = df_compras.copy()
    up = _ultimo_precio(df_historico)

    sin_costo = df["costo_total"].isna() | (df["costo_total"] <= 0)
    df.loc[sin_costo, "costo_total"] = (
        df.loc[sin_costo, "referencia"].map(up).fillna(0)
        * df.loc[sin_costo, "cantidad_sugerida"]
    )
    df["costo_con_prediccion"] = (~sin_costo).astype(bool)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. SEMÁFORO EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_semaforo(
    df_compras: pd.DataFrame,
    df_historico: pd.DataFrame,
    df_op: pd.DataFrame = None,
) -> dict:
    """
    Clasifica referencias con compra sugerida en tres buckets de urgencia.

    Criterios:
      🔴 Urgente    — alerta_stock=True  o  cobertura_real ≤ 0.5 meses
      🟠 Recomendada — cantidad_sugerida > 0  y  no urgente
      🟢 OK/Diferible — sin sugerencia de compra (cobertura suficiente)

    Returns
    -------
    dict con keys: urgente, recomendada, diferible
      Cada bucket: {n_refs, costo_total, tiene_prediccion, pct_costo_total}
    """
    tiene_sug = df_compras["cantidad_sugerida"] > 0
    df_sug    = _enriquecer_costo(df_compras[tiene_sug].copy(), df_historico)

    cob_real   = df_sug["meses_cobertura_real"].apply(_safe_cob)
    alerta_col = df_sug.get("alerta_stock", pd.Series(False, index=df_sug.index)).fillna(False)

    mask_urg = alerta_col | (cob_real <= 0.5)

    costo_total_global = float(df_sug["costo_total"].sum()) or 1.0

    def _bucket(sub):
        costo = float(sub["costo_total"].sum())
        return {
            "n_refs":            len(sub),
            "costo_total":       costo,
            "tiene_prediccion":  bool(sub.get("costo_con_prediccion", pd.Series(False)).any()),
            "pct_costo":         round(costo / costo_total_global * 100, 1),
        }

    return {
        "urgente":     _bucket(df_sug[mask_urg]),
        "recomendada": _bucket(df_sug[~mask_urg]),
        "diferible":   {"n_refs": int((~tiene_sug).sum()), "costo_total": 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPROMISO DE CAJA (próximos 90 días)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_compromiso_caja(
    df_compras: pd.DataFrame,
    df_historico: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribuye el costo de compras sugeridas en 4 buckets temporales basados
    en cobertura real. No requiere predicciones de precio.

    Buckets:
      🔴 Semana 1-2 (Urgente)  — alerta_stock o cobertura ≤ 0.5 m
      🟠 Mes 1 (0-30 días)     — cobertura ≤ 1 mes
      🟡 Mes 2 (30-60 días)    — cobertura ≤ 2 meses
      🟢 Mes 3 (60-90 días)    — resto con sugerencia activa

    Returns
    -------
    pd.DataFrame [periodo, n_refs, costo, costo_acumulado, color]
    """
    df = _enriquecer_costo(
        df_compras[df_compras["cantidad_sugerida"] > 0].copy(),
        df_historico,
    )

    ORDEN   = ["🔴 Sem 1-2 · Urgente", "🟠 Mes 1 · 0-30 días", "🟡 Mes 2 · 30-60 días", "🟢 Mes 3 · 60-90 días"]
    COLORES = ["#d60000", "#f4a261", "#e9c46a", "#2a9d8f"]

    def _bucket(row):
        cob    = _safe_cob(row.get("meses_cobertura_real", row.get("cobertura_meses", 99)))
        alerta = bool(row.get("alerta_stock", False))
        if alerta or cob <= 0.5:
            return ORDEN[0]
        if cob <= 1.0:
            return ORDEN[1]
        if cob <= 2.0:
            return ORDEN[2]
        return ORDEN[3]

    df["periodo"] = df.apply(_bucket, axis=1)

    resumen = (
        df.groupby("periodo", sort=False)
        .agg(n_refs=("referencia", "count"), costo=("costo_total", "sum"))
        .reindex(ORDEN)
        .fillna(0)
        .reset_index()
    )
    resumen["costo_acumulado"] = resumen["costo"].cumsum()
    resumen["color"]           = COLORES
    return resumen


# ══════════════════════════════════════════════════════════════════════════════
# 3. ESTADO DEL INVENTARIO
# ══════════════════════════════════════════════════════════════════════════════

def calcular_estado_inventario(
    df_compras: pd.DataFrame,
    df_historico: pd.DataFrame,
) -> dict:
    """
    Métricas de salud del inventario.

    Returns
    -------
    dict:
      cobertura_promedio   : meses (float) — media sobre refs activas
      n_riesgo_quiebre     : refs con cobertura_real < 1 mes
      n_stock_excesivo     : refs con cobertura_real > 6 meses
      capital_inmovilizado : COP estimado en stock excesivo
      distribucion         : pd.Series con bins para gráfico
    """
    up  = _ultimo_precio(df_historico)
    cob = df_compras["meses_cobertura_real"].apply(_safe_cob)

    cob_activa          = cob[(cob > 0) & (cob < 900)]
    cobertura_promedio  = float(cob_activa.mean()) if not cob_activa.empty else 0.0
    n_riesgo            = int((cob <= 1.0).sum())

    mask_exc  = cob > 6
    n_exceso  = int(mask_exc.sum())
    df_exc    = df_compras[mask_exc].copy()
    df_exc["_up"] = df_exc["referencia"].map(up).fillna(0)
    capital_inmovilizado = float((df_exc["stock_actual"] * df_exc["_up"]).sum())

    # Distribución para gráfico de barras
    bins   = [0, 0.5, 1, 2, 3, 6, 9_999]
    labels = ["< 0.5m ⚠️", "0.5–1m", "1–2m", "2–3m", "3–6m ✅", "> 6m 🔒"]
    dist   = (
        pd.cut(cob, bins=bins, labels=labels, right=True)
        .value_counts()
        .reindex(labels)
        .fillna(0)
        .astype(int)
    )

    return {
        "cobertura_promedio":   round(cobertura_promedio, 1),
        "n_riesgo_quiebre":     n_riesgo,
        "n_stock_excesivo":     n_exceso,
        "capital_inmovilizado": capital_inmovilizado,
        "distribucion":         dist,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPOSICIÓN CAMBIARIA
# ══════════════════════════════════════════════════════════════════════════════

def calcular_exposicion_fx(
    df_historico: pd.DataFrame,
    df_trm: pd.DataFrame,
    df_compras: pd.DataFrame = None,
) -> dict:
    """
    Cuantifica la exposición al riesgo cambiario (USD / EUR):
      - Gasto en divisas extranjeras en los últimos 12 meses
      - TRM actual y variación vs. hace 1 mes (~22 días hábiles)
      - Impacto en COP si TRM sube +5% / +10% sobre compras sugeridas en FX

    Returns
    -------
    dict: trm_actual, var_1m_pct, gasto_usd_12m, gasto_eur_12m,
          gasto_fx_12m, pct_fx, n_refs_fx, impacto_5pct, impacto_10pct
    """
    hoy     = pd.Timestamp(datetime.today().date())
    hace12m = hoy - pd.DateOffset(months=12)

    df_12m    = df_historico[df_historico["fecha"] >= hace12m]
    gasto_usd = float(df_12m[df_12m["moneda"] == "USD"]["precio_cop"].sum())
    gasto_eur = float(df_12m[df_12m["moneda"] == "EUR"]["precio_cop"].sum())
    gasto_fx  = gasto_usd + gasto_eur

    total_12m = float(df_12m["precio_cop"].sum())
    pct_fx    = round(gasto_fx / total_12m * 100, 1) if total_12m > 0 else 0.0

    # TRM actual: df_trm (API) → trm_historica en df_historico → fallback fijo
    trm_actual = 0.0
    trm_serie  = pd.Series(dtype=float)

    if df_trm is not None and not df_trm.empty:
        trm_serie  = df_trm["trm"].dropna().sort_values().reset_index(drop=True)
        trm_actual = float(trm_serie.iloc[-1])
    elif "trm_historica" in df_historico.columns:
        _trm_h = df_historico.sort_values("fecha")["trm_historica"].dropna()
        if not _trm_h.empty:
            trm_actual = float(_trm_h.iloc[-1])
            trm_serie  = _trm_h.reset_index(drop=True)
    elif "trm_aplicada" in df_historico.columns:
        _trm_h = df_historico.sort_values("fecha")["trm_aplicada"].dropna()
        if not _trm_h.empty:
            trm_actual = float(_trm_h.iloc[-1])
            trm_serie  = _trm_h.reset_index(drop=True)

    if trm_actual <= 0:
        trm_actual = 4_200.0

    var_1m = 0.0
    if len(trm_serie) >= 22:
        trm_1m = float(trm_serie.iloc[-22])
        if trm_1m > 0:
            var_1m = round((trm_actual - trm_1m) / trm_1m * 100, 2)

    # Refs FX con compra sugerida y su costo
    n_refs_fx        = 0
    costo_compras_fx = 0.0
    if df_compras is not None and not df_compras.empty:
        moneda_map = (
            df_historico.sort_values("fecha")
            .groupby("referencia")["moneda"]
            .last()
        )
        dc       = df_compras.copy()
        dc["_m"] = dc["referencia"].map(moneda_map).fillna("COP")
        mask     = dc["_m"].isin(["USD", "EUR"]) & (dc["cantidad_sugerida"] > 0)
        n_refs_fx        = int(mask.sum())
        costo_compras_fx = float(dc.loc[mask, "costo_total"].fillna(0).sum())

    # Impacto: sobre las compras FX sugeridas (o gasto mensual FX histórico como proxy)
    base_impacto  = costo_compras_fx if costo_compras_fx > 0 else (gasto_fx / 12)
    impacto_5pct  = base_impacto * 0.05
    impacto_10pct = base_impacto * 0.10

    return {
        "trm_actual":    round(trm_actual, 0),
        "var_1m_pct":    var_1m,
        "gasto_usd_12m": gasto_usd,
        "gasto_eur_12m": gasto_eur,
        "gasto_fx_12m":  gasto_fx,
        "pct_fx":        pct_fx,
        "n_refs_fx":     n_refs_fx,
        "impacto_5pct":  impacto_5pct,
        "impacto_10pct": impacto_10pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. TOP ACCIONES INMEDIATAS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_top_acciones(
    df_compras: pd.DataFrame,
    df_historico: pd.DataFrame,
    df_op: pd.DataFrame = None,
    n: int = 10,
) -> pd.DataFrame:
    """
    Top N referencias con compra sugerida ordenadas por urgencia → criticidad → costo.

    Returns
    -------
    pd.DataFrame limpio listo para mostrar en UI
    """
    df = _enriquecer_costo(
        df_compras[df_compras["cantidad_sugerida"] > 0].copy(),
        df_historico,
    )

    desc_map = (
        df_historico.groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    df["descripcion"] = df["referencia"].map(desc_map).fillna("—")

    senal_map = {}
    if df_op is not None and not df_op.empty and "senal" in df_op.columns:
        senal_map = dict(zip(df_op["referencia"], df_op["senal"]))
    df["senal_precio"] = df["referencia"].map(senal_map).fillna("—")

    _CRIT = {"Alta": 3, "Media": 2, "Baja": 1}
    _EST  = {"🚨 Alerta": 4, "⚠️ Monitorear": 3, "✅ Cubierto (tránsito)": 2, "✅ Cubierto": 1}

    costo_max = df["costo_total"].max()
    df["_sort"] = (
        df.get("alerta_stock", pd.Series(False, index=df.index)).fillna(False).astype(int) * 1000
        + df.get("criticidad", pd.Series("Baja", index=df.index)).map(_CRIT).fillna(1) * 100
        + df.get("estado_stock", pd.Series("", index=df.index)).map(_EST).fillna(0) * 10
        + (df["costo_total"] / max(costo_max, 1) * 9).fillna(0)
    )

    out = df.sort_values("_sort", ascending=False).head(n).reset_index(drop=True)

    cols = [
        "referencia", "descripcion", "criticidad", "estado_stock",
        "meses_cobertura_real", "cantidad_sugerida", "costo_total",
        "senal_precio", "fuente_demanda",
    ]
    return out[[c for c in cols if c in out.columns]]


# ══════════════════════════════════════════════════════════════════════════════
# 6. ALERTAS EJECUTIVAS
# ══════════════════════════════════════════════════════════════════════════════

def calcular_alertas(
    df_compras: pd.DataFrame,
    df_historico: pd.DataFrame,
    df_gap_req: pd.DataFrame = None,
) -> list:
    """
    Genera lista de alertas ejecutivas priorizadas.

    Returns
    -------
    list de dict: [{tipo, titulo, descripcion, nivel}]
      nivel: 'critico' | 'warning' | 'info'
    """
    alertas = []
    up  = _ultimo_precio(df_historico)
    cob = df_compras["meses_cobertura_real"].apply(_safe_cob)

    # ── 1. Concentración de proveedor ──────────────────────────────────────────
    gasto_prov  = df_historico.groupby("proveedor")["precio_cop"].sum()
    total_gasto = float(gasto_prov.sum())
    if total_gasto > 0:
        top1_sh  = float(gasto_prov.max() / total_gasto)
        top1_nom = str(gasto_prov.idxmax())
        top2_sh  = float(gasto_prov.nlargest(2).sum() / total_gasto)
        if top1_sh > 0.60:
            alertas.append({
                "tipo":        "proveedor",
                "titulo":      "Concentración crítica de proveedor",
                "descripcion": (
                    f"**{top1_nom}** representa el **{top1_sh:.0%}** del gasto histórico total. "
                    "Ante desabasto o alza de precios, el impacto es directo sobre toda la operación."
                ),
                "nivel": "critico",
            })
        elif top2_sh > 0.80:
            alertas.append({
                "tipo":        "proveedor",
                "titulo":      "Concentración moderada de proveedores",
                "descripcion": (
                    f"Los 2 proveedores principales concentran el **{top2_sh:.0%}** del gasto. "
                    "Evaluar diversificación de fuentes de suministro."
                ),
                "nivel": "warning",
            })

    # ── 2. Gap operativo ───────────────────────────────────────────────────────
    if df_gap_req is not None and not df_gap_req.empty:
        criticos = df_gap_req[df_gap_req["clasificacion_gap"] == "Crítico"]
        altos    = df_gap_req[df_gap_req["clasificacion_gap"] == "Alto"]
        if not criticos.empty:
            reqs = ", ".join(criticos["requerimiento"].astype(str).tolist()[:3])
            alertas.append({
                "tipo":        "eficiencia",
                "titulo":      "Gap operativo crítico (> 60%)",
                "descripcion": (
                    f"Requerimientos con brecha crítica entre pedidos y consumo real: **{reqs}**. "
                    "Sobreestimación sistemática → riesgo de sobrecompra y capital inmovilizado."
                ),
                "nivel": "critico",
            })
        elif not altos.empty:
            alertas.append({
                "tipo":        "eficiencia",
                "titulo":      f"Gap operativo elevado en {len(altos)} requerimiento(s)",
                "descripcion": (
                    "Brecha del 30–60% entre lo que operaciones solicita y lo que los técnicos consumen. "
                    "Revisar proceso de pedidos antes de aprobar nuevas compras."
                ),
                "nivel": "warning",
            })

    # ── 3. Capital inmovilizado ────────────────────────────────────────────────
    df_exc    = df_compras[cob > 6].copy()
    df_exc["_up"] = df_exc["referencia"].map(up).fillna(0)
    capital   = float((df_exc["stock_actual"] * df_exc["_up"]).sum())
    if capital > 5_000_000 and len(df_exc) > 0:
        alertas.append({
            "tipo":        "capital",
            "titulo":      f"Capital potencialmente inmovilizado — {len(df_exc)} referencias",
            "descripcion": (
                f"Refs. con más de **6 meses de cobertura** representan aprox. "
                f"**{fmt_cop(capital)} COP** en inventario. "
                "Evaluar si corresponde a decisión estratégica o sobrestock a corregir."
            ),
            "nivel": "info",
        })

    # ── 4. Demanda insuficiente con stock bajo ─────────────────────────────────
    if "fuente_demanda" in df_compras.columns:
        mask_ins = (
            df_compras["fuente_demanda"].isin(["Insuficiente", "—", ""])
            & (cob <= 2.0)
            & (df_compras["stock_actual"] > 0)
        )
        n_ins = int(mask_ins.sum())
        if n_ins > 0:
            alertas.append({
                "tipo":        "datos",
                "titulo":      f"{n_ins} referencia(s) — baja cobertura sin historial suficiente",
                "descripcion": (
                    "Tienen ≤ 2 meses de cobertura pero no cuentan con historial de demanda "
                    "suficiente para modelar. Revisión manual recomendada antes de procesar compra."
                ),
                "nivel": "warning",
            })

    return alertas


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRECISIÓN ESTADÍSTICA DEL MODELO (requiere predicciones)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_precision_modelo(
    df_compras: pd.DataFrame,
    df_predicciones: pd.DataFrame,
) -> dict:
    """
    Traduce el MAPE del modelo estadístico a impacto monetario en COP.
    Responde: "¿Cuánto puede estar equivocado el modelo en el costo total proyectado?"

    Requiere df_predicciones con columnas [referencia, mape].

    Returns
    -------
    dict:
      mape_ponderado     : MAPE promedio ponderado por costo de compra
      incertidumbre_cop  : error esperado total sobre el portafolio (COP)
      costo_modelado     : costo total de refs con modelo activo (COP)
      n_refs_modeladas   : número de refs con modelo
      confiabilidad_dist : {'Alta': N, 'Media': N, 'Baja': N}
    """
    if df_predicciones is None or df_predicciones.empty:
        return {}

    mape_ref = df_predicciones.groupby("referencia")["mape"].first()

    dc = df_compras[df_compras["cantidad_sugerida"] > 0].copy()
    dc["mape"] = dc["referencia"].map(mape_ref)
    dc = dc.dropna(subset=["mape", "precio_proyectado_promedio"])
    dc = dc[dc["precio_proyectado_promedio"] > 0]

    if dc.empty:
        return {}

    w                = dc["precio_proyectado_promedio"] * dc["cantidad_sugerida"]
    mape_ponderado   = float((dc["mape"] * w).sum() / w.sum()) if w.sum() > 0 else float(dc["mape"].mean())
    incertidumbre    = float((dc["mape"] / 100 * dc["precio_proyectado_promedio"] * dc["cantidad_sugerida"]).sum())
    costo_modelado   = float(w.sum())

    def _conf(m):
        if m <= 12: return "Alta"
        if m <= 25: return "Media"
        return "Baja"

    dist = dc["mape"].apply(_conf).value_counts().to_dict()

    return {
        "mape_ponderado":    round(mape_ponderado, 1),
        "incertidumbre_cop": incertidumbre,
        "costo_modelado":    costo_modelado,
        "n_refs_modeladas":  len(dc),
        "confiabilidad_dist": dist,
    }
