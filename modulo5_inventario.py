"""
=============================================================
DICO S.A. — Módulo 5: Inventario, Compras Sugeridas y Gap de Eficiencia
=============================================================
Responsabilidades:
  1. Leer stock físico disponible (pbi_saldos, Tipo_bodega='Fisica')
  2. Calcular demanda mensual promedio a partir de actas de asignación + pedidos
  3. Determinar cantidad sugerida a comprar = ceil(demanda * meses_objetivo) - stock
  4. Estimar costo total usando el precio promedio proyectado del módulo predictivo
  5. Calcular gap de eficiencia operativa: pedidos vs consumo real (solo indicador)

Fuentes MariaDB:
  - pbi_saldos                                        → stock físico hoy
  - pbi_detalle_movimientos_actas_asignacion_2025/26  → materiales asignados (fecha: FECHA_CREACION)
  - pbi_detalle_movimientos_pedidos_2025/26            → lo que operaciones solicita (fecha: FECHA_CREACION)
  - pbi_detalle_movimientos_consumos_2025/26           → lo que los técnicos gastaron (fecha: FECHA_CREACION)

IMPORTANTE:
  - Consumos NO se usan para calcular compras. Solo para el indicador de gap.
  - Cantidades negativas en consumos son válidas (devoluciones) y se conservan.
  - No duplica credenciales: importa parámetros de conexión desde modulo1_etl.py

Autor: Data Science DICO S.A.
Versión: 1.0
=============================================================
"""

import math
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import pymysql
import streamlit as st

# ── Parámetros de conexión reutilizados desde modulo1 (sin duplicar credenciales) ──
from modulo1_etl import DB_HOST, DB_USUARIO, DB_CLAVE, DB_BD, DB_PUERTO

log = logging.getLogger(__name__)


# ─── Helper de conexión ────────────────────────────────────────────────────────

def _get_conn() -> pymysql.connections.Connection:
    """Crea y retorna una conexión pymysql con los parámetros de modulo1_etl."""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PUERTO,
        user=DB_USUARIO,
        password=DB_CLAVE,
        database=DB_BD,
        charset="utf8mb4",
        connect_timeout=30,
    )


# ─── Helper interno: limpieza de movimientos ──────────────────────────────────

def _limpiar_movimientos(df: pd.DataFrame, tipo: str, permite_negativos: bool) -> pd.DataFrame:
    """
    Normaliza un DataFrame de movimientos:
      - Tipifica columnas referencia, fecha, cantidad
      - Filtra fechas futuras (warning)
      - Filtra cantidades negativas si no son válidas (warning)
      - Rellena NaN en cantidad con 0
    """
    df = df.copy()
    df["referencia"] = df["referencia"].astype(str).str.strip()
    df["fecha"]      = pd.to_datetime(df["fecha"], errors="coerce")
    df["cantidad"]   = pd.to_numeric(df["cantidad"], errors="coerce")
    df["tipo"]       = tipo

    # Fechas futuras
    hoy = pd.Timestamp(datetime.today().date())
    n_futuras = (df["fecha"] > hoy).sum()
    if n_futuras:
        log.warning(f"[{tipo}] {n_futuras} registros con fecha futura eliminados.")
        df = df[df["fecha"] <= hoy]

    # Filas sin fecha
    df = df.dropna(subset=["fecha"])

    # Cantidades negativas (excluir solo en actas y pedidos)
    if not permite_negativos:
        n_neg = (df["cantidad"] < 0).sum()
        if n_neg:
            log.warning(f"[{tipo}] {n_neg} registros con cantidad negativa excluidos.")
            df = df[df["cantidad"] >= 0]

    df["cantidad"] = df["cantidad"].fillna(0)
    return df.reset_index(drop=True)


# ─── 1. Saldos físicos ────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_saldos() -> pd.DataFrame:
    """
    Lee pbi_saldos filtrando Tipo_bodega = 'Fisica'.
    Agrupa por referencia y suma Saldo_disponible.

    Retorna: DataFrame [referencia, stock_actual]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "descripcion", "stock_actual"])
    try:
        conn = _get_conn()
        # Intentar traer DESCRIPCION; si la columna no existe en la vista, caer al fallback.
        try:
            df = pd.read_sql(
                """
                SELECT codigo_tercero        AS referencia,
                       MAX(DESCRIPCION)      AS descripcion,
                       SUM(Saldo_disponible) AS stock_actual
                FROM   pbi_saldos
                WHERE  Tipo_bodega = 'Fisica'
                GROUP BY codigo_tercero
                """,
                conn,
            )
        except Exception:
            df = pd.read_sql(
                """
                SELECT codigo_tercero        AS referencia,
                       SUM(Saldo_disponible) AS stock_actual
                FROM   pbi_saldos
                WHERE  Tipo_bodega = 'Fisica'
                GROUP BY codigo_tercero
                """,
                conn,
            )
            df["descripcion"] = ""
        conn.close()
        df["referencia"]   = df["referencia"].astype(str).str.strip()
        df["stock_actual"] = pd.to_numeric(df["stock_actual"], errors="coerce").fillna(0)
        df["descripcion"]  = df["descripcion"].fillna("").astype(str).str.strip()
        log.info(f"[saldos] {len(df)} referencias con stock físico cargadas.")
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar saldos de inventario: {e}")
        log.error(f"[saldos] Error: {e}")
        return EMPTY


# ─── 2. Actas de asignación ───────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_actas() -> pd.DataFrame:
    """
    UNION ALL de pbi_detalle_movimientos_actas_asignacion_2025 y _2026.
    Representa materiales asignados a personal administrativo.

    Retorna: DataFrame [referencia, fecha, cantidad, tipo]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "fecha", "cantidad", "tipo"])
    SQL = """
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_actas_asignacion_2025
        UNION ALL
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_actas_asignacion_2026
    """
    try:
        conn = _get_conn()
        df = pd.read_sql(SQL, conn)
        conn.close()
        df = _limpiar_movimientos(df, tipo="acta", permite_negativos=False)
        log.info(f"[actas] {len(df)} registros cargados.")
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar actas de asignación: {e}")
        log.error(f"[actas] Error: {e}")
        return EMPTY


# ─── 3. Pedidos ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_pedidos() -> pd.DataFrame:
    """
    UNION ALL de pbi_detalle_movimientos_pedidos_2025 y _2026.
    Representa lo que operaciones solicita para comprar.

    Retorna: DataFrame [referencia, fecha, cantidad, tipo]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "fecha", "cantidad", "tipo"])
    SQL = """
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_pedidos_2025
        UNION ALL
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_pedidos_2026
    """
    try:
        conn = _get_conn()
        df = pd.read_sql(SQL, conn)
        conn.close()
        df = _limpiar_movimientos(df, tipo="pedido", permite_negativos=False)
        log.info(f"[pedidos] {len(df)} registros cargados.")
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar pedidos: {e}")
        log.error(f"[pedidos] Error: {e}")
        return EMPTY


# ─── 4. Consumos ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_consumos() -> pd.DataFrame:
    """
    Lee pbi_detalle_movimientos_consumos_2026.
    Representa lo que los técnicos realmente gastaron.
    Las cantidades negativas (devoluciones) se conservan.
    Se usa como señal primaria de demanda real.

    Retorna: DataFrame [referencia, fecha, cantidad, tipo]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "fecha", "cantidad", "tipo"])
    SQL = """
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_consumos_2025
        UNION ALL
        SELECT codigo_tercero AS referencia,
               FECHA_CREACION AS fecha,
               CANTIDAD       AS cantidad
        FROM   pbi_detalle_movimientos_consumos_2026
    """
    try:
        conn = _get_conn()
        df = pd.read_sql(SQL, conn)
        conn.close()
        df = _limpiar_movimientos(df, tipo="consumo", permite_negativos=True)
        log.info(f"[consumos] {len(df)} registros cargados.")
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar consumos: {e}")
        log.error(f"[consumos] Error: {e}")
        return EMPTY


# ─── 4b. Pendientes de arribo (compras en tránsito) ──────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def cargar_pendientes_arribo() -> pd.DataFrame:
    """
    Lee pbi_pendientes_arribo.
    Agrupa por referencia y suma CANTIDAD_COMPRADA para obtener
    el total de unidades que ya fueron compradas y están en camino.

    Retorna: DataFrame [referencia, cantidad_transito]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "cantidad_transito"])
    SQL = """
        SELECT REFERENCIA             AS referencia,
               SUM(CANTIDAD_COMPRADA) AS cantidad_transito
        FROM   pbi_pendientes_arribo
        GROUP BY REFERENCIA
    """
    try:
        conn = _get_conn()
        df = pd.read_sql(SQL, conn)
        conn.close()
        df["referencia"]        = df["referencia"].astype(str).str.strip()
        df["cantidad_transito"] = pd.to_numeric(df["cantidad_transito"], errors="coerce").fillna(0)
        log.info(f"[pendientes_arribo] {len(df)} referencias con stock en tránsito cargadas.")
        return df
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar pendientes de arribo: {e}")
        log.error(f"[pendientes_arribo] Error: {e}")
        return EMPTY


# ─── 5. Demanda mensual promedio ───────────────────────────────────────────────

def calcular_demanda_mensual(
    df_actas: pd.DataFrame,
    df_pedidos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina actas + pedidos, agrupa por referencia y mes, y promedia.

    Lógica:
      demanda_mes     = SUM(actas.cantidad) + SUM(pedidos.cantidad) por mes
      demanda_prom    = AVG(demanda_mes) sobre meses con al menos 1 movimiento

    Confiabilidad:
      >= 6 meses → 'Alta'
      >= 3 meses → 'Media'
      >= 2 meses → 'Baja'
      <  2 meses → 'Insuficiente' (se excluye de sugerencias de compra)

    Retorna: DataFrame [referencia, demanda_mensual_promedio, n_meses, confiabilidad]
    """
    EMPTY = pd.DataFrame(columns=[
        "referencia", "demanda_mensual_promedio", "n_meses", "confiabilidad"
    ])

    if df_actas.empty and df_pedidos.empty:
        return EMPTY

    movimientos = pd.concat(
        [df_actas[["referencia", "fecha", "cantidad"]],
         df_pedidos[["referencia", "fecha", "cantidad"]]],
        ignore_index=True,
    )

    movimientos["mes"] = movimientos["fecha"].dt.to_period("M")

    # Demanda total por referencia × mes
    por_mes = (
        movimientos
        .groupby(["referencia", "mes"])["cantidad"]
        .sum()
        .reset_index()
        .rename(columns={"cantidad": "demanda_mes"})
    )

    # Promedio y conteo de meses activos
    resumen = (
        por_mes
        .groupby("referencia")
        .agg(
            demanda_mensual_promedio=("demanda_mes", "mean"),
            n_meses=("demanda_mes", "count"),
        )
        .reset_index()
    )
    resumen["demanda_mensual_promedio"] = resumen["demanda_mensual_promedio"].round(2)

    def _confiabilidad(n: int) -> str:
        if n >= 6: return "Alta"
        if n >= 3: return "Media"
        if n >= 2: return "Baja"
        return "Insuficiente"

    resumen["confiabilidad"] = resumen["n_meses"].apply(_confiabilidad)
    return resumen


# ─── 5b. Demanda promedio últimos 3 meses ─────────────────────────────────────

def calcular_demanda_3meses(
    df_actas: pd.DataFrame,
    df_pedidos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la demanda promedio de los últimos 3 meses por referencia.
    Se usa para determinar si una referencia tiene 'uso activo reciente'.

    Una referencia tiene uso activo (promedio_alto=True) si:
      - tiene demanda > 0 en al menos 1 de los últimos 3 meses.

    Retorna: DataFrame [referencia, demanda_3meses_promedio, meses_activos_3m, promedio_alto]
    """
    EMPTY = pd.DataFrame(columns=[
        "referencia", "demanda_3meses_promedio", "meses_activos_3m", "promedio_alto"
    ])

    if df_actas.empty and df_pedidos.empty:
        return EMPTY

    hoy    = pd.Timestamp(datetime.today().date())
    hace3m = hoy - pd.DateOffset(months=3)

    frames = []
    if not df_actas.empty:
        frames.append(df_actas[["referencia", "fecha", "cantidad"]])
    if not df_pedidos.empty:
        frames.append(df_pedidos[["referencia", "fecha", "cantidad"]])
    if not frames:
        return EMPTY
    movimientos = pd.concat(frames, ignore_index=True)
    recientes = movimientos[movimientos["fecha"] >= hace3m].copy()

    if recientes.empty:
        return EMPTY

    recientes["mes"] = recientes["fecha"].dt.to_period("M")

    por_mes = (
        recientes
        .groupby(["referencia", "mes"])["cantidad"]
        .sum()
        .reset_index()
        .rename(columns={"cantidad": "demanda_mes"})
    )

    resumen = (
        por_mes
        .groupby("referencia")
        .agg(
            demanda_3meses_promedio=("demanda_mes", "mean"),
            meses_activos_3m=("demanda_mes", "count"),
        )
        .reset_index()
    )
    resumen["demanda_3meses_promedio"] = resumen["demanda_3meses_promedio"].round(2)
    # Promedio alto = tiene demanda activa en al menos 1 de los últimos 3 meses
    resumen["promedio_alto"] = resumen["demanda_3meses_promedio"] > 0
    return resumen


# ─── 5b2. Rotación de consumo ────────────────────────────────────────────────

def calcular_rotacion_consumo(df_consumos: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica referencias por rotación de consumo real (señal primaria).

    Criterio de alta rotación:
      1. Consumo neto positivo en ≥ 3 de los últimos 6 meses calendario
      2. Y consumo en al menos 1 de los últimos 2 meses

    Las devoluciones (cantidad negativa) no cuentan como consumo activo.
    El umbral 3/6 (50%) garantiza presencia regular sin ser demasiado estricto.
    El chequeo de 2 meses descarta referencias que pararon recientemente.

    Returns
    -------
    pd.DataFrame [referencia, meses_activos_6m, meses_activos_2m, alta_rotacion]
    """
    EMPTY = pd.DataFrame(columns=["referencia","meses_activos_6m","meses_activos_2m","alta_rotacion"])
    if df_consumos.empty:
        return EMPTY

    hoy    = pd.Timestamp(datetime.today().date())
    hace6m = hoy - pd.DateOffset(months=6)
    hace2m = hoy - pd.DateOffset(months=2)

    df = df_consumos.copy()
    df["mes"] = df["fecha"].dt.to_period("M")

    # Consumo neto por referencia × mes (solo meses con neto positivo)
    por_mes = (
        df.groupby(["referencia", "mes"])["cantidad"]
        .sum()
        .reset_index()
    )
    por_mes_pos = por_mes[por_mes["cantidad"] > 0]

    p6m = pd.Period(hace6m, "M")
    p2m = pd.Period(hace2m, "M")

    activos_6m = (
        por_mes_pos[por_mes_pos["mes"] >= p6m]
        .groupby("referencia")["mes"].nunique()
        .rename("meses_activos_6m")
    )
    activos_2m = (
        por_mes_pos[por_mes_pos["mes"] >= p2m]
        .groupby("referencia")["mes"].nunique()
        .rename("meses_activos_2m")
    )

    result = (
        activos_6m.to_frame()
        .join(activos_2m, how="outer")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    result["alta_rotacion"] = (
        (result["meses_activos_6m"] >= 3) & (result["meses_activos_2m"] >= 1)
    )
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def cargar_refs_alta_rotacion() -> set:
    """
    Devuelve el set de referencias con alta rotación de consumo.
    Cacheado 1 hora. Si falla, devuelve set vacío (no bloquea la app).
    """
    try:
        df_consumos = cargar_consumos()
        rotacion    = calcular_rotacion_consumo(df_consumos)
        return set(rotacion.loc[rotacion["alta_rotacion"], "referencia"].tolist())
    except Exception as e:
        log.warning(f"[rotacion] No se pudo calcular rotación: {e}")
        return set()


# ─── 5c. Demanda real basada en consumos (señal primaria) ─────────────────────

def calcular_demanda_real(
    df_consumos: pd.DataFrame,
    df_actas: pd.DataFrame,
    df_pedidos: pd.DataFrame,
    df_gap_ref: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calcula la demanda mensual promedio usando consumos como fuente primaria.

    Lógica de selección por referencia:
      1. Si hay ≥2 meses de consumo → usa consumos directamente (demanda real).
      2. Si hay consumos pero <2 meses → usa consumos disponibles.
      3. Si no hay consumos → fallback a actas+pedidos corregido por gap histórico.
         demanda_corregida = demanda_actas_pedidos × (1 - gap_pct/100)

    Fuente de demanda:
      'consumos'         → dato directo de técnicos
      'actas_corregido'  → actas+pedidos ajustados por factor de gap
      'actas'            → actas+pedidos sin corrección (sin gap histórico disponible)

    Retorna: DataFrame [referencia, demanda_mensual_promedio, demanda_std,
                        n_meses, confiabilidad, fuente_demanda]
    """
    EMPTY = pd.DataFrame(columns=[
        "referencia", "demanda_mensual_promedio", "demanda_std",
        "n_meses", "confiabilidad", "fuente_demanda",
    ])

    def _confiabilidad(n: int) -> str:
        if n >= 6: return "Alta"
        if n >= 3: return "Media"
        if n >= 2: return "Baja"
        return "Insuficiente"

    # ── Demanda por consumos ──
    dem_consumos = pd.DataFrame()
    if not df_consumos.empty:
        df_c = df_consumos[df_consumos["cantidad"] > 0].copy()  # excluir devoluciones
        df_c["mes"] = df_c["fecha"].dt.to_period("M")
        por_mes_c = (
            df_c.groupby(["referencia", "mes"])["cantidad"]
            .sum().reset_index().rename(columns={"cantidad": "demanda_mes"})
        )
        dem_consumos = (
            por_mes_c.groupby("referencia")
            .agg(
                demanda_mensual_promedio=("demanda_mes", "mean"),
                demanda_std=("demanda_mes", "std"),
                n_meses=("demanda_mes", "count"),
            )
            .reset_index()
        )
        dem_consumos["demanda_std"] = dem_consumos["demanda_std"].fillna(0).round(2)
        dem_consumos["demanda_mensual_promedio"] = dem_consumos["demanda_mensual_promedio"].round(2)
        dem_consumos["fuente_demanda"] = "consumos"
        dem_consumos["confiabilidad"] = dem_consumos["n_meses"].apply(_confiabilidad)

    # ── Demanda por actas + pedidos (fallback) ──
    dem_actas = calcular_demanda_mensual(df_actas, df_pedidos)
    if not dem_actas.empty:
        # Calcular std por actas+pedidos
        _frames = []
        if not df_actas.empty:
            _frames.append(df_actas[["referencia", "fecha", "cantidad"]])
        if not df_pedidos.empty:
            _frames.append(df_pedidos[["referencia", "fecha", "cantidad"]])
        mov = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame(
            columns=["referencia", "fecha", "cantidad"]
        )

        if not mov.empty:
            mov["mes"] = mov["fecha"].dt.to_period("M")
            std_actas = (
                mov.groupby(["referencia", "mes"])["cantidad"].sum()
                .reset_index()
                .groupby("referencia")["cantidad"]
                .std()
                .fillna(0)
                .rename("demanda_std")
            )
            dem_actas = dem_actas.merge(std_actas, on="referencia", how="left")
            dem_actas["demanda_std"] = dem_actas["demanda_std"].fillna(0)
        else:
            dem_actas["demanda_std"] = 0

        # Aplicar corrección por gap histórico
        gap_map = {}
        if df_gap_ref is not None and not df_gap_ref.empty and "gap_pct" in df_gap_ref.columns:
            gap_map = dict(zip(df_gap_ref["referencia"], df_gap_ref["gap_pct"] / 100))

        def _corregir_demanda(row):
            gap = gap_map.get(row["referencia"], 0)
            if gap > 0:
                return round(row["demanda_mensual_promedio"] * (1 - min(gap, 0.9)), 2)
            return row["demanda_mensual_promedio"]

        dem_actas["fuente_demanda"] = dem_actas["referencia"].apply(
            lambda r: "actas_corregido" if gap_map.get(r, 0) > 0 else "actas"
        )
        dem_actas["demanda_mensual_promedio"] = dem_actas.apply(_corregir_demanda, axis=1)

    # ── Combinar: consumos primero, fallback a actas ──
    if dem_consumos.empty and dem_actas.empty:
        return EMPTY

    cols = ["referencia", "demanda_mensual_promedio", "demanda_std",
            "n_meses", "confiabilidad", "fuente_demanda"]

    if dem_consumos.empty:
        return dem_actas[cols].reset_index(drop=True)

    if dem_actas.empty:
        return dem_consumos[cols].reset_index(drop=True)

    refs_con_consumos = set(dem_consumos["referencia"])
    dem_actas_solo = dem_actas[~dem_actas["referencia"].isin(refs_con_consumos)].copy()

    resultado = pd.concat(
        [dem_consumos[cols], dem_actas_solo[cols]],
        ignore_index=True,
    )
    return resultado


# ─── 5d. Criticidad automática por referencia ──────────────────────────────────

def calcular_criticidad(
    df_historico: pd.DataFrame,
    df_demanda: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Clasifica cada referencia en Alta / Media / Baja criticidad de forma automática.

    Criterios (ponderados):
      40% — Precio relativo (precio_promedio / mediana_global). Ítem costoso = más crítico.
      35% — Concentración de proveedor (1/n_proveedores). Un solo proveedor = más riesgo.
      25% — Frecuencia de compra (n_compras normalizadas). Alta rotación = más crítico.

    Umbrales:
      score > 0.60 → Alta   (ítem caro + proveedor único o escaso + alta rotación)
      score > 0.30 → Media
      score ≤ 0.30 → Baja

    Retorna: DataFrame [referencia, criticidad, score_criticidad]
    """
    EMPTY = pd.DataFrame(columns=["referencia", "criticidad", "score_criticidad"])
    if df_historico.empty or "precio_cop" not in df_historico.columns:
        return EMPTY

    if "proveedor" in df_historico.columns:
        stats = (
            df_historico.groupby("referencia")
            .agg(
                precio_promedio=("precio_cop", "mean"),
                n_compras=("precio_cop", "count"),
                n_proveedores=("proveedor", "nunique"),
            )
            .reset_index()
        )
    else:
        stats = (
            df_historico.groupby("referencia")
            .agg(
                precio_promedio=("precio_cop", "mean"),
                n_compras=("precio_cop", "count"),
            )
            .reset_index()
        )
        stats["n_proveedores"] = 1

    mediana_precio = stats["precio_promedio"].median()
    if mediana_precio == 0:
        return EMPTY

    # ── Normalización [0, 1] ──
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    stats["precio_rel"]   = stats["precio_promedio"] / mediana_precio
    stats["p_norm"]       = _norm(stats["precio_rel"])
    stats["freq_norm"]    = _norm(stats["n_compras"])
    stats["prov_conc"]    = 1 / stats["n_proveedores"]  # 1 proveedor → score 1.0
    stats["conc_norm"]    = _norm(stats["prov_conc"])

    # ── Score ponderado ──
    stats["score_criticidad"] = (
        0.40 * stats["p_norm"]
        + 0.25 * stats["freq_norm"]
        + 0.35 * stats["conc_norm"]
    ).round(3)

    def _nivel(s: float) -> str:
        if s > 0.60: return "Alta"
        if s > 0.30: return "Media"
        return "Baja"

    stats["criticidad"] = stats["score_criticidad"].apply(_nivel)

    return stats[["referencia", "criticidad", "score_criticidad"]].reset_index(drop=True)


# ─── 6. Gap de eficiencia operativa ───────────────────────────────────────────

def _clasificar_gap(pct: float) -> str:
    if pct <= 10: return "Eficiente"
    if pct <= 30: return "Moderado"
    if pct <= 60: return "Alto"
    return "Crítico"


def calcular_gap_eficiencia(
    df_pedidos: pd.DataFrame,
    df_consumos: pd.DataFrame,
    df_historico: pd.DataFrame = None,
) -> tuple:
    """
    Calcula el gap entre lo que operaciones pide y lo que los técnicos consumen.
    Un gap alto indica sobreestimación de necesidad real.

    Clasificación:
      <= 10% → Eficiente | <= 30% → Moderado | <= 60% → Alto | > 60% → Crítico

    Retorna: (df_gap_ref, df_gap_req)
      df_gap_ref: [referencia, total_pedido, total_consumido, gap_unidades, gap_pct, clasificacion_gap]
      df_gap_req: [requerimiento, total_pedido, total_consumido, gap_unidades, gap_pct, clasificacion_gap]
    """
    COLS_REF = ["referencia", "total_pedido", "total_consumido",
                "gap_unidades", "gap_pct", "clasificacion_gap"]
    COLS_REQ = ["requerimiento", "total_pedido", "total_consumido",
                "gap_unidades", "gap_pct", "clasificacion_gap"]
    EMPTY_REF = pd.DataFrame(columns=COLS_REF)
    EMPTY_REQ = pd.DataFrame(columns=COLS_REQ)

    if df_pedidos.empty:
        return EMPTY_REF, EMPTY_REQ

    # Totales por referencia
    tot_ped = (
        df_pedidos.groupby("referencia")["cantidad"]
        .sum().reset_index().rename(columns={"cantidad": "total_pedido"})
    )

    if not df_consumos.empty:
        tot_con = (
            df_consumos.groupby("referencia")["cantidad"]
            .sum().reset_index().rename(columns={"cantidad": "total_consumido"})
        )
    else:
        tot_con = pd.DataFrame(columns=["referencia", "total_consumido"])

    gap_ref = tot_ped.merge(tot_con, on="referencia", how="left")
    gap_ref["total_consumido"] = gap_ref["total_consumido"].fillna(0)
    gap_ref["gap_unidades"]    = gap_ref["total_pedido"] - gap_ref["total_consumido"]
    gap_ref["gap_pct"] = (
        gap_ref["gap_unidades"] / gap_ref["total_pedido"].clip(lower=1e-9) * 100
    ).round(1)
    gap_ref["clasificacion_gap"] = gap_ref["gap_pct"].apply(_clasificar_gap)
    gap_ref = gap_ref.sort_values("gap_pct", ascending=False).reset_index(drop=True)

    # Por requerimiento (usa df_historico para mapear referencia → requerimiento)
    if (df_historico is not None
            and not df_historico.empty
            and "requerimiento" in df_historico.columns):
        mapa = (
            df_historico[["referencia", "requerimiento"]]
            .drop_duplicates("referencia")
        )
        gap_rq = gap_ref.merge(mapa, on="referencia", how="left")
        gap_rq["requerimiento"] = gap_rq["requerimiento"].fillna("Sin requerimiento")

        gap_req = (
            gap_rq.groupby("requerimiento")
            .agg(
                total_pedido=("total_pedido", "sum"),
                total_consumido=("total_consumido", "sum"),
            )
            .reset_index()
        )
        gap_req["gap_unidades"]    = gap_req["total_pedido"] - gap_req["total_consumido"]
        gap_req["gap_pct"] = (
            gap_req["gap_unidades"] / gap_req["total_pedido"].clip(lower=1e-9) * 100
        ).round(1)
        gap_req["clasificacion_gap"] = gap_req["gap_pct"].apply(_clasificar_gap)
        gap_req = gap_req.sort_values("gap_pct", ascending=False).reset_index(drop=True)
    else:
        gap_req = EMPTY_REQ

    return gap_ref, gap_req


# ─── 7. Compras sugeridas ─────────────────────────────────────────────────────

def calcular_compras_sugeridas(
    df_saldos: pd.DataFrame,
    df_demanda: pd.DataFrame,
    df_predicciones: pd.DataFrame,
    meses_objetivo: int = 3,
    df_pendientes: pd.DataFrame = None,
    df_demanda_3m: pd.DataFrame = None,
    lead_time_meses: float = 1.0,
    df_criticidad: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Calcula cantidad_sugerida, cobertura real (stock + tránsito) y alertas por referencia.

    Columnas nuevas respecto a v1:
      cantidad_transito    : unidades en camino (pbi_pendientes_arribo)
      meses_cobertura_real : (stock + tránsito) / demanda_mensual
      safety_stock         : stock de seguridad estadístico Z×σ×√(lead_time)
      estado_stock         : '🚨 Alerta' | '✅ Cubierto' | '⚠️ Monitorear' | '➖ Sin demanda'
      alerta_stock         : bool — True si cobertura_real ≤ 1 mes y uso activo reciente
      cobertura_transito_ok: bool — True si cobertura_real ≥ 2 meses (no necesita compra)
      criticidad           : Alta / Media / Baja (score automático)
      fuente_demanda       : 'consumos' | 'actas_corregido' | 'actas'
      ahorro_estimado_cop  : (precio_actual - precio_optimo) × cantidad_sugerida

    Reglas de alerta (por criticidad):
      - Alta: alerta si cobertura_real ≤ 2 meses
      - Media: alerta si cobertura_real ≤ 1 mes
      - Baja:  alerta si cobertura_real ≤ 0.5 meses
    """
    EMPTY = pd.DataFrame(columns=[
        "referencia", "stock_actual", "cantidad_transito",
        "demanda_mensual_promedio", "demanda_std", "n_meses",
        "confiabilidad", "fuente_demanda", "criticidad",
        "cobertura_meses", "meses_cobertura_real", "safety_stock",
        "estado_stock", "alerta_stock", "cobertura_transito_ok",
        "cantidad_sugerida", "precio_proyectado_promedio",
        "mes_precio_optimo", "costo_total", "ahorro_estimado_cop",
    ])

    if df_demanda.empty:
        return EMPTY

    # Solo referencias con suficiente historial
    aptas = df_demanda[df_demanda["confiabilidad"] != "Insuficiente"].copy()
    if aptas.empty:
        return EMPTY

    # ── Join con saldos (referencia sin saldo → stock = 0) ──
    _saldos_cols = ["referencia", "stock_actual"]
    if not df_saldos.empty and "descripcion" in df_saldos.columns:
        _saldos_cols = ["referencia", "stock_actual", "descripcion"]
    df = aptas.merge(
        df_saldos[_saldos_cols] if not df_saldos.empty
        else pd.DataFrame(columns=["referencia", "stock_actual"]),
        on="referencia", how="left",
    )
    df["stock_actual"] = df["stock_actual"].fillna(0)
    if "descripcion" not in df.columns:
        df["descripcion"] = ""
    df["descripcion"] = df["descripcion"].fillna("").astype(str).str.strip()

    # ── Join con tránsito (pendientes de arribo) ──
    if df_pendientes is not None and not df_pendientes.empty:
        df = df.merge(df_pendientes[["referencia", "cantidad_transito"]],
                      on="referencia", how="left")
    else:
        df["cantidad_transito"] = 0.0
    df["cantidad_transito"] = df["cantidad_transito"].fillna(0)

    # ── Join con demanda reciente (últimos 3 meses) ──
    if df_demanda_3m is not None and not df_demanda_3m.empty:
        df = df.merge(
            df_demanda_3m[["referencia", "promedio_alto"]],
            on="referencia", how="left",
        )
    else:
        df["promedio_alto"] = False
    df["promedio_alto"] = df["promedio_alto"].fillna(False)

    # ── Join con criticidad ──
    if df_criticidad is not None and not df_criticidad.empty:
        df = df.merge(df_criticidad[["referencia", "criticidad", "score_criticidad"]],
                      on="referencia", how="left")
    else:
        df["criticidad"]       = "Media"
        df["score_criticidad"] = 0.5
    df["criticidad"] = df["criticidad"].fillna("Media")

    # ── Propagar demanda_std si viene en df_demanda ──
    if "demanda_std" not in df.columns:
        df["demanda_std"] = 0.0
    df["demanda_std"] = df["demanda_std"].fillna(0)

    # ── Propagar fuente_demanda si viene en df_demanda ──
    if "fuente_demanda" not in df.columns:
        df["fuente_demanda"] = "desconocido"

    # ── Cobertura con solo stock ──
    def _cobertura(row):
        if row["demanda_mensual_promedio"] == 0:
            return float("inf")
        return round(row["stock_actual"] / row["demanda_mensual_promedio"], 1)

    df["cobertura_meses"] = df.apply(_cobertura, axis=1)

    # ── Cobertura real (stock + tránsito) ──
    def _cobertura_real(row):
        if row["demanda_mensual_promedio"] == 0:
            return float("inf")
        return round(
            (row["stock_actual"] + row["cantidad_transito"]) / row["demanda_mensual_promedio"], 1
        )

    df["meses_cobertura_real"] = df.apply(_cobertura_real, axis=1)

    # ── Safety stock estadístico: Z × σ_demanda × √(lead_time) ──
    # Z = 1.65 para nivel de servicio del 95%
    _Z_95 = 1.65
    df["safety_stock"] = df["demanda_std"].apply(
        lambda s: max(0, math.ceil(_Z_95 * s * (lead_time_meses ** 0.5)))
    )

    # ── Umbral de alerta según criticidad ──
    _UMBRAL_ALERTA = {"Alta": 2.0, "Media": 1.0, "Baja": 0.5}

    df["umbral_alerta"] = df["criticidad"].map(_UMBRAL_ALERTA).fillna(1.0)

    # ── Flags de alerta ──
    df["alerta_stock"] = (
        (df["meses_cobertura_real"] <= df["umbral_alerta"]) &
        (df["promedio_alto"] == True)
    )
    df["cobertura_transito_ok"] = df["meses_cobertura_real"] >= 2

    # ── Estado legible ──
    def _estado(row):
        if row["demanda_mensual_promedio"] == 0:
            return "➖ Sin demanda"
        if row["cobertura_transito_ok"]:
            return "✅ Cubierto (tránsito)" if row["cantidad_transito"] > 0 else "✅ Cubierto"
        if row["alerta_stock"]:
            return "🚨 Alerta — comprar urgente"
        return "⚠️ Monitorear"

    df["estado_stock"] = df.apply(_estado, axis=1)

    # ── Cantidad sugerida (incluye safety stock) ──
    def _cantidad(row):
        if row["demanda_mensual_promedio"] == 0:
            return 0
        if row["cobertura_transito_ok"]:
            return 0
        # Meses meta: al menos 2 si hay alerta; meses_objetivo en el resto
        meses_meta = max(2, meses_objetivo) if row["alerta_stock"] else meses_objetivo
        base = row["demanda_mensual_promedio"] * meses_meta + row["safety_stock"]
        return max(0, math.ceil(base - row["stock_actual"] - row["cantidad_transito"]))

    df["cantidad_sugerida"] = df.apply(_cantidad, axis=1)

    # ── Precio óptimo proyectado (mínimo en ventana) ──
    df["precio_proyectado_promedio"] = np.nan
    df["mes_precio_optimo"]          = pd.NaT

    if df_predicciones is not None and not df_predicciones.empty:
        hoy    = pd.Timestamp(datetime.today().date())
        limite = hoy + pd.DateOffset(months=meses_objetivo)

        pred_ventana = df_predicciones[
            (df_predicciones["fecha_pred"] >= hoy) &
            (df_predicciones["fecha_pred"] <= limite)
        ]

        if not pred_ventana.empty:
            idx_min    = pred_ventana.groupby("referencia")["yhat"].idxmin()
            precio_opt = pred_ventana.loc[idx_min.values, ["referencia", "yhat", "fecha_pred"]].copy()
            precio_opt.columns = ["referencia", "precio_proyectado_promedio", "mes_precio_optimo"]
            df = df.merge(precio_opt, on="referencia", how="left", suffixes=("_drop", ""))
            if "precio_proyectado_promedio_drop" in df.columns:
                df.drop(columns=["precio_proyectado_promedio_drop"], inplace=True)
            if "mes_precio_optimo_drop" in df.columns:
                df.drop(columns=["mes_precio_optimo_drop"], inplace=True)

    # ── Costo total ──
    df["costo_total"] = df.apply(
        lambda r: (round(r["cantidad_sugerida"] * r["precio_proyectado_promedio"])
                   if pd.notna(r["precio_proyectado_promedio"]) and r["cantidad_sugerida"] > 0
                   else np.nan),
        axis=1,
    )

    # ── Ahorro estimado si se compra en el mes óptimo ──
    # Precio actual = último precio pagado (no disponible aquí; se calcula en app)
    # Dejamos la columna como NaN; la app la enriquece con precio actual del histórico
    df["ahorro_estimado_cop"] = np.nan

    return df.sort_values("costo_total", ascending=False, na_position="last").reset_index(drop=True)


# ─── 8. Pipeline completo ─────────────────────────────────────────────────────

def generar_reporte_inventario(
    df_historico: pd.DataFrame,
    df_predicciones: pd.DataFrame,
    meses_objetivo: int = 3,
    lead_time_meses: float = 1.0,
) -> dict:
    """
    Orquesta todo el pipeline de inventario y retorna un dict con:
      'compras' → DataFrame de compras sugeridas ordenado por costo DESC
      'gap_ref' → DataFrame de gap por referencia
      'gap_req' → DataFrame de gap por requerimiento

    Parameters
    ----------
    df_historico    : DataFrame de compras históricas (de modulo1_etl) — para mapear
                      referencia → requerimiento en el análisis de gap por requerimiento.
    df_predicciones : DataFrame de predicciones masivas (de modulo3_modelo) — para
                      calcular precio proyectado promedio.
    meses_objetivo  : Cuántos meses de cobertura queremos garantizar con la compra.
    """
    EMPTY_RESULT = {
        "compras":    pd.DataFrame(),
        "gap_ref":    pd.DataFrame(),
        "gap_req":    pd.DataFrame(),
        "criticidad": pd.DataFrame(),
    }

    try:
        df_saldos     = cargar_saldos()
        df_actas      = cargar_actas()
        df_pedidos    = cargar_pedidos()
        df_consumos   = cargar_consumos()
        df_pendientes = cargar_pendientes_arribo()
    except Exception as e:
        log.error(f"[generar_reporte] Error cargando tablas base: {e}")
        st.warning(f"⚠️ Error cargando datos de inventario: {e}")
        return EMPTY_RESULT

    try:
        # Gap primero (se necesita para corregir la demanda de actas)
        df_gap_ref, df_gap_req = calcular_gap_eficiencia(
            df_pedidos, df_consumos, df_historico
        )
    except Exception as e:
        log.error(f"[generar_reporte] Error en gap_eficiencia: {e}")
        df_gap_ref = pd.DataFrame()
        df_gap_req = pd.DataFrame()

    try:
        # Demanda real: consumos como señal primaria, actas corregidas por gap como fallback
        df_demanda = calcular_demanda_real(
            df_consumos, df_actas, df_pedidos, df_gap_ref=df_gap_ref
        )
    except Exception as e:
        log.error(f"[generar_reporte] Error en calcular_demanda_real: {e}")
        df_demanda = pd.DataFrame()

    try:
        df_demanda_3m = calcular_demanda_3meses(df_actas, df_pedidos)
    except Exception as e:
        log.error(f"[generar_reporte] Error en calcular_demanda_3meses: {e}")
        df_demanda_3m = pd.DataFrame()

    try:
        df_criticidad = calcular_criticidad(df_historico)
    except Exception as e:
        log.error(f"[generar_reporte] Error en calcular_criticidad: {e}")
        df_criticidad = pd.DataFrame()

    try:
        df_compras = calcular_compras_sugeridas(
            df_saldos, df_demanda, df_predicciones, meses_objetivo,
            df_pendientes=df_pendientes,
            df_demanda_3m=df_demanda_3m,
            lead_time_meses=lead_time_meses,
            df_criticidad=df_criticidad,
        )
    except Exception as e:
        log.error(f"[generar_reporte] Error en calcular_compras_sugeridas: {e}")
        st.warning(f"⚠️ Error calculando compras sugeridas: {e}")
        df_compras = pd.DataFrame()

    return {
        "compras":     df_compras,
        "gap_ref":     df_gap_ref,
        "gap_req":     df_gap_req,
        "criticidad":  df_criticidad,
    }
