"""
=============================================================
DICO S.A. — Módulo 1: ETL & Carga de Datos
=============================================================
Responsabilidades:
  1. Cargar datos históricos de compras desde MariaDB
  2. Normalizar columnas (fechas, precios, monedas)
  3. Obtener TRM histórica desde API datos.gov.co (OData)
  4. Convertir precios USD → COP cuando aplique
  5. Entregar un DataFrame limpio y estandarizado

Autor: Data Science DICO S.A.
Versión: 2.0  (fuente: MariaDB — pbi_ordenes_compra)
=============================================================
"""

import os
import re
import sys
import configparser
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import logging

import pandas as pd
import numpy as np
import requests
import pymysql

warnings.filterwarnings("ignore")

# ─── Configuración de logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Rutas base (portables, independientes del equipo) ───────────────────────
RUTA_BASE = Path(__file__).parent

_config = configparser.ConfigParser()
_config.read(RUTA_BASE / "config.ini", encoding="utf-8")

CACHE_TRM = str(RUTA_BASE / _config.get("datos", "cache_trm", fallback="trm_cache.parquet"))

# ─── Conexión MariaDB ─────────────────────────────────────────────────────────
# Prioridad: Streamlit Secrets (nube) → config.ini (local) → fallback hardcoded
def _get_db_param(key: str, fallback: str, as_int: bool = False):
    """Lee parámetro de BD desde Streamlit Secrets si está disponible, si no desde config.ini."""
    try:
        import streamlit as st
        val = st.secrets["base_datos"][key]
        return int(val) if as_int else str(val)
    except Exception:
        if as_int:
            return _config.getint("base_datos", key, fallback=int(fallback))
        return _config.get("base_datos", key, fallback=fallback)

DB_HOST     = _get_db_param("host",     "46.231.126.165")
DB_USUARIO  = _get_db_param("usuario",  "pbi_user")
DB_CLAVE    = _get_db_param("clave",    "4JQi8C72;.5}")
DB_BD       = _get_db_param("bd",       "dico_siid_col")
DB_CONSULTA = _get_db_param("consulta", "pbi_ordenes_compra")
DB_PUERTO   = _get_db_param("puerto",   "3306", as_int=True)

# ─── Constantes ──────────────────────────────────────────────────────────────
TRM_ODATA_URL = "https://www.datos.gov.co/api/odata/v4/ceyp-9c7c"
EUR_USD_RATE   = 1.08  # Factor de conversión EUR → USD (ajustable)

# Mapeo flexible de nombres de columnas → nombre estándar interno
COLUMN_MAP = {
    # ── Columnas reales de pbi_ordenes_compra (MariaDB) ────
    "fecha_orden_compra": "fecha",
    "valor_compra":       "precio_original",
    "vr_mone":            "trm_compra",
    "deta":               "descripcion",
    "mone":               "moneda",
    "referencia":         "referencia",
    "nit":                "nit",
    "proveedor":          "proveedor",
    "requerimiento":      "requerimiento",
    "entregado":          "cantidad",

    # ── Variantes alternativas ─────────────────────────────
    "fecha":                    "fecha",
    "precio":                   "precio_original",
    "moneda":                   "moneda",
    "descripcion":              "descripcion",
    "trm":                      "trm_compra",
    "fecha de compra":          "fecha",
    "fecha_compra":             "fecha",
    "precio de compra":         "precio_original",
    "precio_compra":            "precio_original",
    "valor":                    "precio_original",
    "valor unitario":           "precio_original",
    "tipo moneda":              "moneda",
    "currency":                 "moneda",
    "nombre proveedor":         "proveedor",
    "referencia interna":       "referencia",
    "código":                   "referencia",
    "descripción":              "descripcion",
    "descripcion del elemento": "descripcion",
    "tasa de cambio":           "trm_compra",
    "tasa cambio":              "trm_compra",
    "nit proveedor":            "nit",
}

COLUMNAS_REQUERIDAS = ["fecha", "precio_original", "moneda", "referencia"]


# ─── 1. Carga desde MariaDB ───────────────────────────────────────────────────

def cargar_desde_mariadb(
    host: str     = DB_HOST,
    usuario: str  = DB_USUARIO,
    clave: str    = DB_CLAVE,
    bd: str       = DB_BD,
    consulta: str = DB_CONSULTA,
    puerto: int   = DB_PUERTO,
) -> pd.DataFrame:
    """
    Carga los datos históricos de compras directamente desde MariaDB.

    Ejecuta ``SELECT * FROM <consulta>`` y devuelve el resultado como
    DataFrame crudo con las columnas originales de la BD.

    Parameters
    ----------
    host, usuario, clave, bd, consulta, puerto
        Parámetros de conexión (por defecto toma valores de config.ini).

    Returns
    -------
    pd.DataFrame
        DataFrame crudo con las columnas originales de pbi_ordenes_compra.
    """
    log.info(f"Conectando a MariaDB: {host}:{puerto} / {bd}")

    try:
        conn = pymysql.connect(
            host=host,
            port=puerto,
            user=usuario,
            password=clave,
            database=bd,
            charset="utf8mb4",
            connect_timeout=30,
        )
        log.info(f"Conexión establecida. Consultando: {consulta}")

        # Solo las columnas necesarias y desde 2018 en adelante
        columnas_bd = (
            "fecha_orden_compra, valor_compra, vr_mone, deta, "
            "mone, referencia, nit, proveedor, requerimiento, entregado"
        )
        df = pd.read_sql(
            f"SELECT {columnas_bd} FROM {consulta} "
            f"WHERE fecha_orden_compra >= '2017-01-01'",
            conn
        )
        conn.close()

        log.info(f"Datos cargados: {len(df):,} filas × {len(df.columns)} columnas")
        log.info(f"Columnas encontradas: {list(df.columns)}")
        return df

    except pymysql.err.OperationalError as e:
        raise ConnectionError(f"No se pudo conectar a MariaDB ({host}:{puerto}): {e}")
    except Exception as e:
        raise RuntimeError(f"Error cargando datos desde MariaDB: {e}")


# ─── 2. Normalización de columnas ─────────────────────────────────────────────

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas del Excel al estándar interno usando COLUMN_MAP.
    Es resistente a mayúsculas, tildes y espacios extras.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        log.warning(f"Columnas no encontradas automáticamente: {faltantes}")
        log.warning(f"Columnas disponibles: {list(df.columns)}")
        raise ValueError(
            f"No se encontraron las columnas requeridas: {faltantes}\n"
            f"Columnas disponibles: {list(df.columns)}\n"
            f"Agrega el mapeo correcto en COLUMN_MAP."
        )

    log.info("Columnas normalizadas correctamente.")
    return df


# ─── 3. Limpieza y validación de datos ────────────────────────────────────────

def limpiar_precio(valor) -> float:
    """
    Limpieza inteligente de precios:
    - Si ya es número (int/float) → lo usa directamente
    - Si es string con formato colombiano ($71.140,48) → convierte correctamente
    - Si es string con punto decimal (71140.48) → convierte directamente
    """
    if isinstance(valor, (int, float)):
        return float(valor)
    s = str(valor).strip()
    s = re.sub(r"[$\s]", "", s)
    if not s or s.lower() == "nan":
        return float("nan")
    if "," in s and "." in s:
        # Formato colombiano (71.140,48): la coma viene después del punto
        if s.rindex(",") > s.rindex("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            # Formato inglés (71,140.48): quitar coma de miles
            s = s.replace(",", "")
    elif "," in s:
        partes = s.split(",")
        if len(partes) == 2 and len(partes[1]) <= 2:
            s = s.replace(",", ".")   # Coma decimal (71140,48)
        else:
            s = s.replace(",", "")   # Coma de miles (71,140)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza integral del DataFrame:
      - Parseo de fechas
      - Conversión numérica de precios
      - Normalización de moneda
      - Eliminación de nulos críticos
    """
    df = df.copy()
    filas_iniciales = len(df)

    # ── Fechas ──
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    nulos_fecha = df["fecha"].isna().sum()
    if nulos_fecha > 0:
        log.warning(f"Se eliminaron {nulos_fecha} filas con fecha inválida.")
    df = df.dropna(subset=["fecha"])

    # ── Precios ──
    df["precio_original"] = df["precio_original"].apply(limpiar_precio)
    nulos_precio = df["precio_original"].isna().sum()
    if nulos_precio > 0:
        log.warning(f"Se eliminaron {nulos_precio} filas con precio inválido.")
    df = df.dropna(subset=["precio_original"])
    df = df[df["precio_original"] > 0]

    # ── Moneda ──
    df["moneda"] = df["moneda"].astype(str).str.strip().str.upper()
    df["moneda"] = df["moneda"].replace({
        "PESO": "COP", "PESOS": "COP", "COLOMBIANO": "COP",
        "DOLLAR": "USD", "DOLARES": "USD", "DÓLAR": "USD", "DÓLARES": "USD",
        "EURO": "EUR", "EUROS": "EUR"
    })

    # ── Proveedor, Referencia y Descripción ──
    for col in ["proveedor", "referencia", "descripcion"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # ── Cantidad entregada ──
    if "cantidad" in df.columns:
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0)

    # ── TRM de compra ──
    if "trm_compra" in df.columns:
        df["trm_compra"] = pd.to_numeric(
            df["trm_compra"].astype(str).str.replace(r"[$,\s]", "", regex=True),
            errors="coerce"
        )
    else:
        df["trm_compra"] = np.nan

    # ── Ordenar por fecha ──
    df = df.sort_values("fecha").reset_index(drop=True)

    filas_finales = len(df)
    log.info(f"Limpieza completada: {filas_iniciales:,} → {filas_finales:,} filas "
             f"({filas_iniciales - filas_finales:,} eliminadas)")
    log.info(f"Rango de fechas: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    log.info(f"Referencias únicas: {df['referencia'].nunique():,}")
    log.info(f"Distribución de moneda:\n{df['moneda'].value_counts().to_string()}")

    return df


# ─── 4. API TRM ───────────────────────────────────────────────────────────────

def obtener_trm_historica(
    fecha_inicio: str = None,
    fecha_fin: str = None,
    cache_path: str = CACHE_TRM
) -> pd.DataFrame:
    """
    Descarga la TRM histórica desde la API OData de datos.gov.co.
    Implementa caché local para evitar llamadas repetidas.

    Parameters
    ----------
    fecha_inicio : str, optional
        Formato 'YYYY-MM-DD'. Default: hace 5 años.
    fecha_fin : str, optional
        Formato 'YYYY-MM-DD'. Default: hoy.
    cache_path : str
        Ruta del archivo de caché local.

    Returns
    -------
    pd.DataFrame
        Columnas: fecha (datetime), trm (float)
    """
    hoy = datetime.today().date()

    if fecha_inicio is None:
        fecha_inicio = str(hoy - timedelta(days=5 * 365))
    if fecha_fin is None:
        fecha_fin = str(hoy)

    # ── Verificar caché ──
    if os.path.exists(cache_path):
        cache_df = pd.read_parquet(cache_path)
        cache_max = cache_df["fecha"].max().date()
        dias_desactualizados = (hoy - cache_max).days

        if dias_desactualizados <= 1:
            log.info(f"TRM desde caché ({len(cache_df):,} registros). Última fecha: {cache_max}")
            return cache_df

        log.info(f"Caché desactualizado ({dias_desactualizados} días). Actualizando TRM...")
        fecha_inicio_actualiz = str(cache_max + timedelta(days=1))
        trm_nueva = _descargar_trm_api(fecha_inicio_actualiz, fecha_fin)
        if not trm_nueva.empty:
            cache_df = pd.concat([cache_df, trm_nueva]).drop_duplicates("fecha")
            cache_df.to_parquet(cache_path, index=False)
        return cache_df

    # ── Primera descarga ──
    log.info(f"Descargando TRM histórica: {fecha_inicio} → {fecha_fin}")
    trm_df = _descargar_trm_api(fecha_inicio, fecha_fin)

    if not trm_df.empty:
        trm_df.to_parquet(cache_path, index=False)
        log.info(f"TRM guardada en caché: {cache_path}")

    return trm_df


def _descargar_trm_api(fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Descarga datos de TRM desde OData en lotes (máx 1000 registros por petición).
    """
    todos_los_registros = []
    skip = 0
    limit = 1000

    while True:
        filtro = (
            f"vigenciadesde ge datetime'{fecha_inicio}T00:00:00' "
            f"and vigenciadesde le datetime'{fecha_fin}T23:59:59'"
        )
        params = {
            "$filter": filtro,
            "$select": "vigenciadesde,valor",
            "$orderby": "vigenciadesde asc",
            "$top": limit,
            "$skip": skip,
        }

        try:
            response = requests.get(TRM_ODATA_URL, params=params, timeout=30)
            response.raise_for_status()
            registros = response.json().get("value", [])

            if not registros:
                break

            todos_los_registros.extend(registros)
            log.info(f"  TRM descargada: {len(todos_los_registros):,} registros...")

            if len(registros) < limit:
                break
            skip += limit

        except requests.exceptions.ConnectionError:
            log.error("Sin conexión a internet. Usando TRM de respaldo.")
            return pd.DataFrame()
        except Exception as e:
            log.error(f"Error en API TRM: {e}")
            return pd.DataFrame()

    if not todos_los_registros:
        log.warning("No se obtuvieron datos de TRM.")
        return pd.DataFrame()

    trm_df = pd.DataFrame(todos_los_registros)
    trm_df = trm_df.rename(columns={"vigenciadesde": "fecha", "valor": "trm"})
    trm_df["fecha"] = pd.to_datetime(trm_df["fecha"], errors="coerce")
    trm_df["trm"] = pd.to_numeric(trm_df["trm"], errors="coerce")
    trm_df = trm_df.dropna().sort_values("fecha").reset_index(drop=True)

    log.info(f"TRM descargada: {len(trm_df):,} registros | "
             f"Rango TRM: {trm_df['trm'].min():,.0f} – {trm_df['trm'].max():,.0f}")
    return trm_df


def rellenar_trm_faltante(trm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una TRM para cada día calendario (forward fill para fines de semana/festivos).
    """
    if trm_df.empty:
        return trm_df

    rango_fechas = pd.date_range(start=trm_df["fecha"].min(), end=trm_df["fecha"].max(), freq="D")
    trm_completa = pd.DataFrame({"fecha": rango_fechas})
    trm_completa = trm_completa.merge(trm_df, on="fecha", how="left")
    trm_completa["trm"] = trm_completa["trm"].ffill().bfill()

    return trm_completa


# ─── 5. Conversión de precios a COP ───────────────────────────────────────────

def convertir_precios_a_cop(
    df: pd.DataFrame,
    trm_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convierte todos los precios a pesos colombianos (COP).

    Lógica:
      - COP → sin conversión
      - USD → precio × TRM del día de compra
      - EUR → precio × TRM_USD × EUR_USD_RATE
      - Otros → se deja como está con advertencia

    Si no hay TRM disponible para una fecha, usa la TRM más cercana (forward fill).
    """
    df = df.copy()

    if trm_df.empty:
        log.warning("TRM vacía. Los precios USD no serán convertidos correctamente.")
        df["precio_cop"] = df["precio_original"]
        df["trm_aplicada"] = np.nan
        return df

    # TRM diaria como Serie indexada por fecha para lookup vectorizado
    trm_diaria = rellenar_trm_faltante(trm_df).set_index("fecha")["trm"]

    # Lookup vectorizado: reindex por fecha normalizada, rellenar huecos residuales
    fechas_norm = df["fecha"].dt.normalize()
    df["trm_aplicada"] = trm_diaria.reindex(fechas_norm).ffill().bfill().values

    condiciones = [
        df["moneda"] == "COP",
        df["moneda"] == "USD",
        df["moneda"] == "EUR",
    ]
    valores = [
        df["precio_original"],
        df["precio_original"] * df["trm_aplicada"],
        df["precio_original"] * df["trm_aplicada"] * EUR_USD_RATE,
    ]

    df["precio_cop"] = np.select(condiciones, valores, default=df["precio_original"])

    monedas_usd = (df["moneda"] == "USD").sum()
    monedas_cop = (df["moneda"] == "COP").sum()
    log.info(f"Conversión completada: {monedas_cop:,} COP | {monedas_usd:,} USD convertidos")

    return df


# ─── 5b. Detección de outliers de precio ──────────────────────────────────────

def detectar_outliers_precio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca precios anómalos por referencia usando el rango intercuartílico (IQR).

    Un precio se considera outlier si está fuera de:
        [Q1 - 2.5 × IQR,  Q3 + 2.5 × IQR]

    Los outliers se CONSERVAN en el dataset (visibilidad histórica completa) pero
    se marcan con ``es_outlier = True`` para que el motor predictivo los excluya
    del entrenamiento, evitando que emergencias o descuentos excepcionales sesguen
    la proyección de precios.

    Requisito mínimo: la referencia debe tener al menos 4 compras; con menos puntos
    el IQR no es estadísticamente confiable y se omite el marcado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame procesado con columna ``precio_cop``.

    Returns
    -------
    pd.DataFrame
        Mismo DataFrame con columna booleana ``es_outlier`` agregada.
    """
    df = df.copy()
    df["es_outlier"] = False

    for ref, grupo in df.groupby("referencia"):
        if len(grupo) < 4:
            continue
        Q1  = grupo["precio_cop"].quantile(0.25)
        Q3  = grupo["precio_cop"].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 2.5 * IQR
        upper = Q3 + 2.5 * IQR
        mask_outlier = (grupo["precio_cop"] < lower) | (grupo["precio_cop"] > upper)
        df.loc[grupo[mask_outlier].index, "es_outlier"] = True

    n_outliers = int(df["es_outlier"].sum())
    if n_outliers:
        pct = n_outliers / len(df) * 100
        log.warning(f"Outliers de precio: {n_outliers:,} registros ({pct:.1f}%) marcados — "
                    f"se excluirán del entrenamiento predictivo.")
    else:
        log.info("Sin outliers de precio detectados.")
    return df


# ─── 6. Pipeline completo ─────────────────────────────────────────────────────

def ejecutar_pipeline_etl(
    cache_trm: str = CACHE_TRM
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline ETL completo. Punto de entrada principal.

    Carga los datos desde MariaDB (pbi_ordenes_compra), los normaliza,
    limpia y convierte precios a COP usando la TRM histórica.

    Returns
    -------
    df_compras : pd.DataFrame
        DataFrame limpio de compras con precio_cop incluido.
    df_trm : pd.DataFrame
        TRM histórica completa (fecha diaria).
    """
    log.info("=" * 55)
    log.info("INICIANDO PIPELINE ETL — DICO S.A.")
    log.info("=" * 55)

    df_raw    = cargar_desde_mariadb()
    df_norm   = normalizar_columnas(df_raw)
    df_limpio = limpiar_datos(df_norm)

    fecha_min = df_limpio["fecha"].min().strftime("%Y-%m-%d")
    fecha_max = datetime.today().strftime("%Y-%m-%d")
    df_trm = obtener_trm_historica(fecha_min, fecha_max, cache_trm)

    df_final = convertir_precios_a_cop(df_limpio, df_trm)
    df_final = detectar_outliers_precio(df_final)

    # trm_historica = misma TRM ya calculada en convertir_precios_a_cop
    if not df_trm.empty:
        df_final["trm_historica"] = df_final["trm_aplicada"]

    log.info("=" * 55)
    log.info("PIPELINE ETL COMPLETADO EXITOSAMENTE")
    log.info(f"Dataset final: {len(df_final):,} filas")
    log.info("=" * 55)

    return df_final, df_trm


# ─── 7. Diagnóstico de calidad ────────────────────────────────────────────────

def reporte_calidad(df: pd.DataFrame) -> dict:
    """
    Genera un reporte de calidad del dataset para mostrar en la app.

    Returns
    -------
    dict con métricas de calidad del dataset.
    """
    refs_con_pocos_datos = (
        df.groupby("referencia").size()
        .reset_index(name="n")
        .query("n < 10")
        ["referencia"].tolist()
    )

    reporte = {
        "total_registros":         len(df),
        "total_referencias":       df["referencia"].nunique(),
        "referencias_pocos_datos": len(refs_con_pocos_datos),
        "rango_fechas": {
            "inicio": df["fecha"].min().strftime("%Y-%m-%d"),
            "fin":    df["fecha"].max().strftime("%Y-%m-%d"),
        },
        "distribucion_moneda":  df["moneda"].value_counts().to_dict(),
        "proveedores_unicos":   df["proveedor"].nunique() if "proveedor" in df.columns else "N/A",
        "precio_promedio_cop":  round(df["precio_cop"].mean(), 0),
        "precio_max_cop":       round(df["precio_cop"].max(), 0),
        "precio_min_cop":       round(df["precio_cop"].min(), 0),
        "nulos_por_columna":    df.isnull().sum().to_dict(),
    }

    log.info("\n── REPORTE DE CALIDAD DEL DATASET ──")
    for k, v in reporte.items():
        log.info(f"  {k}: {v}")

    return reporte


# ─── TEST RÁPIDO (ejecutar directo) ───────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  TEST DEL MÓDULO 1 — ETL DICO S.A.")
    print(f"  Fuente: MariaDB {DB_HOST} / {DB_BD}.{DB_CONSULTA}")
    print("=" * 55)

    df_compras, df_trm = ejecutar_pipeline_etl()

    print("\n── PRIMERAS FILAS DEL DATASET PROCESADO ──")
    cols_mostrar     = ["fecha", "referencia", "descripcion",
                        "nit", "precio_original", "moneda", "precio_cop", "trm_aplicada"]
    cols_disponibles = [c for c in cols_mostrar if c in df_compras.columns]
    print(df_compras[cols_disponibles].head(10).to_string(index=False))

    print("\n── REPORTE DE CALIDAD ──")
    reporte_calidad(df_compras)

    print("\n Módulo 1 ejecutado correctamente.")
    print("   Siguiente paso → Módulo 2: Análisis Exploratorio\n")
