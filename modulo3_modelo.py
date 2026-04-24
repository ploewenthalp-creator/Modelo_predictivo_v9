"""
=============================================================
DICO S.A. — Módulo 3: Motor Predictivo de Precios
=============================================================
Metodología v3.0:
  - Referencias con ≥10 datos → Ensemble ponderado de los 2 mejores
    modelos entre: Prophet, Holt-Winters y SARIMA.
    Selección y pesos por validación cruzada walk-forward (2 folds).
    Pesos = 1/MAPE_cv → el modelo más preciso domina la predicción.
  - Referencias con 5-9 datos → Holt-Winters con trend amortiguado
  - Referencias con <5 datos  → Tendencia lineal robusta
  - Horizonte: configurable (default 8 meses), granularidad mensual
  - Métricas: MAPE walk-forward CV (out-of-sample real)

Fundamento matemático:
  · SARIMA (Box-Jenkins): captura autocorrelación, diferenciación y
    componente MA. Parámetro d determinado por test ADF de estacionariedad.
  · Walk-forward CV: evalúa cada modelo en períodos futuros reales,
    no en datos usados para entrenar. Estimador insesgado del error.
  · Ensemble ponderado: reduce varianza respecto a un modelo único.
    E[error_ensemble] ≤ min(E[error_i]) cuando los errores no están
    perfectamente correlacionados (teorema de diversificación).

Dependencias: pip install prophet scikit-learn statsmodels

Autor: Data Science DICO S.A.
Versión: 3.0
=============================================================
"""

import os
import sys
import pickle
import configparser
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─── Rutas y configuración ─────────────────────────────────────────────────────
RUTA_BASE = Path(__file__).parent
_config = configparser.ConfigParser()
_config.read(RUTA_BASE / "config.ini", encoding="utf-8")

HORIZONTE_MESES     = _config.getint("modelo", "horizonte_meses", fallback=8)
MIN_PUNTOS_PROPHET  = _config.getint("modelo", "min_puntos_prophet", fallback=10)
INTERVALO_CONFIANZA = _config.getfloat("modelo", "intervalo_confianza", fallback=0.80)
CACHE_MODELOS_DIR   = str(RUTA_BASE / _config.get("datos", "carpeta_modelos", fallback="modelos_cache"))

MIN_COMPRAS_PREDICCION = 6     # referencias con ≤5 compras no generan pronóstico
MIN_PUNTOS_HW          = 6     # mínimo para Holt-Winters (alineado con umbral global)
MIN_PUNTOS_SARIMA      = 24    # mínimo para SARIMA estacional (2 ciclos anuales completos)
EUR_USD_RATE           = 1.08  # Factor EUR→USD para conversión a COP (ajustable)
_Z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960}.get(INTERVALO_CONFIANZA, 1.282)
CACHE_VERSION = "7.0"  # v7: Bug fixes — SARIMA grid p/q range(3), MIN_PUNTOS_SARIMA=24, TRM por días, rankings por último valor

# Verificar disponibilidad de statsmodels (HW + SARIMA usan el mismo paquete)
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HoltWinters
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HW_DISPONIBLE     = True
    _SARIMA_DISPONIBLE = True
except ImportError:
    _HW_DISPONIBLE     = False
    _SARIMA_DISPONIBLE = False
    log.warning("statsmodels no instalado. HW y SARIMA no disponibles. "
                "Instala con: pip install statsmodels")


# ─── 1. Preparación de datos ───────────────────────────────────────────────────

def preparar_serie_mensual(
    df: pd.DataFrame,
    referencia: str,
    col_precio: str = "precio_cop",
) -> pd.DataFrame:
    """
    Agrega datos de una referencia a frecuencia mensual.

    Mejoras v4.0:
    - Parámetro col_precio: permite usar precio_original (USD/EUR) en lugar de precio_cop.
    - Relleno de gaps: meses sin compra se interpolan linealmente para que los modelos
      no confundan una pausa de 3 meses con un punto a 1 mes del anterior.
    - Media recortada 10% por mes: robustez ante outliers de precio en el mismo mes.

    Returns
    -------
    pd.DataFrame con columnas: ds (datetime), y (precio en la moneda de col_precio)
    """
    datos = df[df["referencia"] == referencia].copy().sort_values("fecha")
    if datos.empty:
        return pd.DataFrame()

    # Excluir outliers de precio marcados por el ETL
    if "es_outlier" in datos.columns:
        n_out = datos["es_outlier"].sum()
        if n_out:
            log.debug(f"[{referencia}] Excluyendo {n_out} outlier(s) del entrenamiento.")
        datos = datos[~datos["es_outlier"]]
        if datos.empty:
            return pd.DataFrame()

    # Usar col_precio si existe; si no, degradar a precio_cop
    if col_precio not in datos.columns:
        col_precio = "precio_cop"

    datos["mes"] = datos["fecha"].dt.to_period("M").dt.to_timestamp()

    def media_recortada(x):
        if len(x) <= 2:
            return x.mean()
        lo, hi = x.quantile(0.10), x.quantile(0.90)
        filtrado = x[(x >= lo) & (x <= hi)]
        return filtrado.mean() if not filtrado.empty else x.mean()

    serie = (
        datos.groupby("mes")[col_precio]
        .apply(media_recortada)
        .reset_index()
        .rename(columns={"mes": "ds", col_precio: "y"})
    )
    serie["ds"] = pd.to_datetime(serie["ds"])
    serie = serie.dropna().sort_values("ds").reset_index(drop=True)

    # ── Rellenar gaps mensuales con interpolación lineal ──────────────────────
    # Sin esto, SARIMA y HW interpretan dos compras separadas por 3 meses
    # como si fueran puntos consecutivos, distorsionando tendencia y estacionalidad.
    if len(serie) >= 2:
        rango_completo = pd.date_range(
            start=serie["ds"].min(),
            end=serie["ds"].max(),
            freq="MS"
        )
        serie = (
            serie.set_index("ds")
            .reindex(rango_completo)
            .rename_axis("ds")
            .reset_index()
        )
        serie["y"] = (
            serie["y"]
            .interpolate(method="linear")
            .bfill()
            .ffill()
        )
        serie = serie.dropna().reset_index(drop=True)

    return serie


# ─── 2. Holt-Winters Exponential Smoothing ────────────────────────────────────

def entrenar_holtwinters(serie: pd.DataFrame, horizonte: int = None) -> dict:
    """
    Entrena un modelo Holt-Winters (ETS) con trend amortiguado.
    Apropiado para series con 5+ observaciones.
    Intervalos de confianza basados en desviación estándar de residuos,
    con incertidumbre creciente según el horizonte.

    Returns
    -------
    dict con: modelo, forecast, serie, metricas, n_puntos, metodo
    """
    if horizonte is None:
        horizonte = HORIZONTE_MESES
    if not _HW_DISPONIBLE:
        raise ImportError("Instala statsmodels: pip install statsmodels")

    n = len(serie)
    y = serie["y"].values

    modelo = _HoltWinters(
        y,
        trend="add",
        seasonal="add" if n >= 14 else None,
        seasonal_periods=12 if n >= 14 else None,
        damped_trend=True,  # amortigua la extrapolación a largo plazo
    ).fit(optimized=True, use_brute=False)

    # Sigma de residuos (para intervalos de confianza)
    residuos = y - modelo.fittedvalues
    sigma    = np.std(residuos, ddof=1)
    mae      = np.mean(np.abs(residuos))
    # MAPE in-sample solo como referencia interna; será reemplazado por CV MAPE en predecir_referencia
    mape = np.mean(np.abs(residuos / np.clip(y, 1, None))) * 100

    # Forecast con intervalos que crecen con el horizonte
    predicciones_raw = modelo.forecast(horizonte)
    ultima_fecha = serie["ds"].iloc[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=horizonte, freq="MS"
    )

    filas_futuro = []
    for h, (fecha, yhat_raw) in enumerate(zip(fechas_futuras, predicciones_raw), start=1):
        yhat   = max(float(yhat_raw), 0)
        margen = _Z * sigma * np.sqrt(h)
        filas_futuro.append({
            "ds":          fecha,
            "yhat":        round(yhat, 0),
            "yhat_lower":  round(max(yhat - margen, 0), 0),
            "yhat_upper":  round(yhat + margen, 0),
            "es_forecast": True,
        })

    hist_fmt = serie.copy()
    hist_fmt["yhat"]       = hist_fmt["y"]
    hist_fmt["yhat_lower"] = hist_fmt["y"]
    hist_fmt["yhat_upper"] = hist_fmt["y"]
    hist_fmt["es_forecast"] = False

    forecast = pd.concat([
        hist_fmt[["ds", "yhat", "yhat_lower", "yhat_upper", "es_forecast"]],
        pd.DataFrame(filas_futuro)
    ]).reset_index(drop=True)

    return {
        "modelo":   modelo,
        "forecast": forecast,
        "serie":    serie,
        "metricas": {"MAE": round(mae, 0), "MAPE": round(mape, 2)},
        "n_puntos": n,
        "metodo":   "holtwinters",
    }


# ─── 3. Prophet ───────────────────────────────────────────────────────────────

def entrenar_prophet(serie: pd.DataFrame, para_cv: bool = False, horizonte: int = None) -> dict:
    """
    Entrena Prophet con changepoint_prior_scale adaptativo según la volatilidad
    de la serie (series más volátiles reciben mayor flexibilidad).

    Returns
    -------
    dict con: modelo, forecast, serie, metricas, n_puntos, metodo
    """
    if horizonte is None:
        horizonte = HORIZONTE_MESES
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Instala Prophet: pip install prophet")

    n = len(serie)
    y = serie["y"].values

    # Hiperparámetro adaptativo: más flexible para series volátiles
    cv_serie = np.std(y) / np.mean(y) if np.mean(y) > 0 else 0.10
    changepoint_prior_scale = float(np.clip(cv_serie * 0.20, 0.01, 0.50))

    modelo = Prophet(
        interval_width=INTERVALO_CONFIANZA,
        yearly_seasonality=(n >= 14),
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=10.0,
        changepoint_range=0.85,
        n_changepoints=min(15, max(5, n // 3)),           # adaptativo
        uncertainty_samples=0 if para_cv else 200,        # CV: sin intervalos (10x más rápido); final: 200 muestras
    )

    if n >= 14:
        modelo.add_seasonality(name="mensual", period=30.5, fourier_order=3)

    modelo.fit(serie)

    futuro = modelo.make_future_dataframe(periods=horizonte, freq="MS")
    forecast = modelo.predict(futuro)

    forecast["yhat"]       = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)
    forecast["es_forecast"] = forecast["ds"] > serie["ds"].max()

    # MAPE in-sample provisional; será reemplazado por CV MAPE en predecir_referencia
    hist = forecast[forecast["ds"].isin(serie["ds"])][["ds", "yhat"]].merge(serie, on="ds")
    if not hist.empty:
        mae  = np.mean(np.abs(hist["y"] - hist["yhat"]))
        mape = np.mean(np.abs((hist["y"] - hist["yhat"]) / hist["y"].clip(lower=1))) * 100
    else:
        mae, mape = np.nan, np.nan

    return {
        "modelo":   modelo,
        "forecast": forecast,
        "serie":    serie,
        "metricas": {"MAE": round(mae, 0), "MAPE": round(mape, 2)},
        "n_puntos": n,
        "metodo":   "prophet",
    }


# ─── 4. SARIMA (Box-Jenkins) ──────────────────────────────────────────────────

def entrenar_sarima(serie: pd.DataFrame, horizonte: int = None) -> dict:
    """
    SARIMA con orden determinado automáticamente:
      - d: test ADF de Dickey-Fuller (estacionariedad). d=0 si estacionaria, d=1 si no.
      - (p, q): selección por AIC sobre grid reducido {0,1,2} × {0,1,2}
      - Componente estacional (1,1,0,12) cuando hay ≥24 observaciones.
    Intervalos de confianza analíticos de SARIMAX (más precisos que σ√h).
    """
    if horizonte is None:
        horizonte = HORIZONTE_MESES
    if not _SARIMA_DISPONIBLE:
        raise ImportError("Instala statsmodels: pip install statsmodels")

    n = len(serie)
    y = serie["y"].values

    # ── Test ADF: determinar orden de diferenciación d ──
    try:
        p_valor_adf = _adfuller(y, autolag="AIC")[1]
        d = 0 if p_valor_adf < 0.05 else 1
    except Exception:
        d = 1  # asumir no estacionaria por defecto

    usa_estacional = n >= MIN_PUNTOS_SARIMA
    orden_estac    = (1, 1, 0, 12) if usa_estacional else (0, 0, 0, 0)

    # ── Grid search AIC sobre (p, q) — grid 3×3 ──
    mejor_aic   = np.inf
    mejor_orden = (1, d, 1)
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                res = _SARIMAX(
                    y, order=(p, d, q), seasonal_order=orden_estac,
                    trend="c", enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False, method="lbfgs", maxiter=200)
                if res.aic < mejor_aic:
                    mejor_aic   = res.aic
                    mejor_orden = (p, d, q)
            except Exception:
                pass

    # ── Entrenamiento final con mejor orden ──
    modelo_final = _SARIMAX(
        y, order=mejor_orden, seasonal_order=orden_estac,
        trend="c", enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False, method="lbfgs", maxiter=500)

    # ── Forecast con intervalos analíticos ──
    pred_obj    = modelo_final.get_forecast(steps=horizonte)
    yhat_future = pred_obj.predicted_mean.values
    ci          = pred_obj.conf_int(alpha=1 - INTERVALO_CONFIANZA)

    ultima_fecha  = serie["ds"].iloc[-1]
    fechas_futuras = pd.date_range(
        start=ultima_fecha + pd.DateOffset(months=1),
        periods=horizonte, freq="MS"
    )

    filas_futuro = []
    for h, (fecha, yhat_raw, lower, upper) in enumerate(
        zip(fechas_futuras, yhat_future, ci.iloc[:, 0].values, ci.iloc[:, 1].values)
    ):
        yhat = max(float(yhat_raw), 0)
        filas_futuro.append({
            "ds":          fecha,
            "yhat":        round(yhat, 0),
            "yhat_lower":  round(max(float(lower), 0), 0),
            "yhat_upper":  round(max(float(upper), yhat), 0),
            "es_forecast": True,
        })

    # Histórico con valores ajustados por el modelo
    fitted = modelo_final.fittedvalues
    hist_fmt = serie.copy()
    hist_fmt["yhat"]       = np.clip(fitted, 0, None)
    hist_fmt["yhat_lower"] = hist_fmt["yhat"]
    hist_fmt["yhat_upper"] = hist_fmt["yhat"]
    hist_fmt["es_forecast"] = False

    forecast = pd.concat([
        hist_fmt[["ds", "yhat", "yhat_lower", "yhat_upper", "es_forecast"]],
        pd.DataFrame(filas_futuro)
    ]).reset_index(drop=True)

    residuos = y - fitted[:len(y)]
    mae  = float(np.mean(np.abs(residuos)))
    mape = float(np.mean(np.abs(residuos / np.clip(y, 1, None))) * 100)

    return {
        "modelo":   modelo_final,
        "forecast": forecast,
        "serie":    serie,
        "metricas": {
            "MAE":         round(mae, 0),
            "MAPE":        round(mape, 2),
            "orden_sarima": str(mejor_orden),
            "AIC":         round(mejor_aic, 1),
        },
        "n_puntos": n,
        "metodo":   "sarima",
    }


# ─── 5. Validación cruzada walk-forward ───────────────────────────────────────

def _mape_cv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.inf
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _folds_adaptativos(n: int) -> int:
    """
    Número de folds CV según tamaño de serie.
    Más datos → estimación de error más estable y menos ruidosa.
    """
    if n < 15:
        return 2
    if n < 30:
        return 3
    return 4


def _walk_forward_cv(serie: pd.DataFrame, entrenar_fn, n_folds: int = None) -> float:
    """
    Validación cruzada walk-forward con ventana expandida.

    Esquema (n_folds=2):
      Fold 1 → train: [0 … split1),  test: [split1 … split2)
      Fold 2 → train: [0 … split2),  test: [split2 … n)

    Retorna el MAPE promedio de los folds (estimador insesgado del error futuro).
    El entrenamiento siempre crece hacia adelante — nunca usa datos futuros.
    """
    n         = len(serie)
    if n_folds is None:
        n_folds = _folds_adaptativos(n)
    min_train = max(MIN_PUNTOS_HW, n // (n_folds + 1))
    fold_size = max(1, (n - min_train) // n_folds)

    mapes_folds = []
    for fold in range(n_folds):
        train_end  = min_train + fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if train_end >= n or test_end <= test_start:
            continue

        serie_tr = serie.iloc[:train_end].copy()
        serie_te = serie.iloc[test_start:test_end].copy()

        if len(serie_tr) < 3 or len(serie_te) < 1:
            continue

        try:
            res   = entrenar_fn(serie_tr)
            preds = res["forecast"][res["forecast"]["ds"].isin(serie_te["ds"])]["yhat"].values
            n_ok  = min(len(preds), len(serie_te))
            if n_ok >= 1:
                mape_fold = _mape_cv(serie_te["y"].values[:n_ok], preds[:n_ok])
                if np.isfinite(mape_fold):
                    mapes_folds.append(mape_fold)
        except Exception as e:
            log.debug(f"    CV fold {fold} ({entrenar_fn.__name__}) falló: {e}")

    return float(np.mean(mapes_folds)) if mapes_folds else np.inf


# ─── 6. Ensemble ponderado ────────────────────────────────────────────────────

def _ensemble_ponderado(resultados: dict, mapes_cv: dict) -> dict:
    """
    Combina los forecasts de los modelos con pesos inversamente proporcionales
    a su MAPE de walk-forward CV: peso_i = (1/MAPE_i) / Σ(1/MAPE_j)

    Esto maximiza la contribución del modelo más preciso mientras mantiene
    la diversificación que reduce la varianza del ensemble.

    Los intervalos de confianza del ensemble se calculan como el promedio
    ponderado de los intervalos individuales: cuando los modelos discrepan,
    el intervalo se amplía automáticamente (mayor incertidumbre real).
    """
    # Pesos crudos = 1/MAPE (mayor peso al más preciso)
    pesos_crudos = {m: 1.0 / max(mape, 1e-6) for m, mape in mapes_cv.items() if m in resultados}
    total        = sum(pesos_crudos.values())
    pesos        = {m: w / total for m, w in pesos_crudos.items()}

    # Referencia de estructura (fechas) del primer modelo
    modelo_ref = next(iter(resultados))
    fc_ref     = resultados[modelo_ref]["forecast"].copy()

    yhat_e  = np.zeros(len(fc_ref))
    lower_e = np.zeros(len(fc_ref))
    upper_e = np.zeros(len(fc_ref))

    for nombre, peso in pesos.items():
        fc = resultados[nombre]["forecast"].set_index("ds").reindex(fc_ref["ds"].values)
        yhat_e  += fc["yhat"].fillna(0).values       * peso
        lower_e += fc["yhat_lower"].fillna(0).values * peso
        upper_e += fc["yhat_upper"].fillna(0).values * peso

    fc_ensemble = fc_ref.copy()
    fc_ensemble["yhat"]       = np.round(np.clip(yhat_e, 0, None), 0)
    fc_ensemble["yhat_lower"] = np.round(np.clip(lower_e, 0, None), 0)
    fc_ensemble["yhat_upper"] = np.round(np.clip(upper_e, 0, None), 0)

    mape_ensemble = sum(pesos[m] * mapes_cv[m] for m in pesos)
    detalle       = " + ".join(f"{m}({pesos[m]:.0%})" for m in sorted(pesos))

    return {
        "modelo":   None,
        "forecast": fc_ensemble,
        "serie":    resultados[modelo_ref]["serie"],
        "metricas": {
            "MAE":             np.nan,
            "MAPE":            round(mape_ensemble, 2),
            "MAPE_cv":         round(mape_ensemble, 2),
            "detalle_modelos": detalle,
        },
        "n_puntos": resultados[modelo_ref]["n_puntos"],
        "metodo":   f"ensemble({detalle})",
    }


# ─── 7. Helpers para manejo de moneda extranjera ─────────────────────────────

def _detectar_moneda(df: pd.DataFrame, referencia: str) -> str:
    """
    Devuelve la moneda dominante de una referencia (COP, USD o EUR).
    Si más del 50% de las compras son en USD/EUR, usa esa moneda.
    """
    datos = df[df["referencia"] == referencia]
    if datos.empty or "moneda" not in datos.columns:
        return "COP"
    return str(datos["moneda"].value_counts().idxmax()).upper()


def _proyectar_trm(df: pd.DataFrame, n_meses: int) -> np.ndarray:
    """
    Proyecta TRM para los próximos n_meses.

    Bajo la hipótesis de paseo aleatorio (validada empíricamente para tipos de cambio),
    la mejor predicción puntual de la TRM futura es la TRM actual.
    Usamos el promedio de los últimos 90 días de registros para suavizar volatilidad
    de corto plazo sin introducir tendencias espurias.
    """
    FALLBACK_TRM = 4200.0
    if "trm_aplicada" not in df.columns:
        return np.full(n_meses, FALLBACK_TRM)

    fecha_max = df["fecha"].max()
    trm_reciente = (
        df[
            (df["trm_aplicada"] > 0) &
            (df["fecha"] >= fecha_max - pd.Timedelta(days=90))
        ]
        .sort_values("fecha")["trm_aplicada"]
        .mean()
    )
    if pd.isna(trm_reciente) or trm_reciente <= 0:
        trm_reciente = FALLBACK_TRM

    return np.full(n_meses, round(trm_reciente, 0))


def _convertir_forecast_a_cop(
    resultado: dict,
    trm_array: np.ndarray,
    df: pd.DataFrame,
    referencia: str,
    factor_eur: float = 1.0,
) -> dict:
    """
    Convierte un forecast en USD/EUR a COP usando la TRM proyectada.

    Para la parte histórica del gráfico, reemplaza los valores ajustados del modelo
    (que están en USD/EUR) con los precios reales en COP para mantener consistencia visual.

    Parameters
    ----------
    resultado    : dict resultado de cualquier modelo (forecast en USD/EUR)
    trm_array    : array de n_meses con TRM proyectada (misma TRM para todos los períodos)
    df           : DataFrame original (para extraer precios reales en COP)
    referencia   : código de la referencia
    factor_eur   : 1.0 para USD, EUR_USD_RATE para EUR
    """
    fc = resultado["forecast"].copy()

    # ── Parte futura: multiplicar por TRM proyectada ──
    mask_fut   = fc["es_forecast"] == True
    n_futuro   = mask_fut.sum()
    trm_futuro = trm_array[:n_futuro] * factor_eur

    fc.loc[mask_fut, "yhat"]       = np.round(fc.loc[mask_fut, "yhat"].values       * trm_futuro, 0)
    fc.loc[mask_fut, "yhat_lower"] = np.round(fc.loc[mask_fut, "yhat_lower"].values * trm_futuro, 0)
    fc.loc[mask_fut, "yhat_upper"] = np.round(fc.loc[mask_fut, "yhat_upper"].values * trm_futuro, 0)

    # ── Parte histórica: reemplazar con precio_cop real para display correcto ──
    serie_cop = preparar_serie_mensual(df, referencia, col_precio="precio_cop")
    if not serie_cop.empty:
        cop_map   = serie_cop.set_index("ds")["y"]
        mask_hist = fc["es_forecast"] == False
        fc.loc[mask_hist, "yhat"] = (
            fc.loc[mask_hist, "ds"].map(cop_map)
            .fillna(fc.loc[mask_hist, "yhat"])
        )
        fc.loc[mask_hist, "yhat_lower"] = fc.loc[mask_hist, "yhat"]
        fc.loc[mask_hist, "yhat_upper"] = fc.loc[mask_hist, "yhat"]
        resultado["serie"] = serie_cop   # gráfico muestra precios COP reales

    resultado["forecast"] = fc
    return resultado


# ─── 8. Orquestador: CV + ensemble ────────────────────────────────────────────

_FUNCIONES_MODELO = {
    "prophet":     entrenar_prophet,
    "holtwinters": entrenar_holtwinters if _HW_DISPONIBLE else None,
    "sarima":      entrenar_sarima      if _SARIMA_DISPONIBLE else None,
}

def _entrenar_con_ensemble(serie: pd.DataFrame, horizonte: int = None) -> dict:
    """
    Para series con ≥ MIN_PUNTOS_PROPHET observaciones:
      1. Evalúa cada modelo candidato con walk-forward CV (2 folds)
      2. Selecciona los 2 de menor MAPE_cv
      3. Los reentrena sobre la serie COMPLETA
      4. Construye el ensemble ponderado por 1/MAPE_cv
         - Si el mejor supera al 2° por >25% relativo → usa solo el mejor
           (el ensemble no ayuda cuando un modelo domina claramente)
    """
    if horizonte is None:
        horizonte = HORIZONTE_MESES
    n = len(serie)

    # Candidatos disponibles según tamaño de serie
    candidatos = {
        nombre: fn for nombre, fn in _FUNCIONES_MODELO.items()
        if fn is not None
        and not (nombre == "sarima" and n < MIN_PUNTOS_SARIMA)
    }

    # ── Walk-forward CV para cada candidato (2 folds, Prophet sin intervalos = 10x más rápido) ──
    def _prophet_cv(serie): return entrenar_prophet(serie, para_cv=True)

    candidatos_cv = {
        nombre: (_prophet_cv if nombre == "prophet" else fn)
        for nombre, fn in candidatos.items()
    }
    mapes_cv = {}
    for nombre, fn in candidatos_cv.items():
        mape = _walk_forward_cv(serie, fn, n_folds=2)
        if np.isfinite(mape):
            mapes_cv[nombre] = mape
            log.debug(f"  {nombre} CV-MAPE: {mape:.1f}%")

    if not mapes_cv:
        log.warning("  Todos los modelos fallaron en CV. Usando Prophet directo.")
        return entrenar_prophet(serie, horizonte=horizonte)

    # ── Seleccionar top-2 ──
    ordenados   = sorted(mapes_cv.items(), key=lambda x: x[1])
    mape_mejor  = ordenados[0][1]
    usar_solo_1 = (
        len(ordenados) < 2
        or (ordenados[1][1] - mape_mejor) / max(mape_mejor, 1e-6) > 0.25
    )

    seleccionados = dict(ordenados[:1] if usar_solo_1 else ordenados[:2])
    log.debug(f"  Modelos seleccionados: {seleccionados}")

    # ── Reentrenar sobre serie completa ──
    resultados_finales = {}
    for nombre in seleccionados:
        try:
            res = candidatos[nombre](serie, horizonte=horizonte)
            res["metricas"]["MAPE"]    = round(seleccionados[nombre], 2)
            res["metricas"]["MAPE_cv"] = round(seleccionados[nombre], 2)
            resultados_finales[nombre] = res
        except Exception as e:
            log.warning(f"  {nombre} falló en entrenamiento final: {e}")

    if not resultados_finales:
        return entrenar_prophet(serie, horizonte=horizonte)

    if len(resultados_finales) == 1:
        return next(iter(resultados_finales.values()))

    return _ensemble_ponderado(resultados_finales, seleccionados)


# ─── 5. Modelo Fallback (<5 datos) ────────────────────────────────────────────

def modelo_fallback(serie: pd.DataFrame, horizonte: int = None) -> dict:
    """
    Para referencias con <5 observaciones.
    Regresión lineal con intervalos de incertidumbre creciente.
    """
    from sklearn.linear_model import LinearRegression

    if horizonte is None:
        horizonte = HORIZONTE_MESES

    n = len(serie)
    y = serie["y"].values
    ultimo_precio = y[-1]
    fecha_ultima  = serie["ds"].iloc[-1]

    pendiente_mensual = 0.0
    r2 = 0.0
    if n >= 2:
        x = np.arange(n).reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        pendiente_mensual = reg.coef_[0]
        r2 = reg.score(x, y)

    std_hist = np.std(y, ddof=1) if n > 1 else ultimo_precio * 0.10

    fechas_futuras = pd.date_range(
        start=fecha_ultima + pd.DateOffset(months=1),
        periods=horizonte, freq="MS"
    )

    filas_futuro = []
    for i, fecha in enumerate(fechas_futuras, start=1):
        yhat   = max(ultimo_precio + pendiente_mensual * i, 0)
        margen = std_hist * (1 + 0.15 * i)
        filas_futuro.append({
            "ds":          fecha,
            "yhat":        round(yhat, 0),
            "yhat_lower":  round(max(yhat - margen, 0), 0),
            "yhat_upper":  round(yhat + margen, 0),
            "es_forecast": True,
        })

    hist_fmt = serie.copy()
    hist_fmt["yhat"]       = hist_fmt["y"]
    hist_fmt["yhat_lower"] = hist_fmt["y"]
    hist_fmt["yhat_upper"] = hist_fmt["y"]
    hist_fmt["es_forecast"] = False

    forecast = pd.concat([
        hist_fmt[["ds", "yhat", "yhat_lower", "yhat_upper", "es_forecast"]],
        pd.DataFrame(filas_futuro)
    ]).reset_index(drop=True)

    return {
        "modelo":   None,
        "forecast": forecast,
        "serie":    serie,
        "metricas": {"MAE": np.nan, "MAPE": np.nan, "R2": round(r2, 3)},
        "n_puntos": n,
        "metodo":   "fallback_lineal",
    }


# ─── 6. Predicción para una referencia ────────────────────────────────────────

def predecir_referencia(
    df: pd.DataFrame,
    referencia: str,
    forzar_reentrenamiento: bool = False,
    horizonte: int = None,
) -> dict:
    """
    Punto de entrada para predecir el precio de una referencia.
    Selecciona automáticamente el mejor modelo según los datos disponibles.
    Caché con control de versión: invalida automáticamente modelos v1.0.

    Returns
    -------
    dict con: forecast, serie, metricas, metodo, referencia, n_puntos
    """
    _h = horizonte if (horizonte is not None and horizonte > 0) else HORIZONTE_MESES
    os.makedirs(CACHE_MODELOS_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_MODELOS_DIR, f"{referencia.replace('/', '_')}_{_h}m.pkl")

    # ── Verificar caché (con validación de versión) ──
    if os.path.exists(cache_file) and not forzar_reentrenamiento:
        try:
            with open(cache_file, "rb") as f:
                resultado = pickle.load(f)
            if resultado.get("cache_version") == CACHE_VERSION:
                log.debug(f"[{referencia}] Cargado desde caché v{CACHE_VERSION}")
                return resultado
            log.debug(f"[{referencia}] Caché obsoleto (v{resultado.get('cache_version')}). Reentrenando...")
        except Exception:
            pass

    # ── Detectar moneda y preparar serie en moneda original si aplica ──
    moneda     = _detectar_moneda(df, referencia)
    col_precio = (
        "precio_original"
        if moneda in ("USD", "EUR") and "precio_original" in df.columns
        else "precio_cop"
    )
    serie = preparar_serie_mensual(df, referencia, col_precio=col_precio)
    n = len(serie)

    if serie.empty:
        return {
            "forecast":      pd.DataFrame(),
            "serie":         pd.DataFrame(),
            "metricas":      {},
            "metodo":        "sin_datos",
            "referencia":    referencia,
            "n_puntos":      0,
            "cache_version": CACHE_VERSION,
        }

    # ── Horizonte efectivo: cubre desde la última compra hasta HOY + horizonte ──
    # Si la última compra fue hace 2 años, el modelo pronostica hasta hoy + N meses,
    # no solo N meses desde la última compra.
    hoy = pd.Timestamp(datetime.today().date())
    ultima_fecha_serie = serie["ds"].max() if not serie.empty else hoy
    meses_hasta_hoy = max(0, (hoy.year - ultima_fecha_serie.year) * 12
                          + (hoy.month - ultima_fecha_serie.month))
    horizonte_efectivo = meses_hasta_hoy + _h

    # ── Bloqueo por datos insuficientes ──
    if n < MIN_COMPRAS_PREDICCION:
        return {
            "forecast":      pd.DataFrame(),
            "serie":         serie,
            "metricas":      {},
            "metodo":        "datos_insuficientes",
            "referencia":    referencia,
            "n_puntos":      n,
            "mensaje":       (
                f"Esta referencia tiene solo {n} compra(s) registrada(s). "
                f"Se requieren al menos {MIN_COMPRAS_PREDICCION} para generar un pronóstico confiable."
            ),
            "cache_version": CACHE_VERSION,
        }

    # ── Seleccionar y entrenar ──
    try:
        if n >= MIN_PUNTOS_PROPHET:
            resultado = _entrenar_con_ensemble(serie, horizonte=horizonte_efectivo)
        elif n >= MIN_PUNTOS_HW and _HW_DISPONIBLE:
            resultado = entrenar_holtwinters(serie, horizonte=horizonte_efectivo)
            # Reemplazar MAPE in-sample por CV MAPE para métricas honestas
            mape_cv = _walk_forward_cv(serie, entrenar_holtwinters)
            if np.isfinite(mape_cv):
                resultado["metricas"]["MAPE"]    = round(mape_cv, 2)
                resultado["metricas"]["MAPE_cv"] = round(mape_cv, 2)
        else:
            resultado = modelo_fallback(serie, horizonte=horizonte_efectivo)
    except Exception as e:
        log.warning(f"[{referencia}] Error en entrenamiento: {e}. Usando fallback.")
        resultado = modelo_fallback(serie, horizonte=horizonte_efectivo)

    resultado["referencia"]    = referencia
    resultado["cache_version"] = CACHE_VERSION

    # ── Si la serie se entrenó en USD/EUR, convertir forecast a COP ──────────
    if col_precio == "precio_original" and moneda in ("USD", "EUR"):
        trm_proyectada = _proyectar_trm(df, _h)
        factor_eur     = EUR_USD_RATE if moneda == "EUR" else 1.0
        resultado      = _convertir_forecast_a_cop(
            resultado, trm_proyectada, df, referencia, factor_eur=factor_eur
        )
        resultado["metricas"]["moneda_modelo"] = moneda
        resultado["metricas"]["trm_aplicada"]  = round(float(trm_proyectada[0]), 0)
        log.debug(f"[{referencia}] Forecast convertido {moneda}→COP @ TRM={trm_proyectada[0]:,.0f}")

    log.debug(f"[{referencia}] {resultado['metodo']} | n={n} | "
              f"MAPE={resultado['metricas'].get('MAPE', 'N/A')} | moneda={moneda}")

    # ── Guardar en caché ──
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(resultado, f)
    except Exception as e:
        log.warning(f"[{referencia}] No se pudo guardar caché: {e}")

    return resultado


# ─── 7. Predicción masiva ─────────────────────────────────────────────────────

def predecir_todas_las_referencias(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
    solo_aptas: bool = True,
    forzar_reentrenamiento: bool = False,
    modo_rapido: bool = True,
    horizonte: int = None,
) -> pd.DataFrame:
    """
    Entrena modelos para todas las referencias y consolida resultados.

    Parameters
    ----------
    solo_aptas : bool
        Si True, solo modela referencias con ≥ MIN_PUNTOS_PROPHET compras.
    modo_rapido : bool
        Si True, usa solo Holt-Winters (sin Prophet ni SARIMA) para el cálculo
        masivo. ~10x más rápido, con impacto mínimo en la dirección de señales.
        El Predictor individual sigue usando el ensemble completo.

    Returns
    -------
    pd.DataFrame con columnas:
        referencia, descripcion, fecha_pred, yhat, yhat_lower, yhat_upper,
        metodo, mape, n_puntos
    """
    referencias = (
        df_stats[df_stats["apto_para_modelo"]]["referencia"].tolist()
        if solo_aptas else df_stats["referencia"].tolist()
    )

    total = len(referencias)
    log.info(f"Iniciando predicción para {total} referencias (paralelo, modo_rapido={modo_rapido})...")

    resultados_consolidados = []
    errores = []

    _h_masivo = horizonte if (horizonte is not None and horizonte > 0) else HORIZONTE_MESES

    def _procesar_ref_rapido(ref):
        """Holt-Winters directo, sin CV ni ensemble."""
        serie = preparar_serie_mensual(df, ref, col_precio="precio_cop")
        if serie.empty or len(serie) < MIN_COMPRAS_PREDICCION:
            return []
        hoy_r = pd.Timestamp(datetime.today().date())
        ultima_r = serie["ds"].max()
        meses_r = max(0, (hoy_r.year - ultima_r.year) * 12 + (hoy_r.month - ultima_r.month))
        horizonte_r = meses_r + _h_masivo
        try:
            resultado = entrenar_holtwinters(serie, horizonte=horizonte_r)
        except Exception:
            resultado = modelo_fallback(serie, horizonte=horizonte_r)
        resultado["referencia"]    = ref
        resultado["cache_version"] = CACHE_VERSION
        return resultado

    def _procesar_ref(ref):
        if modo_rapido:
            resultado = _procesar_ref_rapido(ref)
            if not resultado or resultado["forecast"].empty:
                return []
        else:
            resultado = predecir_referencia(df, ref, forzar_reentrenamiento, horizonte=_h_masivo)
            if resultado["forecast"].empty:
                return []
        forecast = resultado["forecast"].copy()
        if "es_forecast" not in forecast.columns:
            fecha_corte = resultado["serie"]["ds"].max() if not resultado["serie"].empty else pd.Timestamp.min
            forecast["es_forecast"] = forecast["ds"] > fecha_corte
        futuro = forecast[forecast["es_forecast"]].copy()
        if futuro.empty:
            return []
        df_ref = df[df["referencia"] == ref]
        desc = df_ref["descripcion"].value_counts().idxmax() if not df_ref.empty else "N/A"
        filas = []
        for _, fila in futuro.iterrows():
            filas.append({
                "referencia":  ref,
                "descripcion": desc,
                "fecha_pred":  fila["ds"],
                "yhat":        round(fila["yhat"], 0),
                "yhat_lower":  round(fila["yhat_lower"], 0),
                "yhat_upper":  round(fila["yhat_upper"], 0),
                "metodo":      resultado["metodo"],
                "mape":        resultado["metricas"].get("MAPE", np.nan),
                "n_puntos":    resultado["n_puntos"],
            })
        return filas

    completadas = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {executor.submit(_procesar_ref, ref): ref for ref in referencias}
        for future in as_completed(futuros):
            ref = futuros[future]
            completadas += 1
            if completadas % 25 == 0 or completadas == total:
                log.info(f"  Progreso: {completadas}/{total} referencias procesadas...")
            try:
                filas = future.result()
                resultados_consolidados.extend(filas)
            except Exception as e:
                errores.append(ref)
                log.warning(f"  Error en {ref}: {e}")

    if errores:
        log.warning(f"Referencias con error ({len(errores)}): {errores[:5]}...")

    df_predicciones = pd.DataFrame(resultados_consolidados)

    if not df_predicciones.empty:
        df_predicciones["fecha_pred"] = pd.to_datetime(df_predicciones["fecha_pred"])
        df_predicciones = df_predicciones.sort_values(["referencia", "fecha_pred"])
        log.info(f"Predicción masiva completada: {df_predicciones['referencia'].nunique()} "
                 f"referencias | {len(df_predicciones)} proyecciones generadas")

    return df_predicciones


# ─── 8. Consultas analíticas ──────────────────────────────────────────────────

def referencias_mayor_incremento(
    df_predicciones: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 10,
    meses: int = 2
) -> pd.DataFrame:
    """
    Identifica las referencias con mayor incremento de precio proyectado
    en los próximos N meses respecto al último precio histórico real.
    """
    fecha_limite = pd.Timestamp(datetime.today().date()) + pd.DateOffset(months=meses)

    precio_actual = (
        df.sort_values("fecha")
        .groupby("referencia")
        .last()["precio_cop"]
        .reset_index()
        .rename(columns={"precio_cop": "precio_actual"})
    )

    precio_proyectado = (
        df_predicciones[df_predicciones["fecha_pred"] <= fecha_limite]
        .sort_values("fecha_pred")
        .groupby("referencia")["yhat"]
        .last()
        .reset_index()
        .rename(columns={"yhat": "precio_proyectado"})
    )

    comparacion = precio_actual.merge(precio_proyectado, on="referencia")
    comparacion["incremento_pct"] = (
        (comparacion["precio_proyectado"] - comparacion["precio_actual"])
        / comparacion["precio_actual"].clip(lower=1) * 100
    ).round(2)

    descripciones = (
        df.groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    comparacion = comparacion.merge(descripciones, on="referencia", how="left")

    return (
        comparacion
        .sort_values("incremento_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def resumen_prediccion_referencia(
    df_predicciones: pd.DataFrame,
    referencia: str
) -> dict:
    """
    Resumen ejecutivo de la predicción de una referencia. Usado por el chat IA.
    """
    datos = df_predicciones[df_predicciones["referencia"] == referencia]

    if datos.empty:
        return {"error": f"No hay predicción para la referencia {referencia}"}

    variacion = (
        (datos.iloc[-1]["yhat"] - datos.iloc[0]["yhat"])
        / max(datos.iloc[0]["yhat"], 1) * 100
    )

    return {
        "referencia":             referencia,
        "precio_mes_1":           f"${datos.iloc[0]['yhat']:,.0f}",
        "precio_mes_8":           f"${datos.iloc[-1]['yhat']:,.0f}",
        "precio_min_proyectado":  f"${datos['yhat_lower'].min():,.0f}",
        "precio_max_proyectado":  f"${datos['yhat_upper'].max():,.0f}",
        "tendencia":              "alcista" if variacion > 0 else "bajista",
        "variacion_total_pct":    f"{variacion:.1f}%",
        "metodo_modelo":          datos.iloc[0]["metodo"],
        "precision_mape":         f"{datos.iloc[0]['mape']:.1f}%" if not pd.isna(datos.iloc[0]["mape"]) else "N/A",
    }


# ─── 9. Visualización ─────────────────────────────────────────────────────────

def grafico_prediccion_referencia(
    df: pd.DataFrame,
    referencia: str,
    df_predicciones: pd.DataFrame = None,
    resultado_modelo: dict = None
) -> "go.Figure":
    """
    Gráfico interactivo con histórico real, predicción puntual y banda de confianza.
    """
    import plotly.graph_objects as go

    DICO_ROJO   = "#d60000"
    DICO_NAVY   = "#0d2232"
    DICO_GRIS   = "#b3bbc1"
    DICO_BLANCO = "#ffffff"
    GRIS_CLR    = "#e8ecef"

    if resultado_modelo is None:
        resultado_modelo = predecir_referencia(df, referencia)

    forecast = resultado_modelo.get("forecast", pd.DataFrame())
    serie    = resultado_modelo.get("serie", pd.DataFrame())

    if forecast.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Sin datos suficientes para: {referencia}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=DICO_NAVY)
        )
        return fig

    fecha_corte  = serie["ds"].max() if not serie.empty else forecast["ds"].min()
    hoy_graf     = pd.Timestamp(datetime.today().date())
    hist_fc      = forecast[forecast["ds"] <= fecha_corte].copy()
    # Solo mostrar pronóstico desde HOY en adelante — el tramo pasado no genera valor
    futuro       = forecast[forecast["ds"] > hoy_graf].copy()

    desc = (
        df[df["referencia"] == referencia]["descripcion"].value_counts().idxmax()
        if not df[df["referencia"] == referencia].empty else referencia
    )

    fig = go.Figure()

    # Banda de confianza futura
    if not futuro.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([futuro["ds"], futuro["ds"][::-1]]),
            y=pd.concat([futuro["yhat_upper"], futuro["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(214,0,0,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Intervalo {int(INTERVALO_CONFIANZA*100)}% confianza",
            hoverinfo="skip",
        ))

    # Histórico real
    if not serie.empty:
        fig.add_trace(go.Scatter(
            x=serie["ds"], y=serie["y"],
            mode="lines+markers",
            name="Precio histórico",
            line=dict(color=DICO_ROJO, width=2.5),
            marker=dict(size=8, color=DICO_BLANCO, symbol="circle",
                        line=dict(color=DICO_ROJO, width=2.5)),
            hovertemplate="<b>%{x|%b %Y}</b><br><b>Precio real:</b> $%{y:,.0f}<extra></extra>",
        ))

    # Predicción futura
    if not futuro.empty:
        if not serie.empty:
            # Conector desde última compra hasta primer pronóstico futuro
            # Si hay un gap largo (compra antigua), se muestra en gris tenue
            meses_gap = (hoy_graf.year - fecha_corte.year) * 12 + (hoy_graf.month - fecha_corte.month)
            color_conector = "rgba(179,187,193,0.4)" if meses_gap > 3 else DICO_NAVY
            ancho_conector = 1.2 if meses_gap > 3 else 1.8
            fig.add_trace(go.Scatter(
                x=[serie["ds"].iloc[-1], futuro["ds"].iloc[0]],
                y=[serie["y"].iloc[-1],  futuro["yhat"].iloc[0]],
                mode="lines",
                line=dict(color=color_conector, width=ancho_conector, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
            # Advertencia visible si no hay compras recientes (más de 6 meses)
            if meses_gap > 6:
                fig.add_annotation(
                    x=hoy_graf, y=1, yref="paper",
                    text=f"  ⚠️ Sin compras hace {meses_gap} meses",
                    showarrow=False,
                    font=dict(color="#b8860b", size=10, family="Inter, Segoe UI"),
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,248,220,0.9)",
                )

        fig.add_trace(go.Scatter(
            x=futuro["ds"], y=futuro["yhat"],
            mode="lines+markers",
            name="Precio proyectado",
            line=dict(color=DICO_NAVY, width=2.5, dash="dot"),
            marker=dict(size=9, color=DICO_NAVY, symbol="diamond",
                        line=dict(color=DICO_BLANCO, width=1.5)),
            hovertemplate="<b>%{x|%b %Y}</b><br><b>Proyectado:</b> $%{y:,.0f}<extra></extra>",
        ))

        fig.add_annotation(
            x=futuro["ds"].iloc[-1], y=futuro["yhat"].iloc[-1],
            text=f"  <b>${futuro['yhat'].iloc[-1]:,.0f}</b>",
            showarrow=False,
            font=dict(color=DICO_NAVY, size=12, family="Inter, Segoe UI"),
            xanchor="left",
        )

    # Línea divisoria "Hoy" — siempre en la fecha real de hoy, no en la última compra
    hoy_str = pd.Timestamp(datetime.today().date()).strftime("%Y-%m-%d")
    fig.add_vline(
        x=hoy_str,
        line=dict(color=DICO_GRIS, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=hoy_str, y=1, yref="paper", text=" Hoy",
        showarrow=False,
        font=dict(color=DICO_NAVY, size=11, family="Inter, Segoe UI"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.85)",
    )

    mape    = resultado_modelo["metricas"].get("MAPE", None)
    metodo  = resultado_modelo.get("metodo", "")
    subtitulo = f"Modelo: {metodo}"
    if mape is not None and not (isinstance(mape, float) and np.isnan(mape)):
        subtitulo += f"  ·  Error histórico (MAPE): {mape:.1f}%"

    fig.update_layout(
        paper_bgcolor=DICO_BLANCO,
        plot_bgcolor=DICO_BLANCO,
        font=dict(family="Inter, Segoe UI, Arial", color=DICO_NAVY, size=12),
        title=dict(
            text=f"<b>{referencia}</b>  ·  {desc[:55]}<br>"
                 f"<sup>{subtitulo}</sup>",
            x=0.02, font=dict(size=14, color=DICO_NAVY),
        ),
        xaxis=dict(
            title="Fecha", showgrid=True, gridcolor=GRIS_CLR, tickformat="%b %Y",
            tickfont=dict(color=DICO_NAVY, size=11),
            title_font=dict(color=DICO_NAVY),
            zeroline=False, linecolor=GRIS_CLR,
        ),
        yaxis=dict(
            title="Precio (COP $)", tickformat="$,.0f",
            showgrid=True, gridcolor=GRIS_CLR,
            tickfont=dict(color=DICO_NAVY, size=11),
            title_font=dict(color=DICO_NAVY),
            zeroline=False, linecolor=GRIS_CLR,
        ),
        hovermode="x unified",
        margin=dict(l=64, r=44, t=94, b=60),
        height=460,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=DICO_NAVY, size=11),
        ),
    )
    return fig


# ─── 10. Pipeline completo ────────────────────────────────────────────────────

def ejecutar_motor_predictivo(
    df: pd.DataFrame,
    df_trm: pd.DataFrame,
    df_stats: pd.DataFrame,
    forzar_reentrenamiento: bool = False,
    solo_aptas: bool = True,
    carpeta_salida: str = "."
) -> dict:
    """
    Ejecuta el pipeline completo del motor predictivo.

    Returns
    -------
    dict con: df_predicciones, top_incremento_2m, top_incremento_8m
    """
    log.info("=" * 55)
    log.info("INICIANDO MÓDULO 3 — MOTOR PREDICTIVO v3.0")
    log.info(f"Horizonte: {HORIZONTE_MESES} meses | Confianza: {int(INTERVALO_CONFIANZA*100)}%")
    log.info(f"Modelos: Prophet | HW={_HW_DISPONIBLE} | SARIMA={_SARIMA_DISPONIBLE}")
    log.info("=" * 55)

    df_predicciones = predecir_todas_las_referencias(
        df, df_stats,
        solo_aptas=solo_aptas,
        forzar_reentrenamiento=forzar_reentrenamiento
    )

    if df_predicciones.empty:
        log.error("No se generaron predicciones.")
        return {}

    top_2_meses = referencias_mayor_incremento(df_predicciones, df, top_n=10, meses=2)
    top_8_meses = referencias_mayor_incremento(df_predicciones, df, top_n=10, meses=8)

    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_excel = os.path.join(carpeta_salida, "predicciones_8_meses.xlsx")
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        df_predicciones.to_excel(writer, sheet_name="Predicciones", index=False)
        top_2_meses.to_excel(writer, sheet_name="Top_Incremento_2m", index=False)
        top_8_meses.to_excel(writer, sheet_name="Top_Incremento_8m", index=False)

    log.info(f"Predicciones exportadas: {ruta_excel}")
    log.info("=" * 55)
    log.info("MÓDULO 3 COMPLETADO")
    log.info("=" * 55)

    return {
        "df_predicciones":   df_predicciones,
        "top_incremento_2m": top_2_meses,
        "top_incremento_8m": top_8_meses,
    }


# ─── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(RUTA_BASE))
    from modulo1_etl import ejecutar_pipeline_etl, DB_HOST, DB_BD, DB_CONSULTA
    from modulo2_eda import calcular_estadisticas_por_referencia

    print("\n" + "=" * 55)
    print("  TEST MÓDULO 3 — MOTOR PREDICTIVO v2.0 DICO S.A.")
    print(f"  Fuente: MariaDB {DB_HOST} / {DB_BD}.{DB_CONSULTA}")
    print("=" * 55)

    try:
        print("\n Cargando datos desde MariaDB...")
        df, df_trm = ejecutar_pipeline_etl()
    except Exception as _e:
        print(f"\n Generando datos sintéticos con estacionalidad (error BD: {_e})...")
        np.random.seed(42)
        n = 800
        refs  = [f"REF-{str(i).zfill(3)}" for i in range(1, 51)]
        provs = ["PROVEEDOR A", "PROVEEDOR B", "PROVEEDOR C"]
        fechas     = pd.date_range("2019-01-01", periods=n, freq="7D")
        tendencia  = np.linspace(200_000, 350_000, n)
        estacional = 30_000 * np.sin(2 * np.pi * np.arange(n) / 52)
        ruido      = np.random.normal(0, 15_000, n)
        df = pd.DataFrame({
            "fecha":         fechas,
            "referencia":    np.random.choice(refs, n),
            "descripcion":   "Elemento demo",
            "proveedor":     np.random.choice(provs, n),
            "precio_cop":    (tendencia + estacional + ruido).clip(50_000).round(0),
            "moneda":        "COP",
            "trm_historica": np.random.uniform(3800, 4800, n),
        })
        df_trm = pd.DataFrame(columns=["fecha", "trm"])

    df_stats = calcular_estadisticas_por_referencia(df)
    aptas    = df_stats[df_stats["apto_para_modelo"]]
    print(f"\n   Referencias aptas para modelar: {len(aptas)}")

    ref_prueba = aptas["referencia"].iloc[0]
    print(f"\n── PREDICCIÓN DE PRUEBA: {ref_prueba} ──")
    resultado = predecir_referencia(df, ref_prueba)

    print(f"   Método:   {resultado['metodo']}")
    print(f"   Puntos:   {resultado['n_puntos']}")
    print(f"   Métricas: {resultado['metricas']}")

    futuro = resultado["forecast"][resultado["forecast"]["es_forecast"]].head(HORIZONTE_MESES)
    print(f"\n   Proyección a {HORIZONTE_MESES} meses:")
    print(f"   {'Mes':<12} {'Precio':<15} {'Mínimo':<15} {'Máximo':<15}")
    print(f"   {'-'*55}")
    for _, row in futuro.iterrows():
        print(f"   {str(row['ds'])[:7]:<12} "
              f"${row['yhat']:>12,.0f}  "
              f"${row['yhat_lower']:>12,.0f}  "
              f"${row['yhat_upper']:>12,.0f}")

    try:
        fig = grafico_prediccion_referencia(df, ref_prueba, resultado_modelo=resultado)
        ruta_html = str(RUTA_BASE / f"prediccion_{ref_prueba.replace('/', '_')}.html")
        fig.write_html(ruta_html, include_plotlyjs="cdn")
        print(f"\n   Gráfico exportado: {ruta_html}")
    except Exception as e:
        print(f"\n   (Gráfico no exportado: {e})")

    df_preds = predecir_todas_las_referencias(df, df_stats, solo_aptas=True)
    if not df_preds.empty:
        top = referencias_mayor_incremento(df_preds, df, top_n=5, meses=2)
        print(f"\n── TOP 5 REFERENCIAS CON MAYOR ALZA EN 2 MESES ──")
        print(top[["referencia", "descripcion", "precio_actual",
                    "precio_proyectado", "incremento_pct"]].to_string(index=False))

    print("\n Módulo 3 v2.0 completado exitosamente.")
    print("   Siguiente paso → Módulo 4: App Streamlit\n")
