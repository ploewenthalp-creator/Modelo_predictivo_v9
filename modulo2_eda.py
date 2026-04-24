"""
=============================================================
DICO S.A. — Módulo 2: Análisis Exploratorio de Datos (EDA)
=============================================================
Responsabilidades:
  1. Evolución del precio por referencia en el tiempo
  2. Ranking de referencias más caras y más volátiles
  3. Comparación entre proveedores
  4. Frecuencia de compra por referencia
  5. Reporte de estadísticas descriptivas por referencia
  6. Detección de referencias con datos suficientes para modelar

Dependencias:
  pip install plotly pandas numpy openpyxl

Autor: Data Science DICO S.A.
Versión: 1.0
=============================================================
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import logging

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─── Paleta corporativa DICO ──────────────────────────────────────────────────
DICO_ROJO     = "#d60000"          # Rojo corporativo primario
DICO_ROJO_OSC = "#a30000"          # Rojo oscuro (bordes, acento fuerte)
DICO_NAVY     = "#0d2232"          # Azul marino oscuro (títulos, ejes)
DICO_GRIS     = "#b3bbc1"          # Gris azulado (elementos secundarios)
DICO_GRIS_CLR = "#e8ecef"          # Gris muy claro (grillas)
DICO_BLANCO   = "#ffffff"          # Fondo
DICO_TEXTO    = "#1a1a1a"          # Texto general sobre blanco

TEMPLATE_DICO = dict(
    paper_bgcolor=DICO_BLANCO,
    plot_bgcolor=DICO_BLANCO,
    font=dict(family="Inter, Segoe UI, Arial", color=DICO_NAVY, size=12),
    title_font=dict(family="Inter, Segoe UI, Arial", color=DICO_NAVY, size=15, weight="bold"),
    margin=dict(l=64, r=44, t=74, b=60),
)

# Ejes base reutilizables (legibles sobre fondo blanco)
_EJE_X = dict(
    showgrid=True, gridcolor=DICO_GRIS_CLR, gridwidth=1,
    zeroline=False, linecolor=DICO_GRIS_CLR, linewidth=1,
    tickfont=dict(color=DICO_NAVY, size=11),
    title_font=dict(color=DICO_NAVY, size=12),
)
_EJE_Y = dict(
    showgrid=True, gridcolor=DICO_GRIS_CLR, gridwidth=1,
    zeroline=False, linecolor=DICO_GRIS_CLR, linewidth=1,
    tickfont=dict(color=DICO_NAVY, size=11),
    title_font=dict(color=DICO_NAVY, size=12),
)

PALETA_MULTI = [
    "#d60000", "#0d2232", "#b3bbc1", "#e8534a", "#1a4a6e",
    "#7a8b95", "#f4845f", "#2a9d8f", "#264653", "#e9c46a",
]


# ─── 1. Estadísticas descriptivas por referencia ──────────────────────────────

def calcular_estadisticas_por_referencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas clave por referencia:
      - n_compras, precio_promedio, precio_min, precio_max
      - volatilidad (coef. de variación = std/media)
      - tendencia (pendiente de regresión lineal simple)
      - primera y última compra
      - proveedor más frecuente
    """
    resultados = []

    for ref, grupo in df.groupby("referencia"):
        grupo = grupo.sort_values("fecha")
        precios = grupo["precio_cop"].values
        n = len(precios)

        # Coeficiente de variación (volatilidad relativa)
        cv = (np.std(precios) / np.mean(precios) * 100) if np.mean(precios) > 0 else 0

        # Tendencia: pendiente normalizada por precio promedio
        if n >= 3:
            x = np.arange(n)
            pendiente = np.polyfit(x, precios, 1)[0]
            tendencia_pct = (pendiente / np.mean(precios)) * 100
        else:
            tendencia_pct = 0.0

        # Proveedor más frecuente
        if "proveedor" in grupo.columns:
            _vc_prov = grupo["proveedor"].dropna().value_counts()
            proveedor_top = _vc_prov.idxmax() if not _vc_prov.empty else "N/A"
        else:
            proveedor_top = "N/A"

        # Descripción más frecuente
        if "descripcion" in grupo.columns:
            _vc_desc = grupo["descripcion"].dropna().value_counts()
            descripcion = _vc_desc.idxmax() if not _vc_desc.empty else "N/A"
        else:
            descripcion = "N/A"

        resultados.append({
            "referencia":        ref,
            "descripcion":       descripcion,
            "proveedor_top":     proveedor_top,
            "n_compras":         n,
            "precio_promedio":   round(np.mean(precios), 0),
            "precio_mediano":    round(np.median(precios), 0),
            "precio_min":        round(np.min(precios), 0),
            "precio_max":        round(np.max(precios), 0),
            "volatilidad_pct":   round(cv, 2),
            "tendencia_pct":     round(tendencia_pct, 4),
            "primera_compra":    grupo["fecha"].min(),
            "ultima_compra":     grupo["fecha"].max(),
            "puede_predecir":    n >= 6,    # mínimo absoluto para generar pronóstico
            "apto_para_modelo":  n >= 10,   # mínimo para modelos completos (Prophet/SARIMA)
        })

    df_stats = pd.DataFrame(resultados).sort_values("precio_promedio", ascending=False)
    log.info(f"Estadísticas calculadas: {len(df_stats)} referencias | "
             f"{df_stats['apto_para_modelo'].sum()} aptas para modelar")
    return df_stats


# ─── 2. Evolución del precio por referencia ───────────────────────────────────

def grafico_evolucion_precio(
    df: pd.DataFrame,
    referencia: str,
    mostrar_trm: bool = True
) -> go.Figure:
    """
    Gráfico de línea interactivo: precio histórico de una referencia.
    Si hay TRM disponible, la muestra en eje secundario.
    """
    datos = df[df["referencia"] == referencia].sort_values("fecha").copy()

    if datos.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"Sin datos para referencia: {referencia}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=DICO_NAVY))
        return fig

    descripcion = datos["descripcion"].iloc[0] if "descripcion" in datos.columns else referencia
    tiene_trm = "trm_historica" in datos.columns and datos["trm_historica"].notna().any()

    fig = make_subplots(specs=[[{"secondary_y": tiene_trm and mostrar_trm}]])

    # ── Área bajo la curva ──
    fig.add_trace(go.Scatter(
        x=datos["fecha"],
        y=datos["precio_cop"],
        mode="lines+markers",
        name="Precio COP",
        line=dict(color=DICO_ROJO, width=2.5),
        marker=dict(size=7, color=DICO_BLANCO, symbol="circle",
                    line=dict(color=DICO_ROJO, width=2)),
        fill="tozeroy",
        fillcolor="rgba(214,0,0,0.07)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Precio: $%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    # ── TRM en eje secundario ──
    if tiene_trm and mostrar_trm:
        trm_datos = datos.dropna(subset=["trm_historica"])
        fig.add_trace(go.Scatter(
            x=trm_datos["fecha"],
            y=trm_datos["trm_historica"],
            mode="lines",
            name="TRM (USD/COP)",
            line=dict(color=DICO_NAVY, width=1.5, dash="dot"),
            opacity=0.55,
            hovertemplate="<b>%{x|%d %b %Y}</b><br>TRM: $%{y:,.0f}<extra></extra>",
        ), secondary_y=True)

    # ── Media móvil 4 períodos ──
    if len(datos) >= 4:
        datos["ma4"] = datos["precio_cop"].rolling(4, min_periods=2).mean()
        fig.add_trace(go.Scatter(
            x=datos["fecha"],
            y=datos["ma4"],
            mode="lines",
            name="Media móvil (4 per.)",
            line=dict(color=DICO_GRIS, width=2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>MA4: $%{y:,.0f}<extra></extra>",
        ), secondary_y=False)

    fig.update_layout(
        **TEMPLATE_DICO,
        title=dict(text=f"<b>{referencia}</b>  ·  {descripcion[:60]}", x=0.02),
        hovermode="x unified",
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0)", font=dict(color=DICO_NAVY, size=11),
        ),
    )
    fig.update_xaxes(**_EJE_X, title_text="Fecha", tickformat="%b %Y")
    fig.update_yaxes(
        **_EJE_Y, title_text="Precio (COP $)", tickformat="$,.0f",
        secondary_y=False,
    )
    if tiene_trm and mostrar_trm:
        fig.update_yaxes(
            title_text="TRM (COP/USD)", tickformat="$,.0f",
            tickfont=dict(color=DICO_NAVY, size=10),
            showgrid=False, secondary_y=True,
        )
    return fig


# ─── 3. Ranking de referencias más caras y volátiles ─────────────────────────

def grafico_ranking_referencias(
    df_stats: pd.DataFrame,
    top_n: int = 15,
    modo: str = "precio"   # "precio" | "volatilidad" | "tendencia"
) -> go.Figure:
    """
    Gráfico de barras horizontal con ranking de referencias.
    modo: "precio"       → ordenado por precio promedio
          "volatilidad"  → ordenado por coef. de variación
          "tendencia"    → ordenado por tendencia alcista
    """
    config = {
        "precio":      ("precio_promedio",  "Precio Promedio COP",        "$,.0f", "Referencias más costosas"),
        "volatilidad": ("volatilidad_pct",  "Coef. de Variación (%)",     ".1f%",  "Referencias más volátiles (riesgo de precio)"),
        "tendencia":   ("tendencia_pct",    "Tendencia de Precio (%/comp)", ".2f%", "Referencias con mayor tendencia alcista"),
    }

    col, xtitle, fmt, titulo = config[modo]

    df_plot = (
        df_stats[df_stats["apto_para_modelo"]]
        .nlargest(top_n, col)[["referencia", "descripcion", col, "n_compras", "proveedor_top"]]
        .sort_values(col, ascending=True)
    )

    if df_plot.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin referencias con datos suficientes.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=13, color=DICO_NAVY))
        return fig

    etiquetas_y = (
        df_plot["referencia"].astype(str)
        + "  ·  "
        + df_plot["descripcion"].str[:38].fillna("")
    ).tolist()

    # Degradado de intensidad del rojo corporativo según magnitud
    valores = df_plot[col].values
    v_min, v_max = valores.min(), valores.max()
    alfa = [0.30 + 0.70 * (v - v_min) / (v_max - v_min + 1e-9) for v in valores]
    colores = [f"rgba(214,0,0,{a:.2f})" for a in alfa]

    fig = go.Figure(go.Bar(
        x=valores,
        y=etiquetas_y,
        orientation="h",
        marker=dict(
            color=colores,
            line=dict(color=DICO_ROJO_OSC, width=0.8),
        ),
        text=[f"  {v:,.0f}" for v in valores],
        textposition="outside",
        textfont=dict(size=10, color=DICO_NAVY, family="Inter, Segoe UI"),
        customdata=df_plot[["referencia", "descripcion", "n_compras", "proveedor_top"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<i>%{customdata[1]}</i><br>"
            f"<b>{xtitle}:</b> %{{x:,.2f}}<br>"
            "<b>N° compras:</b> %{customdata[2]}<br>"
            "<b>Proveedor:</b> %{customdata[3]}<extra></extra>"
        ),
    ))

    fig.update_layout(**{
        **TEMPLATE_DICO,
        "title":      dict(text=f"<b>TOP {top_n} — {titulo}</b>", x=0.02),
        "height":     max(430, top_n * 40 + 150),
        "margin":     dict(l=20, r=130, t=74, b=50),
        "showlegend": False,
    })
    fig.update_xaxes(**_EJE_X, title_text=xtitle)
    fig.update_yaxes(
        type="category", title_text="", automargin=True,
        tickfont=dict(size=10, color=DICO_NAVY),
        linecolor=DICO_GRIS_CLR, linewidth=1,
    )
    return fig


# ─── 4. Comparación entre proveedores ────────────────────────────────────────

def grafico_comparacion_proveedores(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    Panel de 2 gráficos:
      - Izq: Precio promedio por proveedor (barras horizontales — nombres legibles)
      - Der: Participación en volumen de compras (donut — % interno, sin etiquetas externas)
    """
    if "proveedor" not in df.columns or df.empty:
        return go.Figure()

    stats_prov = (
        df.groupby("proveedor")
        .agg(
            precio_promedio=("precio_cop", "mean"),
            precio_mediano=("precio_cop", "median"),
            n_compras=("precio_cop", "count"),
            volatilidad=("precio_cop", lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0),
            total_comprado=("precio_cop", "sum"),
        )
        .reset_index()
        .sort_values("precio_promedio", ascending=False)
        .head(top_n)
    )

    # Nombres truncados para los ejes (nombre completo en hover)
    stats_prov["nombre_corto"] = stats_prov["proveedor"].str[:28]

    stats_prov_bar = stats_prov.sort_values("precio_promedio", ascending=True)

    valores = stats_prov_bar["precio_promedio"].values
    v_min, v_max = valores.min(), valores.max()
    colores = [
        f"rgba(214,0,0,{0.28 + 0.72 * (v - v_min) / (v_max - v_min + 1e-9):.2f})"
        for v in valores
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Precio Promedio por Proveedor", "Participación en Volumen de Compras"),
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Bar(
        y=stats_prov_bar["nombre_corto"],
        x=stats_prov_bar["precio_promedio"],
        orientation="h",
        name="Precio Promedio",
        marker=dict(color=colores, line=dict(color=DICO_ROJO_OSC, width=0.8)),
        text=stats_prov_bar["precio_promedio"].apply(lambda v: f"  ${v:,.0f}"),
        textposition="outside",
        textfont=dict(size=10, color=DICO_NAVY, family="Inter, Segoe UI"),
        customdata=stats_prov_bar[["proveedor", "n_compras", "volatilidad"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<b>Precio promedio:</b> $%{x:,.0f}<br>"
            "<b>N° compras:</b> %{customdata[1]}<br>"
            "<b>Volatilidad:</b> %{customdata[2]:.1f}%<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=stats_prov["nombre_corto"],
        values=stats_prov["total_comprado"],
        hole=0.50,
        marker=dict(
            colors=PALETA_MULTI[:len(stats_prov)],
            line=dict(color=DICO_BLANCO, width=2),
        ),
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        textfont=dict(size=10, color=DICO_BLANCO, family="Inter, Segoe UI"),
        customdata=stats_prov[["proveedor", "total_comprado"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "<b>Total comprado:</b> $%{customdata[1]:,.0f}<br>"
            "<b>Participación:</b> %{percent}<extra></extra>"
        ),
        showlegend=True,
    ), row=1, col=2)

    altura = max(500, len(stats_prov) * 40 + 170)

    fig.update_layout(**{
        **TEMPLATE_DICO,
        "title":      dict(text="<b>Análisis de Proveedores</b>", x=0.02),
        "height":     altura,
        "showlegend": True,
        "legend":     dict(
            orientation="v", x=1.01, y=0.5, xanchor="left",
            font=dict(size=10, color=DICO_NAVY),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=DICO_GRIS_CLR, borderwidth=1,
        ),
        "margin": dict(l=60, r=190, t=74, b=40),
    })
    fig.update_xaxes(
        **_EJE_X, tickformat="$,.0f", row=1, col=1
    )
    fig.update_yaxes(
        tickfont=dict(size=11, color=DICO_NAVY),
        linecolor=DICO_GRIS_CLR, linewidth=1, row=1, col=1,
    )
    # Subtítulos de subplots en navy
    for ann in fig.layout.annotations:
        ann.font = dict(color=DICO_NAVY, size=13, family="Inter, Segoe UI")

    return fig


# ─── 5. Frecuencia de compra por referencia ───────────────────────────────────

def grafico_frecuencia_compras(
    df: pd.DataFrame,
    top_n: int = 20
) -> go.Figure:
    """
    Barras horizontales con las Top N referencias por número total de compras.
    Muestra referencia, descripción abreviada, total de compras y último precio.
    """
    df = df.copy()

    agg = (
        df.groupby("referencia")
        .agg(
            n_compras=("referencia", "size"),
            descripcion=("descripcion", "last"),
            ultimo_precio=("precio_original", "last"),
        )
        .nlargest(top_n, "n_compras")
        .reset_index()
        .sort_values("n_compras", ascending=True)  # ascendente para barh (plotly invierte)
    )

    # Etiqueta Y: referencia + descripción recortada
    agg["label"] = agg.apply(
        lambda r: f"{r['referencia']}  —  {str(r['descripcion'])[:40]}",
        axis=1,
    )

    fig = go.Figure(go.Bar(
        x=agg["n_compras"],
        y=agg["label"],
        orientation="h",
        marker_color=DICO_ROJO,
        text=agg["n_compras"].astype(str),
        textposition="outside",
        textfont=dict(color=DICO_NAVY, size=11),
        customdata=agg[["descripcion", "ultimo_precio"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Compras totales: <b>%{x}</b><br>"
            "Último precio: <b>$%{customdata[1]:,.0f}</b>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **{**TEMPLATE_DICO, "margin": dict(l=20, r=60, t=74, b=60)},
        title=dict(text=f"<b>Referencias con Más Compras — Top {top_n}</b>", x=0.02),
        height=max(420, top_n * 38 + 120),
        xaxis=dict(
            title_text="Número de Compras",
            tickfont=dict(size=10, color=DICO_NAVY),
            title_font=dict(color=DICO_NAVY),
            showgrid=True,
            gridcolor=DICO_GRIS_CLR,
        ),
        yaxis=dict(
            title_text="",
            tickfont=dict(size=10, color=DICO_NAVY),
            showgrid=False,
            automargin=True,
        ),
    )
    return fig


# ─── 6. Evolución de TRM histórica ───────────────────────────────────────────

def grafico_trm_historica(df_trm: pd.DataFrame) -> go.Figure:
    """
    Gráfico de TRM histórica con bandas de volatilidad (±1 desv. estándar móvil).
    """
    if df_trm.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="TRM no disponible (sin conexión a API)",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=DICO_NAVY)
        )
        return fig

    df = df_trm.sort_values("fecha").copy()
    df["ma30"]      = df["trm"].rolling(30, min_periods=5).mean()
    df["std30"]     = df["trm"].rolling(30, min_periods=5).std()
    df["banda_sup"] = df["ma30"] + df["std30"]
    df["banda_inf"] = df["ma30"] - df["std30"]

    fig = go.Figure()

    # Banda de volatilidad ±1σ
    fig.add_trace(go.Scatter(
        x=pd.concat([df["fecha"], df["fecha"][::-1]]),
        y=pd.concat([df["banda_sup"], df["banda_inf"][::-1]]),
        fill="toself",
        fillcolor="rgba(13,34,50,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Banda ±1σ (30d)",
        hoverinfo="skip",
    ))

    # TRM real
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["trm"],
        mode="lines",
        name="TRM Oficial",
        line=dict(color=DICO_ROJO, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br><b>TRM:</b> $%{y:,.2f}<extra></extra>",
    ))

    # Media móvil 30d
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["ma30"],
        mode="lines",
        name="Media móvil 30d",
        line=dict(color=DICO_NAVY, width=1.8, dash="dash"),
        opacity=0.65,
        hovertemplate="<b>%{x|%d %b %Y}</b><br><b>MA30:</b> $%{y:,.2f}<extra></extra>",
    ))

    fig.update_layout(
        **TEMPLATE_DICO,
        title=dict(text="<b>TRM Histórica (COP/USD)</b>", x=0.02),
        hovermode="x unified",
        height=380,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=DICO_NAVY, size=11), bgcolor="rgba(255,255,255,0)",
        ),
    )
    fig.update_xaxes(**_EJE_X, title_text="Fecha", tickformat="%b %Y")
    fig.update_yaxes(**_EJE_Y, title_text="TRM (COP $)", tickformat="$,.0f")
    return fig


# ─── 7. Resumen ejecutivo (tarjetas KPI) ─────────────────────────────────────

def calcular_kpis(df: pd.DataFrame, df_trm: pd.DataFrame) -> dict:
    """
    Calcula KPIs de alto nivel para mostrar en la app como tarjetas.
    """
    trm_actual = (
        df_trm["trm"].iloc[-1] if not df_trm.empty else None
    )
    trm_hace_1_mes = (
        df_trm[df_trm["fecha"] <= df_trm["fecha"].max() - pd.Timedelta(days=30)]
        ["trm"].iloc[-1]
        if not df_trm.empty and len(df_trm) > 30 else None
    )
    var_trm = (
        ((trm_actual - trm_hace_1_mes) / trm_hace_1_mes * 100)
        if trm_actual and trm_hace_1_mes else None
    )

    # Referencia más volátil (entre las que tienen suficientes datos)
    stats = calcular_estadisticas_por_referencia(df)
    apto = stats[stats["apto_para_modelo"]]

    ref_mas_volatil = apto.nlargest(1, "volatilidad_pct")["referencia"].values[0] if not apto.empty else "N/A"
    ref_mas_cara    = apto.nlargest(1, "precio_promedio")["referencia"].values[0] if not apto.empty else "N/A"
    _activas = df.groupby("referencia").size()
    ref_mas_activa  = _activas.idxmax() if not _activas.empty else "N/A"

    refs_pueden_predecir = stats[stats["puede_predecir"]].shape[0] if "puede_predecir" in stats.columns else apto.shape[0]
    refs_sin_datos       = stats.shape[0] - refs_pueden_predecir

    kpis = {
        "total_registros":        f"{len(df):,}",
        "total_referencias":      f"{df['referencia'].nunique():,}",
        "total_proveedores":      f"{df['proveedor'].nunique():,}" if "proveedor" in df.columns else "N/A",
        "rango_historico":        f"{df['fecha'].min().strftime('%b %Y')} → {df['fecha'].max().strftime('%b %Y')}",
        "precio_promedio_cop":    f"${df['precio_cop'].mean():,.0f}",
        "refs_aptas_modelo":      f"{apto.shape[0]:,} / {stats.shape[0]:,}",
        "refs_pueden_predecir":   f"{refs_pueden_predecir:,} / {stats.shape[0]:,}",
        "refs_sin_datos_sufic":   f"{refs_sin_datos:,}",
        "trm_actual":             f"${trm_actual:,.2f}" if trm_actual else "N/A",
        "variacion_trm_1mes":     f"{var_trm:+.1f}%" if var_trm else "N/A",
        "ref_mas_volatil":        ref_mas_volatil,
        "ref_mas_cara":           ref_mas_cara,
        "ref_mas_activa":         ref_mas_activa,
    }

    return kpis


# ─── 8. Pipeline completo del EDA ─────────────────────────────────────────────

def ejecutar_eda(
    df: pd.DataFrame,
    df_trm: pd.DataFrame,
    guardar_html: bool = True,
    carpeta_salida: str = "."
) -> dict:
    """
    Ejecuta el EDA completo y retorna todas las figuras en un dict.
    Opcionalmente exporta cada gráfico como HTML interactivo.

    Returns
    -------
    dict con keys: stats, kpis, figs
    """
    import os
    log.info("Iniciando Módulo 2 — Análisis Exploratorio...")

    # ── Estadísticas ──
    df_stats = calcular_estadisticas_por_referencia(df)
    kpis     = calcular_kpis(df, df_trm)

    # ── Gráficos ──
    figs = {
        "evolucion_muestra":    grafico_evolucion_precio(df, df["referencia"].value_counts().idxmax()),
        "ranking_precio":       grafico_ranking_referencias(df_stats, modo="precio"),
        "ranking_volatilidad":  grafico_ranking_referencias(df_stats, modo="volatilidad"),
        "ranking_tendencia":    grafico_ranking_referencias(df_stats, modo="tendencia"),
        "proveedores":          grafico_comparacion_proveedores(df),
        "frecuencia":           grafico_frecuencia_compras(df),
        "trm":                  grafico_trm_historica(df_trm),
    }

    # ── Exportar HTML ──
    if guardar_html:
        os.makedirs(carpeta_salida, exist_ok=True)
        for nombre, fig in figs.items():
            ruta_html = os.path.join(carpeta_salida, f"eda_{nombre}.html")
            fig.write_html(ruta_html, include_plotlyjs="cdn")
            log.info(f"  Exportado: {ruta_html}")

    log.info("EDA completado exitosamente.")
    return {"stats": df_stats, "kpis": kpis, "figs": figs}


# ─── 7. Oportunidades de compra ───────────────────────────────────────────────

def calcular_oportunidades_compra(
    df: pd.DataFrame,
    df_predicciones: pd.DataFrame,
    umbral_ahorro_pct: float = 3.0,
) -> pd.DataFrame:
    """
    Compara el último precio pagado de cada referencia contra el precio
    mínimo proyectado en el horizonte de predicción.

    Retorna una tabla con las oportunidades ordenadas por mayor ahorro
    potencial, incluyendo el mes óptimo para comprar y el ahorro estimado
    en COP basado en la cantidad promedio histórica entregada.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio de compras (debe incluir columna 'cantidad' si está disponible).
    df_predicciones : pd.DataFrame
        Resultado de predecir_todas_las_referencias() con columnas:
        referencia, descripcion, fecha_pred, yhat, yhat_lower, yhat_upper.
    umbral_ahorro_pct : float
        Solo incluye referencias con ahorro proyectado >= este porcentaje.

    Returns
    -------
    pd.DataFrame con columnas:
        referencia, descripcion, proveedor, precio_actual, precio_min_proyectado,
        mes_optimo, ahorro_pct, cantidad_promedio, ahorro_cop_estimado
    """
    if df_predicciones is None or df_predicciones.empty:
        return pd.DataFrame()

    tiene_cantidad = "cantidad" in df.columns
    hoy            = pd.Timestamp(datetime.today().date())
    resultados     = []

    for ref, grp in df_predicciones.groupby("referencia"):
        datos_ref = df[df["referencia"] == ref]
        if datos_ref.empty:
            continue

        datos_ref = datos_ref.sort_values("fecha")
        ultimo    = datos_ref.iloc[-1]

        precio_actual = ultimo["precio_cop"]
        if precio_actual <= 0:
            continue

        descripcion = ultimo["descripcion"]
        proveedor   = ultimo.get("proveedor", "—") if "proveedor" in ultimo.index else "—"
        moneda      = str(ultimo.get("moneda", "COP")).upper() if "moneda" in ultimo.index else "COP"

        # Cantidad promedio (unidades por orden, excluyendo ceros)
        if tiene_cantidad:
            cant_series = datos_ref["cantidad"].replace(0, np.nan)
            cant_prom   = cant_series.mean() if cant_series.notna().any() else 1.0
        else:
            cant_prom = 1.0

        # Filtrar solo predicciones FUTURAS (desde hoy en adelante)
        grp = grp.copy()
        grp["fecha_pred"] = pd.to_datetime(grp["fecha_pred"])
        grp_futuro = grp[grp["fecha_pred"] >= hoy].copy()
        if grp_futuro.empty:
            continue  # sin predicciones futuras, referencia sin valor para oportunidades

        # Mes con precio mínimo proyectado (solo en el futuro real)
        idx_min    = grp_futuro["yhat"].idxmin()
        fila_min   = grp_futuro.loc[idx_min]
        precio_min = fila_min["yhat"]
        mes_optimo = fila_min["fecha_pred"]

        ahorro_pct = ((precio_actual - precio_min) / precio_actual) * 100
        if ahorro_pct < umbral_ahorro_pct:
            continue

        ahorro_cop = ahorro_pct / 100 * precio_actual * cant_prom

        # ── Confiabilidad (basada en MAPE del modelo) ──────────────────────────
        mape_ref = grp["mape"].mean() if "mape" in grp.columns else np.nan
        if np.isnan(mape_ref):
            confiabilidad = "⚠️ Media"
        elif mape_ref <= 12:
            confiabilidad = "✅ Alta"
        elif mape_ref <= 25:
            confiabilidad = "⚠️ Media"
        else:
            confiabilidad = "❌ Baja"

        # ── Señal de acción ────────────────────────────────────────────────────
        # Meses desde hoy hasta el mes óptimo
        meses_hasta_optimo = max(0, round((mes_optimo - hoy).days / 30.4))

        # Precio del primer mes futuro proyectado (¿sube o baja ya?)
        primer_mes  = grp_futuro.sort_values("fecha_pred").iloc[0]["yhat"]
        precio_sube = primer_mes > precio_actual * 1.03  # próximo mes sube >3%

        if meses_hasta_optimo <= 1 or precio_sube:
            # Hay que actuar pronto: el óptimo es ya, o el precio sube antes de bajar
            senal = "🟢 Compra ya"
        elif meses_hasta_optimo >= 3 and confiabilidad != "❌ Baja":
            # Todavía hay tiempo, el precio sigue bajando
            senal = "🔴 Espera"
        else:
            senal = "🟡 Monitorea"

        # Si la confiabilidad es baja, degradar señal verde a monitorea
        if confiabilidad == "❌ Baja" and senal == "🟢 Compra ya":
            senal = "🟡 Monitorea"

        # ── Riesgo cambiario ───────────────────────────────────────────────────
        riesgo_fx = "⚠️ USD/TRM" if moneda in ("USD", "EUR") else "—"

        resultados.append({
            "referencia":            ref,
            "descripcion":           str(descripcion)[:55],
            "proveedor":             str(proveedor)[:35],
            "moneda":                moneda,
            "precio_actual":         round(precio_actual, 0),
            "precio_min_proyectado": round(precio_min, 0),
            "mes_optimo":            mes_optimo,
            "meses_hasta_optimo":    int(meses_hasta_optimo),
            "ahorro_pct":            round(ahorro_pct, 1),
            "cantidad_promedio":     round(cant_prom, 1),
            "ahorro_cop_estimado":   round(ahorro_cop, 0),
            "confiabilidad":         confiabilidad,
            "senal":                 senal,
            "riesgo_fx":             riesgo_fx,
        })

    if not resultados:
        return pd.DataFrame()

    return (
        pd.DataFrame(resultados)
        .sort_values("ahorro_cop_estimado", ascending=False)
        .reset_index(drop=True)
    )


# ─── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from modulo1_etl import ejecutar_pipeline_etl, DB_HOST, DB_BD, DB_CONSULTA

    print("\n" + "="*55)
    print("  TEST MÓDULO 2 — EDA DICO S.A.")
    print(f"  Fuente: MariaDB {DB_HOST} / {DB_BD}.{DB_CONSULTA}")
    print("="*55)

    try:
        print("\n Cargando datos desde MariaDB...")
        df, df_trm = ejecutar_pipeline_etl()
    except Exception as _e:
        print(f"\n⚠️  No se pudo conectar a MariaDB: {_e}")
        print("   Generando datos sintéticos para demo...")
        np.random.seed(42)
        n = 500
        refs  = [f"REF-{str(i).zfill(3)}" for i in range(1, 41)]
        provs = ["PROVEEDOR A", "PROVEEDOR B", "PROVEEDOR C",
                 "PROVEEDOR D", "PROVEEDOR E"]
        nits  = [f"90012345{i}-1" for i in range(5)]

        fechas = pd.date_range("2020-01-01", periods=n, freq="3D")
        df = pd.DataFrame({
            "fecha":       fechas,
            "referencia":  np.random.choice(refs, n),
            "descripcion": "Elemento demo",
            "proveedor":   np.random.choice(provs, n),
            "nit":         np.random.choice(nits, n),
            "precio_cop":  np.random.uniform(80_000, 3_000_000, n).round(0),
            "moneda":      np.random.choice(["COP","COP","COP","USD"], n),
            "trm_historica": np.random.uniform(3800, 4800, n).round(0),
        })
        df_trm = pd.DataFrame({
            "fecha": pd.date_range("2020-01-01", periods=1500, freq="D"),
            "trm":   np.cumsum(np.random.normal(0, 15, 1500)) + 3800,
        })

    # Ejecutar EDA
    resultado = ejecutar_eda(
    df, df_trm,
    guardar_html=True,
    carpeta_salida=r"C:\Users\Usuario\OneDrive - Dico Telecomunicaciones S.A\1.Proyectos_bi\1.Proyecto_Tablero_saldos_Proyectos\OneDrive - Dico Telecomunicaciones S.A\Escritorio\Script Import\modelo predictivo de compras"
)

    # Mostrar KPIs
    print("\n── KPIs PRINCIPALES ──")
    for k, v in resultado["kpis"].items():
        print(f"  {k:<25} {v}")

    # Mostrar top 5 referencias
    print("\n── TOP 5 REFERENCIAS MÁS COSTOSAS ──")
    cols = ["referencia", "descripcion", "precio_promedio", "volatilidad_pct", "n_compras"]
    print(resultado["stats"][cols].head(5).to_string(index=False))

    print("\n✅ Módulo 2 completado.")
    print("   Gráficos exportados en: ./eda_output/")
    print("   Siguiente paso → Módulo 3: Motor Predictivo\n")
