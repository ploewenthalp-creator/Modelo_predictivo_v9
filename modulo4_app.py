"""
=============================================================
DICO S.A. — Módulo 4: Aplicación Streamlit v9.0
=============================================================
Interfaz simplificada para uso sin capacitación. 4 tabs:
  1. 🏠 Resumen            → gerente, ¿cómo estamos hoy? (10 seg)
  2. 🔍 Buscar material    → analista, ¿a qué precio compro X?
  3. 🛒 ¿Qué comprar?      → analista+gerente, ¿qué compro y con qué plata?
  4. 📊 Análisis detallado → analista avanzado (rankings, proveedores, etc.)

Cambios v9.0:
  - Lenguaje simplificado: sin términos técnicos en la UI
  - KPIs del tab 1 reorientados al gerente (3 KPIs sin FX)
  - Semáforo con etiquetas en lenguaje natural
  - Tab 2: confiabilidad sin MAPE, sin nombre de modelo
  - Tab 3: columnas simplificadas, badges en lenguaje natural
  - CACHE_VERSION 8.0 → 9.0 (invalida pkl anteriores)

Autor: Data Science DICO S.A.
Versión: 9.0 — Abril 2026
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import io
import base64
from datetime import datetime
from zoneinfo import ZoneInfo

_BOGOTA = ZoneInfo("America/Bogota")

import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from modulo1_etl import ejecutar_pipeline_etl
from modulo2_eda import (
    calcular_estadisticas_por_referencia,
    grafico_ranking_referencias,
    grafico_comparacion_proveedores,
    grafico_frecuencia_compras,
    grafico_trm_historica,
    calcular_kpis,
    calcular_oportunidades_compra,
    TEMPLATE_DICO, DICO_ROJO, DICO_NAVY, DICO_GRIS, DICO_GRIS_CLR,
)
from modulo3_modelo import (
    predecir_referencia,
    grafico_prediccion_referencia,
    referencias_mayor_incremento,
    predecir_todas_las_referencias,
    HORIZONTE_MESES,
    MIN_COMPRAS_PREDICCION,
)
from modulo5_inventario import (
    generar_reporte_inventario,
    cargar_saldos,
    cargar_actas,
    cargar_pedidos,
    cargar_consumos,
    cargar_pendientes_arribo,
    calcular_demanda_mensual,
    calcular_criticidad,
    calcular_demanda_3meses,
    calcular_gap_eficiencia,
    calcular_compras_sugeridas,
)
from modulo6_homologos import (
    enriquecer_con_homologos,
    calcular_demanda_por_subcategoria,
    tabla_grupos_homologos,
)
from modulo7_presupuesto import (
    calcular_prioridad_compra,
    priorizar_dentro_presupuesto,
    proyectar_cashflow,
    resumen_cashflow_por_mes,
)
from modulo8_ejecutivo import (
    calcular_exposicion_fx,
    calcular_top_acciones,
    calcular_alertas,
)

# ─── Versión de caché ─────────────────────────────────────────────────────────
# Cambiar este valor invalida todos los modelos pkl entrenados con versiones anteriores.
CACHE_VERSION = "9.0"

# ─── Colores señal / gap (complementan la paleta de modulo2_eda) ──────────────
VERDE_SENAL    = "#2a9d8f"
AMARILLO_SENAL = "#e9c46a"
NARANJA_GAP    = "#f4a261"

_COLOR_GAP = {
    "Eficiente": VERDE_SENAL,
    "Moderado":  AMARILLO_SENAL,
    "Alto":      NARANJA_GAP,
    "Crítico":   DICO_ROJO,
}

# ─── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DICO S.A. — Gestión de Compras",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS v10.0 — Enterprise Intelligence Platform ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ══════════════════════════════════════
   VARIABLES — Design tokens v10.0
   ══════════════════════════════════════ */
:root {
    /* Brand */
    --rojo:           #d60000;
    --rojo-osc:       #a30000;
    --rojo-hover:     #bb0000;
    --rojo-light:     rgba(214,0,0,0.07);
    --rojo-glow:      rgba(214,0,0,0.20);
    --navy:           #0d2232;
    --navy-light:     #1a3a52;
    --navy-mid:       #162d42;
    --navy-glass:     rgba(13,34,50,0.03);

    /* Status */
    --verde:          #16a34a;
    --verde-light:    rgba(22,163,74,0.09);
    --verde-border:   rgba(22,163,74,0.22);
    --amarillo:       #d97706;
    --amarillo-light: rgba(217,119,6,0.09);
    --naranja:        #ea580c;
    --naranja-light:  rgba(234,88,12,0.09);

    /* Neutral scale */
    --gris:           #94a3b8;
    --gris-claro:     #e2e8f0;
    --gris-hover:     #f1f5f9;
    --gris-subtle:    #f8fafc;

    /* Surface */
    --bg-main:        #f4f6f9;
    --bg-card:        #ffffff;
    --bg-elevated:    #ffffff;

    /* Text */
    --text-main:      #0f172a;
    --text-body:      #1e293b;
    --text-muted:     #64748b;
    --text-faint:     #94a3b8;

    /* Shadows */
    --shadow-xs:  0 1px 2px rgba(15,23,42,0.04);
    --shadow-sm:  0 1px 3px rgba(15,23,42,0.06), 0 4px 8px rgba(15,23,42,0.04);
    --shadow-md:  0 2px 4px rgba(15,23,42,0.04), 0 8px 20px rgba(15,23,42,0.06);
    --shadow-lg:  0 4px 8px rgba(15,23,42,0.05), 0 16px 36px rgba(15,23,42,0.08);
    --shadow-card:0 1px 3px rgba(15,23,42,0.06), 0 8px 22px rgba(13,34,50,0.05);
    --shadow-red: 0 6px 24px rgba(214,0,0,0.22);

    /* Geometry */
    --radius-xs:  4px;
    --radius-sm:  6px;
    --radius:     10px;
    --radius-lg:  14px;
    --radius-xl:  18px;
    --radius-pill:999px;

    /* Motion */
    --ease:       cubic-bezier(0.16,1,0.3,1);
    --dur-fast:   0.14s;
    --dur:        0.20s;
    --dur-slow:   0.32s;
}

/* ══════════════════════════════════════
   BASE
   ══════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-body) !important;
    -webkit-font-smoothing: antialiased !important;
    text-rendering: optimizeLegibility !important;
}

/* Fondo con textura sutil */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(160deg, #f4f6f9 0%, #eef1f6 100%) !important;
}

/* ── Scrollbar premium ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gris-claro); border-radius: var(--radius-pill); }
::-webkit-scrollbar-thumb:hover { background: var(--rojo); }

/* ── Animaciones globales ── */
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.85); }
}
@keyframes fade-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

/* ══════════════════════════════════════
   HEADER — barra ejecutiva premium
   ══════════════════════════════════════ */
.dico-header {
    background: var(--bg-card);
    border: 1px solid var(--gris-claro);
    border-left: 4px solid var(--rojo);
    padding: 1.05rem 1.8rem 1.05rem 1.5rem;
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: var(--shadow-md);
    animation: fade-in var(--dur-slow) var(--ease) both;
    position: relative;
    overflow: hidden;
}
.dico-header::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 240px; height: 100%;
    background: linear-gradient(270deg, rgba(214,0,0,0.025) 0%, transparent 100%);
    pointer-events: none;
}
.dico-header-left { display: flex; align-items: center; gap: 1.1rem; }
.dico-header h1 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--navy) !important;
    font-size: 1.28rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
    letter-spacing: -0.015em !important;
    line-height: 1.25 !important;
}
.dico-header p {
    color: var(--text-muted) !important;
    font-size: 0.74rem !important;
    margin: 0.18rem 0 0 !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
}
.dico-header-accent {
    width: 3px;
    height: 38px;
    background: linear-gradient(180deg, var(--rojo) 0%, var(--rojo-osc) 100%);
    border-radius: var(--radius-pill);
    flex-shrink: 0;
    box-shadow: 0 0 10px var(--rojo-glow);
}
.dico-header-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.70rem;
    color: var(--text-muted);
    white-space: nowrap;
    font-weight: 500;
    background: var(--gris-subtle);
    border: 1px solid var(--gris-claro);
    border-radius: var(--radius-pill);
    padding: 0.28rem 0.75rem;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--verde);
    display: inline-block;
    flex-shrink: 0;
    animation: pulse-dot 2.2s ease-in-out infinite;
}
.dico-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.61rem;
    color: var(--text-faint);
    background: var(--gris-claro);
    padding: 3px 9px;
    border-radius: var(--radius-sm);
    letter-spacing: 0.05em;
    border: 1px solid transparent;
}

/* ══════════════════════════════════════
   KPI CARDS v10.0 — premium
   ══════════════════════════════════════ */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--gris-claro);
    border-top: 3px solid var(--rojo);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.3rem 1.1rem;
    box-shadow: var(--shadow-card);
    transition: transform var(--dur) var(--ease),
                box-shadow var(--dur) var(--ease),
                border-color var(--dur) var(--ease);
    cursor: default;
    position: relative;
    overflow: hidden;
    animation: fade-in var(--dur-slow) var(--ease) both;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--rojo) 0%, transparent 100%);
    opacity: 0;
    transition: opacity var(--dur) var(--ease);
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: rgba(214,0,0,0.18);
}
.kpi-card:hover::after { opacity: 1; }
.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.5rem;
    min-height: 1.5rem;
}
.kpi-icon {
    font-size: 0.95rem;
    opacity: 0.40;
    line-height: 1;
}
.kpi-trend {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.64rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: var(--radius-pill);
    white-space: nowrap;
    letter-spacing: 0.04em;
}
.kpi-trend.up   { background: rgba(214,0,0,0.08); color: var(--rojo); border: 1px solid rgba(214,0,0,0.16); }
.kpi-trend.down { background: rgba(22,163,74,0.08); color: #15803d; border: 1px solid rgba(22,163,74,0.20); }
.kpi-trend.neutral { background: var(--gris-subtle); color: var(--text-faint); border: 1px solid var(--gris-claro); }
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    font-weight: 400;
    color: var(--navy);
    line-height: 1.08;
    letter-spacing: -0.025em;
}
.kpi-label {
    font-size: 0.62rem;
    color: var(--text-faint);
    margin-top: 0.35rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    line-height: 1.5;
}
.kpi-sub {
    font-size: 0.70rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    font-weight: 400;
}

/* ══════════════════════════════════════
   SEMÁFORO DE INVENTARIO v10.0
   ══════════════════════════════════════ */
.sem-card {
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.2rem 1.15rem 1.4rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 0.6rem;
    border: 1px solid transparent;
    border-left: 4px solid transparent;
    position: relative;
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}
.sem-card:hover { transform: translateX(3px); box-shadow: var(--shadow-lg); }
.sem-card .sem-icon {
    font-size: 1.1rem;
    line-height: 1;
    margin-bottom: 0.45rem;
    opacity: 0.75;
}
.sem-card .sem-label {
    font-size: 0.60rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    font-weight: 700;
    opacity: 0.60;
}
.sem-card .sem-n {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    letter-spacing: -0.03em;
    line-height: 1.0;
    margin: 0.08rem 0 0.12rem;
}
.sem-card .sem-list {
    font-size: 0.74rem;
    margin-top: 0.6rem;
    opacity: 0.80;
    text-align: left;
    line-height: 1.65;
    font-weight: 400;
    border-top: 1px solid rgba(0,0,0,0.06);
    padding-top: 0.5rem;
}
.sem-urgente {
    background: linear-gradient(145deg, #fff8f8 0%, #fff0f0 100%);
    border-color: #fecaca;
    border-left-color: var(--rojo);
    color: #450a0a;
}
.sem-monitorear {
    background: linear-gradient(145deg, #fffdf0 0%, #fef9e7 100%);
    border-color: #fde68a;
    border-left-color: var(--amarillo);
    color: #451a03;
}
.sem-cubierto {
    background: linear-gradient(145deg, #f0fdf8 0%, #e8fdf4 100%);
    border-color: #bbf7d0;
    border-left-color: var(--verde);
    color: #052e16;
}

/* ══════════════════════════════════════
   ALERTAS v10.0
   ══════════════════════════════════════ */
.alerta-critico {
    background: linear-gradient(135deg, #fff8f8 0%, #fff0f0 100%);
    border: 1px solid #fecaca;
    border-left: 4px solid var(--rojo);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow-sm), inset 2px 0 0 var(--rojo);
    transition: transform var(--dur) var(--ease);
}
.alerta-critico:hover { transform: translateX(3px); }
.alerta-warning {
    background: linear-gradient(135deg, #fffdf0 0%, #fef9e7 100%);
    border: 1px solid #fde68a;
    border-left: 4px solid var(--amarillo);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease);
}
.alerta-warning:hover { transform: translateX(3px); }
.alerta-info {
    background: linear-gradient(135deg, #f0fdf8 0%, #e8fdf4 100%);
    border: 1px solid #bbf7d0;
    border-left: 4px solid var(--verde);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease);
}
.alerta-info:hover { transform: translateX(3px); }
.alerta-titulo {
    font-size: 0.81rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 0.22rem;
    letter-spacing: 0.01em;
}
.alerta-desc {
    font-size: 0.77rem;
    color: var(--text-muted);
    line-height: 1.55;
}

/* ══════════════════════════════════════
   PANEL DECISIÓN REFERENCIA v10.0
   ══════════════════════════════════════ */
.precio-probable {
    background: linear-gradient(145deg, #0a1c2a 0%, var(--navy) 50%, var(--navy-light) 100%);
    border-radius: var(--radius-xl);
    padding: 1.6rem 1.4rem 1.4rem;
    text-align: center;
    color: white;
    box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255,255,255,0.04) inset;
    margin-bottom: 1rem;
    border-top: 3px solid var(--rojo);
    position: relative;
    overflow: hidden;
}
.precio-probable::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(214,0,0,0.15) 0%, transparent 65%);
    pointer-events: none;
}
.precio-probable::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 130px; height: 130px;
    background: radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.precio-probable .label {
    font-size: 0.61rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    opacity: 0.55;
    margin-bottom: 0.5rem;
    font-weight: 700;
}
.precio-probable .valor {
    font-family: 'DM Serif Display', serif;
    font-size: 2.35rem;
    font-weight: 400;
    letter-spacing: -0.025em;
    line-height: 1.08;
    text-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.precio-probable .rango {
    font-size: 0.67rem;
    opacity: 0.48;
    margin-top: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}

/* ── OPTIMA CARD v10.0 ── */
.optima-card {
    background: linear-gradient(145deg, #f0fdf8 0%, #e8fdf4 100%);
    border: 1px solid #bbf7d0;
    border-left: 4px solid var(--verde);
    border-radius: var(--radius);
    padding: 1.05rem 1.15rem;
    margin-bottom: 0.9rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease);
}
.optima-card:hover { transform: translateX(3px); }
.optima-card .titulo {
    font-size: 0.60rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #14604e;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.optima-card .mes {
    font-family: 'DM Serif Display', serif;
    font-size: 1.12rem;
    font-weight: 400;
    color: var(--navy);
}
.optima-card .ahorro {
    font-size: 0.79rem;
    color: #14604e;
    font-weight: 600;
    margin-top: 0.2rem;
}
.optima-card .precio-opt {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* ══════════════════════════════════════
   SEÑALES DE COMPRA v10.0 — signal cards
   ══════════════════════════════════════ */
.signal-card {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 0.95rem 1.15rem;
    border-radius: var(--radius-lg);
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--gris-claro);
    border-left: 4px solid transparent;
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
    cursor: default;
}
.signal-card:hover { transform: translateX(3px); box-shadow: var(--shadow-md); }
.signal-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px currentColor;
}
.signal-content { flex: 1; }
.signal-title { font-size: 0.87rem; font-weight: 700; color: var(--text-main); line-height: 1.3; }
.signal-desc  { font-size: 0.73rem; color: var(--text-muted); margin-top: 0.18rem; line-height: 1.45; }
.signal-green  { background: linear-gradient(135deg, #f0fdf8 0%, #e8fdf4 100%); border-left-color: var(--verde); }
.signal-green .signal-dot  { background: var(--verde); color: var(--verde); }
.signal-yellow { background: linear-gradient(135deg, #fffdf0 0%, #fef9e7 100%); border-left-color: var(--amarillo); }
.signal-yellow .signal-dot { background: var(--amarillo); color: var(--amarillo); }
.signal-red    { background: linear-gradient(135deg, #fff8f8 0%, #fff0f0 100%); border-left-color: var(--rojo); }
.signal-red .signal-dot    { background: var(--rojo); color: var(--rojo); }

/* Legacy senal-* (clases usadas en f-strings) */
.senal-compra-ya {
    background: #f0fdf8;
    border: 1px solid #c6f2e8;
    border-left: 4px solid var(--verde);
    border-radius: var(--radius);
    padding: 0.88rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.90rem;
    font-weight: 700;
    color: #0a3d32;
    text-align: center;
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow-card);
    letter-spacing: 0.01em;
}
.senal-monitorea {
    background: #fffbf0;
    border: 1px solid #fde8a0;
    border-left: 4px solid var(--amarillo);
    border-radius: var(--radius);
    padding: 0.88rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.90rem;
    font-weight: 700;
    color: #5c3c00;
    text-align: center;
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow-card);
}
.senal-espera {
    background: #fff5f5;
    border: 1px solid #fdd8d8;
    border-left: 4px solid var(--rojo);
    border-radius: var(--radius);
    padding: 0.88rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.90rem;
    font-weight: 700;
    color: #5a0000;
    text-align: center;
    margin-bottom: 0.7rem;
    box-shadow: var(--shadow-card);
}

/* ═══════════════════════════════════════
   BADGES v9.5 — pills con borde
   ═══════════════════════════════════════ */
.badge-urgente {
    background: rgba(214,0,0,0.07);
    color: var(--rojo);
    border: 1px solid rgba(214,0,0,0.28);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
    white-space: nowrap;
    letter-spacing: 0.04em;
}
.badge-dilema {
    background: rgba(244,162,97,0.10);
    color: #7a3b0a;
    border: 1px solid rgba(244,162,97,0.38);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-alerta {
    background: rgba(214,0,0,0.06);
    color: #7a0000;
    border: 1px solid rgba(214,0,0,0.18);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-monitor {
    background: rgba(233,196,106,0.13);
    color: #5c3c00;
    border: 1px solid rgba(233,196,106,0.42);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-cubierto {
    background: rgba(42,157,143,0.09);
    color: #0a3d32;
    border: 1px solid rgba(42,157,143,0.28);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
    white-space: nowrap;
}
.badge-alta {
    background: rgba(214,0,0,0.06);
    color: #7a0000;
    border: 1px solid rgba(214,0,0,0.18);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
}
.badge-media {
    background: rgba(233,196,106,0.13);
    color: #5c3c00;
    border: 1px solid rgba(233,196,106,0.38);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
}
.badge-baja {
    background: #f3f4f6;
    color: var(--text-muted);
    border: 1px solid var(--gris-claro);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
    font-size: 0.69rem;
    font-weight: 700;
}

/* ═══════════════════════════════════════
   REF CARD v9.5
   ═══════════════════════════════════════ */
.ref-card {
    background: var(--bg-card);
    border: 1px solid var(--gris-claro);
    border-left: 4px solid var(--rojo);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.8rem;
    box-shadow: var(--shadow-card);
}
.ref-card .ref-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: 0.03em;
}
.ref-card .ref-desc {
    font-size: 0.79rem;
    color: var(--gris);
    margin-top: 0.2rem;
    font-weight: 400;
}

/* ══════════════════════════════════════
   DIVIDER & TENDENCIA v10.0
   ══════════════════════════════════════ */
.dico-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(214,0,0,0.45) 0%, rgba(13,34,50,0.10) 50%, transparent 100%);
    margin: 1.5rem 0;
    border-radius: 1px;
}
.tendencia-up {
    background: linear-gradient(135deg, #fff8f8 0%, #fff0f0 100%);
    border: 1px solid #fecaca;
    border-left: 3px solid var(--rojo);
    border-radius: var(--radius-sm);
    padding: 0.65rem 1.05rem;
    color: var(--rojo-osc);
    font-size: 0.81rem;
    font-weight: 600;
    box-shadow: var(--shadow-xs);
}
.tendencia-down {
    background: linear-gradient(135deg, #f0fdf8 0%, #e8fdf4 100%);
    border: 1px solid #bbf7d0;
    border-left: 3px solid var(--verde);
    border-radius: var(--radius-sm);
    padding: 0.65rem 1.05rem;
    color: #14604e;
    font-size: 0.81rem;
    font-weight: 600;
    box-shadow: var(--shadow-xs);
}

/* ═══════════════════════════════════════
   FX CARD v9.5
   ═══════════════════════════════════════ */
.fx-card {
    background: linear-gradient(155deg, var(--navy) 0%, var(--navy-light) 100%);
    border-radius: var(--radius);
    padding: 1.1rem 1.1rem;
    color: white;
    box-shadow: 0 4px 20px rgba(13,34,50,0.22);
    border-top: 3px solid rgba(214,0,0,0.55);
}
.fx-card .fx-title {
    font-size: 0.61rem;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    opacity: 0.52;
    font-weight: 700;
    margin-bottom: 0.45rem;
}
.fx-card .fx-trm {
    font-family: 'DM Serif Display', serif;
    font-size: 1.65rem;
    font-weight: 400;
    letter-spacing: -0.02em;
}
.fx-card .fx-sub {
    font-size: 0.71rem;
    opacity: 0.60;
    margin-top: 0.65rem;
    border-top: 1px solid rgba(255,255,255,0.09);
    padding-top: 0.5rem;
}

/* ══════════════════════════════════════
   RADIO NAV — underline tabs v10.0
   ══════════════════════════════════════ */
div[data-testid="stRadio"] {
    background: transparent !important;
    padding: 0 !important;
    border-radius: 0 !important;
    border-bottom: 2px solid var(--gris-claro) !important;
    margin-bottom: 1.6rem !important;
}
div[data-testid="stRadio"] > label:first-child { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    border-radius: 0 !important;
    padding: 0.65rem 1.25rem !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: var(--text-muted) !important;
    cursor: pointer !important;
    transition: color var(--dur) var(--ease),
                background var(--dur) var(--ease),
                border-bottom-color var(--dur) var(--ease) !important;
    margin: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label p,
div[data-testid="stRadio"] div[role="radiogroup"] label div,
div[data-testid="stRadio"] div[role="radiogroup"] label span { color: inherit !important; }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    color: var(--rojo) !important;
    border-bottom-color: var(--rojo) !important;
    background: transparent !important;
    font-weight: 700 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div,
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span { color: var(--rojo) !important; }
div[data-testid="stRadio"] div[role="radiogroup"] label:hover:not(:has(input:checked)) {
    color: var(--navy) !important;
    background: var(--gris-subtle) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] input[type="radio"] { display: none !important; }

/* ══════════════════════════════════════
   SIDEBAR v10.0 — premium dark
   ══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #07131e 0%, #0a1c2a 40%, #0d2232 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebar"] > div:first-child {
    border-top: 3px solid var(--rojo) !important;
    box-shadow: 0 3px 0 rgba(214,0,0,0.3) !important;
}
[data-testid="stSidebar"] * { color: #c8d8e4 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label {
    color: #4e6d82 !important;
    font-size: 0.60rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stMarkdown strong {
    color: #ffffff !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background: rgba(214,0,0,0.65) !important;
    color: #fff !important;
    border-radius: var(--radius-xs) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 1px 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--dur) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: rgba(214,0,0,0.45) !important;
}
.sidebar-section-title {
    color: #3d5a6e !important;
    font-size: 0.59rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.17em !important;
    font-weight: 800 !important;
    margin: 0.15rem 0 !important;
}
.filter-badge-active {
    background: linear-gradient(135deg, var(--rojo) 0%, var(--rojo-osc) 100%);
    color: #fff;
    padding: 3px 11px;
    border-radius: var(--radius-pill);
    font-size: 0.66rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.05em;
    display: inline-block;
    box-shadow: 0 2px 8px rgba(214,0,0,0.35);
}
.filter-badge-inactive {
    color: rgba(179,187,193,0.30);
    font-size: 0.66rem;
    font-style: italic;
    text-align: center;
    display: block;
}

/* ══════════════════════════════════════
   BOTONES v10.0
   ══════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--rojo) 0%, var(--rojo-osc) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 0.55rem 1.8rem !important;
    letter-spacing: 0.03em !important;
    transition: all var(--dur) var(--ease) !important;
    box-shadow: 0 2px 8px rgba(214,0,0,0.28), 0 1px 2px rgba(0,0,0,0.10) !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--rojo-hover) 0%, var(--rojo-osc) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(214,0,0,0.35), 0 2px 4px rgba(0,0,0,0.12) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.04) !important;
    color: #8aaec4 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    padding: 0.38rem 1rem !important;
    letter-spacing: 0.03em !important;
    transition: all var(--dur) var(--ease) !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
    background: rgba(214,0,0,0.14) !important;
    color: #fff !important;
    border-color: rgba(214,0,0,0.40) !important;
    box-shadow: 0 2px 10px rgba(214,0,0,0.20) !important;
}

/* ══════════════════════════════════════
   DATAFRAME / TABLAS v10.0
   ══════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--gris-claro) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stDataFrame"] table { border-collapse: collapse !important; }
[data-testid="stDataFrame"] thead th {
    background: var(--navy) !important;
    color: #c8d8e4 !important;
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.10em !important;
    padding: 0.65rem 0.9rem !important;
    border-bottom: 2px solid var(--rojo) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background: var(--gris-subtle) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(214,0,0,0.04) !important;
}
[data-testid="stDataFrame"] tbody td {
    font-size: 0.81rem !important;
    color: var(--text-body) !important;
    padding: 0.52rem 0.9rem !important;
    border-bottom: 1px solid var(--gris-claro) !important;
    transition: background var(--dur-fast) !important;
}

/* ══════════════════════════════════════
   MÉTRICAS NATIVAS STREAMLIT v10.0
   ══════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--gris-claro) !important;
    border-top: 3px solid var(--rojo) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.1rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform var(--dur) var(--ease) !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-md) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-faint) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.9rem !important;
    color: var(--navy) !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════
   SELECTBOX / INPUTS v10.0
   ══════════════════════════════════════ */
[data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--gris-claro) !important;
    font-size: 0.85rem !important;
    transition: border-color var(--dur) var(--ease), box-shadow var(--dur) var(--ease) !important;
}
[data-baseweb="select"] > div:hover { border-color: var(--rojo) !important; }
[data-baseweb="select"] > div:focus-within {
    border-color: var(--rojo) !important;
    box-shadow: 0 0 0 3px rgba(214,0,0,0.10) !important;
}
.stTextInput input, .stNumberInput input {
    border-radius: var(--radius-sm) !important;
    border-color: var(--gris-claro) !important;
    font-size: 0.85rem !important;
    transition: border-color var(--dur), box-shadow var(--dur) !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--rojo) !important;
    box-shadow: 0 0 0 3px rgba(214,0,0,0.10) !important;
}

/* ══════════════════════════════════════
   EXPANDERS / SECCIONES v10.0
   ══════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid var(--gris-claro) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    color: var(--navy) !important;
    padding: 0.75rem 1rem !important;
    background: var(--gris-subtle) !important;
    transition: background var(--dur) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--gris-hover) !important; }

/* ══════════════════════════════════════
   FX CARD / DIVIDERS / REF CARD — mejorados
   ══════════════════════════════════════ */
.fx-card {
    background: linear-gradient(145deg, #07131e 0%, var(--navy) 60%, var(--navy-light) 100%);
    border-radius: var(--radius-lg);
    padding: 1.15rem 1.2rem;
    color: white;
    box-shadow: var(--shadow-lg);
    border: 1px solid rgba(255,255,255,0.05);
    border-top: 3px solid rgba(214,0,0,0.60);
    position: relative;
    overflow: hidden;
}
.fx-card::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, rgba(214,0,0,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.fx-card .fx-title {
    font-size: 0.59rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    opacity: 0.45;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.fx-card .fx-trm {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.fx-card .fx-sub {
    font-size: 0.69rem;
    opacity: 0.50;
    margin-top: 0.7rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    padding-top: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
}
.dico-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(214,0,0,0.50) 0%, rgba(13,34,50,0.10) 45%, transparent 100%);
    margin: 1.4rem 0;
    border-radius: 1px;
}
.ref-card {
    background: var(--bg-card);
    border: 1px solid var(--gris-claro);
    border-left: 4px solid var(--rojo);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 0.95rem 1.2rem;
    margin-bottom: 0.85rem;
    box-shadow: var(--shadow-sm);
    transition: transform var(--dur) var(--ease), box-shadow var(--dur) var(--ease);
}
.ref-card:hover {
    transform: translateX(4px);
    box-shadow: var(--shadow-md);
}
.ref-card .ref-code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: 0.03em;
}
.ref-card .ref-desc {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

def imagen_a_base64(ruta: str) -> str:
    try:
        with open(ruta, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def formato_cop(v: float) -> str:
    return f"${v:,.0f}"


def _fmt_millones(v) -> str:
    if pd.isna(v) or v == 0:
        return "$ 0"
    if v >= 1_000_000_000:
        return f"$ {v/1_000_000_000:.1f} B"
    if v >= 1_000_000:
        return f"$ {v/1_000_000:.1f} M"
    return formato_cop(v)


def kpi_html(valor: str, label: str, color_top: str = "#d60000",
             tendencia: str = None, sub: str = None, icono: str = None) -> str:
    """v9.5: KPI card con tipografía DM Serif Display e indicador de tendencia opcional."""
    top_html = ""
    if icono or tendencia:
        _icono = f'<span class="kpi-icon">{icono}</span>' if icono else '<span></span>'
        if tendencia:
            cls = "up" if (tendencia.startswith("+") or "↑" in tendencia) else (
                  "down" if (tendencia.startswith("-") or "↓" in tendencia) else "neutral")
            _trend = f'<span class="kpi-trend {cls}">{tendencia}</span>'
        else:
            _trend = ""
        top_html = f'<div class="kpi-top">{_icono}{_trend}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card" style="border-top-color:{color_top};">'
        f'{top_html}'
        f'<div class="kpi-value">{valor}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _fmt_cobertura(v) -> str:
    """Formatea cobertura en lenguaje natural: inf → Sin consumo, 0 → Sin stock, número → N.N meses."""
    try:
        f = float(v)
        if f > 900 or f == float("inf"):
            return "Sin consumo"
        if f <= 0:
            return "Sin stock"
        return f"{f:.1f} meses"
    except Exception:
        return "—"


def _badge_estado(estado: str, senal: str = "—", alerta: bool = False) -> str:
    """v9.5: badges elegantes con borde semi-transparente."""
    if alerta and senal == "🟢 Compra ya":
        return '<span class="badge-urgente">URGENTE</span>'
    if alerta and senal == "🔴 Espera":
        return '<span class="badge-dilema">DILEMA</span>'
    if "Alerta" in estado:
        return '<span class="badge-alerta">Alerta</span>'
    if "Monitorear" in estado:
        return '<span class="badge-monitor">Monitorear</span>'
    if "Cubierto" in estado:
        return '<span class="badge-cubierto">Cubierto</span>'
    return f'<span class="badge-baja">{estado}</span>'


def senal_html(senal_tipo: str, titulo: str, desc: str = "") -> str:
    """v9.5: signal card ejecutiva para Tab 2.
    senal_tipo: 'green' | 'yellow' | 'red'
    """
    desc_html = f'<div class="signal-desc">{desc}</div>' if desc else ""
    return (
        f'<div class="signal-card signal-{senal_tipo}">'
        f'<div class="signal-dot"></div>'
        f'<div class="signal-content">'
        f'<div class="signal-title">{titulo}</div>'
        f'{desc_html}'
        f'</div>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# CACHÉ — TTL idénticos a v7.0
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner="Cargando datos históricos…")
def cargar_datos():
    df, df_trm = ejecutar_pipeline_etl()
    # calcular_estadisticas_por_referencia usa .value_counts().idxmax() que falla
    # si una referencia tiene TODOS los valores de proveedor/descripcion en NULL.
    # Rellenamos antes de llamarla para evitar "argmax of an empty sequence".
    df_para_stats = df.copy()
    if "proveedor" in df_para_stats.columns:
        df_para_stats["proveedor"] = df_para_stats["proveedor"].fillna("Sin proveedor")
    if "descripcion" in df_para_stats.columns:
        df_para_stats["descripcion"] = df_para_stats["descripcion"].fillna("Sin descripcion")
    df_stats  = calcular_estadisticas_por_referencia(df_para_stats)
    timestamp = datetime.now(_BOGOTA).strftime("%d/%m/%Y %H:%M")
    return df, df_trm, df_stats, timestamp


@st.cache_data(ttl=1800, show_spinner=False)
def calcular_stats_filtrado(_df):
    _df2 = _df.copy()
    if "proveedor" in _df2.columns:
        _df2["proveedor"] = _df2["proveedor"].fillna("Sin proveedor")
    if "descripcion" in _df2.columns:
        _df2["descripcion"] = _df2["descripcion"].fillna("Sin descripcion")
    return calcular_estadisticas_por_referencia(_df2)


@st.cache_data(ttl=28800, show_spinner="Generando predicciones…")
def cargar_predicciones(_df, _df_stats, horizonte: int = 8):
    return predecir_todas_las_referencias(_df, _df_stats, solo_aptas=True, horizonte=horizonte)


@st.cache_data(ttl=1800, show_spinner="Calculando inventario…")
def cargar_reporte_inventario(_df_historico, _df_predicciones, meses_objetivo, lead_time_meses):
    """Reporte usando consumos reales como fuente de demanda (default)."""
    return generar_reporte_inventario(
        _df_historico, _df_predicciones, meses_objetivo,
        lead_time_meses=lead_time_meses,
    )


@st.cache_data(ttl=1800, show_spinner="Calculando inventario (actas)…")
def cargar_reporte_actas(_df_historico, _df_predicciones, meses_objetivo, lead_time_meses):
    """Reporte usando actas+pedidos ajustados por gap como fuente de demanda."""
    EMPTY = {"compras": pd.DataFrame(), "gap_ref": pd.DataFrame(),
             "gap_req": pd.DataFrame(), "criticidad": pd.DataFrame()}
    try:
        df_saldos     = cargar_saldos()
        df_actas_     = cargar_actas()
        df_pedidos_   = cargar_pedidos()
        df_consumos_  = cargar_consumos()
        df_pendientes = cargar_pendientes_arribo()
    except Exception as e:
        st.warning(f"⚠️ Error cargando datos de inventario: {e}")
        return EMPTY

    df_gap_ref, df_gap_req = calcular_gap_eficiencia(df_pedidos_, df_consumos_, _df_historico)
    df_dem = calcular_demanda_mensual(df_actas_, df_pedidos_)

    if not df_dem.empty:
        # Aplicar corrección por gap
        gap_map = {}
        if not df_gap_ref.empty and "gap_pct" in df_gap_ref.columns:
            gap_map = dict(zip(df_gap_ref["referencia"], df_gap_ref["gap_pct"] / 100))

        def _corr(row):
            g = gap_map.get(row["referencia"], 0)
            return round(row["demanda_mensual_promedio"] * (1 - min(g, 0.9)), 2) if g > 0 else row["demanda_mensual_promedio"]

        df_dem["demanda_mensual_promedio"] = df_dem.apply(_corr, axis=1)
        df_dem["fuente_demanda"] = df_dem["referencia"].apply(
            lambda r: "actas_corregido" if gap_map.get(r, 0) > 0 else "actas"
        )
        df_dem["demanda_std"] = 0.0

    df_dem_3m     = calcular_demanda_3meses(df_actas_, df_pedidos_)
    df_criticidad = calcular_criticidad(_df_historico)

    try:
        df_compras = calcular_compras_sugeridas(
            df_saldos, df_dem, _df_predicciones, meses_objetivo,
            df_pendientes=df_pendientes, df_demanda_3m=df_dem_3m,
            lead_time_meses=lead_time_meses, df_criticidad=df_criticidad,
        )
    except Exception as e:
        st.warning(f"⚠️ Error calculando compras sugeridas: {e}")
        df_compras = pd.DataFrame()

    return {"compras": df_compras, "gap_ref": df_gap_ref,
            "gap_req": df_gap_req, "criticidad": df_criticidad}


@st.cache_data(ttl=600, show_spinner="Entrenando modelo… (primera vez ~20 seg)")
def obtener_prediccion_ref(_df, referencia, horizonte: int = 8):
    return predecir_referencia(_df, referencia, horizonte=horizonte)


@st.cache_data(ttl=7200, show_spinner="Agrupando referencias similares…")
def cargar_homologos(_df):
    return enriquecer_con_homologos(_df, umbral=0.75)


def _obtener_reporte(df_hist, df_pred, meses_obj, lead_meses):
    """Devuelve el reporte según la fuente de demanda seleccionada en sidebar."""
    fuente = st.session_state.get("fuente_demanda", "consumos")
    if fuente == "actas":
        return cargar_reporte_actas(df_hist, df_pred, meses_obj, lead_meses)
    return cargar_reporte_inventario(df_hist, df_pred, meses_obj, lead_meses)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_header(ruta_logo: str = None, timestamp: str = ""):
    """v9.5: header limpio con 3 zonas — logo+título | acento | estado+versión."""
    logo_html = ""
    if ruta_logo and os.path.exists(ruta_logo):
        b64  = imagen_a_base64(ruta_logo)
        ext  = ruta_logo.split(".")[-1].lower()
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        logo_html = f'<img src="data:{mime};base64,{b64}" style="height:38px;border-radius:4px;">'

    ts_html = (
        f'<div class="dico-header-status">'
        f'<span class="status-dot"></span>'
        f'{timestamp}'
        f'</div>'
        f'<span class="dico-version">v9.5</span>'
    ) if timestamp else '<span class="dico-version">v9.5</span>'

    st.markdown(f"""
    <div class="dico-header">
        <div class="dico-header-left">
            {logo_html}
            <div class="dico-header-accent"></div>
            <div>
                <h1>Centro de Decisiones de Compra</h1>
                <p>Precios &nbsp;·&nbsp; Stock &nbsp;·&nbsp; Compras &nbsp;—&nbsp; DICO Telecomunicaciones</p>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:0.7rem;">
            {ts_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR v8.0
# ══════════════════════════════════════════════════════════════════════════════

def configurar_sidebar(df: pd.DataFrame, df_stats: pd.DataFrame,
                       ref_to_subcat: dict = None, timestamp: str = "") -> dict:
    with st.sidebar:

        # ── Encabezado ──
        col_t, col_b = st.columns([2, 1])
        with col_t:
            st.markdown("### ⚙️ Filtros")
        badge_ph = col_b.empty()
        st.markdown("**DICO Telecomunicaciones S.A.**")
        st.caption(f"Datos: {df['fecha'].min().strftime('%d/%m/%Y')} — {df['fecha'].max().strftime('%d/%m/%Y')}")

        # ════════════════════
        # SECCIÓN 1 — CONTEXTO
        # ════════════════════
        st.markdown("---")
        st.markdown('<p class="sidebar-section-title">Filtros</p>', unsafe_allow_html=True)

        # Requerimiento
        st.markdown("**Área / Requerimiento**")
        reqs_disponibles = sorted(df["requerimiento"].dropna().astype(str).unique().tolist())
        reqs_sel = st.multiselect(
            "Requerimiento", options=reqs_disponibles, default=[],
            placeholder=f"{len(reqs_disponibles)} requerimientos…",
            key="sb_reqs", label_visibility="collapsed",
        )

        # Proveedor (cascada sobre requerimiento)
        df_filtrado = df.copy()
        if reqs_sel:
            df_filtrado = df_filtrado[df_filtrado["requerimiento"].isin(reqs_sel)]

        st.markdown("**Proveedor**")
        provs_disponibles = sorted(df_filtrado["proveedor"].dropna().unique().tolist())
        provs_sel = st.multiselect(
            "Proveedor", options=provs_disponibles, default=[],
            placeholder=f"{len(provs_disponibles)} proveedores…",
            key="sb_provs", label_visibility="collapsed",
        )
        if provs_sel:
            df_filtrado = df_filtrado[df_filtrado["proveedor"].isin(provs_sel)]

        # ══════════════════════
        # SECCIÓN 2 — PARÁMETROS
        # ══════════════════════
        st.markdown("---")
        st.markdown('<p class="sidebar-section-title">Configuración</p>', unsafe_allow_html=True)

        st.markdown("**¿Para cuántos meses quieres stock?**")
        st.slider(
            "Meses cobertura", min_value=1, max_value=12, value=3,
            key="meses_objetivo", label_visibility="collapsed",
            help="Cuántos meses de inventario quieres tener disponibles después de la compra.",
        )

        st.markdown("**¿En qué basar la demanda?**")
        fuente_opcion = st.radio(
            "Fuente demanda",
            options=["Consumo real de técnicos (recomendado)", "Pedidos de operaciones (ajustados)"],
            index=0,
            key="sb_fuente_demanda",
            label_visibility="collapsed",
        )
        st.session_state["fuente_demanda"] = "consumos" if "Consumo real" in fuente_opcion else "actas"

        # ═════════════════
        # SECCIÓN 3 — INFO
        # ═════════════════
        st.markdown("---")
        if timestamp:
            st.caption(f"🕐 Actualizado: {timestamp}")

        n_activos = sum([bool(reqs_sel), bool(provs_sel)])
        if n_activos:
            badge_ph.markdown(
                f'<div style="text-align:right;margin-top:6px;">'
                f'<span class="filter-badge-active">{n_activos} activo{"s" if n_activos > 1 else ""}</span>'
                f'</div>', unsafe_allow_html=True,
            )
        else:
            badge_ph.markdown('<div class="filter-badge-inactive">sin filtros</div>', unsafe_allow_html=True)

        if st.button("🔄 Quitar filtros", use_container_width=True, type="secondary"):
            for k in ["sb_reqs", "sb_provs", "sb_senal", "sb_busqueda", "sb_referencia"]:
                st.session_state.pop(k, None)
            st.rerun()

    return {
        "requerimiento": reqs_sel,
        "proveedor":     provs_sel,
        "senal_filtro":  ["🟢 Compra ya", "🟡 Monitorea", "🔴 Espera"],
        "df_filtrado":   df_filtrado,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — 🏠 RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════

def tab_ejecutivo(df, df_trm, df_stats, df_filtrado, config):
    """
    Pantalla para el gerente. Sin tecnicismos. Responde: ¿cómo estamos hoy?
    """
    meses_obj      = st.session_state.get("meses_objetivo", 3)
    lead_time_dias = st.session_state.get("lead_time_dias", 30)
    lead_meses     = round(lead_time_dias / 30, 2)

    df_predicciones = st.session_state.get("df_predicciones")

    with st.spinner("Calculando estado actual del inventario…"):
        reporte    = _obtener_reporte(df, df_predicciones, meses_obj, lead_meses)
    df_compras = reporte.get("compras", pd.DataFrame())
    df_gap_req = reporte.get("gap_req", pd.DataFrame())

    # Aplicar filtro de requerimiento si está activo
    if config["requerimiento"] and not df_compras.empty:
        refs_req = set(df_filtrado["referencia"].unique())
        df_compras = df_compras[df_compras["referencia"].isin(refs_req)].copy()
        if not df_gap_req.empty:
            df_gap_req = df_gap_req[df_gap_req["requerimiento"].isin(config["requerimiento"])].copy()

    if df_compras.empty:
        st.info("No hay datos disponibles. Verifica la conexión a la red de DICO.")
        return

    # ── A. KPIs ───────────────────────────────────────────────────────────────
    st.markdown("#### Estado del inventario")

    # Materiales con compra urgente
    n_alerta = int(df_compras.get("alerta_stock", pd.Series(dtype=bool)).fillna(False).sum())

    # Inversión total sugerida
    inversion_total = df_compras.loc[df_compras.get("cantidad_sugerida", pd.Series(0, index=df_compras.index)) > 0,
                                     "costo_total"].sum(skipna=True)

    # Materiales con buen precio ahora
    n_compra_ya = 0
    if df_predicciones is not None and not df_predicciones.empty:
        try:
            _df_op = calcular_oportunidades_compra(df, df_predicciones, umbral_ahorro_pct=1)
            if config["requerimiento"]:
                _df_op = _df_op[_df_op["referencia"].isin(df_filtrado["referencia"])]
            n_compra_ya = int((_df_op["senal"] == "🟢 Compra ya").sum())
        except Exception:
            n_compra_ya = n_alerta  # fallback

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_html(str(n_alerta), "Materiales que necesitan compra urgente", color_top=DICO_ROJO), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_html(_fmt_millones(inversion_total), "Inversión estimada este mes", color_top=DICO_NAVY), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_html(str(n_compra_ya), "Materiales con buen precio ahora", color_top=VERDE_SENAL), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── B. Semáforo de inventario ─────────────────────────────────────────────
    st.markdown("#### Semáforo de inventario")

    # Agrupar por estado
    mask_alerta   = df_compras["estado_stock"].str.contains("Alerta", na=False)
    mask_monitor  = df_compras["estado_stock"].str.contains("Monitorear", na=False)
    mask_cubierto = ~(mask_alerta | mask_monitor)

    df_urg  = df_compras[mask_alerta].copy()
    df_mon  = df_compras[mask_monitor].copy()
    df_cub  = df_compras[mask_cubierto].copy()

    # Última descripción por referencia
    desc_map = (
        df.groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )
    up_map = (
        df.sort_values("fecha")
        .groupby("referencia")["precio_cop"]
        .last()
        .to_dict()
    )

    def _top5_list(df_sub: pd.DataFrame) -> str:
        if df_sub.empty:
            return "<i>Ninguna</i>"
        df_sub = df_sub.copy()
        df_sub["_costo"] = df_sub.apply(
            lambda r: up_map.get(r["referencia"], 0) * r.get("stock_actual", 0), axis=1
        )
        top = df_sub.nlargest(5, "_costo")
        items = []
        for _, r in top.iterrows():
            desc = desc_map.get(r["referencia"], "—")[:35]
            items.append(f"• {desc} ({r['referencia']})")
        return "<br>".join(items)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(
            f'<div class="sem-card sem-urgente">'
            f'<div class="sem-icon">🚨</div>'
            f'<div class="sem-label">Comprar ahora</div>'
            f'<div class="sem-n">{len(df_urg)}</div>'
            f'<div class="sem-list">{_top5_list(df_urg)}</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            f'<div class="sem-card sem-monitorear">'
            f'<div class="sem-icon">⚠️</div>'
            f'<div class="sem-label">Revisar pronto</div>'
            f'<div class="sem-n">{len(df_mon)}</div>'
            f'<div class="sem-list">{_top5_list(df_mon)}</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with sc3:
        st.markdown(
            f'<div class="sem-card sem-cubierto">'
            f'<div class="sem-icon">✅</div>'
            f'<div class="sem-label">Stock suficiente</div>'
            f'<div class="sem-n">{len(df_cub)}</div>'
            f'<div class="sem-list">{_top5_list(df_cub)}</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── C. Top 5 acciones inmediatas ──────────────────────────────────────────
    st.markdown("#### Top 5 acciones inmediatas")

    df_acciones = calcular_top_acciones(df_compras, df, n=5)
    if not df_acciones.empty:
        df_acc_vis = df_acciones.copy()
        df_acc_vis["descripcion"] = df_acc_vis["referencia"].map(desc_map).fillna("—")

        cols_acc = {
            "referencia":           "Código",
            "descripcion":          "Material",
            "meses_cobertura_real": "Stock para N meses",
            "cantidad_sugerida":    "Cantidad sugerida",
            "costo_total":          "Costo aprox.",
        }
        df_acc_vis["meses_cobertura_real"] = df_acc_vis.get(
            "meses_cobertura_real", df_acc_vis.get("cobertura_meses", pd.Series(dtype=float))
        ).apply(_fmt_cobertura)
        df_acc_vis["costo_total"] = df_acc_vis["costo_total"].apply(
            lambda v: _fmt_millones(v) if pd.notna(v) else "Sin precio"
        )

        disp = df_acc_vis[[c for c in cols_acc if c in df_acc_vis.columns]].rename(columns=cols_acc)
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.success("Todos los materiales tienen stock suficiente con los filtros actuales. ✅")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── D. Gap operativo por requerimiento ────────────────────────────────────
    if not df_gap_req.empty:
        st.markdown("#### ¿Cuánto piden vs. cuánto usan realmente?")
        st.caption(
            "Cada barra muestra qué tan lejos está lo que operaciones pide "
            "de lo que los técnicos consumen. Menos es mejor."
        )

        df_gq = df_gap_req.sort_values("gap_pct", ascending=True).copy()
        colores_gap = [_COLOR_GAP.get(c, DICO_GRIS) for c in df_gq["clasificacion_gap"]]

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=df_gq["gap_pct"],
            y=df_gq["requerimiento"],
            orientation="h",
            marker_color=colores_gap,
            text=df_gq["gap_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            textfont=dict(color=DICO_NAVY, size=10),
            customdata=df_gq[["total_pedido", "total_consumido", "gap_pct"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Total solicitado por operaciones: <b>%{customdata[0]:,.0f}</b><br>"
                "Total usado por técnicos: <b>%{customdata[1]:,.0f}</b><br>"
                "Diferencia: <b>%{customdata[2]:.1f}%</b>"
                "<extra></extra>"
            ),
        ))
        fig_gap.update_layout(
            **{**TEMPLATE_DICO, "margin": dict(l=20, r=80, t=50, b=40)},
            title=dict(text="<b>¿Cuánto piden vs. cuánto usan realmente? (%)</b>", x=0.02),
            height=max(280, len(df_gq) * 40 + 100),
            xaxis=dict(title_text="Diferencia (%)", ticksuffix="%", tickfont=dict(size=10, color=DICO_NAVY),
                       title_font=dict(color=DICO_NAVY), showgrid=True, gridcolor=DICO_GRIS_CLR),
            yaxis=dict(title_text="", tickfont=dict(size=10, color=DICO_NAVY), showgrid=False, automargin=True),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"🕐 Datos actualizados: {datetime.now(_BOGOTA).strftime('%d/%m/%Y %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — 🔍 CONSULTAR REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def tab_consultar_referencia(df, df_trm, df_stats, config):
    """
    Analista: ¿a qué precio compro X y cuándo es el mejor momento?
    """
    # ── Buscador de referencia ─────────────────────────────────────────────────
    st.markdown("#### Buscar un material")

    col_bus, col_sel = st.columns([1, 2])
    with col_bus:
        texto_busqueda = st.text_input(
            "¿Qué material necesitas consultar?",
            placeholder="Escribe el nombre o código del material...",
            key="sb_busqueda",
            label_visibility="visible",
        )

    conteo = df.groupby("referencia").size().reset_index(name="n_compras")
    refs_aptas = conteo[conteo["n_compras"] >= MIN_COMPRAS_PREDICCION]["referencia"]
    refs_dispon = (
        df[df["referencia"].isin(refs_aptas)]
        .groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )
    refs_dispon["opcion"] = refs_dispon["referencia"] + " — " + refs_dispon["descripcion"].str[:55]

    if texto_busqueda:
        mask = (
            refs_dispon["descripcion"].str.contains(texto_busqueda, case=False, na=False)
            | refs_dispon["referencia"].str.contains(texto_busqueda, case=False, na=False)
        )
        refs_dispon = refs_dispon[mask]

    with col_sel:
        opcion_sel = st.selectbox(
            "Selecciona el material",
            refs_dispon["opcion"].tolist(),
            index=None,
            placeholder="Elige de la lista...",
            key="sb_referencia",
            label_visibility="visible",
        )

    referencia = opcion_sel.split(" — ")[0].strip() if opcion_sel else None

    if not referencia:
        st.markdown("""
        <div style="background:#f8f9fa; border-left:4px solid #d60000; border-radius:0 10px 10px 0;
             padding:1.2rem 1.5rem; margin-top:1.5rem;">
            <div style="font-size:1rem; font-weight:700; color:#0d2232; margin-bottom:0.4rem;">
                Consulta el precio de cualquier material
            </div>
            <div style="font-size:0.85rem; color:#444; line-height:1.7;">
                1. Escribe el nombre del material en el buscador de arriba.<br>
                2. Selecciónalo de la lista.<br>
                3. Verás el precio probable para el próximo mes y el mejor momento
                   para hacer la compra.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    datos_ref  = df[df["referencia"] == referencia]
    desc       = datos_ref["descripcion"].value_counts().idxmax() if not datos_ref.empty else "—"
    prov_top   = datos_ref["proveedor"].value_counts().idxmax() if not datos_ref.empty else "—"
    n_compras  = len(datos_ref)
    ult_precio = datos_ref.sort_values("fecha")["precio_cop"].iloc[-1] if not datos_ref.empty else 0
    moneda     = datos_ref.sort_values("fecha")["moneda"].iloc[-1] if not datos_ref.empty else "COP"

    # ── Tarjeta de referencia ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ref-card">
        <div class="ref-code">{referencia}</div>
        <div class="ref-desc">{desc[:70]}</div>
    </div>
    """, unsafe_allow_html=True)

    horizonte = st.session_state.get("sb_horizonte", HORIZONTE_MESES)
    with st.spinner("Calculando predicción…"):
        resultado = obtener_prediccion_ref(df, referencia, horizonte)

    metodo = resultado.get("metodo", "—")
    mape   = resultado["metricas"].get("MAPE", None)

    if metodo in ("datos_insuficientes", "sin_datos"):
        st.warning(
            f"Este material tiene muy pocas compras registradas (menos de {MIN_COMPRAS_PREDICCION}). "
            "No es posible generar una estimación de precio confiable."
        )
        return

    forecast = resultado.get("forecast", pd.DataFrame())
    futuro   = pd.DataFrame()
    if not forecast.empty:
        futuro = forecast[forecast["ds"] > pd.Timestamp(datetime.today().date())]

    # ── Layout 2 columnas: gráfico | panel decisión ────────────────────────────
    col_graf, col_panel = st.columns([2.6, 1])

    with col_graf:
        st.markdown("**Histórico de precios y proyección**")
        fig = grafico_prediccion_referencia(df, referencia, resultado_modelo=resultado)
        st.plotly_chart(fig, use_container_width=True)

        # Tabla proyección mes a mes
        if not futuro.empty:
            st.markdown("**Precios estimados por mes**")
            tabla = futuro[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy().head(8)
            tabla.columns = ["Mes", "Precio estimado", "Precio mínimo posible", "Precio máximo posible"]
            tabla["Mes"] = tabla["Mes"].dt.strftime("%b %Y")

            # Resaltar mínimo
            idx_min = tabla["Precio estimado"].idxmin()

            def _fmt(v, i):
                s = formato_cop(v)
                return f"✅ {s} ← mejor mes" if i == idx_min else s

            tabla["Precio estimado"]       = [_fmt(v, i) for i, v in zip(tabla.index, tabla["Precio estimado"])]
            tabla["Precio mínimo posible"] = tabla["Precio mínimo posible"].apply(formato_cop)
            tabla["Precio máximo posible"] = tabla["Precio máximo posible"].apply(formato_cop)
            st.dataframe(tabla, use_container_width=True, hide_index=True, height=260)

    with col_panel:
        if not futuro.empty:
            precio_prox = futuro.iloc[0]["yhat"]
            precio_min  = futuro.iloc[0]["yhat_lower"]
            precio_max  = futuro.iloc[0]["yhat_upper"]

            # Precio estimado próximo mes
            _variacion_rel = (precio_max - precio_min) / max(precio_prox, 1)
            _conf_frase = "Precio con variación esperada: rango amplio" if _variacion_rel > 0.20 else "Estimación confiable"
            st.markdown(f"""
            <div class="precio-probable">
                <div class="label">Precio estimado para el próximo mes</div>
                <div class="valor">{formato_cop(precio_prox)}</div>
                <div class="rango">{_conf_frase}</div>
            </div>
            """, unsafe_allow_html=True)

            # Señal principal
            idx_opt    = futuro["yhat"].idxmin()
            mes_opt    = futuro.loc[idx_opt, "ds"]
            precio_opt = futuro.loc[idx_opt, "yhat"]
            meses_hasta_opt = max(0, round((mes_opt - pd.Timestamp(datetime.today().date())).days / 30.4))

            precio_sube = futuro.iloc[0]["yhat"] > ult_precio * 1.03 if ult_precio > 0 else False

            if meses_hasta_opt <= 1 or precio_sube:
                senal = "🟢 BUEN MOMENTO PARA COMPRAR"
                css_senal = "senal-compra-ya"
            elif meses_hasta_opt >= 3:
                senal = "🔴 ESPERA — EL PRECIO PUEDE BAJAR"
                css_senal = "senal-espera"
            else:
                senal = "🟡 MANTENTE PENDIENTE"
                css_senal = "senal-monitorea"

            st.markdown(f'<div class="{css_senal}">{senal}</div>', unsafe_allow_html=True)

            # Ventana óptima
            if precio_opt < ult_precio and ult_precio > 0:
                ahorro_pct = (ult_precio - precio_opt) / ult_precio * 100
                st.markdown(f"""
                <div class="optima-card">
                    <div class="titulo">📅 Cuándo comprar para ahorrar más</div>
                    <div class="mes">{mes_opt.strftime("%B %Y").capitalize()}</div>
                    <div class="precio-opt">Precio proyectado: {formato_cop(precio_opt)}</div>
                    <div class="ahorro">Comprar en ese mes puede ahorrarte {ahorro_pct:.1f}% vs. comprarlo hoy</div>
                </div>
                """, unsafe_allow_html=True)

            # Tendencia
            if len(futuro) >= 2:
                var_pct = (futuro.iloc[-1]["yhat"] - ult_precio) / max(ult_precio, 1) * 100
                if var_pct >= 0:
                    st.markdown(f'<div class="tendencia-up">📈 El precio podría subir {var_pct:.1f}% en los próximos meses</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="tendencia-down">📉 El precio podría bajar {abs(var_pct):.1f}% en los próximos meses</div>', unsafe_allow_html=True)

            # Alerta FX
            if moneda in ("USD", "EUR"):
                st.warning(
                    f"⚠️ Este material se importa en {moneda}. El precio en pesos puede "
                    "cambiar según la tasa de cambio del día."
                )

            # Confiabilidad
            st.markdown("---")
            if mape and not np.isnan(mape):
                nivel_conf = "Alta" if mape <= 12 else ("Media" if mape <= 25 else "Baja")
                _frase_conf = {
                    "Alta":  "✅ Proyección basada en historial sólido",
                    "Media": "ℹ️ Proyección basada en historial moderado",
                    "Baja":  "⚠️ Historial limitado — tomar como referencia",
                }[nivel_conf]
                st.markdown(
                    f'<span style="font-size:0.82rem;color:#444;">{_frase_conf}</span>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Proveedor principal: {prov_top[:28]}")

    # ── Historial de compras ───────────────────────────────────────────────────
    st.markdown("**Compras anteriores de este material**")
    hist = (
        datos_ref.sort_values("fecha", ascending=False)
        .head(20)[["fecha", "proveedor", "precio_cop", "cantidad"]]
        .copy()
    )
    hist["fecha"]      = hist["fecha"].dt.strftime("%d/%m/%Y")
    hist["precio_cop"] = hist["precio_cop"].apply(formato_cop)
    hist["cantidad"]   = hist["cantidad"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    hist.columns = ["Fecha", "Proveedor", "Precio pagado (COP)", "Cantidad comprada"]
    st.dataframe(hist, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — 🛒 PLAN DE COMPRAS
# ══════════════════════════════════════════════════════════════════════════════

def tab_plan_compras(df, df_trm, df_stats, df_filtrado, config):
    """
    Analista + gerente: ¿qué compro, cuánto y con qué plata?
    """
    meses_obj   = st.session_state.get("meses_objetivo", 3)
    lead_dias   = st.session_state.get("lead_time_dias", 30)
    lead_meses  = round(lead_dias / 30, 2)

    # df_prio se inicializa vacío y se llena si el usuario ingresa presupuesto
    df_prio = pd.DataFrame()

    df_predicciones = st.session_state.get("df_predicciones")

    if df_predicciones is None or df_predicciones.empty:
        st.info(
            "Los precios estimados aún no están calculados. Puedes activarlos "
            "desde la sección **'📊 Análisis detallado'**."
        )

    with st.spinner("Calculando plan de compras…"):
        reporte    = _obtener_reporte(df, df_predicciones, meses_obj, lead_meses)
    df_compras = reporte.get("compras", pd.DataFrame())
    df_gap_ref = reporte.get("gap_ref", pd.DataFrame())
    df_gap_req = reporte.get("gap_req", pd.DataFrame())

    # Filtro por requerimiento
    if config["requerimiento"] and not df_compras.empty:
        refs_req = set(df_filtrado["referencia"].unique())
        df_compras = df_compras[df_compras["referencia"].isin(refs_req)].copy()

    if df_compras.empty:
        st.warning("No hay datos de inventario disponibles. Verifica la conexión a la base de datos.")
        return

    # Señal de precio desde predicciones
    senal_map = {}
    if df_predicciones is not None and not df_predicciones.empty:
        try:
            df_op = calcular_oportunidades_compra(df, df_predicciones, umbral_ahorro_pct=1)
            if not df_op.empty:
                senal_map = dict(zip(df_op["referencia"], df_op["senal"]))
        except Exception:
            pass

    # Último precio y descripción (desde historial de compras)
    desc_map = (
        df.groupby("referencia")["descripcion"]
        .agg(lambda x: x.dropna().value_counts().idxmax() if not x.dropna().empty else "N/A").to_dict()
    )
    # Fallback: descripción desde saldos para refs sin historial de compras
    if not df_compras.empty and "descripcion" in df_compras.columns:
        for ref, desc in df_compras[["referencia", "descripcion"]].drop_duplicates("referencia").values:
            if ref not in desc_map and str(desc).strip():
                desc_map[ref] = desc

    # Precio histórico promedio como fallback cuando no hay predicciones
    precio_hist_map = {}
    if "precio_cop" in df.columns:
        precio_hist_map = (
            df[df["precio_cop"] > 0]
            .groupby("referencia")["precio_cop"]
            .mean()
            .to_dict()
        )

    # Enriquecer precio_proyectado_promedio con precio histórico donde falte
    if not df_compras.empty and precio_hist_map:
        mask_sin_precio = df_compras["precio_proyectado_promedio"].isna()
        df_compras.loc[mask_sin_precio, "precio_proyectado_promedio"] = (
            df_compras.loc[mask_sin_precio, "referencia"].map(precio_hist_map)
        )
        # Recalcular costo_total donde se llenó precio
        mask_recalc = mask_sin_precio & df_compras["precio_proyectado_promedio"].notna() & (df_compras["cantidad_sugerida"] > 0)
        df_compras.loc[mask_recalc, "costo_total"] = (
            df_compras.loc[mask_recalc, "cantidad_sugerida"] * df_compras.loc[mask_recalc, "precio_proyectado_promedio"]
        ).round()

    # ── SECCIÓN B — Tabla de compras sugeridas ────────────────────────────────
    st.markdown("#### ¿Qué comprar este mes?")
    st.caption("Materiales que necesitan reposición basado en el stock actual y la demanda histórica.")

    df_sugeridas = df_compras[df_compras.get("cantidad_sugerida", pd.Series(0, index=df_compras.index)) > 0].copy()

    # Filtros internos del tab
    _col_f1, _col_f2 = st.columns([1, 2])
    with _col_f1:
        mostrar_cubiertas = st.checkbox("Mostrar materiales que no necesitan compra ahora", value=False, key="chk_cubiertas")
    with _col_f2:
        _TODAS_SENALES = ["🟢 Compra ya", "🟡 Monitorea", "🔴 Espera"]
        tab3_senal = st.multiselect(
            "Señal de precio",
            options=_TODAS_SENALES,
            default=_TODAS_SENALES,
            key="tab3_senal",
            help="Filtra la tabla según la señal del modelo predictivo: cuándo conviene comprar según tendencia de precios.",
        )

    if mostrar_cubiertas:
        df_sugeridas = df_compras.copy()

    # KPIs sobre la tabla
    total_refs   = int((df_compras.get("cantidad_sugerida", pd.Series(0)) > 0).sum())
    total_unids  = int(df_sugeridas.get("cantidad_sugerida", pd.Series(0)).sum())
    total_costo  = df_sugeridas["costo_total"].sum(skipna=True) if "costo_total" in df_sugeridas.columns else 0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(kpi_html(str(total_refs), "Materiales a reponer"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html(f"{total_unids:,}", "Unidades totales"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_html(_fmt_millones(total_costo), "Costo total estimado"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df_sugeridas.empty:
        st.success("Todos los materiales tienen stock suficiente con los filtros actuales. ✅")
    else:
        # Construir tabla visible
        t = df_sugeridas.copy()
        t["descripcion"] = t["referencia"].map(desc_map).fillna("—")
        t["senal"]       = t["referencia"].map(senal_map).fillna("—")
        t["alerta"]      = t.get("alerta_stock", pd.Series(False, index=t.index)).fillna(False)

        # Aplicar filtro de señal (solo si no están todas seleccionadas y hay predicciones)
        if tab3_senal and len(tab3_senal) < 3 and senal_map:
            t = t[t["senal"].isin(tab3_senal) | (t["senal"] == "—")].copy()

        # Estado combinado con lenguaje natural
        _ESTADO_MAP = {
            "🚨 Alerta":             "🚨 Comprar pronto",
            "⚠️ Monitorear":         "⚠️ Revisar",
            "✅ Cubierto (tránsito)": "✅ Cubierto",
            "✅ Cubierto":           "✅ Cubierto",
        }
        t["estado_vis"] = t.apply(
            lambda r: (
                "🚨 Comprar ahora"        if r["alerta"] and r["senal"] == "🟢 Compra ya" else
                "⚠️ Precio alto — evaluar" if r["alerta"] and r["senal"] == "🔴 Espera" else
                _ESTADO_MAP.get(r.get("estado_stock", "—"), r.get("estado_stock", "—"))
            ), axis=1
        )

        t["cob_real_fmt"]   = t.get("meses_cobertura_real", t.get("cobertura_meses", pd.Series(dtype=float))).apply(_fmt_cobertura)
        t["precio_fmt"]     = t.get("precio_proyectado_promedio", pd.Series(dtype=float)).apply(
            lambda v: formato_cop(v) if pd.notna(v) else "Sin precio"
        )
        t["costo_fmt"]      = t.get("costo_total", pd.Series(dtype=float)).apply(
            lambda v: _fmt_millones(v) if pd.notna(v) else "Sin precio"
        )
        _CRIT_MAP = {"Alta": "🔴 Alta", "Media": "🟡 Media", "Baja": "⚪ Baja"}
        t["criticidad_fmt"] = t.get("criticidad", pd.Series("—", index=t.index)).map(
            lambda v: _CRIT_MAP.get(v, v)
        )

        t["stock_fmt"] = t.get("stock_actual", pd.Series(0, index=t.index)).apply(
            lambda v: f"{int(v):,}" if pd.notna(v) else "0"
        )
        # Promedio de consumo mensual (fuente: consumos o actas)
        t["consumo_fmt"] = t.apply(
            lambda r: (
                f"{r['demanda_mensual_promedio']:.1f} u/mes"
                + (" ✱" if str(r.get("fuente_demanda", "")).startswith("actas") else "")
                if pd.notna(r.get("demanda_mensual_promedio")) and r.get("demanda_mensual_promedio", 0) > 0
                else "—"
            ), axis=1
        )
        vis = t[[
            "estado_vis", "referencia", "descripcion", "stock_fmt", "cob_real_fmt",
            "consumo_fmt", "cantidad_sugerida", "precio_fmt", "costo_fmt", "criticidad_fmt",
        ]].copy()
        vis.columns = [
            "Situación", "Código", "Material", "Saldo actual (u.)", "Cubre",
            "Prom. consumo", "Cantidad a pedir", "Precio estimado", "Costo total estimado", "Importancia",
        ]
        st.dataframe(vis, use_container_width=True, hide_index=True,
                     column_config={
                         "Material":          st.column_config.TextColumn(width="large"),
                         "Saldo actual (u.)": st.column_config.TextColumn(help="Unidades en bodega física hoy"),
                         "Cubre":             st.column_config.TextColumn(help="Meses de stock disponible con el consumo actual"),
                         "Prom. consumo":     st.column_config.TextColumn(help="Consumo mensual promedio histórico. ✱ = basado en actas (no consumos directos)"),
                     })

    # ── SECCIÓN C — Priorización por presupuesto ──────────────────────────────
    st.markdown("---")
    st.markdown("#### Priorización con presupuesto")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        presupuesto_total = st.number_input(
            "¿Cuánto tienes para gastar este mes? (COP)", min_value=0, value=50_000_000,
            step=5_000_000, format="%d", key="tab3_presupuesto",
            help="Deja en 0 para ver todas las sugerencias sin restricción de presupuesto.",
        )
    with col_p2:
        presupuesto_mensual = st.number_input(
            "¿Cuánto puedes gastar por mes? (COP)", min_value=0, value=20_000_000,
            step=2_000_000, format="%d", key="tab3_pres_mensual",
        )
    with col_p3:
        n_meses_cf = st.slider("¿Para cuántos meses quieres planear?", min_value=1, max_value=12, value=4, key="tab3_horizonte")

    if presupuesto_total > 0 and not df_compras.empty:
        df_op_local = pd.DataFrame()
        if df_predicciones is not None and not df_predicciones.empty:
            try:
                df_op_local = calcular_oportunidades_compra(df, df_predicciones, umbral_ahorro_pct=1)
            except Exception:
                pass

        df_prio = calcular_prioridad_compra(df_compras, df_op_local if not df_op_local.empty else None)  # noqa: F841

        if not df_prio.empty:
            df_dentro, df_fuera, resumen = priorizar_dentro_presupuesto(df_prio, presupuesto_total)

            # Barra de progreso
            pct_uso = resumen.get("pct_presupuesto_uso", 0)
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(kpi_html(str(resumen.get("n_dentro", 0)), "Materiales incluidos"), unsafe_allow_html=True)
            k2.markdown(kpi_html(str(resumen.get("n_ineludibles", 0)), "🚨 Compras obligatorias"), unsafe_allow_html=True)
            k3.markdown(kpi_html(_fmt_millones(resumen.get("costo_dentro", 0)), "Gasto inmediato"), unsafe_allow_html=True)
            k4.markdown(kpi_html(_fmt_millones(resumen.get("costo_diferido", 0)), "Para próximos meses"), unsafe_allow_html=True)
            k5.markdown(kpi_html(f"{pct_uso:.1f}%", "% del presupuesto usado"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(min(pct_uso / 100, 1.0), text=f"Presupuesto utilizado: {pct_uso:.1f}%")

            if resumen.get("hay_sobregiro"):
                st.error(
                    "🚨 Las compras obligatorias (materiales sin stock) superan el presupuesto disponible. "
                    "Se recomienda aprobar un presupuesto adicional."
                )

            if not df_dentro.empty:
                st.markdown("**Tabla de priorización**")
                desc_map2 = (df.groupby("referencia")["descripcion"]
                             .agg(lambda x: x.value_counts().idxmax()).to_dict())

                tbl = df_dentro.copy()
                tbl["descripcion"]    = tbl["referencia"].map(desc_map2).fillna("—")
                tbl["costo_fmt"]      = tbl["costo_total"].apply(lambda v: _fmt_millones(v) if pd.notna(v) else "—")
                tbl["cob_fmt"]        = tbl.get("meses_cobertura_real", pd.Series(dtype=float)).apply(_fmt_cobertura)
                tbl["score_fmt"]      = tbl["prioridad_score"].apply(lambda v: f"{v:.1f}")
                tbl["tipo"] = tbl.apply(
                    lambda r: "🚨 Compra obligatoria" if r.get("ineludible", False) else r.get("prioridad_nivel", "—"), axis=1
                )

                vis_prio = tbl[[
                    "tipo", "referencia", "descripcion", "criticidad",
                    "cob_fmt", "cantidad_sugerida", "costo_fmt",
                ]].rename(columns={
                    "tipo":             "Tipo",
                    "referencia":       "Código",
                    "descripcion":      "Material",
                    "criticidad":       "Importancia",
                    "cob_fmt":          "Stock actual cubre",
                    "cantidad_sugerida": "Cantidad a pedir",
                    "costo_fmt":        "Costo total estimado",
                })
                st.dataframe(vis_prio, use_container_width=True, hide_index=True)

            # Cash flow
            if presupuesto_mensual > 0:
                df_cf = proyectar_cashflow(df_prio, presupuesto_mensual, n_meses_cf)
                if not df_cf.empty:
                    st.markdown("**Proyección de cash flow mensual**")
                    resumen_cf = resumen_cashflow_por_mes(df_cf)

                    _CF_NOMBRES = {
                        "n_ineludibles": "Compras obligatorias",
                        "n_programadas": "Compras planificadas",
                        "n_diferidas":   "Compras para después",
                    }
                    if not resumen_cf.empty:
                        fig_cf = go.Figure()
                        for tipo, col_color in [("n_ineludibles", DICO_ROJO), ("n_programadas", VERDE_SENAL), ("n_diferidas", DICO_GRIS)]:
                            if tipo in resumen_cf.columns:
                                fig_cf.add_trace(go.Bar(
                                    x=resumen_cf["mes"], y=resumen_cf["costo_total"] * resumen_cf[tipo] / resumen_cf["n_referencias"].clip(lower=1),
                                    name=_CF_NOMBRES.get(tipo, tipo),
                                    marker_color=col_color,
                                ))
                        fig_cf.update_layout(
                            **{**TEMPLATE_DICO, "margin": dict(l=20, r=40, t=50, b=40)},
                            title=dict(text="<b>Plan de gastos mes a mes (COP)</b>", x=0.02),
                            barmode="stack", height=320,
                            xaxis=dict(tickfont=dict(size=10, color=DICO_NAVY), title_font=dict(color=DICO_NAVY)),
                            yaxis=dict(tickformat="$,.0f", tickfont=dict(size=10, color=DICO_NAVY),
                                       title_font=dict(color=DICO_NAVY), title_text="COP"),
                        )
                        st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.caption("Escribe cuánto tienes disponible para ver qué comprar primero.")

    # ── SECCIÓN D — Gap de eficiencia ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("📉 ¿Estamos pidiendo más de lo que usamos?", expanded=False):
        if not df_gap_ref.empty:
            st.markdown("**Por material**")
            tg_ref = df_gap_ref.copy()
            tg_ref["total_pedido"]    = tg_ref["total_pedido"].apply(lambda v: f"{v:,.0f}")
            tg_ref["total_consumido"] = tg_ref["total_consumido"].apply(lambda v: f"{v:,.0f}")
            tg_ref["gap_unidades"]    = tg_ref["gap_unidades"].apply(lambda v: f"{v:,.0f}")
            tg_ref["gap_pct"]         = tg_ref["gap_pct"].apply(lambda v: f"{v:.1f}%")
            tg_ref.columns = ["Código", "Total solicitado", "Total usado", "Diferencia (unidades)", "Diferencia (%)", "¿Qué tan eficiente?"]
            st.dataframe(tg_ref, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos suficientes para calcular la diferencia.")

        if not df_gap_req.empty:
            st.markdown("**Por área**")
            tg_req = df_gap_req.copy()
            tg_req["total_pedido"]    = tg_req["total_pedido"].apply(lambda v: f"{v:,.0f}")
            tg_req["total_consumido"] = tg_req["total_consumido"].apply(lambda v: f"{v:,.0f}")
            tg_req["gap_unidades"]    = tg_req["gap_unidades"].apply(lambda v: f"{v:,.0f}")
            tg_req["gap_pct"]         = tg_req["gap_pct"].apply(lambda v: f"{v:.1f}%")
            tg_req.columns = ["Área", "Total solicitado", "Total usado", "Diferencia (unidades)", "Diferencia (%)", "¿Qué tan eficiente?"]
            st.dataframe(tg_req, use_container_width=True, hide_index=True)

    # ── SECCIÓN E — Exportación ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if not df_compras.empty or not df_gap_req.empty:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            if not df_compras.empty:
                df_compras.to_excel(writer, sheet_name="Compras sugeridas", index=False)
            if not df_prio.empty:
                df_prio.to_excel(writer, sheet_name="Priorizacion", index=False)
            if not df_gap_req.empty:
                df_gap_req.to_excel(writer, sheet_name="Gap por requerimiento", index=False)
        buf.seek(0)
        st.download_button(
            label="📥 Descargar plan de compras en Excel",
            data=buf,
            file_name=f"plan_compras_{datetime.today().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 📊 ANÁLISIS PROFUNDO
# ══════════════════════════════════════════════════════════════════════════════

def tab_analisis_profundo(df, df_trm, df_stats, df_filtrado, df_stats_filtrado, config):
    """
    Para analistas avanzados. Todo lo técnico vive aquí.
    """
    df_predicciones = st.session_state.get("df_predicciones")

    # ── A. Rankings ───────────────────────────────────────────────────────────
    with st.expander("🏆 Rankings de materiales", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Materiales más costosos**")
            fig = grafico_ranking_referencias(df_stats_filtrado, top_n=15, modo="precio")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Materiales con precio más variable**")
            fig = grafico_ranking_referencias(df_stats_filtrado, top_n=15, modo="volatilidad")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Materiales con mayor alza histórica**")
        fig = grafico_ranking_referencias(df_stats_filtrado, top_n=15, modo="tendencia")
        st.plotly_chart(fig, use_container_width=True)

        if df_predicciones is not None and not df_predicciones.empty:
            st.markdown('<div class="dico-divider"></div>', unsafe_allow_html=True)
            st.markdown("**🚨 Materiales que más subirán de precio**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("*En los próximos 2 meses*")
                top2 = referencias_mayor_incremento(df_predicciones, df, top_n=10, meses=2)
                if not top2.empty:
                    top2["precio_actual"]     = top2["precio_actual"].apply(formato_cop)
                    top2["precio_proyectado"] = top2["precio_proyectado"].apply(formato_cop)
                    top2["incremento_pct"]    = top2["incremento_pct"].apply(lambda x: f"{x:+.1f}%")
                    top2_vis = top2[["referencia", "descripcion", "precio_actual",
                                     "precio_proyectado", "incremento_pct"]].copy()
                    top2_vis.columns = ["Código", "Material", "Precio actual",
                                        "Precio proyectado", "Variación esperada"]
                    st.dataframe(top2_vis, use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("*En los próximos 8 meses*")
                top8 = referencias_mayor_incremento(df_predicciones, df, top_n=10, meses=8)
                if not top8.empty:
                    top8["precio_actual"]     = top8["precio_actual"].apply(formato_cop)
                    top8["precio_proyectado"] = top8["precio_proyectado"].apply(formato_cop)
                    top8["incremento_pct"]    = top8["incremento_pct"].apply(lambda x: f"{x:+.1f}%")
                    top8_vis = top8[["referencia", "descripcion", "precio_actual",
                                     "precio_proyectado", "incremento_pct"]].copy()
                    top8_vis.columns = ["Código", "Material", "Precio actual",
                                        "Precio proyectado", "Variación esperada"]
                    st.dataframe(top8_vis, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Haz clic en el botón de abajo para calcular los precios proyectados. "
                "Esto puede tomar unos minutos la primera vez."
            )
            if st.button("⚡ Calcular precios proyectados para todos los materiales", type="primary", key="btn_pred_masiva"):
                _h = st.session_state.get("sb_horizonte", HORIZONTE_MESES)
                with st.spinner("Entrenando modelos para todas las referencias… esto puede tardar unos minutos."):
                    st.session_state["df_predicciones"] = cargar_predicciones(df, df_stats, horizonte=_h)
                st.rerun()

    # ── B. Análisis de proveedores ────────────────────────────────────────────
    with st.expander("🏭 Proveedores", expanded=False):
        st.plotly_chart(grafico_comparacion_proveedores(df_filtrado), use_container_width=True)

        resumen_prov = (
            df_filtrado.groupby("proveedor")
            .agg(
                referencias_unicas=("referencia", "nunique"),
                total_compras=("precio_cop", "count"),
                precio_promedio=("precio_cop", "mean"),
                total_invertido=("precio_cop", "sum"),
                volatilidad=("precio_cop", lambda x: x.std()/x.mean()*100 if x.mean() > 0 else 0),
            )
            .reset_index()
            .sort_values("total_invertido", ascending=False)
        )
        resumen_prov["precio_promedio"] = resumen_prov["precio_promedio"].apply(formato_cop)
        resumen_prov["total_invertido"] = resumen_prov["total_invertido"].apply(formato_cop)
        resumen_prov["volatilidad"]     = resumen_prov["volatilidad"].apply(lambda x: f"{x:.1f}%")
        resumen_prov.columns = ["Proveedor", "Materiales", "Compras realizadas",
                                "Precio promedio pagado", "Total comprado", "Variación de precio"]
        st.dataframe(resumen_prov, use_container_width=True, hide_index=True)

    # ── C. Evolución general del mercado ──────────────────────────────────────
    with st.expander("📈 Comportamiento general de precios", expanded=False):
        kpis = calcular_kpis(df_filtrado, df_trm)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(kpi_html(kpis["total_referencias"],    "Materiales analizados"), unsafe_allow_html=True)
        col2.markdown(kpi_html(kpis["total_proveedores"],    "Proveedores activos"), unsafe_allow_html=True)
        col3.markdown(kpi_html(kpis["precio_promedio_cop"],  "Precio promedio"), unsafe_allow_html=True)
        col4.markdown(kpi_html(kpis["trm_actual"],           "Tasa de cambio (TRM)"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not df_trm.empty:
            st.markdown("**Histórico de la tasa de cambio**")
            st.plotly_chart(grafico_trm_historica(df_trm), use_container_width=True)

        st.markdown("**¿Con qué frecuencia compramos cada material?**")
        st.plotly_chart(grafico_frecuencia_compras(df_filtrado, top_n=20), use_container_width=True)

        if df_predicciones is not None and not df_predicciones.empty:
            st.markdown("**Resumen de señales de compra**")
            try:
                df_op_dist = calcular_oportunidades_compra(df_filtrado, df_predicciones, umbral_ahorro_pct=1)
                if not df_op_dist.empty:
                    dist_senal = df_op_dist["senal"].value_counts().reset_index()
                    dist_senal.columns = ["Señal", "Materiales"]
                    st.dataframe(dist_senal, use_container_width=True, hide_index=True)
            except Exception:
                pass

    # ── D. Homólogos / subcategorías ──────────────────────────────────────────
    with st.expander("🔗 Materiales similares agrupados", expanded=False):
        ref_to_subcat = st.session_state.get("ref_to_subcat", {})
        if ref_to_subcat:
            tg = tabla_grupos_homologos(df, ref_to_subcat)
            if not tg.empty:
                st.markdown(
                    "Estos materiales tienen nombres parecidos y pueden ser equivalentes. "
                    "Revisar si se pueden consolidar en una sola referencia."
                )
                col_g1, col_g2 = st.columns([1.5, 1])
                with col_g1:
                    tg_vis = tg.copy()
                    tg_vis.columns = ["Grupo", "Materiales en el grupo", "Códigos", "Descripciones"]
                    st.dataframe(tg_vis, use_container_width=True, hide_index=True)
                with col_g2:
                    try:
                        df_actas_h   = cargar_actas()
                        df_pedidos_h = cargar_pedidos()
                        df_dem_h     = calcular_demanda_mensual(df_actas_h, df_pedidos_h)
                        df_dem_sc    = calcular_demanda_por_subcategoria(df_dem_h, ref_to_subcat)
                        if not df_dem_sc.empty:
                            st.markdown("**Uso mensual por grupo**")
                            dsc_vis = df_dem_sc.copy()
                            dsc_vis["demanda_total_mensual"] = dsc_vis["demanda_total_mensual"].apply(lambda v: f"{v:.1f}")
                            dsc_vis.columns = ["Grupo", "Uso mensual", "Materiales en el grupo", "Códigos"]
                            st.dataframe(dsc_vis, use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            else:
                st.info("No se detectaron materiales similares. Todas las referencias tienen descripciones suficientemente distintas.")
        else:
            st.info("Los grupos se calculan automáticamente al cargar la app. Recarga si no aparecen.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Cargar datos ──────────────────────────────────────────────────────────
    try:
        df, df_trm, df_stats, timestamp = cargar_datos()
    except ConnectionError as e:
        st.error(
            "No se pudo conectar a los datos. Verifica que la VPN esté activa "
            "y vuelve a intentarlo. Si el problema persiste, contacta a TI."
        )
        st.stop()
    except Exception as e:
        st.error(f"**Error cargando datos:** {e}")
        st.stop()

    if df.empty:
        st.warning("No hay datos disponibles. Verifica la conexión a la red de DICO.")
        st.stop()

    # ── Homólogos ─────────────────────────────────────────────────────────────
    try:
        df_enriquecido, ref_to_subcat = cargar_homologos(df)
        st.session_state["ref_to_subcat"] = ref_to_subcat
    except Exception:
        df_enriquecido  = df.copy()
        ref_to_subcat   = {}
        st.session_state["ref_to_subcat"] = {}

    # ── Header ────────────────────────────────────────────────────────────────
    ruta_logo = os.path.join(os.path.dirname(__file__), "logo_dico.png")
    mostrar_header(ruta_logo if os.path.exists(ruta_logo) else None, timestamp)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    config = configurar_sidebar(df, df_stats, ref_to_subcat=ref_to_subcat, timestamp=timestamp)
    df_filtrado = config["df_filtrado"]

    # Estadísticas sobre el subset filtrado
    if len(df_filtrado) < len(df):
        n_refs_filt = df_filtrado["referencia"].nunique()
        st.info(f"🔍 Filtro activo: mostrando **{n_refs_filt}** referencias (de {df['referencia'].nunique()} totales)")

    df_stats_filtrado = calcular_stats_filtrado(df_filtrado)

    # ── Navegación por tabs ───────────────────────────────────────────────────
    tab_sel = st.radio(
        "Navegación",
        ["🏠 Resumen", "🔍 Buscar material", "🛒 ¿Qué comprar?", "📊 Análisis detallado"],
        horizontal=True,
        key="nav_tab",
        label_visibility="collapsed",
    )

    st.markdown('<div class="dico-divider"></div>', unsafe_allow_html=True)

    # ── Despacho de tabs ──────────────────────────────────────────────────────
    if tab_sel == "🏠 Resumen":
        tab_ejecutivo(df, df_trm, df_stats, df_filtrado, config)

    elif tab_sel == "🔍 Buscar material":
        tab_consultar_referencia(df, df_trm, df_stats, config)

    elif tab_sel == "🛒 ¿Qué comprar?":
        tab_plan_compras(df, df_trm, df_stats, df_filtrado, config)

    elif tab_sel == "📊 Análisis detallado":
        tab_analisis_profundo(df, df_trm, df_stats, df_filtrado, df_stats_filtrado, config)


if __name__ == "__main__":
    main()
