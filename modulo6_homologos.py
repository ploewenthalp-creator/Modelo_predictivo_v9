"""
=============================================================
DICO S.A. — Módulo 6: Homólogos (Agrupación por similitud)
=============================================================
Agrupa referencias cuya descripción sea similar en un 75% o más,
usando difflib.SequenceMatcher (librería estándar de Python).

Algoritmo:
  1. Normaliza todas las descripciones (minúsculas, sin caracteres especiales)
  2. Calcula similitud entre pares usando SequenceMatcher
  3. Une pares con ≥ umbral via Union-Find (componentes conexas)
  4. Nombra cada grupo con las 3 palabras más frecuentes del grupo

Uso:
  from modulo6_homologos import enriquecer_con_homologos, calcular_demanda_por_subcategoria

Autor: Data Science DICO S.A.
=============================================================
"""

import re
import difflib
from collections import Counter
from typing import Dict, List

import pandas as pd

# Umbral de similitud por defecto
UMBRAL_DEFAULT = 0.75

# Palabras vacías en español + unidades técnicas comunes
_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "y", "o", "a", "en", "con",
    "por", "para", "un", "una", "al", "se", "que", "es", "no", "si",
    "sin", "sobre", "bajo", "entre", "ante", "hacia", "hasta", "desde",
    # Unidades / abreviaciones técnicas
    "mm", "cm", "mt", "mts", "ft", "m", "pcs", "pc", "und", "unid",
    "x", "n", "no", "ref", "cod", "cat",
}


# ─── Utilidades internas ──────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Normaliza texto: minúsculas, sin caracteres especiales excepto letras y espacios."""
    texto = texto.lower().strip()
    # Conserva letras, dígitos y espacios
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def _similitud(a: str, b: str) -> float:
    """Calcula ratio de similitud entre dos strings normalizados."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def _nombre_grupo(descripciones: List[str]) -> str:
    """
    Genera el nombre de la subcategoría a partir de las descripciones del grupo.
    Toma las 3 palabras más frecuentes que no sean stopwords.
    """
    words = []
    for d in descripciones:
        tokens = re.findall(r"[a-záéíóúüñ]{3,}", _normalizar(d))
        words.extend(t for t in tokens if t not in _STOPWORDS)

    counter = Counter(words)
    top = [w for w, _ in counter.most_common(3)]
    return " ".join(w.upper() for w in top) if top else "SIN CATEGORÍA"


# ─── Union-Find ───────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# ─── Función principal de agrupación ─────────────────────────────────────────

def agrupar_homologos(
    df: pd.DataFrame,
    umbral: float = UMBRAL_DEFAULT,
) -> Dict[str, str]:
    """
    Agrupa referencias cuya descripción sea similar ≥ umbral.

    Parameters
    ----------
    df      : DataFrame con columnas 'referencia' y 'descripcion'.
    umbral  : Umbral de similitud (0.0 – 1.0). Por defecto 0.75 (75%).

    Returns
    -------
    dict {referencia: nombre_subcategoria}
      - Grupos de 1 elemento: subcategoría = descripción truncada (40 chars).
      - Grupos de 2+ elementos: subcategoría = palabras más frecuentes del grupo.
    """
    if df.empty or "descripcion" not in df.columns:
        return {}

    # Descripción más frecuente por referencia
    refs_df = (
        df.groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    refs  = refs_df["referencia"].tolist()
    descs = refs_df["descripcion"].tolist()
    norms = [_normalizar(d) for d in descs]
    n     = len(refs)

    uf = _UnionFind(n)

    # Comparación por pares (O(n²) — eficiente para n < 2000)
    for i in range(n):
        for j in range(i + 1, n):
            # Filtro rápido: deben compartir al menos 1 token no-stopword
            tokens_i = set(re.findall(r"[a-záéíóúüñ]{3,}", norms[i])) - _STOPWORDS
            tokens_j = set(re.findall(r"[a-záéíóúüñ]{3,}", norms[j])) - _STOPWORDS
            if not tokens_i.intersection(tokens_j):
                continue
            if _similitud(norms[i], norms[j]) >= umbral:
                uf.union(i, j)

    # Construir grupos
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    # Asignar nombres
    ref_to_subcat: Dict[str, str] = {}
    for root, members in groups.items():
        if len(members) == 1:
            nombre = descs[members[0]][:40].strip().upper()
        else:
            nombre = _nombre_grupo([descs[m] for m in members])
        for m in members:
            ref_to_subcat[refs[m]] = nombre

    return ref_to_subcat


# ─── Enriquecimiento del DataFrame ───────────────────────────────────────────

def enriquecer_con_homologos(
    df: pd.DataFrame,
    umbral: float = UMBRAL_DEFAULT,
) -> pd.DataFrame:
    """
    Agrega la columna 'subcategoria' al DataFrame.

    Parameters
    ----------
    df     : DataFrame con columnas 'referencia' y 'descripcion'.
    umbral : Umbral de similitud. Por defecto 0.75.

    Returns
    -------
    DataFrame con columna adicional 'subcategoria'.
    """
    ref_to_subcat = agrupar_homologos(df, umbral)
    df = df.copy()
    df["subcategoria"] = df["referencia"].map(ref_to_subcat).fillna("—")
    return df, ref_to_subcat


# ─── Demanda por subcategoría ─────────────────────────────────────────────────

def calcular_demanda_por_subcategoria(
    df_demanda: pd.DataFrame,
    ref_to_subcat: Dict[str, str],
) -> pd.DataFrame:
    """
    Agrega la demanda mensual promedio a nivel de subcategoría.

    Parameters
    ----------
    df_demanda    : Salida de calcular_demanda_mensual() — [referencia, demanda_mensual_promedio, ...]
    ref_to_subcat : Diccionario {referencia: subcategoria} de agrupar_homologos()

    Returns
    -------
    DataFrame [subcategoria, demanda_total_mensual, n_referencias, referencias]
    """
    EMPTY = pd.DataFrame(columns=[
        "subcategoria", "demanda_total_mensual", "n_referencias", "referencias"
    ])
    if df_demanda.empty:
        return EMPTY

    df = df_demanda.copy()
    df["subcategoria"] = df["referencia"].map(ref_to_subcat).fillna("—")

    resumen = (
        df.groupby("subcategoria")
        .agg(
            demanda_total_mensual=("demanda_mensual_promedio", "sum"),
            n_referencias=("referencia", "count"),
            referencias=("referencia", lambda x: ", ".join(sorted(x))),
        )
        .reset_index()
        .sort_values("demanda_total_mensual", ascending=False)
        .reset_index(drop=True)
    )
    resumen["demanda_total_mensual"] = resumen["demanda_total_mensual"].round(2)
    return resumen


# ─── Tabla resumen de grupos ──────────────────────────────────────────────────

def tabla_grupos_homologos(
    df: pd.DataFrame,
    ref_to_subcat: Dict[str, str],
) -> pd.DataFrame:
    """
    Genera tabla de grupos para mostrar en el tablero.

    Returns
    -------
    DataFrame [subcategoria, n_referencias, referencias, descripciones]
      Solo incluye subcategorías con 2 o más referencias (grupos reales).
    """
    if not ref_to_subcat or df.empty:
        return pd.DataFrame(columns=["subcategoria", "n_referencias", "referencias", "descripciones"])

    # Descripción representativa por referencia
    desc_map = (
        df.groupby("referencia")["descripcion"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    grupos: Dict[str, List[str]] = {}
    for ref, subcat in ref_to_subcat.items():
        grupos.setdefault(subcat, []).append(ref)

    rows = []
    for subcat, refs in grupos.items():
        if len(refs) < 2:
            continue
        rows.append({
            "subcategoria":  subcat,
            "n_referencias": len(refs),
            "referencias":   ", ".join(sorted(refs)),
            "descripciones": " | ".join(desc_map.get(r, "—") for r in sorted(refs)),
        })

    if not rows:
        return pd.DataFrame(columns=["subcategoria", "n_referencias", "referencias", "descripciones"])

    return (
        pd.DataFrame(rows)
        .sort_values("n_referencias", ascending=False)
        .reset_index(drop=True)
    )
