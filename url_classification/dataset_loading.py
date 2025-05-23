from urllib.parse import urlparse

import duckdb
from sklearn.model_selection import train_test_split

QUERIES = {
    "huffpo": """
WITH data AS (
SELECT
    *,
    link AS URL
FROM
    'data/raw/news_categories.parquet'
),
total AS (
SELECT COUNT(*) AS total FROM data
),
counts AS (
SELECT category, COUNT(*) AS articles FROM data GROUP BY 1
),
pcts AS (
SELECT category, articles / total.total AS pct FROM counts, total
)
SELECT
    d.*,
    d.headline || ' ' || d.short_description AS text,
    d.headline AS bert_a,
    d.short_description AS bert_b,
    d.URL || ' ' || d.headline || ' ' || d.short_description AS bert_c,
    category AS y
FROM
    data d JOIN pcts
USING (category)
WHERE pct > 0.02
""",
    "uci": """
SELECT *, TITLE AS text, CATEGORY AS y FROM 'data/raw/uci_categories.parquet'
""",
    "recognasumm": """
WITH data AS (
SELECT
    *
FROM
    'data/raw/recognasumm.parquet'
),
total AS (
SELECT COUNT(*) AS total FROM data
),
counts AS (
SELECT Categoria AS category, COUNT(*) AS articles FROM data GROUP BY 1
),
pcts AS (
SELECT category, articles / total.total AS pct FROM counts, total
)
SELECT
    d.*,
    d.Titulo || ' ' || d.Subtitulo AS text,
    d.Titulo AS bert_a,
    d.Sumario AS bert_b,
    d.URL || ' ' || d.Titulo || ' ' || d.Sumario AS bert_c,
    category AS y
FROM
    data d JOIN pcts
ON d.Categoria = pcts.category
WHERE pct > 0.02 AND d.URL LIKE 'http%'
""",
}

MAPPING_PATHS = {
    "huffpo": None,
    "uci": "data/uci_categories_attributed.csv",
    "recognasumm": "data/recognasumm_categories_attributed.csv",
}


def load_data(query_name):
    con = duckdb.connect(database=":memory:", read_only=False)

    q = QUERIES[query_name]

    data = con.execute(q).fetch_df()
    data["parsed"] = data.URL.apply(lambda x: urlparse(x))
    data["x_netloc_path"] = data.parsed.apply(lambda x: x.netloc + x.path)
    data["x_path"] = data.parsed.apply(lambda x: x.path)

    data["x"] = data.x_path.str.replace(r"[/\-\\]", " ", regex=True)
    data = data[data.x.str.contains(r"\w", regex=True)]

    return data, MAPPING_PATHS[query_name]


def make_splits(df, seed=20240823):
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(test, test_size=0.5, random_state=seed)
    return train, val, test
