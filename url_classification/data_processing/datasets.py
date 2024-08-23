import duckdb
from sklearn.model_selection import train_test_split

QUERIES = {
    "huffpo": """
WITH data AS (
SELECT
    *
FROM
    '../../data/raw/news_categories.parquet'
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
SELECT d.*, d.headline || ' ' || d.short_description AS text, REPLACE(REPLACE(SPLIT_PART(d.link, '.com/', 2), '/', ' '), '-', ' ') AS x, category AS y FROM data d JOIN pcts USING (category) WHERE pct > 0.02
""",
    "uci": """
SELECT *, TITLE AS text, REPLACE(REPLACE(SPLIT_PART(URL, '.com/', 2), '/', ' '), '-', ' ') AS x, CATEGORY AS y FROM '../../data/raw/uci_categories.parquet'
""",
    "recognasumm": """
WITH data AS (
SELECT
    *
FROM
    '../../data/raw/recognasumm.parquet'
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
SELECT d.*, d.Titulo || ' ' || d.Subtitulo AS text, REPLACE(REPLACE(SPLIT_PART(REPLACE(d.URL, '.com.br', '.com'), '.com/', 2), '/', ' '), '-', ' ') AS x, category AS y FROM data d JOIN pcts ON d.Categoria = pcts.category WHERE pct > 0.02 AND d.URL LIKE 'http%'
"""
}

MAPPING_PATHS = {
    "huffpo": None,
    "uci": "../../data/uci_categories_attributed.csv",
    "recognasumm": "../../data/recognasumm_categories_attributed.csv"
}

def load_data(query_name):
    con = duckdb.connect(database=':memory:', read_only=False)

    q = QUERIES[query_name]
    return con.execute(q).fetch_df(), MAPPING_PATHS[query_name]

def make_splits(df, seed=20240823):
    train, test = train_test_split(df, test_size=0.2, random_state=seed)
    val, test = train_test_split(test, test_size=0.5, random_state=seed)
    return train, val, test