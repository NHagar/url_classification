from urllib.parse import urlparse

import duckdb
from sklearn.model_selection import train_test_split

from url_classification.model_config import DATASET_CONFIGS, load_configs

# Try to load configs from file, fall back to hardcoded
try:
    configs = load_configs()
    DATASET_CONFIGS_LOADED = configs["datasets"]
except (FileNotFoundError, KeyError):
    DATASET_CONFIGS_LOADED = DATASET_CONFIGS

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
    category AS y
FROM
    data d JOIN pcts
USING (category)
WHERE pct > {pct_threshold}
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
    category AS y
FROM
    data d JOIN pcts
ON d.Categoria = pcts.category
WHERE pct > {pct_threshold} AND d.URL LIKE 'http%'
""",
}

MAPPING_PATHS = {
    "huffpo": None,
    "uci": "data/uci_categories_attributed.csv",
    "recognasumm": "data/recognasumm_categories_attributed.csv",
}


def load_data(dataset_name):
    """Load dataset and apply preprocessing using configuration"""
    if dataset_name not in QUERIES:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Get dataset configuration
    config = DATASET_CONFIGS_LOADED.get(dataset_name, {})

    # Connect to in-memory database
    con = duckdb.connect(database=":memory:", read_only=False)

    # Get and execute query
    # Note: The min_category_percentage is already incorporated in the SQL queries
    # for huffpo and recognasumm datasets
    if dataset_name == "uci":
        q = QUERIES[dataset_name]
    else:
        pct_threshold = config.get("min_category_percentage", 0.02)
        q = QUERIES[dataset_name].format(pct_threshold=pct_threshold)

    data = con.execute(q).fetch_df()

    # Add dataset metadata from config
    data["dataset_name"] = dataset_name
    data["dataset_language"] = config.get("language", "en")

    # Process URLs for feature extraction
    data["parsed"] = data.URL.apply(lambda x: urlparse(x))
    data["x_netloc_path"] = data.parsed.apply(lambda x: x.netloc + x.path)
    data["x_path"] = data.parsed.apply(lambda x: x.path)

    # Clean URL paths for feature extraction
    data["x"] = data.x_path.str.replace(r"[/\-\\]", " ", regex=True)
    data = data[data.x.str.contains(r"\w", regex=True)]

    return data, MAPPING_PATHS[dataset_name]


def make_splits(df, dataset_name="huffpo"):
    """Split data into train, validation, and test sets using configured parameters"""
    config = DATASET_CONFIGS_LOADED.get(dataset_name, {})

    # Get split parameters from config or use defaults
    test_size = config.get("test_size", 0.2)
    val_size = config.get(
        "val_size", 0.5
    )  # proportion of test set to use as validation
    random_seed = config.get("random_seed", 20240823)

    # Split into train and test
    train, test = train_test_split(df, test_size=test_size, random_state=random_seed)

    # Split test into validation and test
    val, test = train_test_split(test, test_size=val_size, random_state=random_seed)

    return train, val, test
