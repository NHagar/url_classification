import argparse

from url_classification.data_processing.datasets import load_data, make_splits
from url_classification.model_training.models import (
    train_bert_clf,
    train_distant_labeler,
    train_xgboost,
)

# command line arguments for selecting which dataset to train on
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, required=True, choices=["huffpo", "uci", "recognasumm"]
)
parser.add_argument("--train_bert", action="store_true", help="Train BERT model")
parser.add_argument(
    "--train_distant", action="store_true", help="Train distant labeler"
)
parser.add_argument("--train_xgboost", action="store_true", help="Train XGBoost model")
parser.add_argument(
    "--bert_variant",
    type=str,
    choices=["a", "b", "c"],
    help="Optional text variants for testing BERT",
)
parser.add_argument(
    "--bert_url",
    type=str,
    choices=["path", "netloc_path"],
    help="Optional URL variants for testing BERT",
)

if __name__ == "__main__":
    args = parser.parse_args()
    df, mapping_fpath = load_data(args.dataset)
    train, val, test = make_splits(df)

    # train BERT model
    if args.train_bert:
        if args.dataset == "recongasumm":
            bert_model_name = "distilbert-base-multilingual-cased"
        else:
            bert_model_name = "distilbert-base-uncased"

        train_bert_clf(
            train.x,
            train.y,
            val.x,
            val.y,
            model_name=bert_model_name,
            output_name=args.dataset,
        )

    # train distant labeler
    if args.train_distant:
        train_distant_labeler(
            train, mapping_file=mapping_fpath, output_name=args.dataset
        )

    # train xgboost model
    if args.train_xgboost:
        train_xgboost(train["text"], train["y"], output_name=args.dataset)

    test.to_csv(f"data/processed/{args.dataset}_test.csv", index=False)
