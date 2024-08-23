import argparse

from data_processing.datasets import load_data, make_splits
from model_training.models import train_bert_clf, train_distant_labeler, train_xgboost

# command line arguments for selecting which dataset to train on
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, choices=["huffpo", "uci", "recognasumm"])

if __name__ == "__main__":
    args = parser.parse_args()
    df, mapping_fpath = load_data(args.dataset)
    train, val, test = make_splits(df)

    # train BERT model
    train_bert_clf(train.x, train.y, val.x, val.y, output_name=args.dataset)

    # train distant labeler
    train_distant_labeler(train, mapping_file=mapping_fpath, output_name=args.dataset)

    # train xgboost model
    train_xgboost(train["text"], train["y"], output_name=args.dataset)