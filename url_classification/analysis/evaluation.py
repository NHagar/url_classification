import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def evaluate_bert(dataset):
    data = con.execute(f"SELECT * FROM 'data/processed/{dataset}_test.csv' ").fetch_df()

    model_path = f"models/bert/{dataset}"

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    label_encoder = torch.load(f"{model_path}/label_encoder.pt")

    # Move model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    texts = data['x'].tolist()
    # Tokenize the input
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    # Create a DataLoader for batch processing
    ds = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(ds, batch_size=64)  # Adjust batch size as needed

    # Run inference
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            input_ids, attention_mask = batch
            # Move batch to GPU
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim=1)
            # Move predictions back to CPU for extending the list
            predictions.extend(predicted_classes.cpu().tolist())

    # Add predictions to the DataFrame
    data['y_pred'] = predictions

    # Encode the labels
    data["y_encoded"] = label_encoder.transform(data["y"])

    # Calculate metrics
    accuracy = accuracy_score(data["y_encoded"], data["y_pred"])
    precision = precision_score(data["y_encoded"], data["y_pred"], average="macro")
    recall = recall_score(data["y_encoded"], data["y_pred"], average="macro")
    f1 = f1_score(data["y_encoded"], data["y_pred"], average="macro")

    return {
        "model": "bert",
        "dataset": dataset,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_distant_labeling(dataset):
    mapping = MAPPING_PATHS[dataset]

    if mapping is not None:
        data = con.execute(f"""
        WITH mapping AS (
            SELECT * FROM '{mapping}'
        )
        SELECT x.text AS x, x.y, m.category AS y_pred FROM 'data/processed/{dataset}_test.csv' x LEFT JOIN mapping m ON SPLIT_PART(SPLIT_PART(x.URL, '://', 2), '/', 2) = m.slug
        """).fetch_df()

        labeled = data[data["y_pred"].notnull()]
        unlabeled = data[data["y_pred"].isnull()]
    else:
        data = con.execute(f"SELECT text AS x, y, NULL AS y_pred FROM 'data/processed/{dataset}_test.csv' ").fetch_df()
        unlabeled = data
        labeled = None
    
    # load in vectorizer and model
    vectorizer = torch.load(f"models/distant/{dataset}/vectorizer.pt")
    model = torch.load(f"models/distant/{dataset}/model.pt")
    # vectorize the text
    X = vectorizer.transform(unlabeled["x"])
    # run inference
    y_pred = model.predict(X)
    # add predictions to the DataFrame
    unlabeled["y_pred"] = y_pred
    # combine labeled and unlabeled data
    if labeled is not None:
        labeled = pd.concat([labeled, unlabeled])
    else:   
        labeled = unlabeled
    
    # Calculate metrics
    accuracy = accuracy_score(labeled["y"], labeled["y_pred"])
    precision = precision_score(labeled["y"], labeled["y_pred"], average="macro")
    recall = recall_score(labeled["y"], labeled["y_pred"], average="macro")
    f1 = f1_score(labeled["y"], labeled["y_pred"], average="macro")

    return {
        "model": "distant_labeling",
        "dataset": dataset,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_xgboost(dataset):
    data = con.execute(f"SELECT * FROM 'data/processed/{dataset}_test.csv' ").fetch_df()

    embedder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    X = embedder.encode(data["text"].tolist())

    # load label encoder and model
    label_encoder = torch.load(f"models/xgboost/{dataset}/label_encoder.pt")
    model = torch.load(f"models/xgboost/{dataset}/model.pt")

    # run inference
    y_pred = model.predict(X)
    # add predictions to the DataFrame
    data['y_pred'] = y_pred

    # Encode the labels
    data["y_encoded"] = label_encoder.transform(data["y"])

    # Calculate metrics
    accuracy = accuracy_score(data["y_encoded"], data["y_pred"])
    precision = precision_score(data["y_encoded"], data["y_pred"], average="macro")
    recall = recall_score(data["y_encoded"], data["y_pred"], average="macro")
    f1 = f1_score(data["y_encoded"], data["y_pred"], average="macro")

    return {
        "model": "xgboost",
        "dataset": dataset,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }




if __name__ == "__main__":
    con = duckdb.connect(':memory:')
    MAPPING_PATHS = {
        "huffpo": None,
        "uci": "data/uci_categories_attributed.csv",
        "recognasumm": "data/recognasumm_categories_attributed.csv"
    }

    datasets = ["huffpo", "uci", "recognasumm"]
    evaluation_metrics = []

    for d in datasets:
        evaluation_metrics.append(evaluate_bert(d))
        evaluation_metrics.append(evaluate_distant_labeling(d))
        evaluation_metrics.append(evaluate_xgboost(d))

    # save to csv
    pd.DataFrame(evaluation_metrics).to_csv("data/processed/evaluation_metrics.csv", index=False)

