import os

from datasets import Dataset, Features, Value
import duckdb
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import xgboost


def train_bert_clf(X_train, y_train, X_val, y_val, model_name='distilbert-base-uncased', epochs=3, output_name="huffpo"):
    # initialize resources
    le = LabelEncoder()
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(y_train.unique()))

    # encode labels
    y_train = le.fit_transform(y_train).astype(int)
    y_val = le.transform(y_val).astype(int)


    # tokenize text and create datasets
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')

    train_dataset = Dataset.from_dict(
        {"text": X_train, "label": y_train.tolist()}, 
        features=Features({"text": Value("string"), "label": Value("int64")})
    )
    val_dataset = Dataset.from_dict(
        {"text": X_val, "label": y_val.tolist()}, 
        features=Features({"text": Value("string"), "label": Value("int64")})
    )

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # train model
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    # save model, tokenizer and label encoder
    # make directory if it doesn't exist
    os.makedirs(f'models/bert/{output_name}', exist_ok=True)

    model.save_pretrained(f'models/bert/{output_name}')
    tokenizer.save_pretrained(f'models/bert/{output_name}')
    torch.save(le, f'models/bert/{output_name}/label_encoder.pt')    

    return model, tokenizer, le


def train_distant_labeler(train, mapping_file="../../data/uci_categories_attributed.csv", output_name="huffpo"):
    con = duckdb.connect(database=':memory:', read_only=False)

    if mapping_file is not None:
        data = con.execute(f"""
        WITH mapping AS (
            SELECT * FROM '{mapping_file}'
        )
        SELECT x.text AS x, m.category AS y FROM train x JOIN mapping m ON SPLIT_PART(SPLIT_PART(x.URL, '://', 2), '/', 2) = m.slug
        """).fetch_df()

        data = data[data.y.notna()]

        X_train= data['x']
        y_train = data['y']
    else:
        X_train = train["text"]
        y_train = train["y"]

    
    # count vectorize X's
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)

    # train model
    clf = MultinomialNB()
    clf.fit(X_train_counts, y_train)

    # save model and vectorizer
    # make directory if it doesn't exist
    os.makedirs(f'models/distant/{output_name}', exist_ok=True)

    torch.save(clf, f'models/distant/{output_name}/model.pt')
    torch.save(vectorizer, f'models/distant/{output_name}/vectorizer.pt')

    return clf, vectorizer


def train_xgboost(X_train, y_train, output_name="huffpo"):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_train = embedder.encode(X_train.tolist())

    # encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # train model
    clf = xgboost.XGBClassifier()
    clf.fit(X_train, y_train)

    # save model and label encoder
    # make directory if it doesn't exist
    os.makedirs(f'models/xgboost/{output_name}', exist_ok=True)

    torch.save(clf, f'models/xgboost/{output_name}/model.pt')
    torch.save(le, f'models/xgboost/{output_name}/label_encoder.pt')

    return clf, le