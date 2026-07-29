from datasets import load_dataset
from transformers import AutoTokenizer
from config import *


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    encoded["labels"] = [LABEL2ID[label] for label in batch["label"]]
    return encoded


def load_data():
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/val.csv",
            "test": "data/test.csv"
        }
    )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text", "label"]
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


if __name__ == "__main__":

    dataset = load_data()

    print(dataset)