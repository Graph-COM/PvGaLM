import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from scipy.special import expit
from argparse import ArgumentParser

from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction
from datasets import load_from_disk, load_dataset
import evaluate


def get_dataset_and_collator(
    data_path,
    model_checkpoints,
    add_prefix_space=True,
    max_length=512,
    truncation=True,
    set_pad_id=False
):
    """
    Load the preprocessed HF dataset with train, valid and test objects
    
    Paramters:
    ---------
    data_path: str 
        Path to the pre-processed HuggingFace dataset 
    model_checkpoints: 
        Name of the pre-trained model to use for tokenization
    """
    # data = load_from_disk(data_path)
    dataset = load_dataset("mehdiiraqui/twitter_disaster")
    data = dataset['train'].train_test_split(train_size=0.8, seed=42)
    data['val'] = data.pop("test")
    data['test'] = dataset['test']

    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoints,
        add_prefix_space=add_prefix_space
    )

    if set_pad_id:
        tokenizer.pad_token = tokenizer.eos_token

    def _preprocesscing_function(examples):
        return tokenizer(examples['text'], truncation=truncation, max_length=max_length)

    col_to_delete = ['id', 'keyword','location', 'text']
    tokenized_datasets = data.map(_preprocesscing_function, batched=False)
    tokenized_datasets = tokenized_datasets.remove_columns(col_to_delete)
    tokenized_datasets = tokenized_datasets.rename_column("target", "label")
    tokenized_datasets.set_format("torch")

    padding_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_datasets, padding_collator

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)

    return np.sum(rr_score) / np.sum(y_true)

def compute_metrics(evalpred: EvalPrediction):
    '''
    This function is for link prediction in batch evaluation.
    '''
    scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]

    if labels.dtype != int:
        predictions = expit(scores).round()
        return {
            "prc": precision_score(predictions, labels),
            "acc": accuracy_score(predictions, labels),
            "auc_roc": roc_auc_score(predictions, labels) if np.unique(labels).size >1 else None,
        }
    else:
        predictions = np.argmax(scores, -1)
        prc = (np.sum((predictions == labels)) / labels.shape[0])

        n_labels = np.max(labels) + (labels[1] - labels[0])
        labels = np.eye(n_labels)[labels]

        # auc_all = [roc_auc_score(labels[i], scores[i]) for i in tqdm(range(labels.shape[0]))]
        # auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_10_all = [ndcg_score(labels[i], scores[i], 10) for i in range(labels.shape[0])]
        ndcg_10 = np.mean(ndcg_10_all)
        ndcg_100_all = [ndcg_score(labels[i], scores[i], 100) for i in range(labels.shape[0])]
        ndcg_100 = np.mean(ndcg_100_all)

        return {
            "prc": prc,
            "mrr": mrr,
            "ndcg_10": ndcg_10,
            "ndcg_100": ndcg_100,
        }
    
def compute_metrics_cls(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric= evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}


def calculate_ncc_metrics(evalpred: EvalPrediction):
    '''
    This function is for coarse-grained classification evaluation.
    '''

    scores, labels = evalpred.predictions[-2], evalpred.predictions[-1]
    preds = np.argmax(scores, 1)

    recall_macro = recall_score(labels, preds, average='macro')
    precision_macro = precision_score(labels, preds, average='macro')
    F1_macro = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)

    return {
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "F1_macro": F1_macro,
        "accuracy": accuracy,
    }


def get_args():
    parser = ArgumentParser(description="Fine-tune pretrained LLMs with PEFT")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface pre-processed dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to store the fine-tuned model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Name of the pre-trained LLM to fine-tune",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the input sequences",
    )
    parser.add_argument(
        "--set_pad_id", 
        action="store_true",
        help="Set the id for the padding token, needed by models such as Mistral-7B",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for training"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Eval batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="Lora rank"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Lora alpha"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.2, help="Lora dropout"
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default='none',
        choices={"lora_only", "none", 'all'},
        help="Layers to add learnable bias"
    )

    arguments = parser.parse_args()
    return arguments