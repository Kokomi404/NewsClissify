import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import os


wandb.login(key="a4d6cd1f4f320c572527e5b2026ea3c0f5d5254d")
wandb.init(project="bert-base-chinese-training")

data_files = {"train": "train.csv", "test": "test.csv"}
raw_datasets = load_dataset("csv", data_files=data_files)

label2id = {
    "体育": 0,
    "财经": 1,
    "房产": 2,
    "家居": 3,
    "教育": 4,
    "科技": 5,
    "时尚": 6,
    "时政": 7,
    "游戏": 8,
    "娱乐": 9,
}

def process_example(example):
    example["label"] = label2id[example["label"]]
    return example


raw_datasets = raw_datasets.map(process_example)


model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters:", total_params)


training_args = TrainingArguments(
    output_dir="./results",              
    num_train_epochs=5,               
    per_device_train_batch_size=64,      
    per_device_eval_batch_size=64,     
    evaluation_strategy="epoch",         
    save_strategy="epoch",               
    logging_dir="./logs",                
    logging_steps=10,                    
    report_to="wandb",                  
    load_best_model_at_end=False,        
)

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(os.path.join(training_args.output_dir, "final_model"))