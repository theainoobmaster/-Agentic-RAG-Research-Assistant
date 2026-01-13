import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from models.base_model import BaseLLM
from models.finetuned_model import LoRALLM
from dotenv import load_dotenv
import os

load_dotenv()   # ✅ THIS is the missing piece

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # -----------------------------
# # 1️⃣ Custom Dataset
# # -----------------------------
class TaskDataset(Dataset):
    """
    Converts processed JSON tasks into a PyTorch Dataset for causal LM fine-tuning.
    Each item returns: input_ids, attention_mask, labels
    """
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, "r",encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        task = self.data[idx]
        prompt = f"Task: {task['task']}\nInput: {task['input']}\nOutput:"
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # For causal LM, labels = input_ids
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# # -----------------------------
# # 2️⃣ Training Function
# # -----------------------------
# def train_lora_model(
#     train_json="data/processed/train.json",
#     val_json="data/processed/val.json",
#     batch_size=4,
#     epochs=1,
#     lr=1e-4,
#     max_length=512,
#     lora_r=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     save_path="models/lora_adapter"
# ):
#     # Load base LLM
#     base_model = BaseLLM()
#     tokenizer = base_model.tokenizer

#     # Create datasets
#     train_dataset = TaskDataset(train_json, tokenizer, max_length)
#     val_dataset = TaskDataset(val_json, tokenizer, max_length)

#     # Initialize LoRA model
#     lora_model = LoRALLM(base_model, lora_r, lora_alpha, lora_dropout)

#     # Train
#     lora_model.train(train_dataset, batch_size=batch_size, epochs=epochs, lr=lr)

#     # Evaluate
#     lora_model.evaluate(val_dataset, batch_size=batch_size)

#     # Save LoRA adapters
#     os.makedirs(save_path, exist_ok=True)
#     lora_model.save(save_path)


# # -----------------------------
# # 3️⃣ Run Trainer
# # -----------------------------
# if __name__ == "__main__":
#     train_lora_model(
#         batch_size=2,  # reduce if GPU is small
#         epochs=1       # increase later
#     )
def train_lora_model(
    train_json="data/processed/research_lora.json",
    batch_size=2,
    epochs=1,
    lr=1e-4,
    max_length=512,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    save_path="models/lora_adapter"
):
    # Load base model
    base_model = BaseLLM()
    tokenizer = base_model.tokenizer

    # Dataset
    train_dataset = TaskDataset(train_json, tokenizer, max_length)

    # LoRA model
    lora_model = LoRALLM(
        base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # Train ONLY
    lora_model.train(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )

    # Save adapters
    os.makedirs(save_path, exist_ok=True)
    lora_model.save(save_path)
if __name__ == "__main__":
    train_lora_model(
        batch_size=2,  # reduce if GPU is small
        epochs=1       # increase later
    )