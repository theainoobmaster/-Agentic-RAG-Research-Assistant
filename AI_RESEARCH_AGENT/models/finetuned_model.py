# import torch
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from peft import LoraConfig, get_peft_model, TaskType
# from tqdm import tqdm

# class LoRALLM:
#     """
#     Adds LoRA adapters to a base LLM for efficient fine-tuning.
#     Includes train() and evaluate() methods.
#     """
#     def __init__(self, base_model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
#         self.model = base_model.model
#         self.tokenizer = base_model.tokenizer
#         self.device = base_model.device

#         # LoRA configuration
#         self.lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=["q_proj", "v_proj"],  # target attention layers
#             lora_dropout=lora_dropout,
#             bias="none"
#         )

#         # Apply LoRA
#         self.model = get_peft_model(self.model, self.lora_config)
#         self.model.to(self.device)

#     # -------------------------
#     # Save / Load
#     # -------------------------
#     def save(self, path: str):
#         """Save LoRA adapters"""
#         self.model.save_pretrained(path)
#         print(f"✅ LoRA adapters saved to {path}")

#     def load(self, path: str):
#         """Load LoRA adapters"""
#         self.model.load_state_dict(torch.load(path, map_location=self.device))
#         print(f"✅ LoRA adapters loaded from {path}")

#     # -------------------------
#     # Training
#     # -------------------------
#     def train(self, dataset, batch_size=4, epochs=1, lr=1e-4, max_grad_norm=1.0):
#         """
#         Fine-tune LoRA adapters on a given dataset.
#         dataset: torch.utils.data.Dataset returning {'input_ids', 'attention_mask', 'labels'}
#         """
#         self.model.train()
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         optimizer = AdamW(self.model.parameters(), lr=lr)

#         for epoch in range(epochs):
#             loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
#             for batch in loop:
#                 input_ids = batch["input_ids"].to(self.device)
#                 attention_mask = batch["attention_mask"].to(self.device)
#                 labels = batch["labels"].to(self.device)

#                 outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )
#                 loss = outputs.loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
#                 optimizer.step()

#                 loop.set_postfix(loss=loss.item())

#         print("✅ LoRA fine-tuning complete!")

#     # -------------------------
#     # Evaluation
#     # -------------------------
#     # def evaluate(self, dataset, batch_size=4):
#     #     """
#     #     Evaluate the fine-tuned model on a dataset.
#     #     """
#     #     self.model.eval()
#     #     dataloader = DataLoader(dataset, batch_size=batch_size)
#     #     total_loss = 0
#     #     with torch.no_grad():
#     #         for batch in dataloader:
#     #             input_ids = batch["input_ids"].to(self.device)
#     #             attention_mask = batch["attention_mask"].to(self.device)
#     #             labels = batch["labels"].to(self.device)

#     #             outputs = self.model(
#     #                 input_ids=input_ids,
#     #                 attention_mask=attention_mask,
#     #                 labels=labels
#     #             )
#     #             total_loss += outputs.loss.item()

#     #     avg_loss = total_loss / len(dataloader)
#     #     print(f"✅ Evaluation complete — Average Loss: {avg_loss:.4f}")
#     #     return avg_loss
#     from torch.utils.data import DataLoader
#     import torch

#     def evaluate(self, dataset, batch_size=4):
#         if len(dataset) == 0:
#             print("⚠️ Validation dataset is empty. Skipping evaluation.")
#             return None

#         self.model.eval()
#         dataloader = DataLoader(dataset, batch_size=batch_size)
#         total_loss = 0

#         with torch.no_grad():
#             for batch in dataloader:
#                 input_ids = batch["input_ids"].to(self.device)
#                 attention_mask = batch["attention_mask"].to(self.device)
#                 labels = batch["labels"].to(self.device)

#                 outputs = self.model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     labels=labels
#                 )
#                 total_loss += outputs.loss.item()

#         avg_loss = total_loss / len(dataloader)
#         print(f"✅ Evaluation complete — Average Loss: {avg_loss:.4f}")
#         return avg_loss
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

class LoRALLM:
    """
    Adds LoRA adapters to a base LLM for efficient fine-tuning.
    Includes train() and evaluate() methods.
    """
    def __init__(self, base_model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.model = base_model.model
        self.tokenizer = base_model.tokenizer
        self.device = base_model.device

        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # target attention layers
            lora_dropout=lora_dropout,
            bias="none"
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.to(self.device)

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, path: str):
        """Save LoRA adapters"""
        self.model.save_pretrained(path)
        print(f"✅ LoRA adapters saved to {path}")

    def load(self, path: str):
        """Load LoRA adapters"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ LoRA adapters loaded from {path}")

    # -------------------------
    # Training
    # -------------------------
    def train(self, dataset, batch_size=4, epochs=1, lr=1e-4, max_grad_norm=1.0):
        """
        Fine-tune LoRA adapters on a given dataset.
        dataset: torch.utils.data.Dataset returning {'input_ids', 'attention_mask', 'labels'}
        """
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

                loop.set_postfix(loss=loss.item())

        print("✅ LoRA fine-tuning complete!")

    # -------------------------
    # Evaluation
    # -------------------------
    # def evaluate(self, dataset, batch_size=4):
    #     """
    #     Evaluate the fine-tuned model on a dataset.
    #     """
    #     self.model.eval()
    #     dataloader = DataLoader(dataset, batch_size=batch_size)
    #     total_loss = 0
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             input_ids = batch["input_ids"].to(self.device)
    #             attention_mask = batch["attention_mask"].to(self.device)
    #             labels = batch["labels"].to(self.device)

    #             outputs = self.model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 labels=labels
    #             )
    #             total_loss += outputs.loss.item()

    #     avg_loss = total_loss / len(dataloader)
    #     print(f"✅ Evaluation complete — Average Loss: {avg_loss:.4f}")
    #     return avg_loss
    from torch.utils.data import DataLoader
    import torch

    def evaluate(self, dataset, batch_size=4):
        if len(dataset) == 0:
            print("⚠️ Validation dataset is empty. Skipping evaluation.")
            return None

        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Evaluation complete — Average Loss: {avg_loss:.4f}")
        return avg_loss