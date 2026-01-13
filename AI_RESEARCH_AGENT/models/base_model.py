# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import os

# class BaseLLM:
#     """
#     Loads a pretrained Hugging Face LLM.
#     Ready for LoRA fine-tuning.
#     """
#     def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
#         self.model_name = model_name
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#         # Load model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None
#         )
#         self.model.to(self.device)

#     def generate(self, prompt: str, max_tokens: int = 100):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class BaseLLM:
    """
    Loads a pretrained Hugging Face LLM.
    Ready for LoRA fine-tuning.
    """
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.to(self.device)

    def generate(self, prompt: str, max_tokens: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)