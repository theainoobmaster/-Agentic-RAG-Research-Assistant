# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel


# class InferenceModel:
#     def __init__(
#         self,
#         base_model_id: str,
#         lora_path: str,
#         device: str | None = None,
#         torch_dtype: torch.dtype = torch.float16,
#     ):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         # Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             base_model_id,
#             use_fast=True
#         )

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         # Base model
#         base_model = AutoModelForCausalLM.from_pretrained(
#             base_model_id,
#             torch_dtype=torch_dtype,
#             device_map="auto" if self.device == "cuda" else None
#         )

#         # Attach LoRA
#         self.model = PeftModel.from_pretrained(
#             base_model,
#             lora_path
#         )

#         self.model.eval()

#         if self.device == "cpu":
#             self.model.to(self.device)

#     @torch.inference_mode()
#     def generate(
#         self,
#         prompt: str,
#         max_new_tokens: int = 256,
#         temperature: float = 0.7,
#         top_p: float = 0.9,
#     ) -> str:
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             padding=True
#         ).to(self.device)

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#             eos_token_id=self.tokenizer.eos_token_id,
#         )

#         return self.tokenizer.decode(
#             outputs[0],
#             skip_special_tokens=True
#         )
# # # import requests
# # # import tempfile
# # # from pypdf import PdfReader
# # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # from langchain_core.documents import Document


# # # def load_paper(paper_url: str):
# # #     response = requests.get(paper_url)
# # #     response.raise_for_status()

# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
# # #         f.write(response.content)
# # #         pdf_path = f.name

# # #     reader = PdfReader(pdf_path)
# # #     full_text = ""

# # #     for page in reader.pages:
# # #         text = page.extract_text()
# # #         if text:
# # #             full_text += text + "\n"

# # #     splitter = RecursiveCharacterTextSplitter(
# # #         chunk_size=1000,
# # #         chunk_overlap=200,
# # #     )

# # #     chunks = splitter.split_text(full_text)

# # #     return [Document(page_content=c) for c in chunks]
import requests, tempfile
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_paper(paper_url: str):
    pdf_bytes = requests.get(paper_url).content

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        path = f.name

    reader = PdfReader(path)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300
    )

    return [Document(page_content=c) for c in splitter.split_text(text)]
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class InferenceModel:
    def __init__(self, base_model_id: str, lora_path: str | None = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if lora_path:
            self.model = PeftModel.from_pretrained(
                base_model,
                lora_path,
                is_trainable=False
            )
        else:
            self.model = base_model

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True
            )

        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()   