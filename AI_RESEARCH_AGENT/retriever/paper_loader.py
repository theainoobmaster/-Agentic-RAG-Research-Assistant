# # # # # import requests
# # # # # import tempfile
# # # # # from pypdf import PdfReader
# # # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # # from langchain_core.documents import Document


# # # # # def load_paper(paper_url: str):
# # # # #     response = requests.get(paper_url)
# # # # #     response.raise_for_status()

# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
# # # # #         f.write(response.content)
# # # # #         pdf_path = f.name

# # # # #     reader = PdfReader(pdf_path)
# # # # #     full_text = ""

# # # # #     for page in reader.pages:
# # # # #         text = page.extract_text()
# # # # #         if text:
# # # # #             full_text += text + "\n"

# # # # #     splitter = RecursiveCharacterTextSplitter(
# # # # #         chunk_size=1000,
# # # # #         chunk_overlap=200,
# # # # #     )

# # # # #     chunks = splitter.split_text(full_text)

# # # # #     return [Document(page_content=c) for c in chunks]
# # # # import requests, tempfile
# # # # from pypdf import PdfReader
# # # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # # from langchain_core.documents import Document

# # # # def load_paper(paper_url: str):
# # # #     pdf_bytes = requests.get(paper_url).content

# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
# # # #         f.write(pdf_bytes)
# # # #         path = f.name

# # # #     reader = PdfReader(path)
# # # #     text = "".join(page.extract_text() or "" for page in reader.pages)

# # # #     splitter = RecursiveCharacterTextSplitter(
# # # #         chunk_size=1200,
# # # #         chunk_overlap=300
# # # #     )

# # # #     return [Document(page_content=c) for c in splitter.split_text(text)]
# # # import torch
# # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # from peft import PeftModel

# # # class InferenceModel:
# # #     def __init__(self, base_model_id: str, lora_path: str | None = None):
# # #         device = "cuda" if torch.cuda.is_available() else "cpu"

# # #         self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
# # #         base_model = AutoModelForCausalLM.from_pretrained(
# # #             base_model_id,
# # #             torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# # #             device_map="auto" if device == "cuda" else None,
# # #         )

# # #         if lora_path:
# # #             self.model = PeftModel.from_pretrained(
# # #                 base_model,
# # #                 lora_path,
# # #                 is_trainable=False
# # #             )
# # #         else:
# # #             self.model = base_model

# # #         self.model.eval()

# # #     def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
# # #         inputs = self.tokenizer(
# # #             prompt,
# # #             return_tensors="pt",
# # #             truncation=True,
# # #             max_length=2048
# # #         ).to(self.model.device)

# # #         with torch.no_grad():
# # #             outputs = self.model.generate(
# # #                 **inputs,
# # #                 max_new_tokens=max_new_tokens,
# # #                 temperature=0.2,
# # #                 top_p=0.9,
# # #                 do_sample=True
# # #             )

# # #         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
# # # import requests
# # # import tempfile
# # # from pypdf import PdfReader

# # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # from langchain_core.documents import Document


# # # def load_paper(paper_url: str):
# # #     """
# # #     Downloads a PDF from a URL, extracts text, splits into chunks,
# # #     and returns a list of LangChain Document objects.
# # #     """

# # #     response = requests.get(paper_url)
# # #     response.raise_for_status()

# # #     # Save PDF temporarily
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
# # #         f.write(response.content)
# # #         pdf_path = f.name

# # #     reader = PdfReader(pdf_path)

# # #     full_text = ""
# # #     for page in reader.pages:
# # #         page_text = page.extract_text()
# # #         if page_text:
# # #             full_text += page_text + "\n"

# # #     splitter = RecursiveCharacterTextSplitter(
# # #         chunk_size=1200,
# # #         chunk_overlap=300
# # #     )

# # #     chunks = splitter.split_text(full_text)

# # #     documents = [
# # #         Document(page_content=chunk)
# # #         for chunk in chunks
# # #         if chunk.strip()
# # #     ]

# # #     return documents
# # import requests
# # import tempfile
# # import os
# # from urllib.parse import urlparse
# # from langchain_community.document_loaders import WebBaseLoader
# # from pypdf import PdfReader


# # def load_paper(url: str):
# #     """
# #     Load paper from arXiv abstract, arXiv PDF, or direct PDF URL.
# #     Returns list of LangChain Documents.
# #     """

# #     # 🔁 Convert arXiv abstract → PDF
# #     if "arxiv.org/abs/" in url:
# #         paper_id = url.split("arxiv.org/abs/")[-1]
# #         url = f"https://arxiv.org/pdf/{paper_id}.pdf"

# #     # 📄 PDF path
# #     if url.endswith(".pdf"):
# #         response = requests.get(url, timeout=30)
# #         response.raise_for_status()

# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #             tmp.write(response.content)
# #             pdf_path = tmp.name

# #         reader = PdfReader(pdf_path)
# #         text = ""

# #         for page in reader.pages:
# #             text += page.extract_text() + "\n"

# #         os.remove(pdf_path)

# #         return [{
# #             "page_content": text,
# #             "metadata": {"source": url}
# #         }]

# #     # 🌐 Fallback: HTML page
# #     loader = WebBaseLoader(url)
# #     return loader.load()
# import requests, tempfile, os
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# def load_paper(url: str) -> list[Document]:

#     # Convert arXiv abstract → PDF
#     if "arxiv.org/abs/" in url:
#         paper_id = url.split("arxiv.org/abs/")[-1]
#         url = f"https://arxiv.org/pdf/{paper_id}.pdf"

#     response = requests.get(url, timeout=30)
#     response.raise_for_status()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#         f.write(response.content)
#         pdf_path = f.name

#     reader = PdfReader(pdf_path)
#     full_text = "".join(page.extract_text() or "" for page in reader.pages)

#     os.remove(pdf_path)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=300
#     )

#     chunks = splitter.split_text(full_text)

#     return [
#         Document(
#             page_content=chunk,
#             metadata={"source": url}
#         )
#         for chunk in chunks if chunk.strip()
#     ]
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# import requests, tempfile, os

# def load_paper(url: str):
#     response = requests.get(url, timeout=30)
#     response.raise_for_status()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#         f.write(response.content)
#         path = f.name

#     reader = PdfReader(path)
#     text = "\n".join(
#         page.extract_text() or ""
#         for page in reader.pages[:3]   # 🔑 ONLY FIRST 3 PAGES
#     )

#     os.remove(path)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=150
#     )

#     chunks = splitter.split_text(text)

#     return [
#         Document(page_content=c, metadata={"source": url})
#         for c in chunks if c.strip()
#     ]
import requests
import tempfile
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from urllib.parse import urlparse

def load_paper(url: str):
    """
    Robust paper loader:
    - Handles arXiv abstract links
    - Handles direct PDF links
    - Rejects non-PDF content safely
    """

    # 🔁 Convert arXiv abstract → PDF
    if "arxiv.org/abs/" in url:
        paper_id = url.split("arxiv.org/abs/")[-1]
        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

    # 📥 Download
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # ❌ Safety check: ensure it's actually a PDF
    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower():
        raise ValueError(
            f"URL did not return a PDF. Content-Type: {content_type}"
        )

    # 💾 Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(response.content)
        path = f.name

    try:
        reader = PdfReader(path)
    except Exception as e:
        os.remove(path)
        raise RuntimeError("Failed to read PDF file") from e

    # 📄 Extract ONLY first pages (important)
    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages[:3]
    )

    os.remove(path)

    # ✂️ Chunk cleanly
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    return [
        Document(page_content=c, metadata={"source": url})
        for c in chunks if c.strip()
    ]