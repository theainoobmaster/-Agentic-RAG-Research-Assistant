# import os
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document  # ✅ FIXED IMPORT

# VECTORSTORE_PATH = "data/vectorstore"

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# def load_or_create_store():
#     if os.path.exists(VECTORSTORE_PATH):
#         return FAISS.load_local(
#             VECTORSTORE_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#     return FAISS.from_texts([], embeddings)


# def ingest_documents(docs):
#     """
#     Normalize incoming docs to LangChain Document objects
#     """

#     normalized_docs = []

#     for d in docs:
#         if isinstance(d, Document):
#             normalized_docs.append(d)

#         elif isinstance(d, dict):
#             normalized_docs.append(
#                 Document(
#                     page_content=d.get("page_content")
#                     or d.get("content")
#                     or "",
#                     metadata=d.get("metadata", {})
#                 )
#             )
#         else:
#             raise TypeError(f"Unsupported document type: {type(d)}")

#     store = load_or_create_store()
#     store.add_documents(normalized_docs)
#     store.save_local(VECTORSTORE_PATH)
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

VECTORSTORE_PATH = "data/vectorstore"

# Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# def load_or_create_store():
#     """
#     Load FAISS store if it exists.
#     DO NOT create an empty FAISS index.
#     """
#     if os.path.exists(VECTORSTORE_PATH):
#         return FAISS.load_local(
#             VECTORSTORE_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#     return None


# def ingest_documents(docs: list):
#     """
#     Safely ingest documents into FAISS.
#     Handles:
#     - First-time creation
#     - Subsequent updates
#     - Dict → Document normalization
#     """

#     if not docs:
#         # Nothing to ingest
#         return

#     normalized_docs: list[Document] = []

#     for d in docs:
#         if isinstance(d, Document):
#             normalized_docs.append(d)

#         elif isinstance(d, dict):
#             normalized_docs.append(
#                 Document(
#                     page_content=d.get("page_content")
#                     or d.get("content")
#                     or "",
#                     metadata=d.get("metadata", {})
#                 )
#             )
#         else:
#             raise TypeError(f"Unsupported document type: {type(d)}")

#     # Remove empty docs
#     normalized_docs = [
#         d for d in normalized_docs if d.page_content.strip()
#     ]

#     if not normalized_docs:
#         return

#     store = load_or_create_store()

#     # 🔑 FIRST TIME → CREATE
#     if store is None:
#         store = FAISS.from_documents(normalized_docs, embeddings)

#     # 🔁 SUBSEQUENT → ADD
#     else:
#         store.add_documents(normalized_docs)

#     store.save_local(VECTORSTORE_PATH)
import os
def load_or_create_store():
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def ingest_documents(docs):
    if not docs:
        raise ValueError("No documents to ingest")

    store = load_or_create_store()

    if store is None:
        store = FAISS.from_documents(docs, embeddings)
    else:
        store.add_documents(docs)

    store.save_local(VECTORSTORE_PATH)