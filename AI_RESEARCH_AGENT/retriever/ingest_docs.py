# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# import pickle

# DATA_DIR = Path("data/processed")
# INDEX_DIR = Path("data/vectorstore")
# INDEX_DIR.mkdir(parents=True, exist_ok=True)


# def main():
#     documents = []

#     for file in DATA_DIR.glob("*.txt"):
#         text = file.read_text(encoding="utf-8")
#         documents.append(Document(page_content=text))

#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     vectorstore = FAISS.from_documents(documents, embeddings)

#     # Save FAISS index
#     vectorstore.save_local(str(INDEX_DIR))

#     print(f"Indexed {len(documents)} documents")


# if __name__ == "__main__":
#     main()
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_DIR = Path("data/processed")
INDEX_DIR = Path("data/vectorstore")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def main():
    documents = []

    for file in DATA_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        documents.append(Document(page_content=text))

    if not documents:
        raise RuntimeError(
            "No documents found in data/processed/. "
            "Add .txt files before ingestion."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(str(INDEX_DIR))

    print(f"Indexed {len(documents)} documents into data/vectorstore")


if __name__ == "__main__":
    main()
