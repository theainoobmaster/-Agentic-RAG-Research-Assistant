# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from pathlib import Path


# class LangChainRetriever:
#     def __init__(self, index_path: str):
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )

#         self.vectorstore = FAISS.load_local(
#             index_path,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

#     def retrieve(self, query: str, k: int = 5) -> list[str]:
#         docs = self.vectorstore.similarity_search(query, k=k)
#         return [doc.page_content for doc in docs]

#     # def retrieve_with_scores(self, query: str, k: int = 5):
#     #     docs_and_scores = self.vectorstore.similarity_search_with_score(
#     #         query, k=k
#     #     )

#     #     docs = [doc.page_content for doc, _ in docs_and_scores]
#     #     scores = [score for _, score in docs_and_scores]

#         return docs, scores
#     def retrieve_with_scores(self, query: str, k: int = 5):
#         docs_and_scores = self.vectorstore.similarity_search_with_score(
#             query, k=k
#         )
#         docs = [doc.page_content for doc, _ in docs_and_scores]
#         scores = [score for _, score in docs_and_scores]
#         return docs, scores
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class LangChainRetriever:
    def __init__(self, index_path: str):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search_with_score(query, k=k)