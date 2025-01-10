
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings


class Retriever:
    def __init__(self, mode, embed_model_name, db = Chroma, top_k = 5):
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.db = db
        self.top_k = top_k

        if self.mode == 'bm25':
            self.embedder = None
        elif 'gecko' in self.embed_model_name: # VertexAI
            self.embedder = VertexAIEmbeddings(model_name=self.embed_model_name)
        else:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)

    def build_schema_corpus(self, df):
        docs = []
        for col_name in df.columns:
            result_text = f'{{"column_name": "{col_name}", "dtype": "{df[col_name].dtype}"}}'
            docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        return docs
    
    def get_retriever(self, df):
        docs = None
        if self.mode == 'embed' or self.mode == 'hybrid':
            docs = self.build_schema_corpus(df)
            db = self.db.from_documents(docs, self.embedder)
            embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        if self.mode == 'bm25' or self.mode == 'hybrid':
            if docs is None:
                docs = self.build_schema_corpus(df)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = self.top_k
        if self.mode == 'hybrid':
            # return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.9, 0.1])
            return EnsembleRetriever(retrievers=[embed_retriever, bm25_retriever], weights=[0.5, 0.5])
        elif self.mode == 'embed':
            return embed_retriever
        elif self.mode == 'bm25':
            return bm25_retriever

    def retrieve_schema(self, query, df):
        results = self.get_retriever(df).invoke(query)
        observations = [doc.metadata['result_text'] for doc in results if 'result_text' in doc.metadata]
        return observations



