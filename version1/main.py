import time
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
            
            
class ProductAnalyzer:
    def __init__(self, 
                 csv_path="product_data.csv", 
                 model_name="llama3.1", 
                 embedding_model="bge-m3", 
                 base_url="http://localhost:11434"
                 ):
        """Initialize the product analyzer"""
        # Add caching mechanism
        self.query_cache = {}   
        self.df = pd.read_csv(csv_path)        
        self.llm = OllamaLLM(
            model=model_name, 
            temperature=0.75
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=base_url
        )
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all necessary components"""
        # Create text descriptions and split
        texts = self._create_detailed_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=100, 
        )
        splits = text_splitter.create_documents(texts)
        
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.qa_chain = self._setup_qa_chain()
        self.agent = self._create_df_agent()
    
    def _create_detailed_text(self):
        """Create detailed text descriptions"""
        texts = []
        for index, row in self.df.iterrows():
            text = f"Product Details - "
            for col in self.df.columns:
                text += f"{col}: {row[col]}. "
            texts.append(text)
        return texts
       
    def _setup_qa_chain(self):
        """Set up the QA chain"""
        template = """
        Answer the question based on the following information:

        Context Information:
        {context}

        Question: {question}

        If there is no relevant information in the data, please clearly state "No information about this product in the database".
        Please answer based on actual data, do not guess or make up information.

        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def _create_df_agent(self):
        """Create DataFrame agent"""
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=False,
            agent_type="zero-shot-react-description",
            max_iterations=20,
            allow_dangerous_code=True
        )
    
    def query_data(self, query_text):
        """
        Query the data with caching
        Args:
            query_text: User's query question
        Returns:
            Dictionary containing answer and context or error information
        """
        # Add query caching
        if query_text in self.query_cache:
            return self.query_cache[query_text]

        try:
            context = self.qa_chain.invoke({"query": query_text})
            
            enhanced_prompt = f"""
            Answer the question based on the retrieved relevant information: {context}
            Provide a concise answer using pandas methods if needed.
            """
            
            response = self.agent.invoke(enhanced_prompt)
            
            result = {
                "answer": response["output"] if isinstance(response, dict) else response,
                "context": context['source_documents']
            }
            
            # Store in cache
            self.query_cache[query_text] = result
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "suggestion": "Please try a more specific question"
            }

    def display_query_result(self, query_text):
        """
        Display query results
        Args:
            query_text: User's query question
        """
        print("\n" + "="*50)
        print(f"📝 Query: {query_text}")
        print("="*50)
        
        result = self.query_data(query_text)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            print(f"💡 Suggestion: {result['suggestion']}")
        else:
            print("\n📊 Answer:")
            print("-"*30)
            print(f"{result['answer']}")
            print("\n📚 Related Context:")
            print("-"*30)
            for i, doc in enumerate(result['context'], 1):
                print(f"{i}. {doc.page_content}\n")
        print("="*50 + "\n")

    def display_batch_query_result(self, queries, delay=0.1):
        """
        Perform batch queries
        Args:
            queries: List of query strings
            delay: Time delay between queries in seconds
        """
        for query in queries:
            self.display_query_result(query)
            time.sleep(delay)