from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.base_language import BaseLanguageModel
from typing import Optional, List
import pandas as pd



class DataFrameQueryAgent:
    """DataFrame query agent class, supporting custom models and data"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        llm: Optional[BaseLanguageModel]
    ):
        """
        Initialize query agent
        
        Args:
            df: DataFrame to query
            llm: Language model instance to use
        """
        
        self.agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            agent_type="zero-shot-react-description",  
            max_iterations=20,
            allow_dangerous_code=True
        )
    
    def query(self, question: str) -> str:
        """
        Query the DataFrame
        
        Args:
            question: Natural language query question
            
        Returns:
            str: Query result
        """
        try:
            response = self.agent.invoke(question)
            return response
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def batch_query(self, questions: List[str]) -> List[str]:
        """
        Batch query multiple questions
        
        Args:
            questions: List of questions to query
            
        Returns:
            List[str]: List of corresponding answers
        """
        return [self.query(q) for q in questions]