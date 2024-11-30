"""
Version 2 of the Amazon Products RAG System
"""
from .raw_data import load_amazon_data
from .interrogate_4agent import DataFrameQueryAgent


__all__ = [
    'load_amazon_data',
    'DataFrameQueryAgent'
]
