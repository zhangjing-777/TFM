import pandas as pd
from pathlib import Path



def load_amazon_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load and merge Amazon products and categories data
    
    Args:
        data_dir (str, optional): Data directory path. If None, uses current directory
        
    Returns:
        pd.DataFrame: Merged product data with categories
    """
    # Set data directory
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent
    
    # Load and merge data
    df1 = pd.read_csv(data_dir / "amazon_products.csv")
    df2 = pd.read_csv(data_dir / "amazon_categories.csv")
    df = df1.merge(df2, left_on="category_id", right_on="id", how="left")
    df.drop(columns=["id"], inplace=True)
    
    return df




# Example usage
if __name__ == "__main__":
    df = load_amazon_data()
    print(f"Loaded {len(df)} products")