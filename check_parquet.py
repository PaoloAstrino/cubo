import pandas as pd
from pathlib import Path

parquet_file = Path(r"c:\Users\paolo\Desktop\cubo\storage\deep\chunks_deep.parquet")
if parquet_file.exists():
    df = pd.read_parquet(parquet_file)
    print(f"Total chunks in parquet: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\nUnique files: {df['filename'].unique() if 'filename' in df.columns else 'N/A'}")
else:
    print("Parquet file not found")
