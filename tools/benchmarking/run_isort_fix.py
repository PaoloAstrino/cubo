from isort import SortImports

paths = [
    "cubo/utils/utils.py",
    "cubo/utils/cpu_tuner.py",
    "cubo/embeddings/model_loader.py",
    "cubo/server/api.py",
    "cubo/ingestion/hierarchical_chunker.py",
    "cubo/ingestion/deep_ingestor.py",
    "cubo/retrieval/retriever.py",
]
for p in paths:
    try:
        si = SortImports(p, settings_path=r"C:\Users\paolo\Desktop\cubo\packaging\pyproject.toml")
        si.sort_file()
        print("OK", p)
    except Exception as e:
        print("ERR", p, e)
