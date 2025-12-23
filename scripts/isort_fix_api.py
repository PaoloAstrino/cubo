from isort import api
from isort.settings import Config

paths = [
    r"C:\Users\paolo\Desktop\cubo\cubo\utils\utils.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\utils\cpu_tuner.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\embeddings\model_loader.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\server\api.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\ingestion\hierarchical_chunker.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\ingestion\deep_ingestor.py",
    r"C:\Users\paolo\Desktop\cubo\cubo\retrieval\retriever.py",
]
cfg = Config(settings_path=r"C:\Users\paolo\Desktop\cubo\pyproject.toml")
for p in paths:
    try:
        changed = api.sort_file(p, config=cfg)
        print(("CHANGED" if changed else "OK"), p)
    except Exception as e:
        print("ERR", p, e)
