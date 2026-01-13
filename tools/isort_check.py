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
cfg = Config(settings_path=r"C:\Users\paolo\Desktop\cubo\packaging\pyproject.toml")
errs = []
for p in paths:
    ok = api.check_file(p, config=cfg)
    print(p, "OK" if ok else "NOT OK")
    if not ok:
        errs.append(p)
print("errors:", errs)
