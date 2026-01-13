import os

from isort import api
from isort.settings import Config

root = r"C:\Users\paolo\Desktop\cubo\cubo"
cfg = Config(settings_path=r"C:\Users\paolo\Desktop\cubo\packaging\pyproject.toml")
errors = []
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith(".py"):
            path = os.path.join(dirpath, fn)
            ok = api.check_file(path, config=cfg)
            if not ok:
                errors.append(path)
                print("NOT OK", path)
print("Total not ok:", len(errors))
if len(errors) > 0:
    raise SystemExit(1)
else:
    print("All files OK")
