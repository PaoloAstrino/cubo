from typing import Dict
from .models.Collection import Collection


class PersistentClient:
    def __init__(self, path: str = None):
        self._collections: Dict[str, Collection] = {}
        self.path = path

    def get_or_create_collection(self, name: str) -> Collection:
        if name not in self._collections:
            self._collections[name] = Collection(name)
        return self._collections[name]
