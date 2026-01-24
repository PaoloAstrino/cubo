import inspect

import isort.api as api

print("sort_file signature:", inspect.signature(api.sort_file))
print(api.sort_file.__doc__)
