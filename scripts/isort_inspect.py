import isort

print("isort version", isort.__version__)
print("isort module attrs:")
print([n for n in dir(isort) if "sort" in n.lower()][:200])
try:
    import isort.api as api

    print("api methods:", [n for n in dir(api) if "sort" in n.lower()][:200])
except Exception as e:
    print("api import error", e)

# show available callables
print("callables in isort:", [n for n in dir(isort) if callable(getattr(isort, n))][:200])
