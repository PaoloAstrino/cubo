from ruff.cli import run

print("running ruff check tests --fix --unsafe-fixes")
run(["check", "tests", "--fix", "--unsafe-fixes", "--quiet"])
