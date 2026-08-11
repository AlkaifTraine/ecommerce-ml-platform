"""Verify every required dependency imports, and report versions."""

import importlib

MODULES = [
    "duckdb", "polars", "pyarrow", "pandas", "numpy",
    "lightgbm", "sklearn", "scipy", "mlflow",
    "fastapi", "uvicorn", "pydantic", "pydantic_settings",
    "psycopg", "sqlalchemy", "redis", "kaggle", "matplotlib", "pytest",
]

missing = []
for name in MODULES:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"{name:<20} OK       {version}")
    except Exception as exc:
        missing.append(name)
        print(f"{name:<20} MISSING  ({type(exc).__name__}: {exc})")

print()
if missing:
    print(f"MISSING {len(missing)}: {', '.join(missing)}")
    raise SystemExit(1)
print("all dependencies present")
