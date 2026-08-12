"""Build-time smoke test for the Airflow image.

pip reports a dependency conflict as a WARNING and still exits 0, so an image
with an incompatible numpy/pandas pair builds "successfully" and only fails
later inside the scheduler, with a binary-incompatibility error that looks
nothing like its cause. This runs as a build layer so the build fails instead.

It happened twice here: scikit-learn pulled numpy 2.5, and then - even with
numpy pinned in an earlier layer - scipy pulled it back, because pip resolves
each RUN independently.
"""

import sys

import duckdb
import lightgbm
import numpy
import pandas
import polars
import pyarrow
import scipy
import sklearn

for name, mod in [
    ("numpy", numpy), ("pandas", pandas), ("scipy", scipy),
    ("scikit-learn", sklearn), ("lightgbm", lightgbm),
    ("duckdb", duckdb), ("polars", polars), ("pyarrow", pyarrow),
]:
    print(f"  {name:<14} {mod.__version__}")

major = int(numpy.__version__.split(".")[0])
if major >= 2:
    sys.exit(
        f"FAIL: numpy {numpy.__version__} is incompatible with the "
        f"pandas {pandas.__version__} that Airflow ships (needs numpy<2)"
    )

# Exercise the actual incompatibility rather than trusting version strings:
# the numpy-2-against-pandas-2.1 failure shows up as a dtype size error here.
df = pandas.DataFrame({"a": numpy.arange(5), "b": numpy.linspace(0, 1, 5)})
assert int(df["a"].sum()) == 10, "pandas/numpy interop is broken"

# LightGBM must be able to consume a numpy matrix, which is how the model is
# both trained and served.
booster = lightgbm.train(
    {"objective": "binary", "verbose": -1, "num_leaves": 2},
    lightgbm.Dataset(numpy.random.rand(50, 3), label=numpy.random.randint(0, 2, 50)),
    num_boost_round=1,
)
assert len(booster.predict(numpy.random.rand(2, 3))) == 2

print("OK: numpy/pandas/scipy/sklearn/lightgbm are mutually compatible")
