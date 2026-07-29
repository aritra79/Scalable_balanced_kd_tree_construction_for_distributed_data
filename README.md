# Supplementary Code — Scalable balanced *k*-d tree construction for distributed data

This archive accompanies the manuscript submitted to the *Journal of Computational
and Graphical Statistics*. It contains two archives: `code.zip` with all Python
source modules, Jupyter notebooks, and this README; and `data.zip` with the
pre-computed benchmark results used to produce the figures in Section 4.1.

---

## Contents

```
code.zip/
├── README.md                        ← this file
├── epstatKDTree.py                  ← epstat k-d tree module (proposed method)
├── canonicalKDTree.py               ← canonical k-d tree module (baseline)
├── pKDTree.py                       ← pkd-tree module (accuracy baseline)
├── epstatKDTreeRunGuide.ipynb       ← epstat benchmark notebook
├── canonicalKDTreeRunGuide.ipynb    ← canonical benchmark notebook
├── pKDTreeRunGuide.ipynb            ← epstat vs pkd accuracy comparison notebook
├── simulationGuide.ipynb            ← synthetic data generation notebook
└── plots.ipynb                      ← figure reproduction notebook

data.zip/
├── epstatKDTree_performance.parquet ← epstat benchmark results (256 rows)
├── canKDTree_performance.parquet    ← canonical benchmark results (160 rows)
└── pKDTree_performance.parquet      ← epstat vs pkd accuracy results (128 rows)
```

---

## Requirements

### Python environment

All modules require **Python 3.9 or later** and have been tested on Python 3.12.9.

### PySpark and cluster

The three `.py` modules are PySpark modules designed to run on a distributed Spark
cluster. The benchmark notebooks were executed on an **AWS EMR** cluster with the
following configuration:

- 1 primary instance: `c8gd.xlarge`
- 10 core instances: `c8gd.48xlarge` (192 vCores, 366 GB RAM each)
- Total vCores available for PySpark tasks: 1,920

A local PySpark installation (e.g. via `pip install pyspark`) is sufficient to run
the modules on a single machine at small dataset sizes for testing purposes.

### Python package dependencies

| Package | Purpose |
|---------|---------|
| `pyspark` | Distributed computing framework |
| `numpy` | Array arithmetic, Chebyshev transforms |
| `pandas` | Result accumulation and Parquet I/O |
| `pyarrow` | Parquet file reading (required by pandas) |

Install with:

```bash
pip install pyspark numpy pandas pyarrow
```

### Reproducing the figures only

To reproduce the paper's figures from the pre-computed Parquet files in `data.zip`
without re-running the benchmark experiments, only a local Python environment is
needed — no Spark cluster is required. Install:

```bash
pip install numpy pandas pyarrow matplotlib
```

---

## Module descriptions

### `epstatKDTree.py` — Proposed method

Implements the embarrassingly parallel (*epstat*) *k*-d tree construction algorithm
introduced in the paper. Attaches the following methods to
`pyspark.sql.DataFrame`:

| Method | Signature | Description |
|--------|-----------|-------------|
| `epstatKDTree` | `(variables, J=[2,4,4], depth=None, batch_size=None, local_depth=None)` | Build an approximate *k*-d tree using SEP sufficient statistics |
| `treeLeafCounts` | `(tree)` | Return a Pandas DataFrame of per-leaf row counts |
| `treePrecision` | `(tree)` | Compute the leaf-balance precision score |
| `treeLeafCells` | `(tree)` | Return a balanced Spark DataFrame with equal rows per leaf |
| `representativeSamples` | `(variables, J, depth, batch_size, local_depth)` | Build a tree and return one representative stratified sample per leaf |
| `randomSamples` | `(variables, sample_size)` | Return a simple random sample partitioned into equal-sized groups |

**Parameters of `epstatKDTree`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variables` | `list[str]` | — | Column names used as splitting axes, cycled in order |
| `J` | `list[int]` | `[2, 4, 4]` | Factorisation vector; effective approximation order is the product of its entries |
| `depth` | `int` | `None` | Total tree depth; if `None`, defaults to `len(variables)` |
| `batch_size` | `int` | `None` | Depth levels processed per MapReduce pass; if `None`, all levels are processed in a single pass |
| `local_depth` | `int` | `None` | Alias for `batch_size` |

**Usage example:**

```python
spark.sparkContext.addPyFile("s3://your-bucket/code/epstatKDTree.py")
from epstatKDTree import *

data = spark.read.parquet("s3://your-bucket/data/normal/2^28/data.parquet/") \
           .repartition(5000)

tree      = data.epstatKDTree(['x', 'y'], J=[3, 5, 5], depth=10)
precision = data.treePrecision(tree)
print(f"Precision: {precision:.4f}")
```

---

### `canonicalKDTree.py` — Canonical baseline

Implements the canonical approximate *k*-d tree construction using PySpark's
built-in `percentile_approx` to estimate splitting medians sequentially, one tree
level at a time. Attaches the following methods to `pyspark.sql.DataFrame`:

| Method | Signature | Description |
|--------|-----------|-------------|
| `canonicalKDTree` | `(variables, depth=10, accuracy=100)` | Build the tree using `percentile_approx` |
| `treeLeafCounts` | `(tree)` | Per-leaf row counts |
| `treePrecision` | `(tree)` | Leaf-balance precision score |

**Parameters of `canonicalKDTree`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variables` | `list[str]` | — | Column names used as splitting axes |
| `depth` | `int` | `10` | Total tree depth |
| `accuracy` | `int` | `100` | Passed to `percentile_approx`; controls the Greenwald–Khanna summary size. Relative error is bounded above by `1/accuracy` |

**Usage example:**

```python
spark.sparkContext.addPyFile("s3://your-bucket/code/canonicalKDTree.py")
from canonicalKDTree import *

data = spark.read.parquet("s3://your-bucket/data/normal/2^28/data.parquet/") \
           .repartition(5000)

tree      = data.canonicalKDTree(['x', 'y'], depth=10, accuracy=1000)
precision = data.treePrecision(tree)
```

> **Note:** Canonical construction requires one full distributed pass per tree level
> and becomes computationally infeasible at dataset sizes beyond approximately
> 2^30 rows on the cluster configuration described above.

---

### `pKDTree.py` — pkd-tree baseline

Implements the sampling-based pkd-tree algorithm (Men et al., 2025) used as a
precision baseline in Section 4.1. Attaches the following methods to
`pyspark.sql.DataFrame`:

| Method | Signature | Description |
|--------|-----------|-------------|
| `pKDTree` | `(variables, depth, lamda=4, sigma=32)` | Build a pkd-tree |
| `treeLeafCounts` | `(tree)` | Per-leaf row counts |
| `treePrecision` | `(tree)` | Leaf-balance precision score |

**Parameters of `pKDTree`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variables` | `list[str]` | — | Column names used as splitting axes |
| `depth` | `int` | — | Total tree depth |
| `lamda` | `int` | `4` | Local batch depth: number of levels built per sampling round |
| `sigma` | `int` | `32` | Per-leaf sample size multiplier; each sampling round draws up to 2^lamda × sigma rows per leaf |

---

## Notebook descriptions

### `epstatKDTreeRunGuide.ipynb` — epstat benchmark

Documents the benchmark loop that produced `epstatKDTree_performance.parquet`.
For each combination of dataset size (2^26 through 2^36), parameter J, distribution,
and depth, the notebook loads data from S3, builds the tree, records wall-clock
runtime and precision, and writes the collected results to Parquet. Requires a Spark
cluster and S3 access.

### `canonicalKDTreeRunGuide.ipynb` — canonical benchmark

Documents the benchmark loop that produced `canKDTree_performance.parquet`.
Covers dataset sizes 2^26 through 2^30, four `accuracy` levels, two distributions,
and four depth values. Requires a Spark cluster and S3 access.

### `pKDTreeRunGuide.ipynb` — epstat vs pkd-tree accuracy comparison

Documents the precision comparison that produced `pKDTree_performance.parquet`.
Benchmarks both epstat and pkd-tree at a fixed dataset size of 2^28 rows, across
two fold schedules (single-pass and two-pass), four parameter settings per method,
and two distributions. Requires a Spark cluster and S3 access.

### `simulationGuide.ipynb` — synthetic data generation

Documents the PySpark pipeline used to generate the `blobs` and `normal`
two-dimensional Parquet datasets at all sizes from 2^26 to 2^36, stored as
5,000-part Parquet files on S3. Each Spark partition independently draws 2^20 rows
using a fixed Gaussian mixture structure (blobs) or a fixed bivariate normal
covariance matrix (normal). Re-running this notebook requires a Spark cluster and
S3 write access.

### `plots.ipynb` — figure reproduction

Reproduces all three figures in Section 4.1 of the manuscript from the
pre-computed Parquet files in `data.zip`. This notebook does **not** require a
Spark cluster and can be run entirely locally.

**To reproduce the figures locally:**

```bash
# 1. Place the three Parquet files from data.zip in the same directory as plots.ipynb
unzip data.zip

# 2. Install dependencies
pip install numpy pandas pyarrow matplotlib

# 3. Open and run the notebook
jupyter lab plots.ipynb
```

---

## Data archive descriptions (`data.zip`)

### `epstatKDTree_performance.parquet`

Benchmark results for the epstat algorithm. Shape: 256 rows × 7 columns.

| Column | Type | Values |
|--------|------|--------|
| `algorithm` | str | `"epstat"` |
| `distribution` | str | `"blobs"`, `"normal"` |
| `size` | str | `"2^26"` through `"2^36"` (8 sizes) |
| `parameter` | str | `"J = [2, 3, 3]"`, `"J = [2, 4, 4]"`, `"J = [3, 4, 4]"`, `"J = [3, 5, 5]"` |
| `depth` | int | 4, 6, 8, 10 |
| `precision` | float | Leaf-balance precision score |
| `runtime` | float | Wall-clock build time (seconds) |

### `canKDTree_performance.parquet`

Benchmark results for canonical construction. Shape: 160 rows × 7 columns.
Same schema as above, with `algorithm = "canonical"`, sizes `"2^26"` through
`"2^30"`, and `parameter` values `"accuracy = 10"` through `"accuracy = 10000"`.

### `pKDTree_performance.parquet`

Precision comparison results for epstat and pkd-tree. Shape: 128 rows × 8 columns.
Same schema as above, fixed at size `"2^28"`, plus one additional column:

| Column | Type | Values |
|--------|------|--------|
| `fold` | int | `1` = single-pass build; `2` = two-pass build at twice the depth |

Algorithm values: `"epstat"` and `"pkd"`. Parameter values for epstat: `"J = ..."`;
for pkd-tree: `"sigma = 16"`, `"sigma = 32"`, `"sigma = 64"`, `"sigma = 128"`.

---

## Registering modules with a Spark cluster

All three `.py` modules must be shipped to every executor before use. The standard
pattern on AWS EMR, or any cluster with S3 access, is:

```python
spark.sparkContext.addPyFile("s3://your-bucket/code/epstatKDTree.py")
from epstatKDTree import *
```

On a local PySpark session, use a local file path:

```python
spark.sparkContext.addPyFile("/path/to/epstatKDTree.py")
from epstatKDTree import *
```

---

## Contact

For questions about the code or data, please contact the corresponding author at
**chakrav0@purdue.edu**.
