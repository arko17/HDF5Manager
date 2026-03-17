# HDF5Manager

Bidirectional HDF5 serialization for Julia and Python with full type preservation. Files written in one language are fully readable in the other, with **automatic column-major/row-major transpose** for multi-dimensional arrays.

## Features

- Real and complex arrays (any dimension)
- Complex scalars
- Numeric scalars (Int, Float, Bool)
- Dict / NamedTuple (recursive)
- Type metadata preserved as HDF5 attributes
- Automatic array transpose between column-major (Julia) and row-major (Python)

## Julia

```julia
using HDF5Manager

# Save
save_hdf5("data.h5"; arr=[1,2,3], z=1+2im, mat=[1.0 2.0; 3.0 4.0])

# Load
data = load_hdf5("data.h5")          # Dict{String,Any}
arr  = load_hdf5_item("data.h5", "arr")
vars = list_hdf5_variables("data.h5") # ["arr", "z", "mat"]
```

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/arko17/HDF5Manager")
```

## Python

```python
from HDF5Manager import save_hdf5, load_hdf5
import numpy as np

# Save
save_hdf5("data.h5", arr=np.array([1,2,3]), z=1+2j, mat=np.eye(3))

# Load
data = load_hdf5("data.h5")
```

### Installation

```bash
pip install git+https://github.com/arko17/HDF5Manager.git#subdirectory=python
```

## Array Memory Layout (Automatic Transpose)

Julia uses **column-major** order, Python/NumPy uses **row-major** order. HDF5Manager handles this transparently:

- Each file is tagged with a `storage_order` attribute (`"column_major"` or `"row_major"`)
- On load, if the storage order differs from the reader's native order, multi-dimensional arrays (ndim >= 2) are automatically transposed
- 1D arrays and scalars are unaffected
- **Legacy files** (without `storage_order`) are read as-is with no transpose

This means a `(2, 3)` matrix saved from Python will be loaded as a `(2, 3)` matrix in Julia, and vice versa, with all elements in the correct positions.

## Complex Number Storage

Complex numbers are stored as separate real/imaginary parts in HDF5 groups:
```
/variable_name/
  +-- real (dataset)
  +-- imag (dataset)
  +-- @attrs: is_complex=true, element_type="Complex{Float64}"
```

## Running Tests

```bash
# Julia
julia --project=. test/runtests.jl

# Python
cd python && pytest tests/ -v

# Cross-language integration
pytest test/test_integration.py -v
```
