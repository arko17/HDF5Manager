# HDF5Manager

Bidirectional HDF5 serialization for Julia and Python with full type preservation. Files written in one language are fully readable in the other.

## Features

- Real and complex arrays (any dimension)
- Complex scalars
- Numeric scalars (Int, Float, Bool)
- Dict / NamedTuple (recursive)
- Type metadata preserved as HDF5 attributes

## Julia

```julia
using HDF5Manager

# Save
save_hdf5("data.h5"; arr=[1,2,3], z=1+2im, scalar=42)

# Load
data = load_hdf5("data.h5")          # Dict{String,Any}
arr  = load_hdf5_item("data.h5", "arr")
vars = list_hdf5_variables("data.h5") # ["arr", "z", "scalar"]
```

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/arko17/HDF5Manager")
```

## Python

```python
from hdfmanager import save_hdf5, load_hdf5
import numpy as np

# Save
save_hdf5("data.h5", arr=np.array([1,2,3]), z=1+2j, scalar=42)

# Load
data = load_hdf5("data.h5")
```

### Installation

```bash
pip install git+https://github.com/arko17/HDF5Manager.git#subdirectory=python
```

## Complex Number Storage

Complex numbers are stored as separate real/imaginary parts in HDF5 groups:
```
/variable_name/
  ├─ real (dataset)
  └─ imag (dataset)
  └─ @attrs: is_complex=true, element_type="Complex{Float64}"
```

## Array Memory Layout

Julia uses **column-major**, Python/NumPy uses **row-major**.
- 1D arrays transfer without issues
- For multi-dimensional arrays, use `.T` in NumPy to recover the Julia shape

## Running Tests

```bash
# Julia
julia --project=. test/runtests.jl

# Python
cd python && pytest tests/ -v

# Cross-language integration
pytest test/test_integration.py -v
```
