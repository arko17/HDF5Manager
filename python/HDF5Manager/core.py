"""
Core HDF5Manager functionality for Python.

Load and save HDF5 files created by Julia HDF5Manager or Python,
preserving Julia datatypes including complex numbers and arrays.
Automatically handles column-major/row-major transpose when reading
files written by Julia.
"""

import h5py
import numpy as np
from typing import Dict, Any, Union
from pathlib import Path

STORAGE_ORDER = "row_major"


# ─── Transpose helpers ──────────────────────────────────────────────────────

def _needs_transpose(file: h5py.File) -> bool:
    """Return True if arrays need transposing (written by column-major writer)."""
    order = file.attrs.get('storage_order')
    if order is not None:
        if isinstance(order, bytes):
            order = order.decode()
        return order == "column_major"
    return False  # fallback: no transpose


def _maybe_transpose(arr: np.ndarray) -> np.ndarray:
    """Transpose a multi-dimensional array (ndim >= 2)."""
    if arr.ndim >= 2:
        return arr.T
    return arr


# ─── Loader class ────────────────────────────────────────────────────────────

class JuliaHDF5Loader:
    """Load and manage HDF5 files created by Julia's HDF5Manager module."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.file = None
        self.data = {}

    def load(self) -> Dict[str, Any]:
        self.data = {}
        with h5py.File(self.filepath, 'r') as file:
            transpose = _needs_transpose(file)
            for name in file.keys():
                self.data[name] = self._load_item(file[name], transpose)
        return self.data

    def _load_group(self, group: h5py.Group, transpose: bool) -> Any:
        attrs = dict(group.attrs)

        # Complex Numbers (Array or Scalar)
        if attrs.get('is_complex'):
            if 'real' in group and 'imag' in group:
                real_part = np.array(group['real'])
                imag_part = np.array(group['imag'])
                result = real_part + 1j * imag_part
                if transpose and isinstance(result, np.ndarray) and result.ndim >= 2:
                    result = _maybe_transpose(result)
                return result

        # Recursive Loading for Dicts & NamedTuples
        result = {}
        for key in group.keys():
            result[key] = self._load_item(group[key], transpose)
        return result

    def _load_item(self, obj: Union[h5py.Group, h5py.Dataset], transpose: bool) -> Any:
        if isinstance(obj, h5py.Group):
            return self._load_group(obj, transpose)
        elif isinstance(obj, h5py.Dataset):
            return self._load_dataset(obj, transpose)
        else:
            return None

    def _load_dataset(self, dataset: h5py.Dataset, transpose: bool) -> Any:
        data = np.array(dataset)

        if data.ndim == 0:
            return data.item()

        if data.dtype.kind == 'S':
            return data.astype(str)

        if transpose and data.ndim >= 2:
            data = _maybe_transpose(data)

        return data

    def __getitem__(self, key: str) -> Any:
        if not self.data:
            self.load()
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        if not self.data:
            self.load()
        return key in self.data

    def keys(self):
        if not self.data:
            self.load()
        return self.data.keys()

    def items(self):
        if not self.data:
            self.load()
        return self.data.items()

    def __repr__(self) -> str:
        if not self.data:
            self.load()
        var_list = ", ".join(self.data.keys())
        return f"JuliaHDF5Loader({self.filepath.name}): {var_list}"


# ─── Load convenience functions ──────────────────────────────────────────────

def load_hdf5(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load all data from an HDF5 file.

    Multi-dimensional arrays are automatically transposed when the file was
    written by a column-major language (Julia). If no ``storage_order``
    attribute is present (legacy files), arrays are returned as-is.

    Example:
        >>> data = load_hdf5("data.h5")
        >>> print(data['my_array'])
    """
    loader = JuliaHDF5Loader(filepath)
    return loader.load()


def load_hdf5_item(filepath: Union[str, Path], name: str) -> Any:
    """
    Load a specific variable from an HDF5 file.

    Example:
        >>> arr = load_hdf5_item("data.h5", "my_array")
    """
    loader = JuliaHDF5Loader(filepath)
    return loader[name]


def list_hdf5_variables(filepath: Union[str, Path]) -> list:
    """
    List all variables in an HDF5 file.

    Example:
        >>> vars = list_hdf5_variables("data.h5")
    """
    with h5py.File(filepath, 'r') as file:
        return list(file.keys())


def inspect_hdf5(filepath: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Inspect an HDF5 file and return detailed information about each variable.

    Example:
        >>> info = inspect_hdf5("data.h5")
        >>> for var_name, var_info in info.items():
        ...     print(f"{var_name}: {var_info}")
    """
    info = {}
    with h5py.File(filepath, 'r') as file:
        for name in file.keys():
            obj = file[name]
            info[name] = _get_object_info(obj)
    return info


def _get_object_info(obj: Union[h5py.Dataset, h5py.Group]) -> Dict[str, Any]:
    """Get information about an HDF5 object."""
    info = {}
    if isinstance(obj, h5py.Group):
        info['type'] = 'group'
        info['contents'] = list(obj.keys())
        attrs = dict(obj.attrs)
        if 'is_complex' in attrs:
            info['is_complex'] = bool(attrs['is_complex'])
        if 'element_type' in attrs:
            info['element_type'] = attrs['element_type']
        if 'shape' in attrs:
            info['shape'] = tuple(attrs['shape'])
    else:
        info['type'] = 'dataset'
        info['dtype'] = str(obj.dtype)
        info['shape'] = obj.shape
        info['size'] = obj.size
    return info


# ─── Save helpers ────────────────────────────────────────────────────────────

def _save_variable(parent: h5py.Group, name: str, value: Any) -> None:
    """Save a single variable into an HDF5 group/file, mirroring Julia HDF5Manager format."""
    if isinstance(value, np.ndarray):
        _save_array(parent, name, value)
    elif isinstance(value, (complex, np.complexfloating)):
        _save_complex_scalar(parent, name, value)
    elif isinstance(value, dict):
        _save_dict(parent, name, value)
    elif isinstance(value, (list, tuple)):
        _save_array(parent, name, np.asarray(value))
    else:
        parent[name] = value
        parent[name].attrs['jl_type'] = str(type(value).__name__)


def _save_array(parent: h5py.Group, name: str, arr: np.ndarray) -> None:
    """Save an array, splitting complex arrays into real/imag groups."""
    if np.issubdtype(arr.dtype, np.complexfloating):
        g = parent.create_group(name)
        g['real'] = arr.real
        g['imag'] = arr.imag
        g.attrs['is_complex'] = True
        g.attrs['element_type'] = f"Complex{{{_numpy_to_julia_type(arr.real.dtype)}}}"
        g.attrs['shape'] = list(arr.shape)
    else:
        parent[name] = arr
        parent[name].attrs['shape'] = list(arr.shape)
        parent[name].attrs['jl_type'] = f"Array{{{_numpy_to_julia_type(arr.dtype)}, {arr.ndim}}}"


def _save_complex_scalar(parent: h5py.Group, name: str, val: complex) -> None:
    """Save a complex scalar as a group with real/imag datasets."""
    g = parent.create_group(name)
    g['real'] = np.float64(val.real)
    g['imag'] = np.float64(val.imag)
    g.attrs['is_complex'] = True
    g.attrs['element_type'] = "ComplexF64"


def _save_dict(parent: h5py.Group, name: str, d: dict) -> None:
    """Save a dictionary as an HDF5 group, recursively saving values."""
    g = parent.create_group(name)
    g.attrs['jl_type'] = "Dict"
    for k, v in d.items():
        _save_variable(g, str(k), v)


_NUMPY_TO_JULIA = {
    np.dtype('float64'): 'Float64', np.dtype('float32'): 'Float32',
    np.dtype('float16'): 'Float16',
    np.dtype('int64'): 'Int64', np.dtype('int32'): 'Int32',
    np.dtype('int16'): 'Int16', np.dtype('int8'): 'Int8',
    np.dtype('uint64'): 'UInt64', np.dtype('uint32'): 'UInt32',
    np.dtype('uint16'): 'UInt16', np.dtype('uint8'): 'UInt8',
    np.dtype('bool'): 'Bool',
}


def _numpy_to_julia_type(dtype: np.dtype) -> str:
    """Map a numpy dtype to its Julia type name string."""
    return _NUMPY_TO_JULIA.get(np.dtype(dtype), str(dtype))


# ─── Save convenience functions ──────────────────────────────────────────────

def save_hdf5(filepath: Union[str, Path], **kwargs: Any) -> None:
    """
    Save multiple variables to an HDF5 file, compatible with Julia HDF5Manager.

    A ``storage_order = "row_major"`` attribute is written to the root group so
    that readers in other languages can automatically transpose multi-dimensional
    arrays.

    Example:
        >>> save_hdf5("data.h5", arr=np.array([1,2,3]), z=1+2j, scalar=42)
    """
    with h5py.File(filepath, 'w') as f:
        f.attrs['storage_order'] = STORAGE_ORDER
        for name, value in kwargs.items():
            _save_variable(f, name, value)


def save_hdf5_item(filepath: Union[str, Path], name: str, value: Any) -> None:
    """
    Append or overwrite a single variable in an existing HDF5 file.

    Example:
        >>> save_hdf5_item("data.h5", "new_arr", np.array([10, 20]))
    """
    with h5py.File(filepath, 'a') as f:
        # Ensure storage_order is set
        if 'storage_order' not in f.attrs:
            f.attrs['storage_order'] = STORAGE_ORDER
        if name in f:
            del f[name]
        _save_variable(f, name, value)
