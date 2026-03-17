"""
HDF5Manager - Load and save HDF5 files with Julia/Python cross-language support.

Bidirectional HDF5 serialization preserving complex numbers, arrays, dicts,
and type metadata across Julia and Python.
"""

from .core import (
    JuliaHDF5Loader,
    load_hdf5,
    load_hdf5_item,
    list_hdf5_variables,
    inspect_hdf5,
    save_hdf5,
    save_hdf5_item,
)

__all__ = [
    "JuliaHDF5Loader",
    "load_hdf5",
    "load_hdf5_item",
    "list_hdf5_variables",
    "inspect_hdf5",
    "save_hdf5",
    "save_hdf5_item",
]
