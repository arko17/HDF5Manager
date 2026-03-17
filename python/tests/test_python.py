"""
test_python.py

Python-only unit tests for HDF5Manager package.
Tests Python save/load functionality independent of Julia.

Run with: pytest python/tests/test_python.py -v
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from HDF5Manager import save_hdf5, load_hdf5


class TestPythonSaveLoad:
    """Test Python save and load functionality."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name: str) -> str:
        return os.path.join(self.tmpdir, name)

    def test_scalar_int(self):
        f = self._path("scalar_int.h5")
        save_hdf5(f, value=42)
        data = load_hdf5(f)
        assert data["value"] == 42

    def test_scalar_float(self):
        f = self._path("scalar_float.h5")
        save_hdf5(f, value=3.14159)
        data = load_hdf5(f)
        assert data["value"] == pytest.approx(3.14159)

    def test_scalar_bool(self):
        f = self._path("scalar_bool.h5")
        save_hdf5(f, value=True)
        data = load_hdf5(f)
        assert data["value"] is True

    def test_scalar_complex(self):
        f = self._path("scalar_complex.h5")
        save_hdf5(f, z=1.5 + 2.5j)
        data = load_hdf5(f)
        assert data["z"] == pytest.approx(1.5 + 2.5j)

    def test_1d_array_float(self):
        f = self._path("array_1d_float.h5")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        save_hdf5(f, data=arr)
        loaded = load_hdf5(f)
        np.testing.assert_array_almost_equal(loaded["data"], arr)

    def test_1d_array_int(self):
        f = self._path("array_1d_int.h5")
        arr = np.array([1, 2, 3, 4, 5])
        save_hdf5(f, data=arr)
        loaded = load_hdf5(f)
        np.testing.assert_array_equal(loaded["data"], arr)

    def test_1d_array_complex(self):
        f = self._path("array_1d_complex.h5")
        arr = np.array([1+2j, 3+4j, 5+6j])
        save_hdf5(f, data=arr)
        loaded = load_hdf5(f)
        np.testing.assert_array_almost_equal(loaded["data"], arr)

    def test_2d_array(self):
        f = self._path("array_2d.h5")
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        save_hdf5(f, matrix=arr)
        loaded = load_hdf5(f)
        assert loaded["matrix"].shape == arr.shape
        np.testing.assert_array_almost_equal(loaded["matrix"], arr)

    def test_2d_array_complex(self):
        f = self._path("array_2d_complex.h5")
        arr = np.array([[1+1j, 2+2j], [3+3j, 4+4j]])
        save_hdf5(f, cmatrix=arr)
        loaded = load_hdf5(f)
        assert loaded["cmatrix"].shape == arr.shape
        np.testing.assert_array_almost_equal(loaded["cmatrix"], arr)

    def test_3d_array(self):
        f = self._path("array_3d.h5")
        arr = np.random.randn(2, 3, 4)
        save_hdf5(f, tensor=arr)
        loaded = load_hdf5(f)
        assert loaded["tensor"].shape == arr.shape
        np.testing.assert_array_almost_equal(loaded["tensor"], arr)

    def test_multiple_variables(self):
        f = self._path("multiple.h5")
        save_hdf5(f, 
                  scalar=42, 
                  vector=np.array([1.0, 2.0, 3.0]),
                  matrix=np.eye(3),
                  complex_val=1+2j)
        data = load_hdf5(f)
        
        assert data["scalar"] == 42
        np.testing.assert_array_almost_equal(data["vector"], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(data["matrix"], np.eye(3))
        assert data["complex_val"] == pytest.approx(1+2j)

    def test_dict_storage(self):
        f = self._path("dict.h5")
        save_hdf5(f, nested={"a": 1, "b": 2.5, "c": np.array([1, 2, 3])})
        data = load_hdf5(f)
        
        assert data["nested"]["a"] == 1
        assert data["nested"]["b"] == pytest.approx(2.5)
        np.testing.assert_array_equal(data["nested"]["c"], np.array([1, 2, 3]))

    def test_overwrite(self):
        f = self._path("overwrite.h5")
        save_hdf5(f, value=1)
        data1 = load_hdf5(f)
        assert data1["value"] == 1
        
        # Save again with different value
        save_hdf5(f, value=2, extra=3)
        data2 = load_hdf5(f)
        assert data2["value"] == 2
        assert data2["extra"] == 3

    def test_empty_file(self):
        f = self._path("empty.h5")
        save_hdf5(f)
        data = load_hdf5(f)
        assert isinstance(data, dict)
