"""
test_roundtrip.py

Round-trip tests: Python save -> Python load for HDF5Manager.
Run with:  pytest python/tests/test_roundtrip.py -v
"""

import tempfile
import numpy as np
import pytest

from HDF5Manager import (
    save_hdf5, load_hdf5, load_hdf5_item, save_hdf5_item,
    list_hdf5_variables, inspect_hdf5,
)

TMPDIR = tempfile.mkdtemp()


def _path(name: str) -> str:
    import os
    return os.path.join(TMPDIR, name)


# ── Scalars ───────────────────────────────────────────────────────────────────

class TestScalars:
    def test_int(self):
        f = _path("scalar_int.h5")
        save_hdf5(f, val=42)
        assert load_hdf5(f)["val"] == 42

    def test_float(self):
        f = _path("scalar_float.h5")
        save_hdf5(f, val=3.14)
        assert load_hdf5(f)["val"] == pytest.approx(3.14)

    def test_bool(self):
        f = _path("scalar_bool.h5")
        save_hdf5(f, val=True)
        assert load_hdf5(f)["val"] == True


# ── Real arrays ───────────────────────────────────────────────────────────────

class TestRealArrays:
    def test_1d(self):
        f = _path("arr1d.h5")
        arr = np.array([1.0, 2.0, 3.0])
        save_hdf5(f, arr=arr)
        np.testing.assert_array_almost_equal(load_hdf5(f)["arr"], arr)

    def test_2d(self):
        f = _path("arr2d.h5")
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        save_hdf5(f, mat=arr)
        np.testing.assert_array_almost_equal(load_hdf5(f)["mat"], arr)

    def test_3d(self):
        f = _path("arr3d.h5")
        arr = np.random.rand(2, 3, 4)
        save_hdf5(f, tensor=arr)
        np.testing.assert_array_almost_equal(load_hdf5(f)["tensor"], arr)

    def test_int_array(self):
        f = _path("arr_int.h5")
        arr = np.array([10, 20, 30], dtype=np.int64)
        save_hdf5(f, arr=arr)
        np.testing.assert_array_equal(load_hdf5(f)["arr"], arr)

    def test_list_input(self):
        f = _path("from_list.h5")
        save_hdf5(f, data=[1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(load_hdf5(f)["data"], [1.0, 2.0, 3.0])


# ── Complex scalars ───────────────────────────────────────────────────────────

class TestComplexScalars:
    def test_complex_scalar(self):
        f = _path("cscalar.h5")
        z = 1.5 + 2.5j
        save_hdf5(f, z=z)
        loaded = load_hdf5(f)["z"]
        assert loaded == pytest.approx(z)

    def test_numpy_complex_scalar(self):
        f = _path("np_cscalar.h5")
        z = np.complex128(3 + 4j)
        save_hdf5(f, z=z)
        loaded = load_hdf5(f)["z"]
        assert loaded == pytest.approx(z)


# ── Complex arrays ────────────────────────────────────────────────────────────

class TestComplexArrays:
    def test_1d(self):
        f = _path("carr1d.h5")
        arr = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        save_hdf5(f, arr=arr)
        np.testing.assert_array_almost_equal(load_hdf5(f)["arr"], arr)

    def test_2d(self):
        f = _path("carr2d.h5")
        arr = np.array([[1 + 0j, 2 + 1j], [3 + 2j, 4 + 3j]])
        save_hdf5(f, mat=arr)
        np.testing.assert_array_almost_equal(load_hdf5(f)["mat"], arr)


# ── Dicts ─────────────────────────────────────────────────────────────────────

class TestDicts:
    def test_flat_dict(self):
        f = _path("dict_flat.h5")
        d = {"a": 1.0, "b": np.array([10, 20, 30])}
        save_hdf5(f, params=d)
        loaded = load_hdf5(f)["params"]
        assert loaded["a"] == 1.0
        np.testing.assert_array_equal(loaded["b"], [10, 20, 30])

    def test_nested_dict(self):
        f = _path("dict_nested.h5")
        d = {"outer": {"inner_val": 42.0}}
        save_hdf5(f, nested=d)
        loaded = load_hdf5(f)["nested"]
        assert loaded["outer"]["inner_val"] == 42.0


# ── Multiple variables ────────────────────────────────────────────────────────

class TestMultipleVars:
    def test_multi(self):
        f = _path("multi.h5")
        save_hdf5(f, a=1, b=np.array([1.0, 2.0]), c=3 + 4j)
        data = load_hdf5(f)
        assert data["a"] == 1
        np.testing.assert_array_almost_equal(data["b"], [1.0, 2.0])
        assert data["c"] == pytest.approx(3 + 4j)


# ── load / list / inspect helpers ─────────────────────────────────────────────

class TestHelpers:
    def test_load_hdf5_item(self):
        f = _path("helpers.h5")
        save_hdf5(f, x=10, y=np.array([1, 2]))
        assert load_hdf5_item(f, "x") == 10

    def test_list_variables(self):
        f = _path("helpers.h5")
        assert sorted(list_hdf5_variables(f)) == ["x", "y"]

    def test_inspect(self):
        f = _path("helpers.h5")
        info = inspect_hdf5(f)
        assert "x" in info
        assert "y" in info

    def test_save_hdf5_item_new(self):
        f = _path("item_append.h5")
        save_hdf5(f, x=1)
        save_hdf5_item(f, "y", np.array([10, 20]))
        data = load_hdf5(f)
        assert data["x"] == 1
        np.testing.assert_array_equal(data["y"], [10, 20])

    def test_save_hdf5_item_overwrite(self):
        f = _path("item_overwrite.h5")
        save_hdf5(f, x=1)
        save_hdf5_item(f, "x", 99)
        assert load_hdf5(f)["x"] == 99


# ── Overwrite ─────────────────────────────────────────────────────────────────

class TestOverwrite:
    def test_full_overwrite(self):
        f = _path("overwrite.h5")
        save_hdf5(f, x=1, y=2)
        save_hdf5(f, x=10)
        data = load_hdf5(f)
        assert data["x"] == 10
        assert "y" not in data
