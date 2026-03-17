"""
test_integration.py

Cross-language integration tests:
  - Julia saves -> Python loads
  - Python saves -> Julia loads

Requires Julia with HDF5.jl available.
Run with:  pytest test/test_integration.py -v
"""

import sys, os, tempfile, subprocess, textwrap
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from HDF5Manager import save_hdf5, load_hdf5

TMPDIR = tempfile.mkdtemp()
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
JULIA = "julia"


def _run_julia(code: str, timeout: int = 120) -> str:
    """Run a Julia snippet and return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        [JULIA, "--project=.", "-e", code],
        capture_output=True, text=True, cwd=ROOT, timeout=timeout,
    )
    if result.returncode != 0:
        pytest.fail(f"Julia error:\n{result.stderr}\n{result.stdout}")
    return result.stdout


def _path(name: str) -> str:
    return os.path.join(TMPDIR, name).replace("\\", "/")


# ══════════════════════════════════════════════════════════════════════════════
# Julia -> Python
# ══════════════════════════════════════════════════════════════════════════════

class TestJuliaToPython:
    """Julia saves an HDF5 file, Python loads and verifies."""

    def test_scalars(self):
        f = _path("jl2py_scalars.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; int_val=42, float_val=3.14, bool_val=true)
        '''))
        data = load_hdf5(f)
        assert data["int_val"] == 42
        assert data["float_val"] == pytest.approx(3.14)
        assert data["bool_val"] == True

    def test_real_arrays(self):
        f = _path("jl2py_real.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; vec=[1.0, 2.0, 3.0], mat=[1.0 2.0; 3.0 4.0])
        '''))
        data = load_hdf5(f)
        np.testing.assert_array_almost_equal(data["vec"], [1.0, 2.0, 3.0])
        mat = np.array(data["mat"])
        assert mat.size == 4
        assert set(mat.flat) == {1.0, 2.0, 3.0, 4.0}

    def test_complex_scalar(self):
        f = _path("jl2py_cscalar.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; z=1.5+2.5im)
        '''))
        data = load_hdf5(f)
        assert data["z"] == pytest.approx(1.5 + 2.5j)

    def test_complex_array(self):
        f = _path("jl2py_carr.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; carr=[1+2im, 3+4im, 5+6im])
        '''))
        data = load_hdf5(f)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        np.testing.assert_array_almost_equal(data["carr"], expected)

    def test_dict(self):
        f = _path("jl2py_dict.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; params=Dict("a" => 1.0, "b" => [10, 20]))
        '''))
        data = load_hdf5(f)
        assert data["params"]["a"] == 1.0
        np.testing.assert_array_equal(data["params"]["b"], [10, 20])

    def test_mixed(self):
        f = _path("jl2py_mixed.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; a=1, b=[1.0,2.0], c=3+4im)
        '''))
        data = load_hdf5(f)
        assert data["a"] == 1
        np.testing.assert_array_almost_equal(data["b"], [1.0, 2.0])
        assert data["c"] == pytest.approx(3 + 4j)


# ══════════════════════════════════════════════════════════════════════════════
# Python -> Julia
# ══════════════════════════════════════════════════════════════════════════════

class TestPythonToJulia:
    """Python saves an HDF5 file, Julia loads and verifies."""

    def test_scalars(self):
        f = _path("py2jl_scalars.h5")
        save_hdf5(f, int_val=42, float_val=3.14, bool_val=True)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            @assert data["int_val"] == 42
            @assert data["float_val"] ≈ 3.14
            @assert data["bool_val"] == true  ||  data["bool_val"] == 1
            println("OK")
        '''))
        assert "OK" in out

    def test_real_array(self):
        f = _path("py2jl_arr.h5")
        save_hdf5(f, vec=np.array([1.0, 2.0, 3.0]))
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            @assert data["vec"] ≈ [1.0, 2.0, 3.0]
            println("OK")
        '''))
        assert "OK" in out

    def test_complex_scalar(self):
        f = _path("py2jl_cscalar.h5")
        save_hdf5(f, z=1.5 + 2.5j)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            @assert data["z"] ≈ 1.5 + 2.5im
            println("OK")
        '''))
        assert "OK" in out

    def test_complex_array(self):
        f = _path("py2jl_carr.h5")
        save_hdf5(f, carr=np.array([1 + 2j, 3 + 4j]))
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            @assert data["carr"] ≈ [1+2im, 3+4im]
            println("OK")
        '''))
        assert "OK" in out

    def test_dict(self):
        f = _path("py2jl_dict.h5")
        save_hdf5(f, params={"a": 1.0, "b": np.array([10, 20])})
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            p = data["params"]
            @assert p["a"] == 1.0
            @assert p["b"] == [10, 20]
            println("OK")
        '''))
        assert "OK" in out

    def test_mixed(self):
        f = _path("py2jl_mixed.h5")
        save_hdf5(f, a=1, b=np.array([1.0, 2.0]), c=3 + 4j)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            @assert data["a"] == 1
            @assert data["b"] ≈ [1.0, 2.0]
            @assert data["c"] ≈ 3 + 4im
            println("OK")
        '''))
        assert "OK" in out


# ══════════════════════════════════════════════════════════════════════════════
# Full round-trip: Python -> Julia -> Python
# ══════════════════════════════════════════════════════════════════════════════

class TestFullRoundTrip:
    """Python saves, Julia loads & re-saves with modifications, Python loads result."""

    def test_roundtrip(self):
        f1 = _path("rt_step1.h5")
        f2 = _path("rt_step2.h5")
        save_hdf5(f1, x=np.array([1.0, 2.0, 3.0]), z=1 + 2j)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f1}")
            new_x = data["x"] .* 2
            new_z = data["z"] + 10
            save_hdf5("{f2}"; x=new_x, z=new_z)
            println("OK")
        '''))
        assert "OK" in out
        data2 = load_hdf5(f2)
        np.testing.assert_array_almost_equal(data2["x"], [2.0, 4.0, 6.0])
        assert data2["z"] == pytest.approx(11 + 2j)
