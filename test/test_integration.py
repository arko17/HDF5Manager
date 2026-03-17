"""
test_integration.py

Cross-language integration tests:
  - Julia saves -> Python loads (with automatic transpose)
  - Python saves -> Julia loads (with automatic transpose)

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
            save_hdf5("{f}"; vec=[1.0, 2.0, 3.0], mat=[1.0 2.0 3.0; 4.0 5.0 6.0])
        '''))
        data = load_hdf5(f)
        np.testing.assert_array_almost_equal(data["vec"], [1.0, 2.0, 3.0])
        # Julia [1 2 3; 4 5 6] is 2×3. After auto-transpose, Python should see (2, 3).
        mat = data["mat"]
        assert mat.shape == (2, 3)
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_almost_equal(mat, expected)

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

    def test_complex_2d_array(self):
        f = _path("jl2py_cmat.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            save_hdf5("{f}"; cmat=[1+0im 2+1im 3+2im; 4+3im 5+4im 6+5im])
        '''))
        data = load_hdf5(f)
        cmat = data["cmat"]
        assert cmat.shape == (2, 3)
        expected = np.array([[1+0j, 2+1j, 3+2j], [4+3j, 5+4j, 6+5j]])
        np.testing.assert_array_almost_equal(cmat, expected)

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

    def test_3d_array(self):
        f = _path("jl2py_3d.h5")
        _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            t = reshape(collect(1.0:24.0), (2, 3, 4))
            save_hdf5("{f}"; tensor=t)
        '''))
        data = load_hdf5(f)
        tensor = data["tensor"]
        assert tensor.shape == (2, 3, 4)
        # Julia reshape(1:24, 2,3,4)[1,1,1] == 1.0, [2,1,1] == 2.0
        assert tensor[0, 0, 0] == pytest.approx(1.0)
        assert tensor[1, 0, 0] == pytest.approx(2.0)


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

    def test_2d_array(self):
        """Python (2,3) array should arrive as Julia (2,3) matrix."""
        f = _path("py2jl_mat.h5")
        save_hdf5(f, mat=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            m = data["mat"]
            @assert size(m) == (2, 3) "expected (2,3) got $(size(m))"
            @assert m ≈ [1.0 2.0 3.0; 4.0 5.0 6.0]
            println("OK")
        '''))
        assert "OK" in out

    def test_3d_array(self):
        """Python (2,3,4) array should arrive as Julia (2,3,4) tensor."""
        f = _path("py2jl_3d.h5")
        t = np.arange(24.0).reshape(2, 3, 4)
        save_hdf5(f, tensor=t)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            t = data["tensor"]
            @assert size(t) == (2, 3, 4) "expected (2,3,4) got $(size(t))"
            @assert t[1, 1, 1] ≈ 0.0   # Python [0,0,0] = 0.0
            @assert t[2, 1, 1] ≈ 12.0  # Python [1,0,0] = 12.0
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

    def test_complex_2d_array(self):
        f = _path("py2jl_cmat.h5")
        save_hdf5(f, cmat=np.array([[1+0j, 2+1j, 3+2j], [4+3j, 5+4j, 6+5j]]))
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f}")
            cm = data["cmat"]
            @assert size(cm) == (2, 3) "expected (2,3) got $(size(cm))"
            @assert cm ≈ [1+0im 2+1im 3+2im; 4+3im 5+4im 6+5im]
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

    def test_roundtrip_1d(self):
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

    def test_roundtrip_2d(self):
        """Python 2D -> Julia modifies -> Python: shape and values preserved."""
        f1 = _path("rt2d_step1.h5")
        f2 = _path("rt2d_step2.h5")
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        save_hdf5(f1, mat=mat)
        out = _run_julia(textwrap.dedent(f'''
            using HDF5Manager
            data = load_hdf5("{f1}")
            m = data["mat"]
            @assert size(m) == (2, 3) "expected (2,3) got $(size(m))"
            save_hdf5("{f2}"; mat=m .* 10)
            println("OK")
        '''))
        assert "OK" in out
        data2 = load_hdf5(f2)
        assert data2["mat"].shape == (2, 3)
        expected = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        np.testing.assert_array_almost_equal(data2["mat"], expected)
