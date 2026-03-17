using Test

include(joinpath(@__DIR__, "..", "src", "HDF5Manager.jl"))
using .HDF5Manager

const TMPDIR = mktempdir()

@testset "HDF5Manager Julia Round-Trip" begin

    # ── Scalars ───────────────────────────────────────────────────────────
    @testset "Scalars" begin
        f = joinpath(TMPDIR, "scalars.h5")
        save_hdf5(f; int_val=42, float_val=3.14, bool_val=true)
        data = load_hdf5(f)

        @test data["int_val"] == 42
        @test data["float_val"] ≈ 3.14
        @test data["bool_val"] == true
    end

    # ── Real arrays ───────────────────────────────────────────────────────
    @testset "Real arrays" begin
        f = joinpath(TMPDIR, "real_arrays.h5")
        v = [1.0, 2.0, 3.0]
        m = [1.0 2.0; 3.0 4.0]
        t = rand(2, 3, 4)
        save_hdf5(f; vec=v, mat=m, tensor=t)
        data = load_hdf5(f)

        @test data["vec"] ≈ v
        @test data["mat"] ≈ m
        @test data["tensor"] ≈ t
    end

    # ── Complex scalars ───────────────────────────────────────────────────
    @testset "Complex scalars" begin
        f = joinpath(TMPDIR, "complex_scalar.h5")
        z = 1.5 + 2.5im
        save_hdf5(f; z=z)
        data = load_hdf5(f)

        @test data["z"] ≈ z
    end

    # ── Complex arrays ────────────────────────────────────────────────────
    @testset "Complex arrays" begin
        f = joinpath(TMPDIR, "complex_arrays.h5")
        cv = [1+2im, 3+4im, 5+6im]
        cm = [1+0im 2+1im; 3+2im 4+3im]
        save_hdf5(f; cvec=cv, cmat=cm)
        data = load_hdf5(f)

        @test data["cvec"] ≈ cv
        @test data["cmat"] ≈ cm
    end

    # ── Dict ──────────────────────────────────────────────────────────────
    @testset "Dict" begin
        f = joinpath(TMPDIR, "dict.h5")
        d = Dict("a" => 1.0, "b" => [10, 20, 30])
        save_hdf5(f; params=d)
        data = load_hdf5(f)

        @test data["params"]["a"] == 1.0
        @test data["params"]["b"] == [10, 20, 30]
    end

    # ── NamedTuple ────────────────────────────────────────────────────────
    @testset "NamedTuple" begin
        f = joinpath(TMPDIR, "namedtuple.h5")
        nt = (alpha=0.5, beta=1.0, data=[1, 2, 3])
        save_hdf5(f; config=nt)
        data = load_hdf5(f)

        @test data["config"].alpha == 0.5
        @test data["config"].beta == 1.0
        @test data["config"].data == [1, 2, 3]
    end

    # ── Multiple variables ────────────────────────────────────────────────
    @testset "Multiple variables" begin
        f = joinpath(TMPDIR, "multi.h5")
        save_hdf5(f; a=1, b=[1.0, 2.0], c=3+4im)
        data = load_hdf5(f)

        @test data["a"] == 1
        @test data["b"] ≈ [1.0, 2.0]
        @test data["c"] ≈ 3 + 4im
    end

    # ── load_hdf5_item ────────────────────────────────────────────────────
    @testset "load_hdf5_item" begin
        f = joinpath(TMPDIR, "multi.h5")  # reuse file from above
        b = load_hdf5_item(f, "b")
        @test b ≈ [1.0, 2.0]

        @test_throws ErrorException load_hdf5_item(f, "nonexistent")
    end

    # ── list_hdf5_variables ───────────────────────────────────────────────
    @testset "list_hdf5_variables" begin
        f = joinpath(TMPDIR, "multi.h5")
        vars = list_hdf5_variables(f)
        @test sort(vars) == sort(["a", "b", "c"])
    end

    # ── Numeric dtypes ────────────────────────────────────────────────────
    @testset "Numeric dtypes" begin
        f = joinpath(TMPDIR, "dtypes.h5")
        save_hdf5(f;
            i8=Int8(1), i32=Int32(2), i64=Int64(3),
            f32=Float32(1.5), f64=Float64(2.5),
        )
        data = load_hdf5(f)
        @test data["i8"] == 1
        @test data["i32"] == 2
        @test data["i64"] == 3
        @test data["f32"] ≈ 1.5
        @test data["f64"] ≈ 2.5
    end

    # ── Overwrite ─────────────────────────────────────────────────────────
    @testset "Overwrite" begin
        f = joinpath(TMPDIR, "overwrite.h5")
        save_hdf5(f; x=1)
        save_hdf5(f; x=2)
        data = load_hdf5(f)
        @test data["x"] == 2
        @test !haskey(data, "y")  # old keys gone after overwrite
    end
end

println("\nAll Julia round-trip tests passed.")
