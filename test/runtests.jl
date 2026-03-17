using Test

include(joinpath(@__DIR__, "..", "src", "HDF5Manager.jl"))
using .HDF5Manager

const TMPDIR = mktempdir()

@testset "HDF5Manager Julia Round-Trip" begin

    @testset "Scalars" begin
        f = joinpath(TMPDIR, "scalars.h5")
        save_hdf5(f; int_val=42, float_val=3.14, bool_val=true)
        data = load_hdf5(f)
        @test data["int_val"] == 42
        @test data["float_val"] ≈ 3.14
        @test data["bool_val"] == true
    end

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

    @testset "2D array shape preserved" begin
        f = joinpath(TMPDIR, "shape2d.h5")
        m = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3
        save_hdf5(f; mat=m)
        data = load_hdf5(f)
        @test size(data["mat"]) == (2, 3)
        @test data["mat"] ≈ m
    end

    @testset "3D array shape preserved" begin
        f = joinpath(TMPDIR, "shape3d.h5")
        t = rand(2, 3, 4)
        save_hdf5(f; tensor=t)
        data = load_hdf5(f)
        @test size(data["tensor"]) == (2, 3, 4)
        @test data["tensor"] ≈ t
    end

    @testset "Complex scalars" begin
        f = joinpath(TMPDIR, "complex_scalar.h5")
        z = 1.5 + 2.5im
        save_hdf5(f; z=z)
        data = load_hdf5(f)
        @test data["z"] ≈ z
    end

    @testset "Complex arrays" begin
        f = joinpath(TMPDIR, "complex_arrays.h5")
        cv = [1+2im, 3+4im, 5+6im]
        cm = [1+0im 2+1im; 3+2im 4+3im]
        save_hdf5(f; cvec=cv, cmat=cm)
        data = load_hdf5(f)
        @test data["cvec"] ≈ cv
        @test data["cmat"] ≈ cm
        @test size(data["cmat"]) == (2, 2)
    end

    @testset "Dict" begin
        f = joinpath(TMPDIR, "dict.h5")
        d = Dict("a" => 1.0, "b" => [10, 20, 30])
        save_hdf5(f; params=d)
        data = load_hdf5(f)
        @test data["params"]["a"] == 1.0
        @test data["params"]["b"] == [10, 20, 30]
    end

    @testset "NamedTuple" begin
        f = joinpath(TMPDIR, "namedtuple.h5")
        nt = (alpha=0.5, beta=1.0, data=[1, 2, 3])
        save_hdf5(f; config=nt)
        data = load_hdf5(f)
        @test data["config"].alpha == 0.5
        @test data["config"].beta == 1.0
        @test data["config"].data == [1, 2, 3]
    end

    @testset "Multiple variables" begin
        f = joinpath(TMPDIR, "multi.h5")
        save_hdf5(f; a=1, b=[1.0, 2.0], c=3+4im)
        data = load_hdf5(f)
        @test data["a"] == 1
        @test data["b"] ≈ [1.0, 2.0]
        @test data["c"] ≈ 3 + 4im
    end

    @testset "load_hdf5_item" begin
        f = joinpath(TMPDIR, "multi.h5")
        b = load_hdf5_item(f, "b")
        @test b ≈ [1.0, 2.0]
        @test_throws ErrorException load_hdf5_item(f, "nonexistent")
    end

    @testset "list_hdf5_variables" begin
        f = joinpath(TMPDIR, "multi.h5")
        vars = list_hdf5_variables(f)
        @test sort(vars) == sort(["a", "b", "c"])
    end

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

    @testset "Overwrite" begin
        f = joinpath(TMPDIR, "overwrite.h5")
        save_hdf5(f; x=1)
        save_hdf5(f; x=2)
        data = load_hdf5(f)
        @test data["x"] == 2
        @test !haskey(data, "y")
    end

    @testset "storage_order attribute written" begin
        f = joinpath(TMPDIR, "storage_order.h5")
        save_hdf5(f; x=1)
        using HDF5
        h5open(f, "r") do file
            @test read(attributes(file)["storage_order"]) == "column_major"
        end
    end

    @testset "Legacy file without storage_order (no transpose)" begin
        f = joinpath(TMPDIR, "legacy.h5")
        m = [1.0 2.0; 3.0 4.0]
        using HDF5
        h5open(f, "w") do file
            file["mat"] = m
        end
        data = load_hdf5(f)
        @test data["mat"] ≈ m
    end
end

println("\nAll Julia round-trip tests passed.")
