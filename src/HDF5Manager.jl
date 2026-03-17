"""
    HDF5Manager.jl

A module for saving and loading Julia objects in HDF5 format with full type preservation.
Supports Arrays (including complex), scalars, and metadata.
Automatically handles column-major/row-major transpose when reading files written by Python.
"""

module HDF5Manager

using HDF5
using Base: eltype

export save_hdf5, load_hdf5, load_hdf5_item, list_hdf5_variables

const STORAGE_ORDER = "column_major"

# ─── Transpose helper ────────────────────────────────────────────────────────

"""Return whether arrays from this file need transposing (written by row-major writer)."""
function _needs_transpose(file)
    root_attrs = attributes(file)
    if "storage_order" in keys(root_attrs)
        return read(root_attrs["storage_order"]) == "row_major"
    end
    return false  # fallback: no transpose
end

"""Transpose an array if ndim >= 2, otherwise return as-is."""
function _maybe_transpose(arr::AbstractArray, transpose::Bool)
    if transpose && ndims(arr) >= 2
        return permutedims(arr, reverse(ntuple(identity, ndims(arr))))
    end
    return arr
end

_maybe_transpose(val, ::Bool) = val  # non-array passthrough

# ─── Save helpers ─────────────────────────────────────────────────────────────

function _save_array(file, name::String, arr::AbstractArray)
    arr = Array(arr)
    is_complex = eltype(arr) <: Complex

    if is_complex
        real_part = real.(arr)
        imag_part = imag.(arr)

        g = create_group(file, name)
        g["real"] = real_part
        g["imag"] = imag_part

        attributes(g)["is_complex"] = true
        attributes(g)["element_type"] = string(eltype(arr))
        attributes(g)["shape"] = collect(size(arr))
    else
        file[name] = arr
        dset = file[name]
        attributes(dset)["shape"] = collect(size(arr))
        attributes(dset)["jl_type"] = string(typeof(arr))
    end
end

function _save_complex(file, name::String, val::Complex)
    g = create_group(file, name)
    g["real"] = real(val)
    g["imag"] = imag(val)
    attributes(g)["is_complex"] = true
    attributes(g)["element_type"] = string(typeof(val))
end

function _save_variable(parent, name::String, value)
    if isa(value, AbstractArray)
        _save_array(parent, name, value)
    elseif isa(value, Complex)
        _save_complex(parent, name, value)
    elseif isa(value, AbstractDict)
        g = create_group(parent, name)
        attributes(g)["jl_type"] = "Dict"
        for (k, v) in value
            _save_variable(g, string(k), v)
        end
    elseif isa(value, NamedTuple)
        g = create_group(parent, name)
        attributes(g)["jl_type"] = "NamedTuple"
        for (k, v) in pairs(value)
            _save_variable(g, string(k), v)
        end
    else
        parent[name] = value
        if haskey(parent, name)
            dset = parent[name]
            attributes(dset)["jl_type"] = string(typeof(value))
        end
    end
end

"""
    save_hdf5(filepath::String; kwargs...)

Save multiple Julia objects to an HDF5 file with type metadata.

A `storage_order = "column_major"` attribute is written to the root group so that
readers in other languages can automatically transpose multi-dimensional arrays.

# Example
```julia
save_hdf5("data.h5"; arr=[1,2,3], z=1+2im, scalar=42)
```
"""
function save_hdf5(filepath::String; kwargs...)
    h5open(filepath, "w") do file
        attributes(file)["storage_order"] = STORAGE_ORDER
        for (name, value) in kwargs
            _save_variable(file, String(name), value)
        end
    end
end


"""
    get_type_info(file, name::String)::Dict

Extract type information stored in the HDF5 file.
"""
function get_type_info(file, name::String)::Dict
    info = Dict()
    if !haskey(file, name)
        error("Variable '\$name' not found in HDF5 file")
    end
    obj = file[name]
    if isa(obj, HDF5.Group)
        attrs = attributes(obj)
        if "is_complex" in keys(attrs)
            info["is_complex"] = read(attrs["is_complex"])
            info["element_type"] = read(attrs["element_type"])
            if "shape" in keys(attrs)
                info["shape"] = read(attrs["shape"])
            end
        end
    else
        file_attrs = attributes(file["/"])
        if (name * "_jl_type") in keys(file_attrs)
            info["jl_type"] = read(file_attrs[name * "_jl_type"])
        end
        if (name * "_shape") in keys(file_attrs)
            info["shape"] = read(file_attrs[name * "_shape"])
        end
    end
    return info
end


# ─── Load helpers ─────────────────────────────────────────────────────────────

const _COMPLEX_TYPE_MAP = Dict(
    "ComplexF64" => ComplexF64, "ComplexF32" => ComplexF32,
    "ComplexF16" => ComplexF16, "Complex{Float64}" => ComplexF64,
    "Complex{Float32}" => ComplexF32, "Complex{Float16}" => ComplexF16,
    "Complex{Int64}" => Complex{Int64}, "Complex{Int32}" => Complex{Int32},
)

function _load_variable(obj, transpose::Bool)
    if isa(obj, HDF5.Group)
        return _load_group(obj, transpose)
    else
        return _load_dataset(obj, transpose)
    end
end

function _load_group(group::HDF5.Group, transpose::Bool)
    attrs = attributes(group)
    attr_keys = keys(attrs)

    # Complex number / array
    if "is_complex" in attr_keys && Bool(read(attrs["is_complex"]))
        real_part = read(group["real"])
        imag_part = read(group["imag"])
        result = complex.(real_part, imag_part)
        return _maybe_transpose(result, transpose)
    end

    # Dict
    if "jl_type" in attr_keys && read(attrs["jl_type"]) == "Dict"
        d = Dict{String, Any}()
        for k in keys(group)
            d[k] = _load_variable(group[k], transpose)
        end
        return d
    end

    # NamedTuple
    if "jl_type" in attr_keys && read(attrs["jl_type"]) == "NamedTuple"
        ks = Symbol[]
        vs = Any[]
        for k in keys(group)
            push!(ks, Symbol(k))
            push!(vs, _load_variable(group[k], transpose))
        end
        return NamedTuple{Tuple(ks)}(Tuple(vs))
    end

    # Fallback: treat unknown group as Dict
    d = Dict{String, Any}()
    for k in keys(group)
        d[k] = _load_variable(group[k], transpose)
    end
    return d
end

function _load_dataset(dataset::HDF5.Dataset, transpose::Bool)
    data = read(dataset)
    return _maybe_transpose(data, transpose)
end


"""
    load_hdf5(filepath::String) -> Dict{String, Any}

Load all variables from an HDF5 file saved with `save_hdf5`.

Multi-dimensional arrays are automatically transposed when the file was written
by a row-major language (Python). If no `storage_order` attribute is present
(legacy files), arrays are returned as-is.

# Example
```julia
data = load_hdf5("data.h5")
data["arr"]          # [1, 2, 3]
data["complex_arr"]  # [1+2im, 3+4im]
```
"""
function load_hdf5(filepath::String)
    result = Dict{String, Any}()
    h5open(filepath, "r") do file
        transpose = _needs_transpose(file)
        for name in keys(file)
            result[name] = _load_variable(file[name], transpose)
        end
    end
    return result
end


"""
    load_hdf5_item(filepath::String, name::String)

Load a single variable by name from an HDF5 file.

# Example
```julia
arr = load_hdf5_item("data.h5", "arr")
```
"""
function load_hdf5_item(filepath::String, name::String)
    h5open(filepath, "r") do file
        if !haskey(file, name)
            error("Variable '\$name' not found in \$filepath")
        end
        transpose = _needs_transpose(file)
        return _load_variable(file[name], transpose)
    end
end


"""
    list_hdf5_variables(filepath::String) -> Vector{String}

List all top-level variable names stored in an HDF5 file.

# Example
```julia
list_hdf5_variables("data.h5")  # ["arr", "complex_arr", "scalar"]
```
"""
function list_hdf5_variables(filepath::String)
    h5open(filepath, "r") do file
        return collect(keys(file))
    end
end

end  # module HDF5Manager
