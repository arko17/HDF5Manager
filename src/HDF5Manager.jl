"""
    HDF5Manager.jl

A module for saving and loading Julia objects in HDF5 format with full type preservation.
Supports Arrays (including complex), scalars, and metadata.
"""

module HDF5Manager

using HDF5
using Base: eltype

export save_hdf5, load_hdf5, load_hdf5_item, list_hdf5_variables

# Helper function to save complex arrays
function _save_array(file, name::String, arr::AbstractArray)
    # Convert to concrete array if needed (e.g., from reshape)
    arr = Array(arr)

    # Determine if complex
    is_complex = eltype(arr) <: Complex

    if is_complex
        # Split complex array into real and imaginary parts
        real_part = real.(arr)
        imag_part = imag.(arr)

        # Create a group for this complex array
        g = create_group(file, name)
        g["real"] = real_part
        g["imag"] = imag_part

        # Store metadata
        attributes(g)["is_complex"] = true
        attributes(g)["element_type"] = string(eltype(arr))
        attributes(g)["shape"] = collect(size(arr))
    else
        # Real-valued array
        file[name] = arr
        # Get the dataset object to attach attributes
        dset = file[name]
        attributes(dset)["shape"] = collect(size(arr))
        attributes(dset)["jl_type"] = string(typeof(arr))
    end
end

# Helper function to save complex scalars
function _save_complex(file, name::String, val::Complex)
    g = create_group(file, name)
    g["real"] = real(val)
    g["imag"] = imag(val)
    attributes(g)["is_complex"] = true
    attributes(g)["element_type"] = string(typeof(val))
end

# Helper function to save a single variable
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
        # Scalar (Int, Float, etc.)
        parent[name] = value
        # Try to attach type info to the created dataset
        if haskey(parent, name)
            dset = parent[name]
            attributes(dset)["jl_type"] = string(typeof(value))
        end
    end
end

"""
    save_hdf5(filepath::String; kwargs...)

Save multiple Julia objects to an HDF5 file with type metadata.

Automatically detects and handles:
- Arrays (real and complex, any dimension)
- Complex scalars
- Numeric scalars (Int, Float, Bool)

# Arguments
- `filepath::String`: Path to the HDF5 file to create/overwrite
- Keyword arguments: Variable name => Variable value pairs to save

# Example
```julia
arr = [1, 2, 3]
complex_arr = [1+2im, 3+4im]
scalar = 42

save_hdf5("data.h5"; arr=arr, complex_arr=complex_arr, scalar=scalar)
```
"""
function save_hdf5(filepath::String; kwargs...)
    h5open(filepath, "w") do file
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

    # Check if object exists
    if !haskey(file, name)
        error("Variable '\$name' not found in HDF5 file")
    end

    obj = file[name]

    # Check if it's a group (complex or array)
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
        # Dataset - check for type metadata
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

# Map stored type strings back to Julia types for complex reconstruction
const _COMPLEX_TYPE_MAP = Dict(
    "ComplexF64" => ComplexF64, "ComplexF32" => ComplexF32,
    "ComplexF16" => ComplexF16, "Complex{Float64}" => ComplexF64,
    "Complex{Float32}" => ComplexF32, "Complex{Float16}" => ComplexF16,
    "Complex{Int64}" => Complex{Int64}, "Complex{Int32}" => Complex{Int32},
)

function _load_variable(obj)
    if isa(obj, HDF5.Group)
        return _load_group(obj)
    else
        return _load_dataset(obj)
    end
end

function _load_group(group::HDF5.Group)
    attrs = attributes(group)
    attr_keys = keys(attrs)

    # Complex number / array
    if "is_complex" in attr_keys && Bool(read(attrs["is_complex"]))
        real_part = read(group["real"])
        imag_part = read(group["imag"])
        return complex.(real_part, imag_part)
    end

    # Dict
    if "jl_type" in attr_keys && read(attrs["jl_type"]) == "Dict"
        d = Dict{String, Any}()
        for k in keys(group)
            d[k] = _load_variable(group[k])
        end
        return d
    end

    # NamedTuple
    if "jl_type" in attr_keys && read(attrs["jl_type"]) == "NamedTuple"
        ks = Symbol[]
        vs = Any[]
        for k in keys(group)
            push!(ks, Symbol(k))
            push!(vs, _load_variable(group[k]))
        end
        return NamedTuple{Tuple(ks)}(Tuple(vs))
    end

    # Fallback: treat unknown group as Dict
    d = Dict{String, Any}()
    for k in keys(group)
        d[k] = _load_variable(group[k])
    end
    return d
end

function _load_dataset(dataset::HDF5.Dataset)
    return read(dataset)
end


"""
    load_hdf5(filepath::String) -> Dict{String, Any}

Load all variables from an HDF5 file saved with `save_hdf5`.

Returns a `Dict{String, Any}` mapping variable names to their values.
Complex arrays/scalars are automatically reconstructed.

# Example
```julia
data = load_hdf5("data.h5")
data["arr"]          # [1, 2, 3]
data["complex_arr"]  # [1+2im, 3+4im]
data["scalar"]       # 42
```
"""
function load_hdf5(filepath::String)
    result = Dict{String, Any}()
    h5open(filepath, "r") do file
        for name in keys(file)
            result[name] = _load_variable(file[name])
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
        return _load_variable(file[name])
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
