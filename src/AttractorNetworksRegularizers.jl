module AttractorNetworksRegularizers
using Calculus

abstract type Regularizer end

struct SigmoidalPlus <: Regularizer
    gain::Float64
    steepness::Float64
    p1::Float64
    pr::Float64
    function SigmoidalPlus(gain,steepness)
        p1=0.5gain
        pr = steepness/p1
        new(gain, steepness,p1,pr)
    end
end
function reg(x::Float64,sp::SigmoidalPlus)
     sp.p1*(1.0 + tanh(x*sp.pr))
end
function dreg(x::Float64,sp::SigmoidalPlus)
     _th = tanh(x*sp.pr)
    sp.steepness*(1. - _th*_th)
end
function ireg(y::Float64,sp::SigmoidalPlus)
    atanh( y/sp.p1 - 1.)/sp.pr
end


struct SigmoidalMinus <: Regularizer
    gain::Float64
    steepness::Float64
    p1::Float64
    pr::Float64
    function SigmoidalMinus(gain,steepness)
        p1=0.5gain
        pr = steepness/p1
        new(gain, steepness,p1,pr)
    end
end
function reg(x::Float64,sp::SigmoidalMinus)
     -sp.p1*(1.0 + tanh(x*sp.pr))
end
function dreg(x::Float64,sp::SigmoidalMinus)
     _th = tanh(x*sp.pr)
    sp.steepness*(_th*_th - 1.0)
end
function ireg(y::Float64,sp::SigmoidalMinus)
    atanh(-y/sp.p1 - 1.)/sp.pr
end
struct NoRegu <: Regularizer  end
reg(x::Float64,sp::NoRegu) = x
dreg(x::Float64,sp::NoRegu) = 1.0
ireg(x::Float64,sp::NoRegu) = x

struct SigmoidalBoth <: Regularizer
    gain_p::Float64
    gain_m::Float64
    steepness::Float64
    dh::Float64
    fact::Float64
    function SigmoidalBoth(gp,gm,steep)
        dh = (gp-gm)/2
        fact = steep/dh
        new(gp,gm,steep,dh,fact)
    end
end

function reg(x::Float64,sp::SigmoidalBoth)
  sp.gain_m+sp.dh*(1.0 + tanh(sp.fact*x))
end
function dreg(x::Float64,sp::SigmoidalBoth)
  th=tanh(sp.fact*x)
  sp.fact*sp.dh*(1.0 - th*th)
end
function ireg(y::Float64,sp::SigmoidalBoth)
  inv(sp.fact)*atanh( (y- sp.gain_m)/sp.dh -1.0 )
end

function gradient_test(x,regu::Regularizer)
  grad_num = Calculus.gradient(xx->reg(xx,regu), x )
  grad_an = dreg(x,regu)
  grad_num,grad_an, 2. * abs((grad_num-grad_an)/(grad_num + grad_an))
end

for fname in (:reg,:dreg)
  eval( :(
    function $fname(mah::Bool)
      123.4
    end ))
end

#=



# Pack that contains the parameters (arrays of floats)
# and the regularizers to use on each.
# It pushes everything into a long vector that can be used for optimization
struct RegularizerPack{Tf<:NamedTuple, Ti<:NamedTuple,
        Ta<:NamedTuple, Tt<:NamedTuple, V<:AbstractVector}
    regu_defs::Ta # symbol to regularizer
    unpacked::Tf  # network components in their natural shape
    indexes_local::Ti # which of the above should be packed
    indexes_packed::Ti # where they go in the packed vector
    g_unpacked::Tf  # space allocated for gradients
    regu_targets::Tt # used to link local indexes with regularizer type
    x_alloc::V  # allocation for finv, or for argument
    y_alloc::V  # allocation for f
    g_alloc::V # allocation for df
end
function RegularizerPack(regu_def,
    regu_attrib,
    unpacked) where Regu<:Regularizer

    # create the allocations for the gradient
    g_unpacked = NamedTuple{ keys(regu_attrib)}([ zero(unpacked[k])
                        for k in keys(regu_attrib) ])
    indexes_local=Dict{Symbol,Vector{Int64}}()
    indexes_packed=Dict{Symbol,Vector{Int64}}()
    _idx_loc = Vector{Int64}[]
    _idx_pac = Vector{Int64}[]
    all_symb = Symbol[]
    pack_size = 0
    for (key,rmat) in pairs(regu_attrib)
        idxs = let
            lin = LinearIndices(rmat)
            lin[.!ismissing.(rmat)]
        end
        push!(_idx_loc,idxs)
        n_els = length(idxs)
        push!(_idx_pac, collect(1:n_els) .+ pack_size)
        pack_size += n_els
        append!(all_symb , convert(Vector{Symbol}, rmat[idxs]))
    end
    indexes_local = NamedTuple{keys(regu_attrib)}(_idx_loc)
    indexes_packed = NamedTuple{keys(regu_attrib)}(_idx_pac)
    @assert length(all_symb) == pack_size
    x_alloc,y_alloc,g_alloc = (Vector{Float64}(undef,pack_size) for i in 1:3)

    regu_targets = let  _usymb = unique(all_symb)
        NamedTuple{Tuple(_usymb)}( [ findall(all_symb .==s) for s in _usymb ])
    end
    RegularizerPack(regu_def,unpacked,indexes_local,
        indexes_packed,g_unpacked, regu_targets, x_alloc,y_alloc,g_alloc)
end

Base.length(pr::RegularizerPack) = length(pr.x_alloc)
Base.keys(pr::RegularizerPack) = keys(pr.g_unpacked)

"""
Makes a tuple with same entries as val_dict,
that can be used to initialize a RegularizerPack object
"""
function make_regus_tuple(val_tuple)
    NamedTuple{keys(val_tuple)}(
     [ fill!(similar(arr,Union{Missing,Symbol}),missing) for arr in values(val_tuple)] )
end

"""
Converts the elements of x using the regularizers specified by the tuple
at the indexes specified by the other tuple
"""
function reg!(y,regu_defs,regu_targ,x)
    for (k,idx) in pairs(regu_targ)
        regu=regu_defs[k]
        reg!(view(y,idx),view(x,idx),regu)
    end
    y
end
function dreg!(y,regu_defs,regu_targ,x)
    for (k,idx) in pairs(regu_targ)
        regu=regu_defs[k]
        dreg!(view(y,idx),view(x,idx),regu)
    end
    y
end
function reginv!(y,regu_defs,regu_targ,x)
    for (k,idx) in pairs(regu_targ)
        regu=regu_defs[k]
        reginv!(view(y,idx),view(x,idx),regu)
    end
    y
end


"""
Auxiliary function: reads the internal unpacked tuple, or the one
    specified, and fills the interal y_alloc vector with the right order
"""
function _load_y(y,rp::RegularizerPack,unpacked)
    for (k,arr) in pairs(unpacked)
        idx_loc = rp.indexes_local[k]
        idx_pack = rp.indexes_packed[k]
        y[idx_pack] = arr[idx_loc]
    end
    nothing
end
function _load_y(rp::RegularizerPack,unpacked)
    y = rp.y_alloc
    _load_y(y, rp, unpacked)
end
function _load_y(rp::RegularizerPack)
    y = rp.y_alloc
    _load_y(y,rp,rp.unpacked)
    nothing
end
function _load_y(y::AbstractVector,
        rp::RegularizerPack,unp,symbs::AbstractVector{S}) where S<: Symbol
    for k in symbs
        arr=unp[k]
        idx_loc = rp.indexes_local[k]
        idx_pack = rp.indexes_packed[k]
        y[idx_pack] = arr[idx_loc]
    end
    nothing
end

"""
Auxiliary function that does the opposite of loading y
that is: takes the contents of y and writes them in a dictionary
It basically transfer the value in the opposite direction
"""
function _unload_y(rp::RegularizerPack,unpacked)
    y = rp.y_alloc
    for (k,arr) in pairs(unpacked)
        idx_loc = rp.indexes_local[k]
        idx_pack = rp.indexes_packed[k]
        arr[idx_loc] = y[idx_pack]   # literally the same of load, except I invert this!
    end
    nothing
end
function _unload_y(rp::RegularizerPack)
    _unload_y(rp,rp.unpacked)
    nothing
end
"""
Needed for initialization: reads the internal dictionary,
    uses it to produce a parameter vector x
"""
function pack(rp::RegularizerPack)
    _load_y(rp)
    reginv!(rp.x_alloc,rp.regu_defs,rp.regu_targets ,rp.y_alloc)
end

"""
After a computation of the gradients into the
arrays in g_unpacked, call this function to  pack the elements
of the unpacked arrays in the vector specified
"""
function g_pack!(g::AbstractArray{T,1},rp::RegularizerPack) where T<:Real
    _load_y(g,rp,rp.g_unpacked)
end
function g_pack!(g::AbstractArray{T,1},
        rp::RegularizerPack,symbs::AbstractVector{S}) where {T<:Real,S<:Symbol}
    _load_y(g,rp,rp.g_unpacked,symbs)
end

"""
Inverts the contents of the vector x and
writes them  in the unpacked dictionary
elements of the packaging
"""
function unpack(x::AbstractVector,rp::RegularizerPack)
    copyto!(rp.x_alloc,x)
    reg!(rp.y_alloc,rp.regu_defs,rp.regu_targets , x)
    _unload_y(rp)
    nothing
end

function unpack_grad(x::AbstractVector,rp::RegularizerPack)
    unpack(x,rp) # this updates x_alloc and y_alloc
    dreg!(rp.g_alloc, rp.regu_defs,rp.regu_targets,rp.x_alloc) # this returns g_alloc
end

"""
Constructors that also fills the packed part, and
returns the inital x vector along with the packed object
"""
function RegularizerPack_init(regu_defs, regu_attrib, unpacked)
    rp=RegularizerPack(regu_defs,regu_attrib,unpacked)
    x0 = pack(rp)  # careful, this is also x_alloc... maybe I should make a copy?
    (rp,x0)
end

function gradient_test(rp::RegularizerPack,x::AbstractVector)
    @assert length(rp) == length(x)
    g_an = unpack_grad(x,rp) # careful, this is g_alloc
    eps_test=1E-8
    # all grads are independent, so I can vectorize
    xpm=copy(x)
    xpm .+= eps_test
    f_p=similar(xpm)
    reg!(f_p, rp.regu_defs,rp.regu_targets, xpm)
    xpm .-= 2eps_test
    f_m=similar(xpm)
    reg!(f_m,rp.regu_defs,rp.regu_targets, xpm)
    g_num = @. (f_p-f_m) / (2*eps_test)
    diff_g = @. abs( 2. * (g_num-g_an)/(g_num+g_an) )
    g_num , g_an , diff_g
end
=#
end # module