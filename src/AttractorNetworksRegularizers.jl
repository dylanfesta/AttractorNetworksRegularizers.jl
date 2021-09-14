module AttractorNetworksRegularizers
export RegularizerPack

abstract type Regularizer{R} end
Base.Broadcast.broadcastable(g::Regularizer)=Ref(g)


struct NoRegu{R} <: Regularizer{R}  end
@inline function (sp::NoRegu{R})(x::R) where R
  return x
end
NoRegu() = NoRegu{Float64}()

dreg(x::R,sp::NoRegu{R}) where R = one(R)
ireg(x::R,sp::NoRegu{R}) where R = x

in_bounds(x::Real,r::NoRegu) = isfinite(x)


struct SigmoidalPlus{R} <: Regularizer{R}
    gain::R
    steepness::R
    p1::R
    pr::R
    function SigmoidalPlus(gain::R,steepness::R) where R
        p1=0.5gain
        pr = steepness/p1
        new{R}(gain, steepness,p1,pr)
    end
end
@inline function (sp::SigmoidalPlus{R})(x::R) where R
  return sp.p1*(1.0 + tanh(x*sp.pr))
end
@inline function dreg(x::R,sp::SigmoidalPlus{R}) where R
     _th = tanh(x*sp.pr)
    return sp.steepness*(1. - _th*_th)
end
@inline function ireg(y::R,sp::SigmoidalPlus{R}) where R
    return atanh( y/sp.p1 - 1.)/sp.pr
end

in_bounds(y::R,r::SigmoidalPlus{R}) where R = (zero(R) < y < r.gain)


struct SigmoidalMinus{R} <: Regularizer{R}
    gain::R
    steepness::R
    p1::R
    pr::R
    function SigmoidalMinus(gain::R,steepness::R) where R<:Real
        p1=0.5gain
        pr = steepness/p1
        new{R}(gain, steepness,p1,pr)
    end
end
@inline function (sp::SigmoidalMinus{R})(x::R) where R
  return -sp.p1*(1.0 + tanh(x*sp.pr))
end
@inline function dreg(x::R,sp::SigmoidalMinus{R}) where R
     _th = tanh(x*sp.pr)
    return sp.steepness*(_th*_th - 1.0)
end
@inline function ireg(y::R,sp::SigmoidalMinus{R}) where R
    return atanh(-y/sp.p1 - 1.)/sp.pr
end
in_bounds(y::R,r::SigmoidalMinus{R}) where R = (-r.gain < y < zero(R))


struct SigmoidalBoth{R} <: Regularizer{R}
    gain_m::R
    gain_p::R
    steepness::R
    dh::R
    fact::R
    function SigmoidalBoth(gm::R,gp::R,steep::R) where R<:Real
      @assert gm<gp "first argument should be the lower bound"
      dh = (gp-gm)/2
      fact = steep/dh
      new{R}(gm,gp,steep,dh,fact)
    end
end

@inline function (sp::SigmoidalBoth{R})(x::R) where R
  return sp.gain_m+sp.dh*(1.0 + tanh(sp.fact*x))
end
@inline function dreg(x::R,sp::SigmoidalBoth{R}) where R
  th=tanh(sp.fact*x)
  return sp.fact*sp.dh*(1.0 - th*th)
end
@inline function ireg(y::R,sp::SigmoidalBoth{R}) where R
  return atanh( (y- sp.gain_m)/sp.dh -1.0 )/sp.fact
end
in_bounds(y::R,r::SigmoidalBoth{R}) where R = (r.gain_m < y < r.gain_p)

# vectorized versions!
@inline function reg!(dest::AbstractVector{R},x::AbstractVector{R},
    reg::Regularizer{R}) where R
  for (i,_x) in enumerate(x)
    dest[i]=reg(_x)
  end
  return dest
end
@inline function dreg!(dest::AbstractVector{R},x::AbstractVector{R},
    reg::Regularizer{R}) where R
  for (i,_x) in enumerate(x)
    dest[i]=dreg(_x,reg)
  end
  return dest
end
function dreg!(x::AbstractVector{R},reg::Regularizer{R}) where R
  return dreg!(simular(x),x,reg)
end
@inline function ireg!(dest::AbstractVector{R},x::AbstractVector{R},
    reg::Regularizer{R}) where R
  for (i,_x) in enumerate(x)
    dest[i]=ireg(_x,reg)
  end
  return dest
end
function ireg(x::AbstractVector{R},reg::Regularizer{R}) where R
  return ireg!(similar(x),x,reg)
end


struct RegularizedUnit{D,N,R<:Number}
  M::Array{R,D}
  Mg::Array{R,D}
  regularizers::NTuple{N,Regularizer{R}}
  globals::NTuple{N,Vector{Int64}}
  locals::NTuple{N,Vector{Int64}}
  xs::NTuple{N,Vector{R}}
  ys::NTuple{N,Vector{R}}
  gs::NTuple{N,Vector{R}}
end

function Base.length(ru::RegularizedUnit)
  return sum(length.(ru.xs))
end


# move values from global vector to M or Mg , and other way round

function partial_pack!(x_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  for (x,glob) in zip(ru.xs,ru.globals)
    for (k,g) in enumerate(glob)
      x_glo[g] = x[k]
    end
  end
  return nothing
end
function partial_unpack!(x_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  for (x,glob) in zip(ru.xs,ru.globals)
    for (k,g) in enumerate(glob)
      x[k] = x_glo[g]
    end
  end
  return nothing
end
function partial_packgrad!(grad_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  for (gra,glob) in zip(ru.gs,ru.globals)
    for (k,g) in enumerate(glob)
      grad_glo[g] = gra[k]
    end
  end
  return nothing
end

function local_pack!(ru::RegularizedUnit)
  for (y,loc) in zip(ru.ys,ru.locals)
    for (k,l) in enumerate(loc)
      y[k] = ru.M[l]
    end
  end
  return nothing
end
function local_unpack!(ru::RegularizedUnit)
  for (y,loc) in zip(ru.ys,ru.locals)
    for (k,l) in enumerate(loc)
      ru.M[l] = y[k]
    end
  end
  return nothing
end
function local_packgrad_chain!(ru::RegularizedUnit)
  for (g,loc) in zip(ru.gs,ru.locals)
    for (k,l) in enumerate(loc)
      g[k] *= ru.Mg[l]
    end
  end
  return nothing
end

# compute regularized functions
function reg!(ru::RegularizedUnit)
  for (reg,x,y) in zip(ru.regularizers,ru.xs,ru.ys)
    reg!(y,x,reg)
  end
  return nothing
end
function dreg!(ru::RegularizedUnit)
  for (reg,x,g) in zip(ru.regularizers,ru.xs,ru.gs)
    dreg!(g,x,reg)
  end
  return nothing
end
function ireg!(ru::RegularizedUnit)
  for (reg,x,y) in zip(ru.regularizers,ru.xs,ru.ys)
    ireg!(x,y,reg)
  end
  return nothing
end

function pack!(x_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  # copies the M elements to y 
  local_pack!(ru)
  #  x = ireg(y) , now x is the unbound variable
  ireg!(ru)
  # copies the xs into x_glo
  partial_pack!(x_glo,ru)
  return nothing
end
function unpack!(x_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  # copies the x_glo to x
  partial_unpack!(x_glo,ru)
  # y = reg(x) , compute the bound variable
  reg!(ru)
  # copies the y into the M array
  local_unpack!(ru)
  return nothing
end

function packgrad_chain!(g_glo::Vector{R},ru::RegularizedUnit{D,N,R}) where {D,N,R}
  # g = f'(x)
  dreg!(ru)
  # dM/dx = dM/dy * f'(x) 
  local_packgrad_chain!(ru)
  # copies g to g_glo
  partial_packgrad!(g_glo,ru)
  return nothing
end

struct RegularizerPack{N,R<:Real}
  units::NTuple{N,RegularizedUnit}
  x_global::Vector{R}  # variables in unbound form (can be bypassed)
  g_global::Vector{R}
end
Base.length(rp::RegularizerPack) = length(rp.x_global)


# bypass x_global
function pack!(x_global::Vector{<:Real},regp::RegularizerPack)
  for u in regp.units
    pack!(x_global,u)
  end
  return nothing
end
pack!(regp::RegularizerPack) = pack!(regp.x_global,regp)

function unpack!(x_global::Vector{<:Real},regp::RegularizerPack)
  for u in regp.units
    unpack!(x_global,u)
  end
  return nothing
end
unpack!(regp::RegularizerPack) = unpack!(regp.x_global,regp)

function pack_gradient_chain!(g_global::Vector{<:Real},regp::RegularizerPack)
  for u in regp.units
    packgrad_chain!(g_global,u)
  end
  return nothing
end
pack_gradient_chain!(regp::RegularizerPack) = pack_gradient_chain!(regp.g_global,regp)

#######################
# constructors!

## sanity check functions
#  same size
_check_mat1(M1,M2) = (@assert size(M1) == size(M2);)
# no sumperimpositions
function _check_mat2(Ms)
  nm = length(Ms) 
  if nm==1
    return nothing
  end
  Mref = copy(Ms[1])
  for j in 2:nm
    @assert !any(Mref .& Ms[j]) "Superimposed indexes!"
    Mref = Mref .| Ms[j]
  end
  return nothing
end
# both
function _check_mats(M,Ms)
  _check_mat1.(Ref(M),Ms)
  _check_mat2(Ms)
end
# check that regularizers are within boundaries
function regularizers_in_bounds(ru::RegularizedUnit)
  # matrix elements sent to ys
  local_pack!(ru)
  for (regu,y) in zip(ru.regularizers,ru.ys)
    for yi in y
      if !in_bounds(yi,regu)
        @error "$yi is offbound!"
        return false
      end
    end
  end
  return true
end

# globals of different regularized units should be different!
# this applies an offset to all elements to shift them forward
function fix_global_offset(ru::RegularizedUnit,glob_offset::Integer)
  for glo in ru.globals
    glo .+= glob_offset
  end
  return nothing
end


# M , and pairs (Regularizer, Mask)
function RegularizedUnit(M::Array{R},Mg::Array{R},
    (rlist::Tuple{Re,B} where {Re<:Regularizer{R},B<:BitArray} )...;
    global_offset::Integer=0) where {R<:Real}
  # check sizes and indexes
  _check_mats(M,getindex.(rlist,2))
  rlist = filter(rl-> count(rl[2])>0 ,rlist)
  @assert !isempty(rlist)
  regularizers = Tuple(r[1] for r in rlist)
  ns = map(r->count(r[2]),rlist)
  npre = global_offset
  globals = map(ns) do n
    ret = collect(1:n) .+ npre
    npre += n 
    return ret
  end
  globals = Tuple(globals)
  locals_cart = [findall(r[2]) for r in rlist]
  lin = LinearIndices(M)
  locals = Tuple(lin[cart] for cart in locals_cart)
  xs = ntuple(k -> Vector{R}(undef,ns[k]),length(ns))
  ys = deepcopy(xs)
  gs = deepcopy(xs)
  ret = RegularizedUnit(M,Mg,regularizers,globals,locals,xs,ys,gs)
  @assert regularizers_in_bounds(ret) "The matrix is outside the regularizers domain!"
  return ret
end

# if no bit array specified, just regularize all
function RegularizedUnit(M::Array{R},Mg::Array{R},reg::Regularizer{R};
    global_offset::Integer=0) where R<:Real
  return RegularizedUnit(M,Mg,(reg,trues(size(M))); global_offset=global_offset)
end

function regularizers_in_bounds(rp::RegularizerPack)
  return all(u->regularizers_in_bounds(u),rp.units)
end

function good_global_counter(rp::RegularizerPack)
  gl=Vector{Int64}[]
  for unit in rp.units
    for glo in unit.globals
      push!(gl,glo)
    end
  end
  gl_all = sort(vcat(gl...))
  return length(gl_all) == gl_all[end] == length(rp)
end

function RegularizerPack(units...)
  ntot = sum(length.(units))
  x_global = Vector{Float64}(undef,ntot)
  g_global = similar(x_global)
  ret = RegularizerPack(units,x_global,g_global)
  if ! good_global_counter(ret)
    c=0
    for unit in units
      fix_global_offset(unit,c)
      c += length(unit)
    end
  end
  @assert good_global_counter(ret)
  @assert regularizers_in_bounds(ret)
  return ret
end



end # of module