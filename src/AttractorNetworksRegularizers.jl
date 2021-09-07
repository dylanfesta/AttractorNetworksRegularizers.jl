module AttractorNetworksRegularizers
using Calculus

abstract type Regularizer{R} end
Base.Broadcast.broadcastable(g::Regularizer)=Ref(g)


struct NoRegu{R} <: Regularizer{R}  end
@inline function (sp::NoRegu{R})(x::R) where R
  return x
end
dreg(x::R,sp::NoRegu{R}) where R = one(R)
ireg(x::R,sp::NoRegu{R}) where R = x

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


struct SigmoidalBoth{R} <: Regularizer{R}
    gain_p::R
    gain_m::R
    steepness::R
    dh::R
    fact::R
    function SigmoidalBoth(gp::R,gm::R,steep::R) where R<:Real
        dh = (gp-gm)/2
        fact = steep/dh
        new{R}(gp,gm,steep,dh,fact)
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


struct RegularizedUnit{D,N, R<:Number}
  M::Array{R,D}
  Mg::Array{R,D}
  regularizers::NTuple{N,Regularizer{R}}
  globals::NTuple{N,Int64}
  locals::NTuple{N,Int64}
  xs::NTuple{N,Vector{R}}
  ys::NTuple{N,Vector{R}}
  gs::NTuple{N,Vector{R}}
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
  for (g,glob) in zip(ru.gs,ru.globals)
    for (k,g) in enumerate(glob)
      grad_glo[g] = g[k]
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
function local_packgrad!(ru::RegularizedUnit)
  for (g,loc) in zip(ru.gs,ru.locals)
    for (k,l) in enumerate(loc)
      g[k] = ru.Mg[l]
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
  for (reg,x,y) in zip(ru.regularizers,ru.xs,ru.ys)
    dreg!(y,x,reg)
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
  # copies the Mg elements to g
  local_packgrad!(ru)
  # chain rule  ****(TODO)****
  # copies g to g_glo
  partial_packgrad!(g_glo,ru)
  return nothing
end



struct RegularizerPack{N}
  units::NTuple{N,RegularizedUnit}
  x::Vector{R}  # variables in unbound form
  y::Vector{R}  # variables in regularized form
  xgrad::Vector{R}  # gradient of regularizer w.r.t. y
  xgrad_units::Vector{R}  # allocated space for gradient of variables
  regus::Vector{Regularizer{R}}  # regularizers, element by element
end
Base.length(rp::RegularizerPack) = sum(length.(rp.units))


#=
abstract type AbstractUnit{N,R,I} end
Base.Broadcast.broadcastable(x::AbstractUnit)=Ref(g)

function get_unit(nm::Symbol,us::Vector{V}) where V<:AbstractUnit
  idx=findfirst(u->u.name==nm,us)
  if isnothing(idx)
    error("element $nm not found!")
  end
  return us[idx]
end




struct RegularizedUnit{N,R,I} <: AbstractUnit{N,R,I}
  name::Symbol
  M::Array{R,N}
  Mg::Array{R,N} #allocated space for gradient
  loc::Vector{I}
  glo::Vector{I}
end
Base.Broadcast.broadcastable(x::RegularizedUnit)=Ref(x)
Base.length(ru::RegularizedUnit)=length(ru.loc)

function find_nonmissing(bu::Array{Union{Missing,Symbol}})
  lin = LinearIndices(bu)
  idx = findall(.!(ismissing.(bu)))
  return lin[idx]
end
function RegularizedUnit(name::Symbol,A::Array{R,N},
    A_regus::Array{Union{Missing,Symbol},N}; offset::Integer=0) where {N,R}
  loc=find_nonmissing(A_regus)
  glo=offset .+ collect(1:length(loc))
  return RegularizedUnit(name,A,similar(A),loc,glo)
end
function RegularizedUnit(name::Symbol,A::Array{<:Real,N},
    A_regus::Array{Union{Missing,Symbol},N},pre::RegularizedUnit) where N
  return RegularizedUnit(name,A,A_regus;offset=length(pre))
end

function pack!(x::Vector{R},ru::RegularizedUnit{N,R,I}) where {N,R,I}
  # note: works well with this, or with views
  for (g,l) in zip(ru.glo,ru.loc)
    x[g] = ru.M[l]
  end
  return nothing
end
function unpack!(ru::RegularizedUnit{N,R,I},x::Vector{R}) where {N,R,I}
  for (g,l) in zip(ru.glo,ru.loc)
    ru.M[l]=x[g]
  end
  return nothing
end

function pack_grad_array!(x::Vector{R},ru::RegularizedUnit{N,R,I}) where {N,R,I}
  for (g,l) in zip(ru.glo,ru.loc)
    x[g]=ru.Mg[l]
  end
  return nothing
end

function make_empty_selection(regu_dict::Dict)
  out=Dict{Symbol,Array{Union{Missing,Symbol}}}()
  for (k,v) in pairs(regu_dict)
    ss =  similar(v,Union{Missing,Symbol})
    out[k] = fill!(ss,missing)
  end
  return out
end

#=
The final goal here is to build the gradient of the packed x, for different cost
 component.  In practice each single cost component might take only a subset of elements.
 So I use a UnitSelector to specify the subset. Then the key operation is to
 pack the gradient for that subset, and propagate it.
=#
struct UnitSelector{N,R,I} <: AbstractUnit{N,R,I}
  name::Symbol
  S::BitArray{N}
  Mg::Array{R,N}
  loc::Vector{I}
  glo::Vector{I}
end
Base.isempty(us::UnitSelector) = isempty(us.glo)
Base.isempty(us::RegularizedUnit) = isempty(us.glo)

function UnitSelector(name::Symbol,S::BitArray{N},u::RegularizedUnit{N,R,I}) where {N,R,I}
  loc=I[]
  glo=I[]
  for (l,g) in zip(u.loc,u.glo)
    if S[l]
      push!(loc,l) ; push!(glo,g)
  end; end
  if isempty(loc)
    @warn "The selector $name is empty! This migth generate errors"
  end
  return UnitSelector{N,R,I}(name,S,u.Mg,loc,glo)
end

function UnitSelector(name::Symbol,u::RegularizedUnit{N,R,I}) where {N,R,I}
  S=trues(size(u.M))
  return UnitSelector(name,S,u)
end
function UnitSelector(u::RegularizedUnit{N,R,I}) where {N,R,I}
  name=u.name
  return UnitSelector(name,u)
end


function pack_grad_array!(x::Vector{R},sel::AbstractUnit{N,R,I}) where {N,R,I}
  for (g,l) in zip(sel.glo,sel.loc)
    x[g]=sel.Mg[l]
  end
  return nothing
end

struct RegularizerPack{R,I}
  # units::Vector{Union{RegularizedUnit{1,R,I},RegularizedUnit{2,R,I}}}
  units::Vector{RegularizedUnit{N,R,I} where N}
  xreg::Vector{R}  # variables in regu form
  xnonreg::Vector{R} # variables in unbound form
  xgrad::Vector{R}  # gradient of regularizer w.r.t. xnonreg
  xgrad_units::Vector{R}  # allocated space for gradient of variables
  regus::Vector{Regularizer{R}}  # regularizers, element by element
end
Base.length(rp::RegularizerPack) = sum(length.(rp.units))

get_unit(nm::Symbol,rp::RegularizerPack)=get_unit(nm,rp.units)

function RegularizerPack(dic_m::Dict{Symbol,Array{R}},
    dic_regu::Dict{Symbol,Array{Union{Missing,Symbol}}},
    dic_regu_type::Dict{Symbol,Regularizer{R}}) where R
  # create both number units and symbol units
  units =Vector{RegularizedUnit{N,R,typeof(0)} where N}(undef,0)
  units_reg = Vector{RegularizedUnit{N,Union{Missing,Symbol},typeof(0)} where N}(undef,0)
  myoff=0
  for k in keys(dic_regu)
    push!(units,RegularizedUnit(k,dic_m[k],dic_regu[k];offset=myoff))
    push!(units_reg,RegularizedUnit(k,dic_regu[k],dic_regu[k];offset=myoff))
    myoff+=length(units[end])
  end
  ntot=sum(length.(units))
  xreg=Vector{R}(undef,ntot)
  xnonreg,xgrad,xgrad_units=similar(xreg),similar(xreg),similar(xreg)
  xsymb=Vector{Union{Symbol,Missing}}(undef,ntot)
  for u in units_reg
    pack!(xsymb,u)
  end
  regus = map(s->getindex(dic_regu_type,s), xsymb)
  regus=convert(Vector{Regularizer{R}},regus) # make sure it is the generic type
  RegularizerPack(units,xreg,xnonreg,xgrad,xgrad_units,regus)
end

# get the global indexes of the selected units only
function global_idx_less(rp::RegularizerPack{R,I},idx_unit::Vector{I}) where {R,I}
   glos=[x.glo for x in  rp.units[idx_unit]]
   return vcat(glos...)
end

# fills xreg and xnonreg
function pack!(rn::RegularizerPack)
  for rnu in rn.units
    pack!(rn.xreg,rnu)
  end
  # now fill xnonreg
  for (i,x) in enumerate(rn.xreg)
    @inbounds rn.xnonreg[i]=ireg(x,rn.regus[i])
  end
  return nothing
end

function unpack!(rn::RegularizerPack)
  for rnu in rn.units
    unpack!(rnu,rn.xreg)
  end
  return nothing
end

#assumes pack! has happened and xnonreg is valid
function pack_grad!(rn::RegularizerPack)
  for (i,regu) in enumerate(rn.regus)
    @inbounds rn.xgrad[i]=dreg(rn.xnonreg[i],regu)
  end
  return nothing
end

# assumes units have been filled, but the rest needs to be done
function pack_xandgrad!(rn::RegularizerPack)
  pack!(rn)
  return pack_grad!(rn)
end

function pack_grad_array!(rp::RegularizerPack)
  for rnu in rp.units
    pack_grad_array!(rp.xgrad_units,rnu)
  end
  return nothing
end
function pack_grad_array!(rp::RegularizerPack{R,I},sel::AbstractUnit{N,R,I}) where {N,R,I}
  return pack_grad_array!(rp.xgrad_units,sel)
end


# the input is written on xnonreg ,
# it regularizes, does the unpacking
# and computes the gradient
# this should be run before computing any objective function over
# the elements
function unpack_reguandgrad!(rn::RegularizerPack)
  for (i,regu) in enumerate(rn.regus)
    @inbounds rn.xreg[i]=regu(rn.xnonreg[i])
  end
  unpack!(rn)
  return pack_grad!(rn)
end
function unpack_reguandgrad!(rp::RegularizerPack{R,I},x::Vector{R}) where {R,I}
  copy!(rp.xnonreg,x)
  return unpack_reguandgrad!(rp)
end


"""
    propagate_gradient!(rp::RegularizerPack)

This function is called after the gradient is computed and stored in the gradient
allocation arrays .  It unpacks the gradient according to the specified
assignment , multiplies each element by the matching gradient of regularizer
functions (i.e. `rp.xgrad`), and adds the result to  `g`
"""
function propagate_gradient!(g::Vector{R},rp::RegularizerPack{R,I}) where {R,I}
  pack_grad_array!(rp) # fills rp.xgrad_units
  for i in eachindex(g)
    g[i] += rp.xgrad_units[i] * rp.xgrad[i]
  end
  return nothing
end

function propagate_gradient!(g::Vector{R},rp::RegularizerPack{R,I},
    sel::AbstractUnit{N,R,I}) where {N,R,I}
  pack_grad_array!(rp,sel) # fills rp.xgrad_units
  for i in sel.glo
    g[i] += rp.xgrad_units[i] * rp.xgrad[i]
  end
  return nothing
end
=#

end # module
