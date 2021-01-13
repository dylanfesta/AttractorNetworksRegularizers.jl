module AttractorNetworksRegularizers
using Calculus

abstract type Regularizer{R} end
Base.Broadcast.broadcastable(g::Regularizer)=Ref(g)

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
struct NoRegu{R} <: Regularizer{R}  end
@inline function (sp::NoRegu{R})(x::R) where R
  return x
end
dreg(x::R,sp::NoRegu{R}) where R = one(R)
ireg(x::R,sp::NoRegu{R}) where R = x

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


struct RegularizedUnit{N,R,I}
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
  return RegularizedUnit{N,R,typeof(loc[1])}(name,A,similar(A),loc,glo)
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
struct UnitSelector{N,R,I}
  name::Symbol
  S::BitArray{N}
  Mg::Array{R,N}
  loc::Vector{I}
  glo::Vector{I}
end

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


function pack_grad_array!(x::Vector{R},sel::UnitSelector{N,R,I}) where {N,R,I}
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

function RegularizerPack(dic_m::Dict{Symbol,Array{R}},
    dic_regu,dic_regu_type) where R
  # create both number units and symbol units
  kall=collect(keys(dic_m))
  myoff=0
  units=map(kall) do k
    ret=RegularizedUnit(k,dic_m[k],dic_regu[k];offset=myoff)
    myoff+=length(ret)
    ret
  end
  myoff=0
  units_reg=map(kall) do k
    ret=RegularizedUnit(k,dic_regu[k],dic_regu[k];offset=myoff)
    myoff+=length(ret)
    ret
  end
  ntot=sum(length.(units))
  xreg=Vector{R}(undef,ntot)
  xnonreg,xgrad,xgrad_units=similar(xreg),similar(xreg),similar(xreg)
  xsymb=Vector{Union{Symbol,Missing}}(undef,ntot)
  for u in units_reg
    pack!(xsymb,u)
  end
  regus = map(s->getindex(dic_regu_type,s), xsymb)
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
function pack_grad_array!(rp::RegularizerPack{R,I},sel::UnitSelector{N,R,I}) where {N,R,I}
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
function unpack_reguandgrad!(rn::RegularizerPack{R,I},x::Vector{R}) where {R,I}
  copy!(nr.xnonreg,x)
  return unpack_reguandgrad!(rn)
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
    sel::UnitSelector{N,R,I}) where {N,R,I}
  pack_grad_array!(rp,sel) # fills rp.xgrad_units
  for i in sel.glo
    g[i] += rp.xgrad_units[i] * rp.xgrad[i]
  end
  return nothing
end



end # module
