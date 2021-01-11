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

function pack_grad_array!(np::RegularizerPack)
  for rnu in rn.units
    pack_grad_array!(rn.xgrad_units,rnu)
  end
  return nothing
end
function pack_grad_array!(rp::RegularizerPack{R,I},units_idx::Vector{I}) where {R,I}
  for rpu in rp.units[units_idx]
    pack_grad_array!(rp.xgrad_units,rpu)
  end
  return nothing
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
function propagate_gradient!(g::Vector{R},rp::RegularizerPack{R,I},
    unit_idxs::Vector{I}) where {R,I}
  pack_grad_array!(rp,unit_idxs) # fills rp.xgrad_units
  for i in global_idx_less(rp,unit_idxs)
    g[i] += rp.xgrad_units[i] * rp.xgrad[i]
  end
  return nothing
end

function propagate_gradient!(g::Vector,rp::RegularizerPack)
  return propagate_gradient!(g,rp,collect(1:length(rp.units)))
end

"""
  add_gradient!(rp::RegularizerPack,assignment)

This function is called after the gradient is computed and stored in the
named tuple `rp.allocs` .  It unpacks the gradient according to the specified
assignment , multiplies each element by the matching gradient of regularizer
functions (i.e. `rp.xgrad`), and adds the result to  `g`
"""
function add_gradient!(g::AbstractVector,rp::RegularizerPack,assignment::NamedTuple)
  for (sglo,sloc) in zip(glo_loc_split(assignment)...)
    _ref = key_nosuffix(sloc)
    for (iglo,iloc) in zip( assignment[sglo],assignment[sloc])
      gval = rp.allocs[_ref][iloc]
      g[iglo] += gval * rp.xgrad[iglo]
    end
  end
  return g
end



#
# #=
# Assignment is a named tuple of indexes.
# names refer to arrays of another tuple,
# plus suffix _loc _glo
# _loc refers to the position of the selected elements on the tuple (linear index)
# _glo refers to the position of the respective element on a global 1D vector
# that will be used for optimization.
#
# Plan: one full assignment covers all the optimized variables,
# however there can be "regional" assignments smaller in size
# that can be used for "regional" operations over the elements.
#
# A function ensures that the global indices of the regional assignment still point
# at the correct positions of the global vector.
# =#
#
# # appends glo and loc to the symbol name
# _gloloc(nm::Symbol) = Symbol(nm,:_glo) , Symbol(nm,:_loc)
# function _gloloc(v::AbstractVector{<:Symbol})
#   glo,loc = copy(v) , copy(v)
#   for i in eachindex(v)
#     glo[i],loc[i] = _gloloc(v[i])
#   end
#   glo,loc
# end
# _gloloc(v) = _gloloc([v...])
#
# key_isglo(k) = occursin(r"\_glo\b",String(k))
# key_isloc(k) = occursin(r"\_loc\b",String(k))
# key_nosuffix(k) = Symbol(String(k)[1:end-4])
# key_toglobal(k) = Symbol(key_nosuffix(k),:_glo)
# key_tolocal(k) = Symbol(key_nosuffix(k),:_loc)
#
# function glo_loc_split(assignments::NamedTuple)
#   locs = filter(key_isloc,[keys(assignments)...])
#   glos = map(key_toglobal,locs)
#   glos,locs
# end
#
# # linear indexes of nonmissing values
# # converts cartesian index to linear index
# function get_nonmissing(bu::AbstractMatrix)
#   idx = findall( .!(ismissing.(bu)) )
#   lin = LinearIndices(bu)
#   return lin[idx]
# end
#
# # see large comment above
# function make_assignment(selections)
#   _keys = keys(selections)
#   selec = [selections[k] for k in _keys]
#   names_glo,names_loc = _gloloc(_keys)
#   idx_loc = get_nonmissing.(selec)
#   # global: one index for each local element
#   lgs = length.(idx_loc)
#   nglo = sum(lgs)
#   whole = collect(1:nglo)
#   idx_glo = [splice!(whole,1:l) for l in lgs]
#   names_all = vcat(names_loc,names_glo)
#   idx_all =vcat(idx_loc, idx_glo)
#   # put them together in a named tuple
#   (nglo, NamedTuple{Tuple(names_all)}(idx_all) )
# end
#
# function make_assignment2(selections)
#   _keys = keys(selections)
#   _selec = [selections[k] for k in _keys]
#   names_glo,names_loc = _gloloc(_keys)
#   idx_loc = map(get_nonmissing,_selec)
#   # global: one index for each local element
#   lgs = length.(idx_loc)
#   nglo = sum(lgs)
#   whole = collect(1:nglo)
#   idx_glo = [splice!(whole,1:l) for l in lgs]
#   names_all = vcat(names_loc,names_glo)
#   idx_all =vcat(idx_loc, idx_glo)
#   # put them together in a named tuple
#   namtup = NamedTuple{Tuple(names_all)}(idx_all)
#   dict_out= Dict(pairs(namtup)...)
#   (nglo,dict_out)
# end
#
#
# """
#     globalize(assignment_reg, assignment_cap)
#
# modifies the global idexes of regional assignment
# so that they are in line with the "global" assignment
# The local indexes of regional assignment MUST be present in the
# global assignment
# """
# function globalize!(assignment_reg, assignment_cap)
#   # separate local and global, take only "regional" elements
#   keys_loc = filter(key_isloc, [keys(assignment_reg)...] )
#   keys_glo = key_toglobal.(keys_loc)
#   for klocglo in zip(keys_loc,keys_glo)
#     # I extract the indexes
#     reg_loc,reg_glo =[ assignment_reg[k] for k in klocglo ]
#     cap_loc,cap_glo =[ assignment_cap[k] for k in klocglo ]
#     # find the position of reg local
#     # in cap local, replace reg global with the
#     # cap global for that index
#     for (k,idxreg) in enumerate(reg_loc)
#       _idxcap = findfirst(idxreg .== cap_loc)
#       @assert !isnothing(_idxcap)
#       reg_glo[k] = cap_glo[_idxcap]
#     end
#   end
#   return nothing
# end
#
# """
#       packing!(v::AbstractVector,elements::NamedTuple,
#             assignment::NamedTuple , forward_mode::Bool=true)
#
# Using the indexes in `assignment` , picks the relevant values in `elements` and
# either packs them in the `v` vector `forward_mode=true` or from `v` writes them
# back into `elements`  `forward_mode=false`
# """
# function packing!(v::AbstractVector,elements::NamedTuple,
#       assignment::NamedTuple , forward_mode::Bool=true)
#   isglo(s) = occursin(r"\_glo\b",String(s))
#   for (k_glo,idxs) in pairs(assignment)
#     if key_isglo(k_glo) && (!isempty(idxs))
#       k_el = key_nosuffix(k_glo)
#       k_loc = Symbol(k_el,:_loc)
#       idx_glo = getfield(assignment,k_glo)
#       idx_loc =  getfield(assignment,k_loc)
#       if forward_mode
#         view(v,idx_glo) .= view(getfield(elements,k_el), idx_loc)
#       else
#         view(getfield(elements,k_el), idx_loc) .= view(v,idx_glo)
#       end
#     end
#   end
#   return nothing
# end
# unpacking!(v,els,as) = packing!(v,els,as,false)
#
#
#
# struct RegularizerPack{R,I}
#   elements::NamedTuple # the elements that are packed, all arrays
#   selection::NamedTuple # same as elements but select which get packed
#   assignments::NamedTuple # indexes , poisitions in x and in elements
#   xreg::Vector{R}  # variables in regu form
#   xnonreg::Vector{R} # variables in unbound form
#   xgrad::Vector{R}  # gradient of regularizer w.r.t. xnonreg
#   regus::Vector{Regularizer{R}}  # regularizers, element by element
#   allocs::NamedTuple  # space used for gradients and stuff, same size as elements
# end
#
#
#
# """
#   make_empty_selection(elements)
# `elements` is a named tuple of arrays. Returns a named tuple with same keys, and same
# array size, but where arrays are of type Union{Missing,Symbol}
# and initialized with missings.  This will be the `selections` element in the
# constructor of a `RegularizerPack`.
# """
# function make_empty_selection(elements)
#   function makeone(x)
#     o = similar(x,Union{Missing,Symbol})
#     fill!(o,missing)
#   end
#   NamedTuple{keys(elements)}(map(makeone,values(elements)))
# end
#
# function make_empty_selection2(elements)
#   function makeone(x)
#     o = similar(x,Union{Missing,Symbol})
#     fill!(o,missing)
#   end
#   ret = NamedTuple{keys(elements)}( map(makeone,values(elements)) )
#   return Dict(pairs(ret)...)
# end
#
#
#
# """
#     RegularizerPack(elements, selections, regudefs)
# Main constructor of a regularizer pack
#
# # inputs
#   + `elements` : named tuple of arrays that are targeted by the optimization
#   + `selections` : named tuples of arrays of type `Union{Missing,Symbol}`,
#       missing elements are ignored, symbols should match with the keys in
#       `regudef` and indicate with regularizer will be used
#       use `make_empty_selection(elements)` to initialize an empty structure
#   + `regudefs`: named tuple that associates the regularizer name to a specific
#     regularizer object. Names should be the same as symbol in `selections`
#
# """
# function RegularizerPack(elements, selection, regudefs)
#   # type - check the inputs
#   # all selection should refer to an element
#   for nm in keys(selection)
#     @assert nm in keys(elements)
#   end
#   for vv in values(elements)
#     @assert typeof(vv) <: AbstractArray{Float64}
#   end
#   for vv in values(selection)
#     @assert typeof(vv) <: AbstractArray{Union{Missing,Symbol}}
#   end
#   for vv in values(regudefs)
#     @assert typeof(vv) <: Regularizer
#   end
#   # easy part, just duplicate elements for the allocs
#   dups = [ similar(x) for x in  values(elements)]
#   allocs = NamedTuple{keys(elements)}(dups)
#   # now, the indexes tuple...
#   (nglobs, assignments) = make_assignment(selection)
#   # now I need to assign the regularizers
#   # this means packing!
#   _regus_aux = Vector{Symbol}(undef,nglobs)
#   packing!(_regus_aux , selection, assignments)
#   for regunm in unique(_regus_aux)
#     @assert regunm in keys(regudefs)
#   end
#   regus = Vector{Regularizer}(undef,nglobs)
#   for (i,rr) in enumerate(_regus_aux)
#     regus[i] = getfield(regudefs,rr)
#   end
#   xreg,xnonreg,xgrad = [Vector{Float64}(undef,nglobs) for _ in 1:3 ]
#   #define it
#   rpack=RegularizerPack(elements,selection,
#     assignments,xreg,xnonreg,xgrad,regus,allocs)
#   # initialize it !
#   pack_allx_grad!(rpack)
#   rpack
# end
# Base.length(p::RegularizerPack) = length(p.xnonreg)
#
# ##
# function pack_allx!(xreg,xnonreg,elements,assignments,regus)
#   # first extract the variables from elements, place them in the buffer
#   # elements are already regularized!
#   packing!(xreg,elements,assignments)
#   # now convert!
#   @. xnonreg =  ireg(xreg,regus)
#   return nothing
# end
# function pack_allx!(rp::RegularizerPack)
#   pack_allx!(rp.xreg,rp.xnonreg,rp.elements,rp.assignments,rp.regus)
# end
# function pack_calc_grad!(xgrad, xnonreg, regus)
#   @. xgrad = dreg(xnonreg,regus)
#   return nothing
# end
#
# function pack_allx_grad!(xreg,xnonreg,xgrad,elements,assignments,regus)
#   pack_allx!(xreg,xnonreg,elements,assignments,regus)
#   pack_calc_grad!(xgrad, xnonreg, regus)
# end
# function pack_allx_grad!(rp::RegularizerPack)
#    pack_allx_grad!(rp.xreg,rp.xnonreg,rp.xgrad,rp.elements,rp.assignments,rp.regus)
# end
#
# # the input here is xnonreg , that is the current point in the unbounded parameter
# # space, this should be run before computing any objective function over
# # the elements
# function unpack_allx_grad!(xreg,xnonreg,xgrad,elements,assignments,regus)
#   # fill x reg from xnonreg
#   @. xreg = reg(xnonreg,regus)
#   # unpack x reg into elements according to assignment
#   packing!(xreg,elements,assignments,false)
#   # update xgrad
#   @. xgrad = dreg(xnonreg,regus)
#   return nothing
# end
#
# function unpack_allx_grad!(x::AbstractVector,rp::RegularizerPack)
#   copy!(rp.xnonreg,x)
#   unpack_allx_grad!(rp)
# end
#
# function unpack_allx_grad!(rp::RegularizerPack)
#   unpack_allx_grad!(rp.xreg,rp.xnonreg,rp.xgrad,rp.elements,rp.assignments,rp.regus)
# end
#


# function gradient_test(rp::RegularizerPack,x::AbstractVector)
#     copy!(rp.xnonreg ,x)
#     pack_calc_grad!(rp.xgrad,rp.xnonreg,rp.regus)
#     g_an = copy(rp.xgrad)
#     eps_test=1E-8
#     # all grads are independent, so I can vectorize
#     xpm=copy(x)
#     xpm .+= eps_test
#     f_p = reg.(xpm,rp.regus)
#     # f_p = map( (x,re)->reg(x,re), zip(xpm,rp.regus) )
#     xpm .-= 2eps_test
#     f_m = reg.(xpm,rp.regus)
#     g_num = @. (f_p-f_m) / (2*eps_test)
#     diff_g = @. abs( 2. * (g_num-g_an)/(g_num+g_an) )
#     g_num , g_an , diff_g
# end
#

#simple query of indexes for regularizer pack
#=


indexes_currents_loc = mypack[Local(:currents)]
indexes_currents_glo = mypack[Global(:currents)]

etc

=#
#
# abstract type PackQuery end
# struct Local <: PackQuery
#   x::Symbol
#   function Local(s::Symbol)
#     new(Symbol(s,:_loc))
#   end
# end
# struct Global <: PackQuery
#   x::Symbol
#   function Global(s::Symbol)
#     new(Symbol(s,:_glo))
#   end
# end
#
# function Base.getindex(rp::RegularizerPack,x)
#   getindex(rp.assignments,x)
# end
# function Base.getindex(rp::RegularizerPack, pq::PackQuery)
#   getindex(rp.assignments, pq.x)
# end
# function Base.keys(rp::RegularizerPack)
#   keys(rp.assignments)
# end
#
#


end # module
