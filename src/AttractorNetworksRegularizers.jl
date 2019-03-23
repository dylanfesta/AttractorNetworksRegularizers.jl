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

# vectorized versions!
for fname in (:reg,:dreg,:ireg)
  fname! = Symbol(fname,"!")
  eval( :(
    function $fname!(xnew::AbstractVector{Float64}, x::AbstractVector{Float64},
                        regu::Regularizer)
      map!(xx->$fname(xx,regu), xnew, x )
    end))
  eval(:(
    function $fname(x::AbstractVector{Float64},regu::Regularizer)
      $fname!(similar(x),x,regu)
    end ))
end

function gradient_test(x::Real,regu::Regularizer)
  grad_num = Calculus.gradient(xx->reg(xx,regu), x )
  grad_an = dreg(x,regu)
  grad_num,grad_an, 2.0abs((grad_num-grad_an)/(grad_num + grad_an))
end

#=
Assignment is a named tuple of indexes.
names refer to arrays of another tuple,
plus suffix _loc _glo
_loc refers to the position of the selected elements on the tuple (linear index)
_glo refers to the position of the respective element on a global 1D vector
that will be used for optimization.

Plan: one full assignment covers all the optimized variables,
however there can be "regional" assignments smaller in size
that can be used for "regional" operations over the elements.

A function ensures that the global indices of the regional assignment still point
at the correct positions of the global vector.
=#

# appends glo and loc to the symbol name
_gloloc(nm::Symbol) = Symbol(nm,:_glo) , Symbol(nm,:_loc)
function _gloloc(v::AbstractVector)
  glo,loc = copy(v) , copy(v)
  for i in eachindex(v)
    glo[i],loc[i] = _gloloc(v[i])
  end
  glo,loc
end
_gloloc(v) = _gloloc([v...])

key_isglo(k) = occursin(r"\_glo\b",String(k))
key_isloc(k) = occursin(r"\_loc\b",String(k))
key_nosuffix(k) = Symbol(String(k)[1:end-4])
key_toglobal(k) = Symbol(key_nosuffix(k),:_glo)
key_tolocal(k) = Symbol(key_nosuffix(k),:_loc)

# linear indexes of nonmissing values
function get_nonmissing(bu::AbstractVector)::Vector{Int64}
  findall( .!(ismissing.(bu)) )
end
# converts cartesian index to linear index
function get_nonmissing(bu::AbstractMatrix)::Vector{Int64}
  idx = findall( .!(ismissing.(bu)) )
  lin = LinearIndices(bu)
  lin[idx]
end
function get_nonmissing(m::AbstractArray)
  error("It only works on vector and matrices, for now!")
end
# see large comment above
function make_assignment(elements,selections)
  _keys = keys(selections)
  for nm in _keys
    @assert nm in keys(elements)
  end
  _selec = [selections[k] for k in _keys]
  names_glo,names_loc = _gloloc(_keys)
  idx_loc = map(get_nonmissing,_selec)
  # global: one index for each local element
  lgs = length.(idx_loc)
  nglo = sum(lgs)
  whole = collect(1:nglo)
  idx_glo = [splice!(whole,1:l) for l in lgs]
  names_all = vcat(names_loc,names_glo)
  idx_all =vcat(idx_loc, idx_glo)
  # put them together in a named tuple
  (nglo, NamedTuple{Tuple(names_all)}(idx_all) )
end

"""
    globalize(assignent_reg, assignment_cap)

modifies the global idexes of regional assignment
so that they are in line with the "capital" assignment
The local indexes of regional assignment MUST be present in the
capital assignment
"""
function globalize(assignent_reg, assignment_cap)
  # extract the relevant names of keys
  keys_loc = filter(key_isloc, keys(assignent_reg))
  keys_glo = map(keys_loc) do k
    Symbol(String(k)[1:end-4]*"_glo")
  end
  for klocglo in zip(keys_loc,keys_glob)
    # I extract the indexes
    reg_loc,reg_glo =[ assignent_reg[k] for k in klocglo ]
    cap_loc,cap_glo =[ assignent_cap[k] for k in klocglo ]
    # find the position of reg local
    # in cap local, replace reg global with the
    # cap global for that index
    for (k,idxreg) in enumerate(reg_loc)
      _idxcap = findfirst(idxreg .== cap_loc)
      @assert !isnothing(_idxcap)
      reg_glo[k] = cap_glo[_idxcap]
    end
  end
end

"""
      packing!(v::AbstractVector,elements::NamedTuple,
            assignment::NamedTuple , forward_mode::Bool=true)

Using the indexes in `assignment` , picks the relevant values in `elements` and
either packs them in the `v` vector `forward_mode=true` or from `v` writes them
back into `elements`  `forward_mode=false`
"""
function packing!(v::AbstractVector,elements::NamedTuple,
      assignment::NamedTuple , forward_mode::Bool=true)
  isglo(s) = occursin(r"\_glo\b",String(s))
  for (k_glo,idxs) in pairs(assignment)
    if key_isglo(k_glo) && (!isempty(idxs))
      k_el = key_nosuffix(k_glo)
      k_loc = Symbol(k_el,:_loc)
      idx_glo = getfield(assignment,k_glo)
      idx_loc =  getfield(assignment,k_loc)
      if forward_mode
        view(v,idx_glo) .= view(getfield(elements,k_el), idx_loc)
      else
        view(getfield(elements,k_el), idx_loc) .= view(v,idx_glo)
      end
    end
  end
  nothing
end
unpack!(v,els,as) = packing!(v,els,as,false)


struct RegularizerPack
  elements::NamedTuple # the elements that are packed, all arrays
  assignments::NamedTuple  # Global assignement that regulates the poisitions in x
  xreg::Vector{Float64}  # variables in regu form
  xnonreg::Vector{Float64} # variables in unbound form
  xgrad::Vector{Float64}  # gradient of regularizer w.r.t. xnonreg
  regus::Vector{Regularizer}  # regularizes, element by element
  allocs::NamedTuple  # space used for gradients and stuff, same size as elements
end

# takes a named tuple of arrays
# returns tuple with same keys, but where the arrays are
# of type Union{Missing,Symbol} and initialized with missings
function make_empty_selection(elements)
  function makeone(x)
    o = similar(x,Union{Missing,Symbol})
    fill!(o,missing)
  end
  NamedTuple{keys(elements)}( map(makeone,values(elements)) )
end


"""
    RegularizerPack(elements, selections, regudefs)
Main constructor of a regularizer pack

# inputs
  + `elements` : named tuple of arrays that are targeted by the optimization
  + `selections` : named tuples of arrays of type `Union{Missing,Symbol}`,
      missing elements are ignored, symbols should match with the keys in
      `regudef` and indicate with regularizer will be used
      use `make_empty_selection(elements)` to initialize an empty structure
  + `regudefs`: named tuple that associates the regularizer name to a specific
    regularizer object. Names should be the same as symbol in `selections`

"""
function RegularizerPack(elements, selections, regudefs)
  # type - check the inputs
  # all selections should refer to an element
  for nm in keys(selections)
    @assert nm in keys(elements)
  end
  for vv in values(elements)
    @assert typeof(vv) <: AbstractArray{Float64}
  end
  for vv in values(selections)
    @assert typeof(vv) <: AbstractArray{Union{Missing,Symbol}}
  end
  for vv in values(regudefs)
    @assert typeof(vv) <: Regularizer
  end
  # easy part, just duplicate elements for the allocs
  dups = [ similar(x) for x in  values(elements)]
  allocs = NamedTuple{keys(elements)}(dups)
  # now, the indexes tuple...
  (nglobs, assignments) = make_assignment(elements,selections)
  # now I need to assign the regularizers
  # this means packing!
  _regus_aux = Vector{Symbol}(undef,nglobs)
  packing!(_regus_aux , selections, assignments)
  for regunm in unique(_regus_aux)
    @assert regunm in keys(regudefs)
  end
  regus = Vector{Regularizer}(undef,nglobs)
  for (i,rr) in enumerate(_regus_aux)
    regus[i] = getfield(regudefs,rr)
  end
  xreg,xnonreg,xgrad = [ Vector{Float64}(undef,nglobs) for _ in 1:3 ]
  #define it
  rpack=RegularizerPack(elements,assignments,xreg,xnonreg,xgrad,regus,allocs)
  # initialize it !
  pack_allx_grad!(rpack)
  rpack
end
Base.length(p::RegularizerPack) = length(p.xnonreg)
Base.keys(p::RegularizerPack) = keys(p.elements)

##

function pack_allx!(xreg,xnonreg,elements,assignments,regus)
  # first extract the variables from elements, place them in the buffer
  # elements are already regularized!
  packing!(xreg,elements,assignments)
  # now convert!
  @. xnonreg =  ireg(xreg,regus)
  nothing
end
function pack_allx!(rp::RegularizerPack)
  pack_allx!(rp.xreg,rp.xnonreg,rp.elements,rp.assignments,rp.regus)
end
function pack_calc_grad!(xgrad, xnonreg, regus)
  @. xgrad = dreg(xnonreg,regus)
end

function pack_allx_grad!(xreg,xnonreg,xgrad,elements,assignments,regus)
  pack_allx!(xreg,xnonreg,elements,assignments,regus)
  pack_calc_grad!(xgrad, xnonreg, regus)
end
function pack_allx_grad!(rp::RegularizerPack)
   pack_allx_grad!(rp.xreg,rp.xnonreg,rp.xgrad,rp.elements,rp.assignments,rp.regus)
end

function gradient_test(rp::RegularizerPack,x::AbstractVector)
    copy!(rp.xnonreg ,x)
    pack_calc_grad!(rp.xgrad,rp.xnonreg,rp.regus)
    g_an = copy(rp.xgrad)
    eps_test=1E-8
    # all grads are independent, so I can vectorize
    xpm=copy(x)
    xpm .+= eps_test
    f_p = reg.(xpm,rp.regus)
    # f_p = map( (x,re)->reg(x,re), zip(xpm,rp.regus) )
    xpm .-= 2eps_test
    f_m = reg.(xpm,rp.regus)
    g_num = @. (f_p-f_m) / (2*eps_test)
    diff_g = @. abs( 2. * (g_num-g_an)/(g_num+g_an) )
    g_num , g_an , diff_g
end


end # module
