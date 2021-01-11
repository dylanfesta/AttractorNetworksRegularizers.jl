
using Pkg ; Pkg.activate(joinpath(@__DIR__(),".."))
using AttractorNetworksRegularizers ; const AA = AttractorNetworksRegularizers
using LinearAlgebra, StatsBase, Statistics
using Calculus
using BenchmarkTools,Cthulhu
using Plots ; theme(:dark)
using Test

function gradient_test(x::R,regu::AA.Regularizer{R}) where R
  grad_num = Calculus.gradient(xx->regu(xx), x )
  grad_an = AA.dreg(x,regu)
  grad_num,grad_an
end


function plotvs(x::AbstractArray{<:Real},y::AbstractArray{<:Real})
  x,y=x[:],y[:]
  @info """
  The max differences between the two are $(extrema(x .-y ))
  """
  plt=plot()
  scatter!(plt,x,y;leg=false,ratio=1,color=:white)
  lm=xlims()
  plot!(plt,identity,range(lm...;length=3);linestyle=:dash,color=:yellow)
  return plt
end
##

A = rand(100,100)
B = randn(50)
C = .- rand(3,122)

things =  Dict(:A=>A,:B=>B,:C=>C)

regurefs = Dict( :n  => AA.NoRegu{Float64}() ,
                   :b => AA.SigmoidalBoth(10.,-20.,30.),
                   :p => AA.SigmoidalPlus(10.0,2.),
                   :m => AA.SigmoidalMinus(40.0,2.))

myselections = AA.make_empty_selection(things)

# let's do masks
Amask=Int64[]
Agood=Int64[]
for i in eachindex(A)
  rand(Bool) ? push!(Amask,i) : push!(Agood,i)
end
Bmask=Int64[]
Bgood=Int64[]
for i in eachindex(B)
  rand() < 0.2 ?  push!(Bmask,i) : push!(Bgood,i)
end
Cmask=Int64[]
Cgood=Int64[]
for i in eachindex(C)
  rand() < 0.9 ?  push!(Cmask,i) : push!(Cgood,i)
end


things[:A][Amask] .= Inf
myselections[:A][Agood] .= :p
things[:B][Bmask] .= Inf
myselections[:B][Bgood] .= :b
things[:C][Cmask] .= Inf
myselections[:C][Cgood] .= :m
A0,B0,C0 = [copy(things[x]) for x in  [:A,:B,:C]]


pk=AA.RegularizerPack(things,myselections,regurefs)
##
# test 3, gradient
for u in pk.units
  fill!(u.Mg,0.3456)
end

g_an1=fill(-2.34,length(pk))
AA.pack_xandgrad!(pk)
AA.propagate_gradient!(g_an1,pk)

@test all(pk.xgrad_units .== 0.3456)
# now the numeric
g_num = similar(g_an1)
for i in eachindex(g_num)
  g_num[i] = Calculus.gradient(pk.regus[i],pk.xnonreg[i])
end
g_an2=pk.xgrad
@test all(isapprox.(g_num,g_an2;atol=1E-3))
g_an_fix = @. (g_an1+2.34)/0.3456
@test all(isapprox.(g_num,g_an_fix;atol=1E-3))




##

myselections.A[1,:] .= :nope
myselections.B[1:4] .= :nope
myselections.C[18:49] .= :plus
myselections.D[1:2,1:2] .= :minus


##

A = rand(10,10)
B = randn(50)
C = fill(7.,88)
D = fill(-33.,3,3)

mythings =  (A=A,B=B,C=C,D=D)
myregudefs = ( nope = AA.NoRegu{Float64}() ,
               plus = AA.SigmoidalPlus(10.0,2.),
               minus = AA.SigmoidalMinus(40.0,2.))

myselections = AA.make_empty_selection(mythings)
myselections.A[:,4] .= :plus
myselections.A[1,:] .= :nope
myselections.B[1:4] .= :nope
myselections.C[18:49] .= :plus
myselections.D[1:2,1:2] .= :minus

##
Aru=AA.RegularizedUnit(:A,A,myselections.A)
Bru=AA.RegularizedUnit(:B,B,myselections.B,Aru)

Aregu=AA.RegularizedUnit(:A,myselections.A,myselections.A)
xregus = Vector{Union{Missing,Symbol}}(undef,length(Aregu))
AA.pack!(xregus,Aregu)

x0 = fill(NaN,length(Aru)+length(Bru))

@btime AA.unpack!($x0,$Aru)

##
things = Dict(pairs(mythings)...)
dict_sels = Dict(pairs(myselections)...)
AA.RegularizerPackNew(things,dict_sels,myregudefs)


## pack testing!

A = rand(10,10)
B = randn(50)
C = fill(7.,88)
D = fill(-33.,3,3)

mythings =  (A=A,B=B,C=C,D=D)
myregudefs = ( nope = AA.NoRegu{Float64}() ,
               plus = AA.SigmoidalPlus(10.0,2.),
               minus = AA.SigmoidalMinus(40.0,2.))

myselections = AA.make_empty_selection(mythings)
myselections.A[:,4] .= :plus
myselections.A[1,:] .= :nope
myselections.B[1:4] .= :nope
myselections.C[18:49] .= :plus
myselections.D[1:2,1:2] .= :minus

test_regupack = AA.RegularizerPack(mythings,myselections,myregudefs)

mythings2 = deepcopy(mythings)

x0 = copy(test_regupack.xnonreg)

AA.pack_allx_grad!(test_regupack)
x0b = copy(test_regupack.xnonreg)
extrema(x0 .- x0b)

AA.unpack_allx_grad!(x0,test_regupack)
extrema(test_regupack.xnonreg .- x0b)

xnoise = x0 .+ 0randn(size(x0))

AA.unpack_allx_grad!(xnoise,test_regupack)
extrema(test_regupack.xnonreg .- xnoise)

display(mythings.C[18:30])
display(mythings.D)
extrema(x0b .- xnoise)


## test regional assignments
A = rand(100,100)
B = -rand(500)
mythings =  (A=A,B=B)
myregudefs = ( nope = AA.NoRegu{Float64}() ,
               plus = AA.SigmoidalPlus(3.0,2.),
               minus = AA.SigmoidalMinus(5.0,2.))
myselection = AA.make_empty_selection(mythings)
n_pick = 300
_ = let idx_a =  sample(eachindex(A),n_pick;replace=false),
  idx_b = sample(eachindex(B),n_pick;replace=false)
  myselection.A[idx_a] .= :plus
  myselection.B[idx_b] .= :minus
  nothing
end
myselection_local = deepcopy(myselection)
myselection_local.A[1:50,1:50] .= missing
myselection_local.B[1:100] .= missing

myassignments = AA.make_assignment(myselection)[2]
myassignments_local = AA.make_assignment(myselection_local)[2]
myassignments_local_bad = deepcopy(myassignments_local)


myassignments = AA.make_assignment_dict(myselection)

AA.globalize!(myassignments_local,myassignments)

# global regu pack
test_regupack = AA.RegularizerPack(mythings,myselection,myregudefs)

# now if I pack again with assignment local , I should see no change
# with assignment_bad the elements should be misplaced

x0 = copy(test_regupack.xnonreg)

AA.pack_allx!(test_regupack.xreg,test_regupack.xnonreg,
  mythings, myassignments,test_regupack.regus)
x0b = copy(test_regupack.xnonreg)
extrema(x0-x0b)

AA.pack_allx!(test_regupack.xreg,test_regupack.xnonreg,
  mythings, myassignments_local,test_regupack.regus)
x1 = copy(test_regupack.xnonreg)
extrema(x0-x1)

# this will not work!
AA.pack_allx!(test_regupack.xreg,test_regupack.xnonreg,
  mythings, myassignments_local_bad,test_regupack.regus)
x10 = copy(test_regupack.xnonreg)
extrema(x0-x1)


#


##
