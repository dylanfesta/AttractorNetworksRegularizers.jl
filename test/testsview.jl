
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

## Selector

A = rand(100,100)
B = randn(50)
things =  Dict(:A=>A,:B=>B)
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
things[:A][Amask] .= Inf
myselections[:A][Agood] .= :p
things[:B][Bmask] .= Inf
myselections[:B][Bgood] .= :b
A0,B0 = [copy(things[x]) for x in  [:A,:B]]

pk=AA.RegularizerPack(things,myselections,regurefs)
AA.pack!(pk)
# two complementary selections for A
As2 = A0 .> 0.5
As2c = .! As2

selA1 = AA.UnitSelector(:As2,isfinite.(A0),pk.units[1])
selA2=AA.UnitSelector(:As1,As2,pk.units[1])
selA2c=AA.UnitSelector(:As1c,As2c,pk.units[1])
selB =  AA.UnitSelector(:Bs,isfinite.(B0),pk.units[2])


##

AA.pack_xandgrad!(pk)
# numeric first
g_num = similar(pk.xnonreg)
for i in eachindex(g_num)
  g_num[i] = Calculus.gradient(pk.regus[i],pk.xnonreg[i])
end
# now analytic
for u in pk.units
  fill!(u.Mg,0.333)
end
g_an1=fill(-0.789,length(pk))
myfix(x) = (x + 0.789)/0.333
##

# add B first
AA.propagate_gradient!(g_an1,pk,selB)
AA.propagate_gradient!(g_an1,pk,selA1)
@test all(isapprox.(g_num,myfix.(g_an1);atol=0.01))
plotvs(g_num,myfix.(g_an1))

g_an1=fill(-0.789,length(pk))
# add B first
AA.propagate_gradient!(g_an1,pk,selB)
# selection is only partial
AA.propagate_gradient!(g_an1,pk,selA2)
@test !all(isapprox.(g_num,myfix.(g_an1);atol=0.01))
# slection should be full
AA.propagate_gradient!(g_an1,pk,selA2c)
@test all(isapprox.(g_num, myfix.(g_an1) ;atol=0.01))
