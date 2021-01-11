
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
