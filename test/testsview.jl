
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
