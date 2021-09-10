push!(LOAD_PATH, abspath(@__DIR__,".."))
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

regp = AA.SigmoidalPlus(10.0,2.)
regp2 = AA.NoRegu()

regp4A = let _A = falses(size(A))
  for i in eachindex(_A)
    if rand(Bool)
      _A[i] = true
    end
  end
  _A
end
regp24A = let _A = falses(size(A))
  for i in eachindex(_A)
    if (!regp4A[i]) && (rand() > 0.6)  # must not superimpose with previous
      _A[i]=true
    end
  end
  _A
end
reguA = AA.RegularizedUnit(A,[(regp,regp4A),(regp2,regp24A)])

A[regp4A] .= 8.888
A[regp24A] .= NaN

# matrix elements sent to ys
AA.local_pack!(reguA)

@test all(reguA.ys[1] .== 8.888)
@test all(isnan.(reguA.ys[2]))
@test count(isnan.(A)) == length(reguA.ys[2])

# revert
reguA.ys[1] .= 0.0
reguA.ys[2] .= -123.
AA.local_unpack!(reguA)

@test all(iszero.(A[regp4A]))
@test all(A[regp24A] .== -123.0)

##

AA.in_bounds(0.5,regp)

0 < 0.5 < regp.gain