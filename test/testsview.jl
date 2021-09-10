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
C = -10 .*  rand(3,122)

regp = AA.SigmoidalPlus(10.0,2.)
regno = AA.NoRegu()
regm =  AA.SigmoidalMinus(10.0,2.)

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
reguA = AA.RegularizedUnit(A,(regp,regp4A),(regno,regp24A))
reguB = AA.RegularizedUnit(B,regno)
reguC = AA.RegularizedUnit(C,regm)
packed_all = AA.RegularizerPack(reguA,reguB,reguC)
@test true


# matrix to packed test (no regu)
A[reguA.locals[2]] .= 8.88
AA.pack!(packed_all)
@test all(packed_all.x_global[reguA.globals[2]] .== 8.88)

# packed to matrix test (sigmoidal regu)
packed_all.x_global[reguC.globals[1]] .= 12.123
AA.unpack!(packed_all)
@test all(reguC.xs[1] .== 12.123)
@test all(isapprox.(C[reguC.locals[1]],regm(12.123)))

