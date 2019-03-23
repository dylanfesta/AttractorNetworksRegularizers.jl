
using Revise
using AttractorNetworksRegularizers ; const AA = AttractorNetworksRegularizers
using LinearAlgebra
using BenchmarkTools
using Plots
using EponymTuples

##
# _regu = AA.SigmoidalPlus(3.0,3.)
_regu = AA.SigmoidalMinus(3.0,3.)
# _regu = AA.NoRegu()
# _regu = AA.SigmoidalBoth(10.,-20.,30.)

x0 = range(-3.;stop=3,length=500)   |> collect
plot(x0,AA.reg(x0,_regu),leg=false)

AA.ireg( AA.reg(2.01,_regu) , _regu )
AA.gradient_test(0.0001, _regu)

@code_warntype AA.reg!(fill(0.0,3) ,[1,2,0.0],_regu)

@code_warntype AA.reg(1.3,_regu)

_  = let  boh=fill(0.0,400),
  mah = randn(400)
  @btime AA.reg!($boh,$mah,_regu)
end


## pack testing!

A = rand(10,10)
B = randn(50)
C = fill(7.,88)

mythings =  @eponymtuple(A,B,C)
myregudefs  = (nope = AA.NoRegu() , plus = AA.SigmoidalPlus(10.0,2.))

myselections = AA.make_empty_selection(mythings)
myselections.A[:,4] .= :plus
myselections.A[1,:] .= :nope
myselections.B[1:4] .= :nope
myselections.C[18:49] .= :plus

test_regupack = AA.RegularizerPack(mythings,myselections,myregudefs)
##
xtest  = let n=length(test_regupack)
  randn(n)
end
whatevs  = AA.gradient_test(test_regupack,xtest)
count(whatevs[2].==1)

count(test_regupack.regus .== [AA.NoRegu()] )
