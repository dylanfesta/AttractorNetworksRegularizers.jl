
using Revise
using BenchmarkTools
using AttractorNetworksRegularizers ; const AA = AttractorNetworksRegularizers
using LinearAlgebra
using Plots

##
# _regu = AA.SigmoidalPlus(3.0,3.)
_regu = AA.SigmoidalMinus(3.0,3.)
# _regu = AA.NoRegu()
# _regu = AA.SigmoidalBoth(10.,-20.,30.)

x0 = range(-3.;stop=3,length=500)   |> collect
plot(x0,AA.reg(x0,_regu),leg=false)

AA.ireg( AA.reg(2.01,_regu) , _regu )
AA.gradient_test(-1., _regu)

@code_warntype AA.reg!(fill(0.0,3) ,[1,2,0.0],_regu)

@code_warntype AA.reg(1.3,_regu)

_  = let  boh=fill(0.0,400),
  mah = randn(400)
  @btime AA.reg!($boh,$mah,_regu)
end


## pack testing!

A = rand(10,10)
B = randn(50)
C = fill(-3.,88)

mythings = (A=A,B=B,C=C)
myregudefs  = (nope = AA.NoRegu() , plus = AA.SigmoidalPlus(4.0,10.))

myselections = AA.make_empty_selection(mythings)
myselections.A[1,:] .= :nope
myselections.A[:,4] .= :plus
myselections.B[1:4] .= :nope
myselections.C[18:49] .= :plus


whatevs = AA.RegularizerPack(mythings,myselections,myregudefs)

size(whatevs.C)

boh = NamedTuple{(:ciao,:uff)}([12,"lalal"])
mah = :ciao
getfield(boh,mah)


boh=Vector{AA.Regularizer}(undef,10)
boh[1] = AA.NoRegu()
boh[2] =  myregudefs.plus

boh[1]
