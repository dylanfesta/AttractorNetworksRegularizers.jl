
using Revise
using AttractorNetworksRegularizers ; const AA = AttractorNetworksRegularizers
using LinearAlgebra, StatsBase, Statistics
using BenchmarkTools
# using Plots
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
D = fill(-33.,3,3)

mythings =  @eponymtuple(A,B,C,D)
myregudefs = ( nope = AA.NoRegu() ,
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
mythings =  @eponymtuple(A,B)
myregudefs = ( nope = AA.NoRegu() ,
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
