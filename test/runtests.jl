using AttractorNetworksRegularizers ; const AA = AttractorNetworksRegularizers
using Test
using Calculus

function gradient_test(x::R,regu::AA.Regularizer{R}) where R
  grad_num = Calculus.gradient(xx->regu(xx), x )
  grad_an = AA.dreg(x,regu)
  return grad_num,grad_an
end

@testset "regularizer functions" begin
  x0 = range(-3.;stop=3,length=500)
  for regu in ( AA.SigmoidalPlus(3.0,3.),
      AA.SigmoidalMinus(3.0,3.),AA.NoRegu{typeof(0.0)}(), AA.SigmoidalBoth(10.,-20.,30.))
    # invert
    rx = regu.(x0)
    @test all(isapprox.(AA.ireg(rx,regu),x0;atol=0.001))
    # gradient
    gt = gradient_test.(x0, regu)
    @test all(ab->isapprox(ab...;atol=0.01),gt)
  end
end


@testset "Packing" begin
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
  reguA = AA.RegularizedUnit(A,similar(A),(regp,regp4A),(regp2,regp24A))

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
end

@testset "Full Pack" begin
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
  reguA = AA.RegularizedUnit(A,similar(A),(regp,regp4A),(regno,regp24A))
  reguB = AA.RegularizedUnit(B,similar(B),regno)
  reguC = AA.RegularizedUnit(C,similar(C),regm)
  packed_all = AA.RegularizerPack(reguA,reguB,reguC)
  # matrix to packed test (no regu)
  A[reguA.locals[2]] .= 8.88
  AA.pack!(packed_all)
  @test all(packed_all.x_global[reguA.globals[2]] .== 8.88)
  # packed to matrix test (sigmoidal regu)
  packed_all.x_global[reguC.globals[1]] .= 12.123
  AA.unpack!(packed_all)
  @test all(reguC.xs[1] .== 12.123)
  @test all(isapprox.(C[reguC.locals[1]],regm(12.123)))
end

#=

@testset "Packing" begin
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
  # revert pack unpack
  AA.pack!(pk)
  fill!(things[:A],Inf)
  fill!(things[:B],Inf)
  fill!(things[:C],Inf)
  AA.unpack!(pk)

  @test all(A0 .== things[:A])
  @test all(B0 .== things[:B])
  @test all(C0 .== things[:C])

  # test 2, packing of nonreg
  xreg2 = similar(pk.xreg)
  for i in eachindex(xreg2)
    xreg2[i]=pk.regus[i](pk.xnonreg[i])
  end
  @test all(isapprox.(xreg2,pk.xreg;atol=1E-4))
  # test 3 , gradients
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
end

@testset "Partial selections" begin
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
  # test the empty case
  @test isempty(@test_logs (:warn,) AA.UnitSelector(:Az,falses(size(A)),pk.units[1]))
  #setup complete
  AA.pack_xandgrad!(pk)
  # constructor
  selBB = AA.UnitSelector(pk.units[2])
  @test all( selBB.loc .== selB.loc)
  @test all( selBB.glo .== selB.glo)
  # numeric gradient first
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
  # test 1 , full selection
  # add B first
  AA.propagate_gradient!(g_an1,pk,selB)
  AA.propagate_gradient!(g_an1,pk,selA1)
  @test all(isapprox.(g_num,myfix.(g_an1);atol=0.01))
  # test 2
  g_an1=fill(-0.789,length(pk))
  # add B first
  AA.propagate_gradient!(g_an1,pk,selB)
  # test 2  partial but complementary selections
  AA.propagate_gradient!(g_an1,pk,selA2)
  @test !all(isapprox.(g_num,myfix.(g_an1);atol=0.01))
  # slection should be full
  AA.propagate_gradient!(g_an1,pk,selA2c)
  @test all(isapprox.(g_num, myfix.(g_an1) ;atol=0.01))
end

=#