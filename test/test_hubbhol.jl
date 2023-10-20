using PauliOperators 
using Test
using LinearAlgebra
using Random
using BlockDavidson
using HubbardHolstein

@testset "Hubbard-Holstein" begin
    H, Hf, Hb, Dens, Disp, N, g, Nqubit, Nf, Nb, L = hubbard_holstein_1d(4, 1, -1.0, 1.0, 0.5, 0.0, true)
    function mymatvec(v)
    return H*v
    end
    lmat = LinOpMat{ComplexF64}(mymatvec, 2^Nqubit, true)
    dav = Davidson(lmat, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    ei,vi = eigs(dav)
    function mymatvec_fermi(v)
    return Hf*v
    end
    lf = LinOpMat{ComplexF64}(mymatvec_fermi, 2^Nf, true)
    df = Davidson(lf, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    ej,vj = eigs(df)
    function mymatvec_bos(v)
    return Hb*v
    end
    lb = LinOpMat{ComplexF64}(mymatvec_bos, 2^Nb, true)
    db = Davidson(lb, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    ek,vk = eigs(db)
    ei = real.(ei[1])
    ej = real.(ej[1])
    ek = real.(ek[1])
    
    @test ei-ej-ek < 1e-10 

    H, Hf, Hb, Dens, Disp, N, g, Nqubit, Nf, Nb, L = hubbard_holstein_1d(2, 1, -1.0, 1.0, 0.5, 0.2, false)
    lmat = LinOpMat{ComplexF64}(mymatvec, 2^Nqubit, true)
    dav = Davidson(lmat, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    ei,vi = eigs(dav)
    lf = LinOpMat{ComplexF64}(mymatvec_fermi, 2^Nf, true)
    df = Davidson(lf, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    ef,vf = eigs(df)
    lb = LinOpMat{ComplexF64}(mymatvec_bos, 2^Nb, true)
    db = Davidson(lb, T=ComplexF64, max_ss_vecs=8, max_iter=500, tol=1e-6, nroots=1)
    eb,vb = eigs(db)

    e_c, vb_c, vf_c, Hf_c, Hb_c = cluster_mean_field(Hf, Hb, Dens, Disp, vf, vb, g, Nf, Nb, L)
    nf = real.(diag(vf_c'*(N*vf_c)))
    Nfull = N ⊗ Pauli(Nb)
    n = real.(diag(vi'*(Nfull*vi)))
    
    @test n[1]-nf[1] < 1e-10
    
    vprod = vprod = kron(vb_c, vf_c)
    eprod = vprod'*(H*vprod) 
    e_c = real.(e_c[1])
    eprod = real.(eprod[1])
    @test e_c-eprod < 1e-10

    gradf, gradb = gradient_check(Nqubit,Nf,H,vprod)
    @test gradf < 1e-6
    @test gradb < 1e-6
    
    dEpq, ΔEpq = finite_difference_comparison(Nqubit, H, vprod, 6, 7)
    @test (real.(ΔEpq[1]) - real.(dEpq[1])) < 1e-11

    vrand = rand(ComplexF64, 2^Nqubit,1)
    de_pq, Δe_pq = finite_difference_comparison(Nqubit, H, vrand, 6, 7)
    @test (real.(Δe_pq[1]) - real.(de_pq[1])) < 1e-11

    H1d = H    
    connect = [(1,2)]
    Hnew, Hf, Hb, Dens, Disp, N, g, Nqubit, Nf, Nb, L = hubbard_holstein(connect, 1, -1.0, 1.0, 0.5, 0.2)
    @test Hnew ≈ H1d
end
