using PauliOperators
using LinearAlgebra
using BlockDavidson
using HubbardHolstein

"""
    function hubbard_holstein_1d(L::Int, Nbos::Int, t::Float, U::Float, ω::Float, g::Float, pbc::Boolean)

Builds the 1D Hubbard-Holstein Hamiltonian for given parameters and conditions
# Arguments
- `L`: Number of lattice sites, default is 4
- `Nb`: Number of Bosons allowed per lattice site (currently uniform across each site), default is 1
- `t`: Hopping parameter (currently uniform across each site), default is -1.0
- `U`: Repulsion parameter (currently uniform across each site), default is 1.0
- `ω`: Oscillator frequency (currently uniform across each site), default is 0.5
- `g`: Hubbard-Holstein interaction Boson-Fermion parameter (currently uniform across the lattice), default is 0.2
- `pbc`: Periodic boundary conditions for when L>2, default is true
"""
function hubbard_holstein_1d(L=4, Nbos=1, t=-1.0, U=1.0, ω=0.5, g=0.2, pbc=true)

    Nf = 2*L
    Nb = Nbos*L
    Nqubit = Nf + Nb

    Hf = 0.0*Pauli(Nf)    
    # Kinetic Hopping Term
    if L > 1
        for i in 1:L-1
            aiα = jordan_wigner(i,Nf)
            ajα = jordan_wigner(i+1,Nf)
            aiβ = jordan_wigner(i+L,Nf)
            ajβ = jordan_wigner(i+L+1,Nf)
            Hf += t * (aiα*ajα' + ajα*aiα')  
            Hf += t * (aiβ*ajβ' + ajβ*aiβ')  
        end
    end
    if L > 2
        if pbc
            Hf += t * (jordan_wigner(L,Nf)*jordan_wigner(1,Nf)' + jordan_wigner(1,Nf)*jordan_wigner(L,Nf)')
            Hf += t * (jordan_wigner(Nf,Nf)*jordan_wigner(L+1,Nf)' + jordan_wigner(L+1,Nf)*jordan_wigner(Nf,Nf)')
        end
    end 

    # Coulomb repulsion term
    for i in 1:L
        ai = jordan_wigner(i,Nf)
        aj = jordan_wigner(i+L,Nf)
        Hf += U * (ai*ai' * aj*aj') 
    end

    # Harmonic oscillator
    bi = boson_binary_transformation(Nbos)'
    Hb = ω * bi' * bi
    for i in 2:L
        Hb = Hb ⊕ (ω * bi' * bi)
    end

    # Interaction Hamiltonian
    aα = jordan_wigner(1, Nf)
    aβ = jordan_wigner(L+1, Nf)
    H_inter = g * ((aα*aα' + aβ*aβ') ⊗ (bi + bi') ⊗ Pauli((L-1)*Nbos))
    for i in 2:L-1
        aα = jordan_wigner(i, Nf)
        aβ = jordan_wigner(i+L, Nf)
        H_inter += g * ((aα*aα' + aβ*aβ') ⊗ Pauli((i-1)*Nbos) ⊗ (bi + bi') ⊗ Pauli((L-i)*Nbos))
    end    
    aα = jordan_wigner(L, Nf)
    aβ = jordan_wigner(Nf, Nf)
    H_inter += g * ((aα*aα' + aβ*aβ') ⊗ Pauli((L-1)*Nbos) ⊗ (bi + bi'))

    # Assembling the full Hubbard-Holstein Hamiltonian
    H = Hf ⊕ Hb
    H += H_inter

    # Forming the partial-traced interaction Hamiltonian terms
    Disp = Dict()
    Disp[1] = ((bi + bi') ⊗ Pauli((L-1)*Nbos))
    for i in 2:L-1
        Disp[i] = (Pauli((i-1)*Nbos) ⊗ (bi + bi') ⊗ Pauli((L-i)*Nbos))
    end
    Disp[L] = (Pauli((L-1)*Nbos) ⊗ (bi + bi'))
    
    Dens = Dict()
    for i in 1:L
        aα = jordan_wigner(i, Nf)
        aβ = jordan_wigner(i+L, Nf)
        Dens[i] = (aα*aα' + aβ*aβ')
    end

    # Generating the fermionic particle number operator
    N = 0.0*Pauli(Nf)
    for i in 1:L
        a = jordan_wigner(i, Nf)'
        b = jordan_wigner(i+L, Nf)'
        N += a'*a + b'*b
    end
 
    clip!(Hb)
    clip!(Hf)
    clip!(H)
    clip!(N)
    clip!(H_inter)

    return H, Hf, Hb, Dens, Disp, N, g, Nqubit, Nf, Nb, L
end

"""
    function hubbard_holstein(lattice::List, Nbos::Int, t::Float, U::Float, ω::Float, g::Float)

Builds the Hubbard-Holstein Hamiltonian for given parameters and conditions
# Arguments
- `lattice`: A list of the edges that connect a system of L vertices
- `Nb`: Number of Bosons allowed per lattice site (currently uniform across each site), default is 1
- `t`: Hopping parameter (currently uniform across each site), default is -1.0
- `U`: Repulsion parameter (currently uniform across each site), default is 1.0
- `ω`: Oscillator frequency (currently uniform across each site), default is 0.5
- `g`: Hubbard-Holstein interaction Boson-Fermion parameter (currently uniform across the lattice), default is 0.2
"""
function hubbard_holstein(lattice, Nbos=1, t=-1.0, U=1.0, ω=0.5, g=0.2)

    L = maximum(maximum(lattice))
    Nf = 2*L 
    Nb = Nbos*L
    Nq = Nf + Nb

    Hf = 0.0*Pauli(Nf)    
    # Kinetic Hopping Term
    if L > 1
        for (i,j) in lattice
            aiα = jordan_wigner(i,Nf)
            ajα = jordan_wigner(j,Nf)
            aiβ = jordan_wigner(i+L,Nf)
            ajβ = jordan_wigner(j+L,Nf)
            Hf += t * (aiα*ajα' + ajα*aiα')  
            Hf += t * (aiβ*ajβ' + ajβ*aiβ')  
        end
    else
        return hubbard_holstein_1d(1, Nbos, t, U, ω, g, pbc)
    end

    # Coulomb repulsion term
    for i in 1:L
        aα = jordan_wigner(i,Nf)
        aβ = jordan_wigner(i+L,Nf)
        Hf += U * (aα*aα' * aβ*aβ') 
    end

    # Harmonic oscillator
    bi = boson_binary_transformation(Nbos)'
    Hb = ω * bi' * bi
    for i in 2:L
        Hb = Hb ⊕ (ω * bi' * bi)
    end

    # Interaction Hamiltonian
    aα = jordan_wigner(1, Nf)
    aβ = jordan_wigner(L+1, Nf)
    H_inter = g * ((aα*aα' + aβ*aβ') ⊗ (bi + bi') ⊗ Pauli((L-1)*Nbos))
    for i in 2:L-1
        aα = jordan_wigner(i, Nf)
        aβ = jordan_wigner(i+L, Nf)
        H_inter += g * ((aα*aα' + aβ*aβ') ⊗ Pauli((i-1)*Nbos) ⊗ (bi + bi') ⊗ Pauli((L-i)*Nbos))
    end    
    aα = jordan_wigner(L, Nf)
    aβ = jordan_wigner(Nf, Nf)
    H_inter += g * ((aα*aα' + aβ*aβ') ⊗ Pauli((L-1)*Nbos) ⊗ (bi + bi'))

    # Assembling the full Hubbard-Holstein Hamiltonian
    H = Hf ⊕ Hb
    H += H_inter

    # Forming the partial-traced interaction Hamiltonian terms
    Disp = Dict()
    Disp[1] = ((bi + bi') ⊗ Pauli((L-1)*Nbos))
    for i in 2:L-1
        Disp[i] = (Pauli((i-1)*Nbos) ⊗ (bi + bi') ⊗ Pauli((L-i)*Nbos))
    end
    Disp[L] = (Pauli((L-1)*Nbos) ⊗ (bi + bi'))
    
    Dens = Dict()
    for i in 1:L
        aα = jordan_wigner(i, Nf)
        aβ = jordan_wigner(i+L, Nf)
        Dens[i] = (aα*aα' + aβ*aβ')
    end

    # Generating the fermionic particle number operator
    N = 0.0*Pauli(Nf)
    for i in 1:L
        a = jordan_wigner(i, Nf)'
        b = jordan_wigner(i+L, Nf)'
        N += a'*a + b'*b
    end
 
    clip!(Hb)
    clip!(Hf)
    clip!(H)
    clip!(N)
    clip!(H_inter)

    return H, Hf, Hb, Dens, Disp, N, g, Nq, Nf, Nb, L
end

"""
    guess_ket(Nqubit::Int, L::Int)

Builds a guess vector for a half-filled Hubbard-Holstein system
# Arguments
- `Nqubit`: Number of qubits
- `L`: Number of lattice sites
"""
function guess_ket(Nqubit, L)
    dim = 2^Nqubit
    v0 = Array{ComplexF64}(undef, dim, 1)
    for i in 1:dim
        v0[i] = 0.0
    end
    flip = []
    for i in 1:L
        if isodd(i)
            push!(flip,i)
        end
    end
    for i in L+1:2*L
        if iseven(i)
            push!(flip,i)
        end
    end
    #println("list is", flip)    
    θ,ket = Pauli(Nqubit, X=flip) * KetBitString(Nqubit,0)
    v0[ket.v+1] = 1

    return v0
end

"""
    gradient(Hg::PauliSum, vg::Vector, Nf::Int, Nb::Int, L::Int, Nbos::Int, g_thresh::Float)

Computes the energy gradient with respect to CI coefficients
# Arguments
- `H`: Hamiltonian operator
- `v`: Vector used to evaluate the matrix elements of the commutator
- `Nf`: Number of Fermionic qubits
- `Nb`: Number of Bosonic qubits
- `L`: Number of lattice sites
- `Nbos`: Number of bosonic excitations permitted on each lattice site, default is 1
- `g_thresh`: Numerical threshhold (to be implemented)
"""
function gradient(Hg, vg, Nf, Nb, L, Nbos=1, g_thresh=1.0e-5)
    #Nbos = Nb / L
    Ograd = 0.0*Hf
    for i in 1:L
        p = jordan_wigner(i, Nf)
        r = jordan_wigner(i+L, Nf)
        for j in 1:L
            q = jordan_wigner(j, Nf)
            s = jordan_wigner(j+L, Nf)
            Ograd += p*q' + r*s'
        end
    end

    bi = boson_binary_transformation(Nbos)'
    Oient = bi' * bi
    for i in 2:L
        Oient = Oient ⊕ (bi' * bi)
    end
    Og = Ograd ⊕ Oient


    op = Hg*Og - Og*Hg
    g = (vg'*(op*vg))[1]
    return g
end

"""
    cluster_mean_field(Hf::PauliSum, Hb::PauliSum, Dens::PauliSum, Disp::PauliSum, vf::Vector, vb::Vector, g::Float, Nf::Int, Nb::Int, L::Int, max_scf_iter::Int, e_thresh::Float)

Computes the cluster mean-field energy and returns the individual cluster wavefunctions that form the product wavefunction
Note: Currently set up for only two clusters, a bosonic and a fermionic
# Arguments
- `Hf`: Purely Fermionic Hamiltonian operator
- `Hb`: Purely Bosonic Hamiltonian operator
- `Dens`: Fermionic density operator
- `Disp`: Bosonic displacement operator
- `vf`: Eigenvector of Hf
- `vb`: Eigenvector of Hb
- `g`: Hubbard-Holstein interaction Boson-Fermion parameter (currently uniform across the lattice)
- `Nf`: Number of qubits needed to span the Fermionic subspace
- `Nb`: Number of qubits needed to span the Bosonic subspace
- `L`: Number of lattice sites
- `max_scf_iter`: Number of SCF iterations allowed to reach the cluster mean-field energy convergence, default is 10
- `e_thresh`: Numeric threshold that defines convergence of the cluster mean-field energy, defalut is 1.0e-6
"""
function cluster_mean_field(Hf, Hb, Dens, Disp, vf, vb, g, Nf, Nb, L, max_scf_iter=10, e_thresh=1.0e-6)

    Hf_eff = 0.0*Hf
    Hb_eff = 0.0*Hb
    function mymatvec_f(v)
        return Hf_eff*v
    end
    function mymatvec_b(v)
        return Hb_eff*v
    end
    vc_bos = vb*1.0
    vc_fer = vf*1.0
    vf_c = vc_fer
    vb_c = vc_bos
    eprev = 0.0
    for i in 1:max_scf_iter
        # Optimizing the Fermionic cluster in the potential of the Bosonic displacement
        Hf_prime = 0.0*Hf
        for i in 1:L
            ε_disp = (vb_c'*(Disp[i]*vb_c))
            #println("Bosonic shift:", ε_disp)
            Hf_prime += (g * ε_disp[1] * Dens[i])
        end
        clip!(Hf_prime)
        Hf_eff = Hf + Hf_prime
        lmatf = LinOpMat{ComplexF64}(mymatvec_f, 2^Nf, true)
        davf = Davidson(lmatf, T=ComplexF64, max_iter=1000, v0=vf_c, tol=1e-6, nroots=1)
        ef_c,vf_c = eigs(davf) 

        # Optimizing the Bosonic cluster in the potential of the Fermionic density
        Hb_prime = 0.0*Hb
        for i in 1:L
            ε_den = (vf_c'*(Dens[i]*vf_c))
            #println("Fermionic density:", ε_den)
            Hb_prime += (g * ε_den[1] * Disp[i])             
        end
        clip!(Hb_prime)
        Hb_eff = Hb + Hb_prime
        lmatb = LinOpMat{ComplexF64}(mymatvec_b, 2^Nb, true)
        davb = Davidson(lmatb, T=ComplexF64, max_iter=1000, v0=vb_c, tol=1e-6, nroots=1)
        eb_c,vb_c = eigs(davb)

        # Computing the cMF energy        
        e_f = vf_c'*(Hf*vf_c)
        e_b = vb_c'*(Hb*vb_c)
        e_i = 0.0*eb_c
        for i in 1:L
            e_i += vb_c'*(Disp[i]*vb_c) * vf_c'*(Dens[i]*vf_c)
        end 
        e_c = e_f + e_b + (g * e_i)
        de = abs.(real.(e_c)[1] - real.(eprev)[1])

        # Checking convergence
        if de < e_thresh
            println("SCF completed at iteration ", i)
            eprev = e_c
            vc_bos = vb_c
            vc_fer = vf_c
            println("Pure Fermion energy:", real.(e_f[1]))
            println("Pure Boson energy:", real.(e_b[1]))
            println("Interaction energy:", real.(g * e_i[1]))
            println("Cluster mean-field energy: ", real.(e_c[1]))
            break
        else
            println("Error in cluster mean-field energy:", de)
            println("Current cluster mean-field energy:", e_c)
            eprev = e_c
        end
    end

    return eprev, vc_bos, vc_fer, Hf_eff, Hb_eff
end

"""
    gradient_check(Nq::Int, H::PauliSum, vprod::Vector)

Computes the gradient of the energy wrt the CI coefficients
# Arguments
- `Nq`: Number of qubits
- `H`: Hamiltonian
- `vprod`: the product wavefunction
"""
function gradient_check(Nq,Nf,H,vcmf)

    σ = H*vcmf
    df = 2^Nf
    d = 2^Nq
    Nb = Nq-Nf
    db = 2^Nb
    grad = zeros(ComplexF64, df, df)
    for i in 1:df
        p = zeros(ComplexF64, d, 1)
        p[i] = 1
        σp = σ'*p
        Cp = vcmf[i]
        for j in 1:df
            q = zeros(ComplexF64, d, 1)
            q[j] = 1
            if (i == j)
            else
                σq = σ'*q
                Cq = vcmf[j]
                dEpq = 2 * real.(σp*Cq - σq*Cp)
                grad[i,j] = dEpq[1]
                if abs(dEpq[1]) > 1e-6
                    println("Site: ", i, ",", j)
                end
            end
        end
    end
    gnormf = norm(grad)
    #println("gradient norm wrt fermions is: ", gnormf)
    grad = zeros(ComplexF64, db, db)
    for i in 1:db
        p = zeros(ComplexF64, d, 1)
        p[(i-1)*df+1] = 1
        σp = σ'*p
        Cp = vcmf[(i-1)*df+1]
        for j in 1:db
            q = zeros(ComplexF64, d, 1)
            θ,b = Pauli(Nb) * KetBitString(Nb,j-1)
            q[(j-1)*df+1] = 1
            if (i == j)
            else
                σq = σ'*q
                Cq = vcmf[(j-1)*df+1]
                dEpq = 2 * real.(σp*Cq - σq*Cp)
                grad[i,j] = dEpq[1]
                if abs(dEpq[1]) > 1e-6
                    println("Site: ", i, ",", j)
                end
            end
        end
    end
    gnormb = norm(grad)
    #println("gradient norm wrt bosons is: ", gnormb)
    return gnormf, gnormb
end

"""
    finite_difference_comparison(Nq::Int, H::PauliSum, vprod::Vector, state_p::Int, state_q::Int)

Computes the gradient of the energy wrt the CI coefficients
# Arguments
- `Nq`: Number of qubits
- `H`: Hamiltonian
- `vprod`: the product wavefunction
- `state_p`: the p-th state
- `state_q`: the q-th state
"""
function finite_difference_comparison(Nq, H, vprod, state_p, state_q)

    # Defining states and sigma vector
    d = 2^Nq
    σ = H*vprod
    p = zeros(ComplexF64, d, 1)
    q = zeros(ComplexF64, d, 1)
    p[state_p] = 1
    q[state_q] = 1

    # Analytical gradient
    σp = σ'*p
    σq = σ'*q
    Cp = vprod[state_p]
    Cq = vprod[state_q]
    dEpq = 2*real(σp*Cq - σq*Cp)

    # 5-point stencil
    gen = p*q' - q*p'
    δ = 0.001
    G1 = 2*δ
    G2 = δ
    G3 = -1*δ
    G4 = -2*δ
    U1 = exp(G1 * gen)
    U2 = exp(G2 * gen)
    U3 = exp(G3 * gen)
    U4 = exp(G4 * gen)
    e_p2δ = vprod'*((inv(U1)*(H*U1))*vprod)
    e_pδ = vprod'*((inv(U2)*(H*U2))*vprod)
    e_mδ = vprod'*((inv(U3)*(H*U3))*vprod)
    e_m2δ = vprod'*((inv(U4)*(H*U4))*vprod)
    numer = (8*e_pδ) - (8*e_mδ) + e_m2δ - e_p2δ
    denom = 12*δ

    ΔEpq = numer/denom
    #println("Analytic: ", dEpq)
    #println("Finite Difference: ", ΔEpq)
    return dEpq, ΔEpq
end
