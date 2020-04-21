import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

def F(Nx, Ny, Nz, alpha, gamma, V):
    N = Nx + Ny + Nz
    term1 = Nx*np.log(alpha*Nx/V)
    term2 = Ny*np.log(alpha*Ny/V)
    term3 = Nz*np.log(alpha*Nz/V)
    term4 = (gamma/V)*(Nx*Ny + Ny*Nz + Nz*Nx)
    return term1 + term2 + term3 + term4

def P(Nx, Ny, Nz, gamma, V):
    N = Nx + Ny + Nz
    return (gamma/V**2)*(Nx*Ny + Ny*Nz + Nz*Nx) + N*V

def Ncomb(n):
    x = np.linspace(1, n, n)
    a,b,c = np.meshgrid(x,x,x)
    indices = np.where(a + b + c == n)
    return a[indices], b[indices], c[indices]

def getFeq(N, V, alpha, gamma):
    Feq2 = np.zeros(len(V))
    Ns = np.zeros((len(V), 3))
    Nx, Ny, Nz = Ncomb(N)
    for i, v in enumerate(tqdm(V)):
        Feq = np.zeros(len(Nx))
        for j in range(len(Nx)):
            Feq[j] = F(Nx[j], Ny[j], Nz[j], alpha, gamma, v)
        Feq2[i] = np.min(Feq)
        Ns[i] = [Nx[np.argmin(Feq)], Ny[np.argmin(Feq)], Nz[np.argmin(Feq)]]
    return Feq2, Ns

N = 100
V = np.linspace(0.01, 0.5, 1000)
alpha = 0.1
gamma = 0.001

load = True

Gibbs = False
Helmholtz = True
Hisotrop = False
equicomb = False

if equicomb:
    if load:
        Feq = np.load('Feq.npy')
        NS = np.load('Ns.npy')
    else:
        Feq, NS = getFeq(N, V, alpha, gamma)
    Ns = np.max(NS, axis=1)
    plt.plot(N/V, Ns/N)
    plt.xlabel('N/V', fontsize=14)
    plt.ylabel(r'max$(N_x, N_y, N_z)/N$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('Nequi_a=' + str(alpha) + '_g=' + str(gamma) + '.pdf')
    plt.show()

if Gibbs:
    if load:
        Feq = np.load('Feq.npy')
        NS = np.load('Ns.npy')
    else:
        Feq, NS = getFeq(N, V, alpha, gamma)
    p = np.empty_like(Feq)
    pd = p.copy()
    p[:-1] = -np.diff(Feq) / (np.abs(V[2] - V[1]))
    p[-1] = (Feq[-1] - Feq[-2]) / (np.abs(V[2] - V[1]))
    pd[:-1] = np.diff(p) / (np.abs(V[2] - V[1]))
    pd[-1] = (p[-2] - p[-1]) / (np.abs(V[2] - V[1]))
    G = np.empty_like(Feq)

    for i in range(len(V)):
        G[i] = Feq[i] + p[i]*V[i]

    idxmax = np.min(np.where(p < 4400))
    idxmin = np.max(np.where(p > 5300))

    plt.plot(p, G)
    plt.xlabel('P', fontsize=14)
    plt.ylabel('G/T', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig('G.pdf')
    plt.show()

    plt.plot(p[idxmin:idxmax], G[idxmin:idxmax])
    plt.xlabel('P', fontsize=14)
    plt.ylabel('G/T', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig('Gzoom.pdf')
    plt.show()

if Hisotrop:
    n = N/3
    Nx = N - 2
    Ny = 1
    Nz = Ny
    helmiso = F(n, n, n, alpha, gamma, V)
    helmother = F(Nx, Ny, Nz, alpha, gamma, V)
    idx = np.argwhere(np.diff(np.sign(helmiso - helmother))).flatten()
    plt.plot((N/V), helmiso, label = r'$N_x = N_y = N_z = N/3$')
    plt.plot((N/V), helmother, label = r'$N_x = N - 2, N_y = N_z = 1$')
    #plt.plot((N/V)[idx], helmiso[idx, 'ro', label= r'$N/V = %3.4f$' %(N/V)[idx])
    plt.legend(loc='best')
    plt.xlabel(r'N/V')
    plt.ylabel('F')
    plt.tight_layout()
    plt.savefig('HelmInt_a=' + str(alpha) + '_g=' + str(gamma) + '.pdf')
    plt.show()

if Helmholtz:
    if load:
        Feq = np.load('Feq.npy')
        NS = np.load('Ns.npy')
    else:
        Feq, NS = getFeq(N, V, alpha, gamma)
    Ns = np.max(NS)
    grad = np.gradient(Feq)
    grad2 = np.gradient(grad)

    p = np.empty_like(Feq)
    pd = p.copy()
    p[:-1] = -np.diff(Feq) / (np.abs(V[2] - V[1]))
    p[-1] = -(Feq[-1] - Feq[-2])/ (np.abs(V[2] - V[1]))
    pd[:-1] = np.diff(p) / (np.abs(V[2] - V[1]))
    pd[-2] = pd[-3]
    pd[-1] = pd[-2]

    grad = p
    grad2 = pd

    plt.figure(figsize=(6,4))
    plt.plot(N/V, Feq)

    plt.xlabel(r'N/V', fontsize=14)
    plt.ylabel(r'F/T', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('3bF.pdf')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(N/V, grad, 'bo')
    plt.xlabel(r'N/V', fontsize=14)
    plt.ylabel(r'$-\left(\frac{\partial F}{\partial V}\right) = P$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('3bDF.pdf')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(N/V, grad2)
    plt.xlabel(r'N/V', fontsize=14)
    plt.ylabel(r'$-\left(\frac{\partial^2 F}{\partial V^2}\right) = \left(\frac{\partial P}{\partial V}\right)$', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('3bDDF.pdf')
    plt.show()
