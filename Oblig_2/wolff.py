import numpy as np
import matplotlib.pyplot as plt
import sys


sys.setrecursionlimit(1500)
from tqdm import tqdm

class Wolff:
    '''
    Class for solving the Ising model in 1D or 2D,
    using the Wolff algorithm
    '''
    def __init__(self, Lx, Ly, T, J, MCsteps):
        self.Lx = Lx                                        #Spin in x-direction
        self.Ly = Ly                                        #Spin in y-direction
        self.N = Lx*Ly                                      #Total number of spins
        self.T = T                                          #Temperature
        self.prob = 1 - np.exp(-2*J/T)                      #Probability to add bond
        self.steps = MCsteps                                #Number of Monte Carlo steps
        s = np.random.randint(0, 2, (self.Lx, self.Ly))     #Spin matrix
        self.s = np.where(s==0, -1, s)
        self.c = np.ones((self.Lx, self.Ly), dtype=bool)    #Cluster matrix


        if self.Ly or self.Lx == 1:
            self.L = np.maximum(self.Lx, self.Ly)
            self.correlation = self.correlation1D
        else:
            self.correlation = self.correlation2D


    def add_to_cluster(self, i, j, spin):
        '''
        Function for flipping spin with probability p = 1 - e^(-2J/T)
        '''
        if self.s[i][j] == spin:
            if np.random.rand(1) < self.prob:
                self.Wolff_alg(i, j, spin)

    def Wolff_alg(self, i, j, spin):
        '''
        Wolff algorithm with periodic boundary conditions. Two matrices are used.
        s is the spin matrix, holding the spin values for each site.
        c is the cluster matrix, where spins are added according to the algorithm.
        '''
        self.c[i][j] = True
        self.s[i][j] *= -1

        iP = self.Lx - 1 if i == 0 else i - 1
        iN = 0 if i == self.Lx - 1 else i + 1
        jP = self.Ly - 1 if j == 0 else j - 1
        jN = 0 if j == self.Ly - 1 else j + 1


        if not self.c[iP][j]:
            self.add_to_cluster(iP, j, spin)
        if not self.c[iN][j]:
            self.add_to_cluster(iN, j, spin)
        if not self.c[i][jP]:
            self.add_to_cluster(i, jP, spin)
        if not self.c[i][jN]:
            self.add_to_cluster(i, jN, spin)

    def MC(self):
        '''
        Monte Carlo simulation using the Wolff algorithm.
        Observables are also measured here
        '''
        self.cr = 0

        # Equilibrium cycles
        for k in range(int(self.steps / 5)):
            # Reset the cluster matrix
            self.c[:] = False

            # Select site by random and run the Wolff algorithm
            i = int(np.random.rand(1)*self.Lx)
            j = int(np.random.rand(1)*self.Ly)
            self.Wolff_alg(i, j, self.s[i][j])

        # Measure observables
        M = np.mean(self.s)
        self.M1 = M             # Mean magnetization per spin
        self.M2 = M**2
        self.M4 = M**4
        self.correlation()
        for k in range(self.steps):
            # Reset the cluster matrix
            self.c[:] = False

            # Select site by random and run the Wolff algorithm
            i = int(np.random.rand(1)*self.Lx)
            j = int(np.random.rand(1)*self.Ly)
            self.Wolff_alg(i, j, self.s[i][j])

            # Measure observables
            self.correlation()
            M = np.mean(self.s)
            self.M1 += M
            self.M2 += M**2
            self.M4 += M**4

        # Normalize observables
        self.cr /= (self.steps +1)
        self.M1 /= (self.steps +1)
        self.M2 /= (self.steps +1)
        self.M4 /= (self.steps +1)

    def correlation2D(self):
        '''
        Function for calculating the correlation function for two dimensions.
        '''
        r = np.zeros(self.Lx)
        samp = np.zeros(self.Lx)
        for i in range(self.Lx):
            for j in range(self.Ly):
                r[np.abs(i - j)] += self.s[i][i]*self.s[i][j]
                r[np.abs(i - j)] += self.s[i][i]*self.s[j][i]
                samp[np.abs(i - j)] += 2
        self.cr += r/samp

    def correlation1D(self):
        '''
        Function for calculating the correlation function for one dimension.
        '''
        r = np.zeros(self.L)
        for i in range(self.L):
            r[i] += self.s[0][0]*self.s[i][0]
        self.cr += r


    def analytic_corr(self, r):
        '''
        Exact solution to the correlation function in one dimension.
        '''
        enum = (np.cosh(1/self.T)/np.sinh(1/self.T))**r*np.tanh(1/self.T)**self.Lx + np.tanh(1/self.T)**r
        denum = 1 + np.tanh(1/self.T)**Lx
        return enum/denum

if __name__ == "__main__":

    figpath = "Plots/"
    fontsize = 15

    b = True
    c = False
    d = False
    f = False
    fc = False

    if b:
        Lx = 16
        Ly = 1
        T1 = 0.5
        T2 = 1
        J = 1
        MCsteps = 30000

        a1 = Wolff(Lx, Ly, T1, J, MCsteps)
        a1.MC()
        a2 = Wolff(Lx, Ly, T2, J, MCsteps)
        a2.MC()

        r = np.arange(0, 16)
        analcorr1 = a1.analytic_corr(r)
        analcorr2 = a2.analytic_corr(r)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(r, analcorr1, label="Analytic")
        ax[0].plot(r, a1.cr, label="MC model")
        ax[0].set_xlabel(r"$r$", fontsize=fontsize)
        ax[0].set_ylabel(r'$C(r)$', fontsize=fontsize)
        ax[0].legend(fontsize=fontsize)
        ax[0].tick_params('both', labelsize= 15)
        ax[0].grid()
        ax[0].text(0.2, 0.95, r'$T/J = 0.5$',transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

        ax[1].plot(r, analcorr2, label="Analytic")
        ax[1].plot(r, a2.cr, label="MC model")
        ax[1].set_xlabel(r"$r$", fontsize=fontsize)
        ax[1].set_ylabel(r'$C(r)$', fontsize=fontsize)
        ax[1].legend(fontsize=fontsize)
        ax[1].tick_params('both', labelsize= 15)
        ax[1].grid()
        ax[1].text(0.2, 0.95, r'$T/J = 1$',transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


        plt.tight_layout()
        #plt.savefig(figpath + "b.pdf")
        plt.show()

    if c:
        save = False

        Lx = Ly = 16
        J = 1
        MCsteps = 20000
        T = np.linspace(0.1, 5, 40)
        m = np.zeros(len(T))
        if save:
            for i, t in enumerate(tqdm(T)):
                a = Wolff(Lx, Ly, t, J, MCsteps)
                a.MC()
                m[i] = a.M1
            np.save("m1_C", m)
        else:
            m = np.load("m1_C.npy")
        plt.scatter(T, m, s = 6)
        plt.plot(T, m)
        plt.xlabel(r"$T/J$", fontsize=fontsize)
        plt.ylabel(r"$\langle m\rangle$", fontsize=fontsize)
        plt.grid()
        plt.tight_layout()
        #plt.savefig(figpath + "c.pdf")
        plt.show()

    if d:
        save = False

        Lx = Ly = 16
        J = 1
        MCsteps = 20000
        T = np.linspace(0.1, 5, 40)
        m = np.zeros(len(T))
        if save:
            for i, t in enumerate(tqdm(T)):
                a = Wolff(Lx, Ly, t, J, MCsteps)
                a.MC()
                m[i] = a.M2
            np.save("m2_D", m)
        else:
            m = np.load("m2_D.npy")
        plt.scatter(T, m, s = 6)
        plt.plot(T, m)
        plt.xlabel(r"$T/J$", fontsize=fontsize)
        plt.ylabel(r"$\langle m^2\rangle$", fontsize=fontsize)
        plt.grid()
        plt.tight_layout()
        #plt.savefig(figpath + "d.pdf")
        plt.show()

    if f:
        save = False

        J = 1
        MCsteps = 20000
        L = [8, 16, 32]
        T = np.linspace(2.0, 3.0, 30)
        m2 = np.zeros((len(L), len(T)))
        m4 = m2.copy()
        if save:
            for i, l in enumerate(L):
                for j, t in enumerate(tqdm(T)):
                    a = Wolff(l, l, t, J, MCsteps)
                    a.MC()
                    m2[i][j] = a.M2
                    m4[i][j] = a.M4
            np.save("m2_F", m2)
            np.save("m4_F", m4)
        else:
            m2 = np.load("m2_F.npy")
            m4 = np.load("m4_F.npy")

        for i in range(3):
            gamma = m4[i]/m2[i]**2
            plt.scatter(T, gamma, s = 6)
            plt.plot(T, gamma, label= "L = %d" % L[i])
        plt.xlabel(r"$T/J$", fontsize=fontsize)
        plt.ylabel(r"$\Gamma$", fontsize=fontsize)
        plt.axvline(x = 2/(np.log(1 + np.sqrt(2))))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        #plt.savefig(figpath + "fa.pdf")
        plt.show()

    if fc:
        save = True

        J = 1
        MCsteps = 20000
        L = [8, 16, 32]
        T = np.linspace(2.2, 2.3, 50)
        m2 = np.zeros((len(L), len(T)))
        m4 = m2.copy()
        if save:
            for i, l in enumerate(L):
                for j, t in enumerate(tqdm(T)):
                    a = Wolff(l, l, t, J, MCsteps)
                    a.MC()
                    m2[i][j] = a.M2
                    m4[i][j] = a.M4
            np.save("m2_Fc", m2)
            np.save("m4_Fc", m4)
        else:
            m2 = np.load("m2_Fc.npy")
            m4 = np.load("m4_Fc.npy")

        for i in range(3):
            lim = 3
            avg_mask = np.ones(4) / 4
            gamma = m4[i]/m2[i]**2
            gammac = np.convolve(gamma, avg_mask, 'same')
            plt.scatter(T[2:-2], gamma[2:-2], s = 6)
            plt.plot(T[2:-2], gammac[2:-2], label= "L = %d" % L[i])
        plt.xlabel(r"$T/J$", fontsize=fontsize)
        plt.ylabel(r"$\Gamma$", fontsize=fontsize)
        plt.axvline(x = 2/(np.log(1 + np.sqrt(2))))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        #plt.savefig(figpath + "fc.pdf")
        plt.show()
