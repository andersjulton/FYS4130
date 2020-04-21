import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Wolff(object):
    def __init__(self, Lx, Ly, T, J, MCsteps):
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx*Ly
        self.T = T
        self.prob = 1 - np.exp(-2*J/T)
        self.steps = MCsteps
        s = np.random.randint(0, 2, (self.Lx, self.Ly))
        self.s = np.where(s==0, -1, s)
        self.c = np.ones((self.Lx, self.Ly), dtype=bool)


        if self.Ly or self.Lx == 1:
            self.L = np.maximum(self.Lx, self.Ly)
            self.correlation = self.correlation1D
        else:
            self.correlation = self.correlation2D

        # Observables

    def add_to_cluster(self, i, j, spin):
        if self.s[i][j] == spin:
            if np.random.rand(1) < self.prob:
                self.Wolff_alg(i, j, spin)

    def Wolff_alg(self, i, j, spin):
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

    def correlation2D(self):
        r = np.zeros(self.Lx)
        samp = np.zeros(self.Lx)
        for i in range(self.Lx):
            for j in range(self.Ly):
                r[np.abs(i - j)] += self.s[i][i]*self.s[i][j]
                r[np.abs(i - j)] += self.s[i][i]*self.s[j][i]
                samp[np.abs(i - j)] += 2
        self.cr += r/samp

    def correlation1D(self):
        r = np.zeros(self.L)
        samp = np.zeros(self.L)
        for i in range(self.L):
            r[i] += self.s[0][0]*self.s[i][0]
            samp[i] += 1
        self.cr += r/samp

    def anal_corr(self, r):
        enum = (np.cosh(1/self.T)/np.sinh(1/self.T))**r*np.tanh(1/self.T)**self.Lx + np.tanh(1/self.T)**r
        denum = 1 + np.tanh(1/self.T)**Lx
        return enum/denum

    def MC(self):
        self.cr = 0
        self.Msum = 0
        for k in tqdm(range(int(self.steps / 5))):
            self.c[:] = False

            i = int(np.random.rand(1)*self.Lx)
            j = int(np.random.rand(1)*self.Ly)
            self.Wolff_alg(i, j, self.s[i][j])
        for k in tqdm(range(self.steps)):
            self.c[:] = False

            i = int(np.random.rand(1)*self.Lx)
            j = int(np.random.rand(1)*self.Ly)
            self.Wolff_alg(i, j, self.s[i][j])
            self.correlation()
            M = np.sum(self.s)
            self.Msum += M / self.N
        self.cr /= self.steps
        self.Msum /= self.steps


if __name__ == "__main__":

    figpath = "Plots/"

    b = True
    c = False
    d = False
    f = False

    if b:
        Lx = 16
        Ly = 1
        T1 = 0.5
        T2 = 1
        J = 1
        MCsteps = 100000

        a1 = Wolff(Lx, Ly, T1, J, MCsteps)
        a1.MC()
        a2 = Wolff(Lx, Ly, T2, J, MCsteps)
        a2.MC()

        r = np.arange(0, 16)
        analcorr1 = a1.anal_corr(r)
        analcorr2 = a2.anal_corr(r)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].set_title("T/J = 0.5")
        ax[0].plot(r, analcorr1, label="Analytic")
        ax[0].plot(r, a1.cr, label="MC model")
        ax[0].set_xlabel(r"$r$")
        ax[0].set_ylabel(r'$C(r)$')
        ax[0].legend()

        ax[1].set_title("T/J = 1")
        ax[1].plot(r, analcorr2, label="Analytic")
        ax[1].plot(r, a2.cr, label="MC model")
        ax[1].set_xlabel(r"$r$")
        ax[1].set_ylabel(r'$C(r)$')
        ax[1].legend()
        plt.savefig("2b.pdf")
        plt.show()

    if c:
        Lx = Ly = 16
        J = 1
        MCsteps = 30000
        T = np.linspace(0.5, 3, 20)
        m = np.zeros(len(T))
        for i, t in enumerate(T):
            a = Wolff(Lx, Ly, t, J, MCsteps)
            a.MC()
            m[i] = np.abs(a.Msum)
        plt.plot(T, m)
        plt.show()
