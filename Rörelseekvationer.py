import numpy as np
import pandas as pd
import sympy as sp
from IPython.core.display_functions import display
import matplotlib.pyplot as plt
from matplotlib import rc
from mpmath import arange
from sympy import symbols, Eq, Function
import math
plt.style.use('dark_background')

# plt.style.use('dark_background')
font_properties = {'family': 'Times New Roman', 'weight': 'roman', 'size': '11'}
rc('font', **font_properties)
"""
t, I0, k, l, m_G, g, c = symbols('t I_0 k l m_G g c')
theta = symbols('θ', cls=Function)
fi = symbols('φ', cls=Function)

φ = fi(t)
θ = theta(t)
fdot = sp.diff(φ, t)
fddot = sp.diff(fdot, t)
tdot = sp.diff(θ, t)
tddot = sp.diff(tdot, t)

T = 0.5 * I0 * fdot**2 + 0.5 * I0 * tdot**2
U = 0.5 * k * l**2 * (sp.sin(φ) - sp.cos(θ))**2 - m_G * g * c * sp.cos(φ) - m_G * g * c * sp.cos(θ)

L = T - U

eq_fi = sp.simplify(sp.diff(sp.diff(L, fdot), t) - sp.diff(L, φ))
eq_theta = sp.simplify(sp.diff(sp.diff(L, tdot), t) - sp.diff(L, θ))
"""


class Coupled_Oscillators():
    def __init__(self, I0, k, m_G, l, g, c, φ0, θ0, φ_dot0, θ_dot0):
        """

        :param I0:
        :param k:
        :param m_G:
        :param l:
        :param g:
        :param c:
        :param φ0:
        :param θ0:
        :param φ_dot0:
        :param θ_dot0:
        :param r_dot0:
        """
        self.c = c
        self.g = g
        self.I0 = I0
        self.m_G = m_G
        self.φ0 = φ0
        self.θ0 = θ0
        self.φ_dot0 = φ_dot0
        self.θ_dot0 = θ_dot0
        self.l = l
        self.k = k

    def potential_energi(self):
        """Beräknar systemets potentiella energi"""
        m_G = self.m_G
        l = self.l
        φ0 = self.φ0
        θ0 = self.θ0
        k = self.k
        g = self.g
        c = self.c

        fjäder = 0.5 * k * l ** 2 * (math.sin(φ0) - math.cos(θ0)) ** 2
        tyngdkraft = -m_G * g * c * math.cos(φ0) - m_G * g * c * math.cos(φ0)
        return fjäder + tyngdkraft

    def kinetisk_energi(self):
        I0 = self.I0
        φdot = self.φ_dot0
        θdot = self.θ_dot0

        T = 0.5 * I0 * φdot ** 2 + 0.5 * I0 * θdot ** 2

        return T

    def mekanisk_energi(self):
        return self.kinetisk_energi() + self.potential_energi()

    def Lagrange_HL(self, φ0, θ0, φ_dot0, θ_dot0):
        # x0 = self.x0
        # θ0 = self.θ0
        # r0 = self.r0

        c = self.c
        g = self.g
        I0 = self.I0
        m_G = self.m_G
        l = self.l
        k = self.k

        # x_dot0 = self.x_dot0
        # r_dot0 = self.r_dot0
        # θ_dot0 = self.θ_dot0
        #g1 = (1 / I0) * (- m_G * g * c * math.sin(φ0) + k * l ** 2 * (math.sin(φ0) - math.sin(φ0)) * math.cos(φ0))
        #g2 = (1 / I0) * (- m_G * g * c * math.sin(θ0) - k * l ** 2 * (math.sin(θ0) - math.sin(φ0)) * math.sin(θ0))
        g1 = (1/I0) * (- m_G * g * c) * φ0 - (1/I0) * k * l ** 2 * (φ0 - θ0)
        g2 = (1/I0) * (- m_G * g * c) * θ0 - (1/I0) * k * l ** 2 * (θ0 - φ0)
        return np.array([φ_dot0, θ_dot0, g1, g2])

    def tidssteg(self, dt):
        φ0 = self.φ0
        θ0 = self.θ0
        φ_dot0 = self.φ_dot0
        θ_dot0 = self.θ_dot0

        y = np.array([φ0, θ0, φ_dot0, θ_dot0])

        k1 = self.Lagrange_HL(*y)
        k2 = self.Lagrange_HL(*(y + dt * k1 / 2))
        k3 = self.Lagrange_HL(*(y + dt * k2 / 2))
        k4 = self.Lagrange_HL(*(y + dt * k3))

        R = 1.0 / 6.0 * dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.φ0 += R[0]
        self.θ0 += R[1]
        self.φ_dot0 += R[2]
        self.θ_dot0 += R[3]


def I0(k, l, c, mc, mr, R1, R2, hc, sc, y1, R3, R4, hr, sr):
    Ic = (1 / 12) * mc * (3 * (R1 ** 2 + R2 ** 2) + hc ** 2)
    Ir = (1 / 12) * mr * (3 * (R3 ** 2 + R4 ** 2) + hr ** 2)

    return Ic + Ir + mc * sc ** 2 + mr * sr ** 2


k = 3.04
l = 12.5 / 100
c = 54 / 100
mc = 72 / 1000
R1 = (0.5 / 2) / 100
R2 = 0.012
hc = 0.018
sc = 0.549

R4 = 0.0030
R3 = 0.0029
hr = 0.558
sr = 0.279
mr = 31 / 1000
y1 = 54 / 100
m_G = mr + mc
g = 9.82

I0c = I0(k, l, c, mc, mr, R1, R2, hc, sc, y1, R3, R4, hr, sr)

θ0 = 0
φ0 = 0.25
φdot = 0
θdot = 0

dt = 0.0001

S = Coupled_Oscillators(I0c, k, m_G, l, g, c, φ0, θ0, φdot, θdot)

E0 = S.mekanisk_energi()
max_dE = 0

step = 0
max_iter = 1000000
θ1 = np.zeros(max_iter)
φ1 = np.zeros(max_iter)
"""
while 1:
    Et = S.mekanisk_energi()
    φ1[step] = S.φ0
    θ1[step] = S.θ0
    # print(
    #    f'Tidsteg: {step * dt}, E_t = {max(abs(Et - E0), max_dE)}')  # x = {S.x0},$x.$ = {S.x_dot0}, α = {-S.x0/R}, r = {S.r0}, θ = {S.θ0}')
    S.tidssteg(dt)
    step += 1

    if step >= max_iter:
        break

t = np.arange(0, step * dt, dt)
Ω = np.fft.fftfreq(len(t), 0.0001)
Φ = np.fft.fft(φ1)
Φ_φ = np.fft.fftshift(Φ)
Ω_φ = np.fft.fftshift(Ω)

#φ1 += 0.2 * np.random.randn(len(t))

ax1 = plt.subplot(211)
ax1.plot(t, φ1)


# plt.title(f'Egenfrekvenser')
ax2 = plt.subplot(212)
ax2.plot(Ω_φ, np.abs(Φ_φ))
ax2.set_xlim(-15, 15)
"""












plt.show()