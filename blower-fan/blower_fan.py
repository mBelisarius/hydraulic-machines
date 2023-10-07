import math
import numpy as np
import matplotlib.pyplot as plt


class VelocityTriangle:
    def __init__(self, alpha, beta, c, cm, u, w):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.cm = cm
        self.u = u
        self.w = w


class Fan:
    def __init__(self, rho, Q, PTV, N, eta_v, eta_m, eta_ele, r_D, tau_adm):
        self.rho = rho
        self.Q = Q
        self.PTV = PTV
        self.N = N
        self.eta_v = eta_v
        self.eta_m = eta_m
        self.eta_ele = eta_ele
        self.r_D = r_D
        self.tau_adm = tau_adm

        self.phi = None
        self.psi = None
        self.eta_t = None
        self.Q_r = None
        self.eta_i = None
        self.Y = None
        self.Y_pa = None
        self.P_efe = None
        self.P_ele = None
        self.d = None
        self.u_5 = None
        self.d_5 = None
        self.d_4 = None
        self.c_m5_ = None
        self.Y_pa_inf = None
        self.beta_5_ = None
        self.beta_5 = None
        self.c_u6 = None
        self.beta_6_ = None
        self.beta_6 = None
        self.u_4 = None
        self.c_m4_ = None
        self.beta_4 = None
        self.r = None
        self.z = None
        self.t5 = None
        self.t4 = None
        self.b = None
        self.d_3 = None
        self.c_m3 = None
        self.c_m4 = None
        self.c_m5 = None

    def calculate(self):
        self.phi = 0.85

        k_z = 7.5
        k_b = 0.45

        self.eta_t = 1.09656 * self.phi ** 3 - 3.971 * self.phi ** 2 + 4.38123 * self.phi - 0.804413
        self.psi = 3.201 - 1.9842 * self.phi ** -1 + 2.5216 * self.phi ** -2 - 0.9475 * self.phi ** -3

        self.Q_r = self.Q / self.eta_v
        self.eta_i = self.eta_t / self.eta_m
        self.Y = self.PTV / self.rho
        self.Y_pa = self.Y / self.eta_i
        self.P_efe = self.rho * self.Y_pa * self.Q_r / self.eta_m
        self.P_ele = self.P_efe / self.eta_ele
        self.d = 36.502 * (self.P_efe / (self.tau_adm * self.N)) ** (1 / 3)
        self.u_5 = (2 * self.Y / self.psi) ** (1 / 2)
        self.d_5 = 60 * self.u_5 / (np.pi * self.N)
        self.d_4 = self.r_D * self.d_5
        self.c_m5_ = self.phi * self.u_5
        self.Y_pa_inf = 2 * self.u_5 ** 2
        self.beta_5_ = math.atan(self.c_m5_ / self.u_5)
        self.beta_5 = np.pi - self.beta_5_
        self.c_u6 = self.Y_pa / self.u_5
        self.beta_6_ = math.atan(self.c_m5_ / (self.c_u6 - self.u_5))
        self.beta_6 = np.pi - self.beta_6_
        self.u_4 = np.pi * self.d_4 * self.N / 60
        self.c_m4_ = self.c_m5_ / r_D
        self.beta_4 = math.atan(self.c_m4_ / self.u_4)
        self.r = (self.d_5 - self.d_4) / (2 * (np.cos(self.beta_4) - np.cos(self.beta_5)))
        self.z = round(k_z * (np.cos(self.beta_4) - np.cos(self.beta_5)) / (1 - self.r_D))
        self.t5 = np.pi * self.d_5 / self.z
        self.t4 = np.pi * self.d_4 / self.z
        self.b = k_b * self.d_5
        self.d_3 = 0.95 * self.d_4
        self.c_m3 = 4 * self.Q / (np.pi * self.d_3 ** 2)
        self.c_m4 = 0.44 * self.d_4 * self.c_m3 / self.b
        self.c_m5 = self.c_m4 * self.d_4 / self.d_5

    def voluta_ret(self, n=100):
        theta = np.linspace(0, 2 * np.pi, n)
        r_theta = 0.516 * self.d_5 * np.exp(self.c_m5 * theta / self.c_u6)

        x = r_theta * np.cos(theta + np.pi / 2)
        y = r_theta * np.sin(theta + np.pi / 2)

        plt.plot(x, y)
        plt.plot([0.0, np.min(x)], [np.max(y), np.max(y)])
        plt.plot([0.0, np.min(x)], [y[0], y[0]])

        plt.plot(0, 0)
        plt.plot(0.5 * self.d_4 * np.cos(theta), 0.5 * self.d_4 * np.sin(theta))
        plt.plot(0.5 * self.d_5 * np.cos(theta), 0.5 * self.d_5 * np.sin(theta))

        plt.show()

    def voluta_quad(self, n=100):
        b_es = self.b
        for _ in range(n):
            b_es = 0.516 * self.d_5 * (np.exp(2 * np.pi * self.c_m5 / (b_es * self.c_u6)) - 1)

        theta = np.linspace(0, 2 * np.pi, n)
        r_theta = 0.516 * self.d_5 * np.exp((self.b / b_es) * self.c_m5 * theta / self.c_u6)

        x = r_theta * np.cos(theta + np.pi / 2)
        y = r_theta * np.sin(theta + np.pi / 2)

        plt.plot(x, y)
        plt.show()

    def print_all(self):
        print(f'     rho {eng_units(self.rho * 1000)}g/m3')
        print(f'       Q {eng_units(self.Q)}m3/s')
        print(f'     PTV {eng_units(self.PTV)}Pa')
        print(f'       N {self.N:11} rpm')
        print(f'   eta_v {self.eta_v:11.6f}')
        print(f'   eta_m {self.eta_m:11.6f}')
        print(f'   eta_t {self.eta_t:11.6f}')
        print(f' eta_ele {self.eta_ele:11.6f}')
        print(f'     r_D {self.r_D:11}')
        print(f' tau_adm {eng_units(self.tau_adm)}Pa')
        print(f'     Q_r {eng_units(self.Q_r)}m3/s')
        print(f'   eta_i {self.eta_i:11.6f}')
        print(f'       Y {eng_units(self.Y)}J/kg')
        print(f'    Y_pa {eng_units(self.Y_pa)}J/kg')
        print(f'   P_efe {eng_units(self.P_efe)}W')
        print(f'   P_ele {eng_units(self.P_ele)}W')
        print(f'       d {eng_units(self.d)}m')
        print(f'     u_5 {eng_units(self.u_5)}m/s')
        print(f'     d_5 {eng_units(self.d_5)}m')
        print(f'     d_4 {eng_units(self.d_4)}m')
        print(f'   c_m5* {eng_units(self.c_m5_)}m / s')
        print(f'Y_pa_inf {eng_units(self.Y_pa_inf)}J/kg')
        print(f' beta_5* {self.beta_5_ * 180 / np.pi:11.6f} deg')
        print(f'  beta_5 {self.beta_5 * 180 / np.pi:11.6f} deg')
        print(f'    c_u6 {eng_units(self.c_u6)}m/s')
        print(f' beta_6* {self.beta_6_ * 180 / np.pi:11.6f} deg')
        print(f'  beta_6 {self.beta_6 * 180 / np.pi:11.6f} deg')
        print(f'     u_4 {eng_units(self.u_4)}m/s')
        print(f'   c_m4* {eng_units(self.c_m4_)}m/s')
        print(f'  beta_4 {self.beta_4 * 180 / np.pi:11.6f} deg')
        print(f'       r {eng_units(self.r)}m')
        print(f'       z {self.z:11}')
        print(f'      t5 {eng_units(self.t5)}m')
        print(f'      t4 {eng_units(self.t4)}m')
        print(f'       b {eng_units(self.b)}m')
        print(f'     d_3 {eng_units(self.d_3)}m')
        print(f'    c_m3 {eng_units(self.c_m3)}m/s')
        print(f'    c_m4 {eng_units(self.c_m4)}m/s')
        print(f'    c_m5 {eng_units(self.c_m5)}m/s')


def eng_units(x):
    n = math.floor(math.log10(x) / 3)

    prefix = {10: 'Q', 9: 'R', 8: 'Y', 7: 'Z', 6: 'E', 5: 'P', 4: 'T', 3: 'G', 2: 'M', 1: 'k',
              0: '', -1: 'm', -2: 'u', -3: n, -4: p, -5: 'f', -6: 'a', -7: 'z', -8: 'y', -9: 'r', -10: 'q'}

    return f'{x * 10**(-3 * n):11.6f} {prefix[n]}'


if __name__ == '__main__':
    Q = 3.6
    PTV = 552 * 9.81
    N = 600

    eta_v = 0.80
    eta_m = 0.95
    eta_ele = 0.96

    r_D = 0.86
    tau_adm = 205e6 / math.sqrt(3)

    p = 101.325e3
    T = 20
    R = 287.052874

    rho = p / (R * (T + 273.15))

    fan = Fan(rho, Q, PTV, N, eta_v, eta_m, eta_ele, r_D, tau_adm)
    fan.calculate()
    fan.print_all()
    fan.voluta_ret()
