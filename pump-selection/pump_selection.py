from utils import friction
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def r_squared(func, xdata, ydata, popt):
    residuals = ydata - func(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return r_squared


def curve_fit(func, xdata, ydata, **kwargs):
    popt, pcov = opt.curve_fit(func, xdata, ydata, **kwargs)
    r2 = r_squared(func, xdata, ydata, popt)
    return popt, pcov, r2


def poly3(x, a, b, c, d):
    return ((a * x + b) * x + c) * x + d


def hydraulic_system(flow_rate, a, h):
    return a * flow_rate**2 + h


def system_height(flow_rate, elevation,
                  duct_lenght, duct_diameter, duct_rugosity,
                  equivalent_lenght_const, equivalent_lenght_inverse,
                  pressure1, pressure8,
                  density=997, viscosity=8.96e-7, gravity=9.80665):
    Re = 4 * flow_rate / (np.pi * viscosity * duct_diameter)
    friction_coeff = friction.solve_friction_coeff(duct_rugosity, duct_diameter, Re) ** -2
    lenght_local_loss = (equivalent_lenght_const + equivalent_lenght_inverse / friction_coeff) * duct_diameter
    height_loss = 8 * friction_coeff * (lenght_local_loss + duct_lenght) * flow_rate**2 / (np.pi**2 * gravity * duct_diameter**5)
    height_system = (pressure8 - pressure1) / (density * gravity) + elevation + height_loss
    return height_system
    

# System specification
density = 997
viscosity = 8.96e-7
pressuve_vapor = 3.17e3
gravity = 9.80665

elevation = 13.7 + 0.5 * 16
duct_lenght = 10 + 190 + (13.5 + 0.5 * 16)
duct_diameter = 150e-3
duct_rugosity = 0.045e-3

equivalent_lenght_const = 30 + 600
equivalent_lenght_inverse = 0.78 + 0.38 + 0.53 + 1

pressure1 = 1e5 + density * gravity * 5
pressure8 = 1.5e5 + density * gravity * 16.3

velocity = 1.5

flow_rate = np.linspace(6e-6, 200/3600, int(1e5))

# System
Hm_system = system_height(flow_rate, elevation,
                          duct_lenght, duct_diameter, duct_rugosity,
                          equivalent_lenght_const, equivalent_lenght_inverse,
                          pressure1, pressure8,
                          density, viscosity, gravity)

Re = 4 * flow_rate / (np.pi * viscosity * duct_diameter)
friction_coeff = friction.solve_friction_coeff(duct_rugosity, duct_diameter, Re) ** -2
equivalent_lenght = 0.78 * duct_diameter / friction_coeff
heigh_loss = 8 * friction_coeff * (10 + equivalent_lenght) * flow_rate**2 / (np.pi**2 * gravity * duct_diameter**5)
npsh_system = (pressure1 - pressuve_vapor) / (density * gravity) + 5 - heigh_loss

# Pump data
pump_flow_rate = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                          dtype=np.float64)
pump_height = np.array([4.9333e+1,
                        4.9091e+1,
                        4.7758e+1,
                        4.5939e+1,
                        4.3636e+1,
                        4.1333e+1,
                        3.8182e+1,
                        3.3818e+1,
                        2.9212e+1,
                        2.4000e+1],
                       dtype=np.float64)

pump_power = np.array([6.3539e+0,
                       7.8016e+0,
                       9.3029e+0,
                       1.0912e+1,
                       1.2520e+1,
                       1.4021e+1,
                       1.5362e+1,
                       1.6273e+1,
                       1.6756e+1,
                       1.7024e+1],
                      dtype=np.float64)

npsh_flow_rate = np.array([35, 50, 65, 80, 95, 110, 125, 140, 150, 160],
                   dtype=np.float64)
npsh132 = np.array([3.3824e+0,
                    3.6765e+0,
                    3.5294e+0,
                    3.9706e+0,
                    4.4118e+0,
                    5.1471e+0,
                    5.7353e+0,
                    6.7647e+0,
                    8.2353e+0,
                    1.5882e+1],
                   dtype=np.float64)
npsh174 = np.array([2.7941e+0,
                    2.3529e+0,
                    2.3529e+0,
                    2.5000e+0,
                    3.0882e+0,
                    3.8235e+0,
                    4.5588e+0,
                    5.5882e+0,
                    6.6176e+0,
                    7.6471e+0],
                   dtype=np.float64)

npsh155 = np.array([((npsh174[i] - npsh132[i]) / (174 - 132)) * (155 - 132) + npsh132[i]
                    for i in range(len(npsh_flow_rate))],
                   dtype=np.float64)

# Pump regression   
Hm_pump_popt, Hm_pump_pcov, Hm_pump_r2 = curve_fit(hydraulic_system, pump_flow_rate, pump_height)
Hm_pump = hydraulic_system(flow_rate * 3600, *Hm_pump_popt)
P_pump_popt, P_pump_pcov, P_pump_r2 = curve_fit(poly3, pump_flow_rate, pump_power)
P_pump = lambda x: poly3(x, *P_pump_popt)
npsh_pump_popt, npsh_pump_pcov, npsh_pump_r2 = curve_fit(poly3, npsh_flow_rate, npsh155)
npsh_pump = poly3(flow_rate * 3600, *npsh_pump_popt)

# Operation point
operation_point = np.argmin(np.abs(Hm_system - Hm_pump))
operation_point_flow_rate = flow_rate[operation_point] * 3600
operation_point_Hm = Hm_system[operation_point]

max_point = np.argmin(np.abs(npsh_system - npsh_pump))
max_point_flow_rate = flow_rate[max_point] * 3600
max_point_Hm = Hm_system[max_point]

# Ideal diameter
# print(np.sqrt(4 * (operation_point_flow_rate / 3600) / (np.pi * 1.5)))

# Efficiency and cost
power_hydraulic = (density * gravity) * (operation_point_flow_rate / 3600) * operation_point_Hm / 1000
power_pump = P_pump(operation_point_flow_rate)
efficiency = power_hydraulic / power_pump

# Taxa de energia eletrica classe A3 (69kV)
# Referencia: AGÊNCIA NACIONAL DE ENERGIA ELÉTRICA – ANEEL. RESOLUÇÃO HOMOLOGATÓRIA Nº 3.049, DE 21 DE JUNHO DE 2022.
# Horário Fora Ponta: tarifa mais barata compreendido entre às 22:00 e 17:00 horas;
# Horário de Ponta: tarifa mais cara compreendido entre às 18:00 e 21:00 horas.
# TUSD_P  = 81.73 R$/MWh    |   TE_P  = 388.08 R$/MWh
# TUSD_FP = 81.73 R$/MWh    |   TE_FP = 247.20 R$/MWh

energy_tax = (4 * (81.73 + 388.08) + 20 * (81.73 + 247.20)) / (24 * 1000)
energy_cost = power_pump * (24 * 30) * energy_tax
energy_cost_volume = power_pump * energy_tax / operation_point_flow_rate

# Results
print(f'Vazao de operacao: {operation_point_flow_rate:.3f} m3/h')
print(f'Hm de operacao: {operation_point_Hm:.3f} m')
print(f'Vazao maxima: {max_point_flow_rate:.3f} m3/h')
print(f'Hm para vazao maxima: {max_point_Hm:.3f} m')
print(f'Eficiencia da bomba no ponto de operacao: {100 * efficiency:.1f}%')
print(f'Potencia consumida no ponto de operacao: {power_pump:.3f} kW')
print(f'Custo operacional: R$ {energy_cost:.2f}')
print(f'Custo operacional por volume: R$/m3 {energy_cost_volume:.5f}')
print(f'Parametros da regressao [Q - Hm]pump: {Hm_pump_popt}')
print(f'R2 da regressao [Q - Hm]pump: {Hm_pump_r2}')
print(f'Parametros da regressao [Q -NPSHr]pump: {npsh_pump_popt}')
print(f'R2 da regressao [Q - NPSHr]pump: {npsh_pump_r2}')
print(f'Parametros da regressao [Q - P]pump: {P_pump_popt}')
print(f'R2 da regressao [Q - P]pump: {P_pump_r2}')


# Plot hydraulic system
plt.plot(flow_rate * 3600, Hm_system,
         linestyle='-', marker='', label='Sistema')

plt.xlim([0, 100])
plt.ylim([38, 45])
plt.grid()
plt.margins(x=0., y=0., tight=True)
plt.xlabel('Q [m3/h]')
plt.ylabel('Hm [m]')
plt.title('Q - Hm')
plt.show()

# Plot pump + hydraulic system
plt.plot(flow_rate * 3600, Hm_system,
         linestyle='-', marker='', label='Sistema')

plt.plot(pump_flow_rate, pump_height,
         linestyle='', marker='+', label='Bomba - Catalogo')

plt.plot(flow_rate * 3600, Hm_pump,
         linestyle='-', marker='', label='Bomba - Regressao')

plt.plot(operation_point_flow_rate, Hm_pump[operation_point],
         linestyle='', marker='x', label='Ponto de operacao')

plt.grid()
plt.margins(x=0., y=0., tight=True)
plt.xlabel('Q [m3/h]')
plt.ylabel('Hm [m]')
plt.title('Q - Hm')
plt.legend(loc='lower left')
plt.show()

# Plot NPSH
plt.plot(flow_rate * 3600, npsh_system,
         linestyle='-', marker='', label='Sistema')

plt.plot(npsh_flow_rate, npsh132,
         linestyle='', marker='+', label='Bomba D132')

plt.plot(npsh_flow_rate, npsh174,
         linestyle='', marker='+', label='Bomba D174')

plt.plot(npsh_flow_rate, npsh155,
         linestyle='', marker='+', label='Bomba D155')

plt.plot(flow_rate * 3600, npsh_pump,
         linestyle='-', marker='', label='Bomba D155 - Regressao')


plt.plot(max_point_flow_rate, npsh_pump[max_point],
         linestyle='', marker='x', label='Vazao maxima')

plt.grid()
plt.margins(x=0., y=0., tight=True)
plt.xlabel('Q [m3/h]')
plt.ylabel('NPSH [m]')
plt.title('Q - NPSH')
plt.legend(loc='upper left')
plt.show()
