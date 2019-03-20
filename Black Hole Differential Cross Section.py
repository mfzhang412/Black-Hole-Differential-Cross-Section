"""
Relativistic scattering off a non-rotating, uniform, and point-like black hole
using midpoint and adaptive integration.

@author: Michael Zhang
"""
import numpy as np
import matplotlib.pyplot as plt
import time

m_sun = 1.989e30
G = 6.673e-11
M = 10*m_sun
m = 1
c = 299792458
gamma = G*M*m
r_schwarz = 2*G*M/c**2


def calc_sig(dr_list):
    sig_mag2 = 0
    dr_A = dr_list[0]
    dr_B = dr_list[1]
    dr_C = dr_list[2]
    for i in range(len(dr_A)):
        sig_mag2 = sig_mag2 + (2*dr_B[i] - dr_A[i] - dr_C[i]) ** 2
    sig = 1/81 * np.sqrt(sig_mag2)
    return sig


def to_spher(cartesian):
    if cartesian.ndim != 1:
        r = np.linalg.norm(cartesian, axis=1)
    else:
        r = np.linalg.norm(cartesian)
    theta = np.arctan(np.linalg.norm(cartesian.T[:1], axis=0)/cartesian.T[-1])
    if theta != 0 and theta != np.pi:
        phi = np.arctan(cartesian.T[1] / cartesian.T[0])
    else:
        phi = 0
    return np.array([r,theta,phi])


def force(r, v, b):
    r_magn = np.linalg.norm(r)
    r_hat = r/r_magn
    F = -r_hat*(gamma/r_magn**2 + 3*gamma/r_magn**4*b**2*(v/c)**2)
    return F


def acc(f, v):
    A = f * (1-(v/c)**2)**(3/2)/(1+(v/c)**2*(c**2-1))
    return A


b_points = r_schwarz * np.linspace(1, 100, 5, endpoint=True)
v_points = c * np.linspace(1e-8, .9, 5, endpoint=True)

data = []
null_data = [None]*3


z_0 = -100*r_schwarz
t_0 = 0.

dt_0 = .1
dt_fine = 1e-5
sig_max = 1e-6
sig_min = 1e-9

start_time = time.time()

#time = 9.1*N/5 seconds where N is dimension of b_points and v_points
for v_0 in v_points:
    for b in b_points:
        
        r = [np.array([b,0,z_0])]
        v = [np.array([0,0,v_0])]
        t = [t_0]
        dt_curr = dt_0
        dr_hist = []
        dt_hist = []
        data_flag = True
        
        while (t[-1] <= 200*abs(z_0)/v_0):
            r_curr = r[-1]
            v_curr = v[-1]
            r_curr_magn = np.linalg.norm(r_curr)
            v_curr_magn = np.linalg.norm(v_curr)
            F_curr = force(r_curr, v_curr_magn, b)
            a_curr = acc(F_curr, v_curr_magn)
            t_curr = t[-1]
            
            if (r_curr_magn <= r_schwarz):
                data.append(null_data)
                data_flag = False
                break
            
            r_mid = r_curr + v_curr * dt_curr/2
            v_mid = v_curr + a_curr * dt_curr/2
            v_mid_magn = np.linalg.norm(v_mid)
            F_mid = force(r_mid, v_mid_magn, b)
            a_mid = acc(F_mid, v_mid_magn)
            
            dr = v_mid * dt_curr
            dv = a_mid * dt_curr
            
            r.append(r_curr + dr)
            v.append(v_curr + dv)
            t.append(t_curr + dt_curr)
            dr_hist.append(dr)
            dt_hist.append(dt_curr)
            
            r_magn = np.linalg.norm(r[-1])
            v_magn = np.linalg.norm(v[-1])
            fine_factor = (r_magn/b)*(c/v_magn)
            if (fine_factor <= 1e-2):
                dt_curr = dt_fine
            elif (len(dt_hist) >= 3):
                dt = dt_hist[-1]
                if (dt == dt_hist[-2] and dt == dt_hist[-3]):
                    dr_ABC = dr_hist[-3:]
                    sig_curr = calc_sig(dr_ABC)
                    if (sig_curr > sig_max):
                        dt_curr = dt_curr / 2
                    elif (sig_curr < sig_min):
                        dt_curr = dt_curr * 2
                    else:
                        pass
        
        if data_flag:
            v_spher_final = to_spher(v[-1])
            scatter_angle = v_spher_final[1]
            insertion_data = [scatter_angle, b, v_0]
            data.append(insertion_data)

end_time = time.time()
print("Computation time:", end_time-start_time, "sec")
#print(data)
