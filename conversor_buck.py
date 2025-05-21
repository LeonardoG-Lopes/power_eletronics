# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# --- Classe de Parâmetros para o Buck DC-DC ---
@dataclass
class ParametrosBuckDC:
    Vin: float         # Tensão de entrada DC (V)
    L: float           # Indutância do filtro (H)
    C: float           # Capacitância do filtro (F)
    R: float           # Resistência de carga (Ω)
    duty: float        # Duty cycle (0…1)
    f_sw: float        # Frequência de chaveamento (Hz)
    Rds_on: float      # Resistência de condução do MOSFET (Ω)
    Vd: float          # Queda de tensão do diodo (V)
    R_esr: float       # ESR do capacitor (Ω)
    tempo_total: float # Tempo total de simulação (s)
    dt: float          # Passo de tempo (s)

# --- Equações de Estado (modelo com MOSFET e diodo) ---
def estados_buck(t, x, p: ParametrosBuckDC):
    iL, vC = x
    periodo = 1.0 / p.f_sw
    on = (t % periodo) < (p.duty * periodo)

    # Estado ON: MOSFET conduz, diodo reverso
    if on:
        v_switch = p.Vin - iL * p.Rds_on  # Vin menos queda no MOSFET
    else:
        # Estado OFF: MOSFET bloqueia, diodo conduz
        v_switch = -p.Vd                # Queda de diodo

    # Tensão efetiva no indutor
    vL = v_switch - vC

    # Derivadas
    diL = vL / p.L
    i_load = vC / p.R
    iC = iL - i_load                      # Corrente no capacitor
    dvC = iC / p.C                        # Despreza perdas extras

    return [diL, dvC]

# --- Interface Streamlit ---
st.title("Simulação de Conversor Buck DC-DC com MOSFET")
st.markdown(
    """
    Ajuste os parâmetros do conversor incluindo características do MOSFET e do diodo.  
    - Ganho teórico: $V_{out}=D\cdot V_{in}$  
    - Simulação em regime permanente  
    """
)

# Widgets de parâmetros
duty     = st.slider("Duty Cycle (%)", 0, 100, 50)
Vin      = st.number_input("Tensão de Entrada DC (V)", value=36.0)
L_mH     = st.number_input("Indutância L (mH)", value=0.22)
C_uF     = st.number_input("Capacitância C (µF)", value=47.0)
R_load   = st.number_input("Carga R (Ω)", value=Vin * duty / 100 / 2.0)
f_sw_k   = st.number_input("Freq. de Chaveamento (kHz)", value=50.0)
Rds_on   = st.number_input("Rds(on) do MOSFET (mΩ)", value=50.0)
Vd       = st.number_input("Queda de tensão do diodo (V)", value=0.7)
R_esr    = st.number_input("ESR do capacitor (Ω)", value=0.01)
t_tot_ms = st.number_input("Tempo Total (ms)", value=5.0)
dt_us    = st.number_input("Passo de Tempo (µs)", value=0.1)

# Conversão de unidades
def to_params():
    return ParametrosBuckDC(
        Vin=Vin,
        L=L_mH * 1e-3,
        C=C_uF * 1e-6,
        R=R_load,
        duty=duty / 100,
        f_sw=f_sw_k * 1e3,
        Rds_on=Rds_on * 1e-3,
        Vd=Vd,
        R_esr=R_esr,
        tempo_total=t_tot_ms * 1e-3,
        dt=dt_us * 1e-6
    )

# Simulação
if st.button("Executar Simulação com MOSFET e Diodo"):
    p = to_params()
    t_eval = np.arange(0, p.tempo_total, p.dt)
    sol = solve_ivp(lambda t, x: estados_buck(t, x, p),
                    [0, p.tempo_total], [0.0, 0.0],
                    t_eval=t_eval, method='RK45')
    t = sol.t
    iL = sol.y[0]
    vC = sol.y[1]

    # Descartar transiente
    Ts = 1.0 / p.f_sw
    idx0 = int(10 * Ts / p.dt)
    t2, iL2, vC2 = t[idx0:], iL[idx0:], vC[idx0:]

    # Cálculo de tensão na carga
    v_load = vC2  # tensão no resistor de carga (mesma de vC)

    # Médias e valores teóricos
    Vout_sim = np.mean(vC2)
    Vout_teo = p.duty * p.Vin
    I_sim = np.mean(iL2)
    I_teo = Vout_teo / p.R

    # Plot: Corrente, Tensão do Capacitor e Tensão na Carga
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    # 1) Corrente no indutor
    ax1.plot(t2, iL2, label='i_L simulado')
    ax1.axhline(I_sim, linestyle='--', label=f'Média Simulada={I_sim:.2f} A')
    ax1.axhline(I_teo, linestyle=':', label=f'I Teórica={I_teo:.2f} A')
    ax1.set_ylabel('i_L (A)')
    ax1.legend(); ax1.grid(True)

    # 2) Tensão no capacitor
    ax2.plot(t2, vC2, label='v_C simulado')
    ax2.axhline(Vout_sim, linestyle='--', label=f'Média Simulada={Vout_sim:.2f} V')
    ax2.axhline(Vout_teo, linestyle=':', label=f'V Teórica={Vout_teo:.2f} V')
    ax2.set_ylabel('v_C (V)')
    ax2.legend(); ax2.grid(True)

    # 3) Tensão na carga (resistor)
    ax3.plot(t2, v_load, label='v_R carga')
    ax3.axhline(Vout_sim, linestyle='--', label=f'Média Simulada={Vout_sim:.2f} V')
    ax3.axhline(Vout_teo, linestyle=':', label=f'V Teórica={Vout_teo:.2f} V')
    ax3.set_ylabel('v_R (V)')
    ax3.set_xlabel('Tempo (s)')
    ax3.set_title('Tensão na Carga (Resistor)')
    ax3.legend(); ax3.grid(True)

    st.pyplot(fig)