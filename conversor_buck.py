import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# --- Classe de Parâmetros para o Buck CC-CC ---
@dataclass
class ParametrosBuckDC:
    Vin: float               # Tensão de entrada DC (V)
    L: float                 # Indutância (H)
    C: float                 # Capacitância (F)
    R: float                 # Carga (Ohm)
    duty: float              # Duty cycle (0 a 1)
    f_sw: float              # Frequência de chaveamento (Hz)
    tempo_total: float       # Tempo total de simulação (s)
    dt: float                # Passo de tempo (s)

# --- Equações de Estado ---
def estados_buck(t, x, p: ParametrosBuckDC):
    iL, vC = x
    # Determina se switch está ON ou OFF
    periodo = 1.0 / p.f_sw
    ligado = (t % periodo) < (p.duty * periodo)
    # Tensões
    if ligado:
        vL = p.Vin - vC
    else:
        vL = -vC  # diodo conduz para aterramento
    diL = vL / p.L
    dvC = (iL - vC / p.R) / p.C
    return [diL, dvC]

# --- Interface Streamlit ---
st.title("Simulação de Conversor Buck DC-DC")
st.markdown(
    """
    Ajuste os parâmetros do conversor e veja as formas de onda de corrente no indutor e tensão no capacitor.
    """
)

# Widgets de parâmetros
duty = st.slider("Duty Cycle (%)", 0, 100, 50)
Vin = st.number_input("Tensão de Entrada DC (V)", min_value=0.0, value=12.0)
L   = st.number_input("Indutância L (mH)", min_value=0.1, value=1.0)
C   = st.number_input("Capacitância C (µF)", min_value=0.1, value=10.0)
R   = st.number_input("Carga R (Ω)", min_value=0.1, value=10.0)
f_sw= st.number_input("Freq. de Chaveamento (kHz)", min_value=1.0, value=100.0)
t_tot = st.number_input("Tempo Total (ms)", min_value=1.0, value=5.0)
dt  = st.number_input("Passo de Tempo (µs)", min_value=0.1, value=0.1)

# Conversão de unidades
params = ParametrosBuckDC(
    Vin=Vin,
    L=L*1e-3,
    C=C*1e-6,
    R=R,
    duty=duty/100,
    f_sw=f_sw*1e3,
    tempo_total=t_tot*1e-3,
    dt=dt*1e-6
)

# Simulação com RK4 via solve_ivp
if st.button("Executar Simulação"):
    t_span = (0, params.tempo_total)
    t_eval = np.arange(0, params.tempo_total, params.dt)
    sol = solve_ivp(lambda t, x: estados_buck(t, x, params),
                    t_span, [0.0, 0.0], t_eval=t_eval, method='RK45')

    t = sol.t
    iL = sol.y[0]
    vC = sol.y[1]

        # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,6))
    # Corrente no indutor
    ax1.plot(t, iL, label='i_L (A)')
    ax1.set_title("Corrente no Indutor")
    ax1.set_ylabel("i_L (A)")
    ax1.grid(True)
    ax1.legend()

    # Tensão no capacitor (saída)
    ax2.plot(t, vC, color='orange', label='v_C (V)')
    # Linha do valor médio de saída
    Vout_medio = np.mean(vC)
    ax2.axhline(Vout_medio, color='blue', linestyle='--',
                label=f'Média saída = {Vout_medio:.2f} V')
    ax2.set_title("Tensão no Capacitor (Saída)")
    ax2.set_ylabel("v_C (V)")
    ax2.set_xlabel("Tempo (s)")
    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig)
