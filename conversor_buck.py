# -*- coding: utf-8 -*-
import streamlit as st                      # Biblioteca para criar interface web interativa
import numpy as np                          # Operações numéricas e criação de vetores
import matplotlib.pyplot as plt             # Plotagem de gráficos
from dataclasses import dataclass           # Definição de classes de parâmetros de forma concisa
from scipy.integrate import solve_ivp       # Solver de equações diferenciais

# --- 1) Definição da classe de parâmetros ---
@dataclass
class ParametrosBuckDC:
    Vin: float         # Tensão de entrada DC (V)
    L: float           # Indutância do filtro (H)
    C: float           # Capacitância do filtro (F)
    R: float           # Resistência de carga (Ω)
    duty: float        # Duty cycle (fração do período ON, 0…1)
    f_sw: float        # Frequência de chaveamento (Hz)
    Rds_on: float      # Resistência de condução do MOSFET (Ω)
    Vd: float          # Queda de tensão do diodo (V)
    R_esr: float       # ESR do capacitor (resistência série equivalente, Ω)
    tempo_total: float # Tempo total de simulação (s)
    dt: float          # Passo de tempo para amostragem (s)

# --- 2) Função de equações de estado ---
def estados_buck(t, x, p: ParametrosBuckDC):
    """
    Calcula as derivadas de iL e vC no instante t, dado o estado x e parâmetros p.
    Inclui características de MOSFET (Rds_on) e diodo (Vd).

    x: vetor [iL, vC]
    p: parâmetros do conversor
    """
    iL, vC = x                        # Desempacota estado: corrente no indutor e tensão no capacitor
    periodo = 1.0 / p.f_sw           # Período de chaveamento
    on = (t % periodo) < (p.duty * periodo)  # True se dentro do tempo ON do PWM

    # --- Estado ON: MOSFET conduz, diodo bloqueado ---
    if on:
        # A tensão no switch é Vin menos a queda I*Rds_on no MOSFET
        v_switch = p.Vin - iL * p.Rds_on
    else:
        # --- Estado OFF: MOSFET bloqueia, diodo conduz ---
        # A tensão no switch é a queda do diodo (negativa)
        v_switch = -p.Vd

    # Tensão efetiva sobre o indutor: vL = v_switch - vC
    vL = v_switch - vC

    # Equação de corrente no indutor
    diL = vL / p.L

    # Corrente de carga: i_load = vC / R
    i_load = vC / p.R
    # Corrente que vai para o capacitor: iC = iL - i_load
    iC = iL - i_load

    # Equação de tensão no capacitor considerando ESR negligenciado
    dvC = iC / p.C

    return [diL, dvC]

# --- 3) Interface Streamlit ---
st.title("Simulação de Conversor Buck DC-DC com MOSFET")
st.markdown(
    """
    Ajuste os parâmetros do conversor incluindo características do MOSFET e do diodo.
    - Ganho teórico de saída: $V_{out} = D \cdot V_{in}$
    - Simulação em regime permanente (descartando transientes)
    """
)

# --- 4) Widgets para entrada de parâmetros ---
duty     = st.slider("Duty Cycle (%)", 0, 100, 50)
Vin      = st.number_input("Tensão de Entrada DC (V)", value=36.0)
L_mH     = st.number_input("Indutância L (mH)", value=0.22)
C_uF     = st.number_input("Capacitância C (µF)", value=47.0)
# Define carga para metade da potência como default
R_load   = st.number_input("Carga R (Ω)", value=Vin * duty / 100 / 2.0)
f_sw_k   = st.number_input("Freq. de Chaveamento (kHz)", value=50.0)
Rds_on   = st.number_input("Rds(on) do MOSFET (mΩ)", value=50.0)
Vd       = st.number_input("Queda de tensão do diodo (V)", value=0.7)
R_esr    = st.number_input("ESR do capacitor (Ω)", value=0.01)
t_tot_ms = st.number_input("Tempo Total (ms)", value=5.0)
dt_us    = st.number_input("Passo de Tempo (µs)", value=0.1)

# --- 5) Função auxiliar para converter unidades e criar objeto de parâmetros ---
def to_params():
    return ParametrosBuckDC(
        Vin=Vin,
        L=L_mH * 1e-3,        # converte de mH para H
        C=C_uF * 1e-6,        # µF para F
        R=R_load,
        duty=duty / 100,      # % para fração
        f_sw=f_sw_k * 1e3,    # kHz para Hz
        Rds_on=Rds_on * 1e-3, # mΩ para Ω
        Vd=Vd,
        R_esr=R_esr,
        tempo_total=t_tot_ms * 1e-3, # ms para s
        dt=dt_us * 1e-6       # µs para s
    )

# --- 6) Execução da simulação ao clicar no botão ---
if st.button("Executar Simulação com MOSFET e Diodo"):
    # 6.1) Monta parâmetros e vetor de tempo
    p = to_params()
    t_eval = np.arange(0, p.tempo_total, p.dt)

    # 6.2) Resolve as ODEs com solver RK45 adaptativo
    sol = solve_ivp(lambda t, x: estados_buck(t, x, p),
                    [0, p.tempo_total], [0.0, 0.0],
                    t_eval=t_eval, method='RK45')
    t = sol.t
    iL = sol.y[0]  # corrente no indutor
    vC = sol.y[1]  # tensão no capacitor

    # 6.3) Descartar transitório (primeiros ciclos)
    Ts = 1.0 / p.f_sw
    idx0 = int(10 * Ts / p.dt)   # descarta 10 ciclos
    t2, iL2, vC2 = t[idx0:], iL[idx0:], vC[idx0:]

    # 6.4) Cálculo de valores médios e teóricos
    Vout_sim = np.mean(vC2)
    Vout_teo = p.duty * p.Vin
    I_sim = np.mean(iL2)
    I_teo = Vout_teo / p.R
    v_load = vC2  # tensão na carga é mesma de vC

    # --- 7) Plotagem dos resultados ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    # 7.1) Corrente no indutor
    ax1.plot(t2, iL2, label='i_L simulado')
    ax1.axhline(I_sim, linestyle='--', label=f'Média Simulada = {I_sim:.2f} A')
    ax1.axhline(I_teo, linestyle=':', label=f'I Teórica = {I_teo:.2f} A')
    ax1.set_ylabel('i_L (A)')
    ax1.legend()
    ax1.grid(True)

    # 7.2) Tensão no capacitor
    ax2.plot(t2, vC2, label='v_C simulado')
    ax2.axhline(Vout_sim, linestyle='--', label=f'Média Simulada = {Vout_sim:.2f} V')
    ax2.axhline(Vout_teo, linestyle=':', label=f'V Teórica = {Vout_teo:.2f} V')
    ax2.set_ylabel('v_C (V)')
    ax2.legend()
    ax2.grid(True)

    # 7.3) Tensão na carga (resistor)
    ax3.plot(t2, v_load, label='v_R carga')
    ax3.axhline(Vout_sim, linestyle='--', label=f'Média Simulada = {Vout_sim:.2f} V')
    ax3.axhline(Vout_teo, linestyle=':', label=f'V Teórica = {Vout_teo:.2f} V')
    ax3.set_ylabel('v_R (V)')
    ax3.set_xlabel('Tempo (s)')
    ax3.set_title('Tensão na Carga (Resistor)')
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig)
