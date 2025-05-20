import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass

# --- Classe de Parâmetros ---
@dataclass
class ParametrosBuck:
    L: float                # Indutância (H)
    C: float                # Capacitância (F)
    R: float                # Resistência de carga (Ohm)
    tensao_entrada: float  # Tensão de entrada (V)
    ciclo_trabalho: float  # Duty cycle (0 a 1)
    frequencia_comutacao: float  # Frequência de comutação (Hz)
    tempo_total: float     # Tempo total de simulação (s)
    passo_tempo: float     # Passo de tempo da simulação (s)

# --- Funções do modelo ---
def esta_chave_ligada(t, p):
    periodo = 1.0 / p.frequencia_comutacao
    return (t % periodo) < (p.ciclo_trabalho * periodo)

def derivadas(t, estado, p):
    iL, vC = estado
    if esta_chave_ligada(t, p):
        diL = (p.tensao_entrada - vC) / p.L
    else:
        diL = -vC / p.L
    dvC = (iL - vC / p.R) / p.C
    return np.array([diL, dvC])

def rk4_step(estado, t, dt, deriv_func, p):
    k1 = deriv_func(t, estado, p)
    k2 = deriv_func(t + dt / 2, estado + dt / 2 * k1, p)
    k3 = deriv_func(t + dt / 2, estado + dt / 2 * k2, p)
    k4 = deriv_func(t + dt, estado + dt * k3, p)
    return estado + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def simular(p):
    N = int(p.tempo_total / p.passo_tempo)
    ts = np.linspace(0, p.tempo_total, N + 1)
    iL = np.zeros(N + 1)
    vC = np.zeros(N + 1)
    estado = np.array([0.0, 0.0])
    for idx, t in enumerate(ts):
        iL[idx], vC[idx] = estado
        estado = rk4_step(estado, t, p.passo_tempo, derivadas, p)
    return ts, iL, vC

# --- Interface Streamlit ---

st.title("Simulação de Retificador com Filtro LC")
st.markdown(
    """
    Ajuste os parâmetros abaixo e clique em **Rodar Simulação** para visualizar:
    1. Tensão da fonte AC
    2. Tensão após retificação
    3. Tensão na carga com filtro LC
    """
)

# Sliders para parâmetros do retificador
Vrms = st.slider("Tensão RMS da Fonte (Vrms, V)", 1.0, 240.0, 36.0)
freq = st.slider("Frequência da Fonte (Hz)", 10.0, 400.0, 60.0)
R = st.slider("Resistência de Carga (Ω)", 1.0, 100.0, 10.0)
L = st.slider("Indutância do Filtro (H)", 0.0, 10.0, 1.0)
C = st.slider("Capacitância do Filtro (μF)", 1.0, 5000.0, 1000.0)
Vd_schottky = st.slider("Queda de Tensão Schottky (V)", 0.0, 1.0, 0.3)
Vd_common = st.slider("Queda de Tensão Diodo Comum (V)", 0.0, 1.5, 0.7)

def simular_retificador(Vrms, freq, R, L, C_uF, Vd_sch, Vd_com):
    # Conversão de unidades
    C = C_uF * 1e-6
    Vp = Vrms * np.sqrt(2)
    T = 1.0 / freq
    omega = 2 * np.pi * freq
    # Vetor de tempo para 4 períodos
    t = np.linspace(0, 50 * T, 10000)

    # Função de derivadas do circuito RLC retificador
    def circuito_deriv(y, t):
        i_L, v_C = y
        v_in = Vp * np.sin(omega * t)
        if v_in > Vd_sch:
            v_rect = v_in - Vd_sch
        else:
            v_rect = -Vd_com
        di_Ldt = (v_rect - v_C) / L if L > 0 else 0.0
        dv_Cdt = (i_L - v_C / R) / C
        return [di_Ldt, dv_Cdt]

    # Resolver ODE
    y0 = [0.0, 0.0]
    sol = odeint(circuito_deriv, y0, t)
    i_L = sol[:, 0]
    v_C = sol[:, 1]

    # Tensão retificada (meia onda) para plot
    V_rect = np.where(Vp * np.sin(omega * t) > Vd_sch,
                      Vp * np.sin(omega * t) - Vd_sch,
                      -Vd_com)
    # Tensão da fonte
    V_ac = Vp * np.sin(omega * t)
    return t, V_ac, V_rect, v_C

# Color pickers dinâmicos para as curvas
c1 = st.color_picker("Cor da Fonte AC", "#1f77b4")
c2 = st.color_picker("Cor da Tensão Retificada", "#ff7f0e")
c3 = st.color_picker("Cor da Tensão na Carga", "#2ca02c")

# Botão para executar simulação
if st.button("Rodar Simulação"):
    t, V_ac, V_rect, V_R = simular_retificador(
        Vrms, freq, R, L, C, Vd_schottky, Vd_common)

    # Plot com três subplots e cores dinâmicas
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax1.plot(t, V_ac,   color=c1, label='Fonte AC')
    ax1.set_title("Tensão da Fonte AC")
    ax1.set_ylabel("V (V)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, V_rect, color=c2, label='Retificada')
    ax2.set_title("Tensão Retificada")
    ax2.set_ylabel("V (V)")
    ax2.legend()
    ax2.grid(True)

    ax3.plot(t, V_R,    color=c3, label='Na Carga')
    ax3.axhline(np.mean(V_R[len(V_R)//2:]), linestyle='--', label=f"Média = {np.mean(V_R[len(V_R)//2:]):.2f} V")
    ax3.set_title("Tensão na Carga (Filtro LC)")
    ax3.set_ylabel("V (V)")
    ax3.set_xlabel("Tempo (s)")
    ax3.legend()
    ax3.grid(True)

    st.pyplot(fig)
