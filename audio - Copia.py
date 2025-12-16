# app.py
# Requisitos: Python 3.x
# Instale dependências: pip install streamlit numpy matplotlib

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import tempfile

# Configurações
SAMPLE_RATE = 44100     # Hz
DURATION_AUDIO = 3.0    # segundos de áudio
DURATION_GRAPH = 15.0   # segundos mostrados no gráfico

# Funções permitidas e constantes
ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil,
}
ALLOWED_CONSTS = {"pi": np.pi, "e": np.e, "tau": np.pi * 2.0}

def evaluate_equation(expr: str, t: np.ndarray) -> np.ndarray:
    """
    Avalia a expressão do usuário como função de t, usando apenas funções/constantes permitidas.
    Suporta definições com ';' (ex.: A=0.5; omega=2*pi*220; A*sin(omega*t)).
    """
    safe_locals = {}
    safe_globals = {}
    safe_globals.update(ALLOWED_FUNCS)
    safe_globals.update(ALLOWED_CONSTS)
    safe_globals["t"] = t
    safe_globals["np"] = np

    parts = [p.strip() for p in expr.split(";") if p.strip()]
    final_expr = parts[-1] if parts else expr

    for p in parts[:-1]:
        if "=" in p:
            name, rhs = p.split("=", 1)
            name = name.strip()
            rhs = rhs.strip()
            value = eval(rhs, safe_globals, safe_locals)
            safe_globals[name] = value
        else:
            eval(p, safe_globals, safe_locals)

    y = eval(final_expr, safe_globals, safe_locals)
    y = np.asarray(y)
    if y.shape != t.shape:
        y = np.broadcast_to(y, t.shape)
    return y

def to_int16_no_normalization(y: np.ndarray) -> np.ndarray:
    """
    Converte diretamente o sinal para int16 sem normalizar o volume.
    Amplitudes maiores geram som mais alto; aplica clip em [-1, 1] para evitar estouro.
    """
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_clipped = np.clip(y, -1.0, 1.0)
    return (y_clipped * 32767.0).astype(np.int16)

def write_wav_temp(audio_int16: np.ndarray, sample_rate: int) -> str:
    """
    Grava um WAV temporário e retorna o caminho do arquivo.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return tmp.name

# ------------------- INTERFACE STREAMLIT -------------------

st.title("Equação → Gráfico (15s) → Áudio (3s)")

expr = st.text_input(
    "Digite a equação em função de t (suporta variáveis com ';'):",
    "sin(2*pi*220*t)"
)

# Vetores de tempo
t_graph = np.linspace(0.0, DURATION_GRAPH, int(SAMPLE_RATE * DURATION_GRAPH), endpoint=False)
t_audio = np.linspace(0.0, DURATION_AUDIO, int(SAMPLE_RATE * DURATION_AUDIO), endpoint=False)

# Escalas ajustáveis (ordem de grandeza)
col1, col2 = st.columns(2)
with col1:
    y_scale_exp = st.slider("Escala Y (potência de 10)", -6, 6, 0)
with col2:
    x_scale_exp = st.slider("Escala X (potência de 10)", -3, 4, 1)

# Botões de ação
plot = st.button("Plotar gráfico")
play = st.button("Tocar áudio (3s)")

# Plot
if plot:
    try:
        y = evaluate_equation(expr, t_graph)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_graph, y, color="tab:blue", linewidth=1.0)
        # eixo horizontal destacado
        ax.axhline(0, color="black", linewidth=2)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Sinal no tempo (0 a 15 s)")
        ax.set_ylim(-10**y_scale_exp, 10**y_scale_exp)
        ax.set_xlim(0, 10**x_scale_exp)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao avaliar a expressão: {e}")

# Áudio
if play:
    try:
        y_audio = evaluate_equation(expr, t_audio)
        audio_int16 = to_int16_no_normalization(y_audio)
        wav_path = write_wav_temp(audio_int16, SAMPLE_RATE)
        st.audio(wav_path, format="audio/wav")
        st.info("O áudio reflete diretamente a amplitude da equação (sem normalização).")
    except Exception as e:
        st.error(f"Erro ao gerar áudio: {e}")
