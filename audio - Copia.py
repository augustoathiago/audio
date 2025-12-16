# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import tempfile

SAMPLE_RATE = 44100
DURATION_AUDIO = 3.0
DURATION_GRAPH = 15.0

ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs
}
ALLOWED_CONSTS = {"pi": np.pi, "e": np.e, "tau": np.pi * 2.0}

def evaluate_equation(expr, t):
    safe_globals = {}
    safe_globals.update(ALLOWED_FUNCS)
    safe_globals.update(ALLOWED_CONSTS)
    safe_globals["t"] = t
    safe_globals["np"] = np
    return eval(expr, safe_globals)

def to_int16_no_normalization(y):
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_clipped = np.clip(y, -1.0, 1.0)
    return (y_clipped * 32767.0).astype(np.int16)

def write_wav_temp(audio_int16, sample_rate):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return tmp.name

# ------------------- INTERFACE -------------------

st.title("Equação → Gráfico (15s) → Áudio (3s)")

expr = st.text_input("Digite a equação em função de t:", "sin(2*pi*220*t)")

t_graph = np.linspace(0.0, DURATION_GRAPH, int(SAMPLE_RATE * DURATION_GRAPH), endpoint=False)
t_audio = np.linspace(0.0, DURATION_AUDIO, int(SAMPLE_RATE * DURATION_AUDIO), endpoint=False)

y_scale_exp = st.slider("Escala Y (potência de 10)", -6, 6, 0)
x_scale_exp = st.slider("Escala X (potência de 10)", -3, 4, 1)

# --- Plot sempre renderizado ---
try:
    y = evaluate_equation(expr, t_graph)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_graph, y, color="tab:blue")
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

# --- Áudio opcional ---
if st.button("Tocar áudio (3s)"):
    try:
        y_audio = evaluate_equation(expr, t_audio)
        audio_int16 = to_int16_no_normalization(y_audio)
        wav_path = write_wav_temp(audio_int16, SAMPLE_RATE)
        st.audio(wav_path, format="audio/wav")
    except Exception as e:
        st.error(f"Erro ao gerar áudio: {e}")
