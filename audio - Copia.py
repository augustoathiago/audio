# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import tempfile

SAMPLE_RATE = 44100
DURATION_AUDIO = 3.0
DURATION_GRAPH = 15.0

def evaluate_equation(expr, t):
    safe_globals = {"np": np, "t": t, "pi": np.pi}
    return eval(expr, safe_globals)

def to_int16_no_normalization(y):
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767).astype(np.int16)

st.title("Equação → Gráfico → Áudio")

expr = st.text_input("Digite a equação em função de t:", "np.sin(2*np.pi*220*t)")

t_graph = np.linspace(0, DURATION_GRAPH, int(SAMPLE_RATE * DURATION_GRAPH), endpoint=False)
t_audio = np.linspace(0, DURATION_AUDIO, int(SAMPLE_RATE * DURATION_AUDIO), endpoint=False)

if st.button("Plotar"):
    y = evaluate_equation(expr, t_graph)
    fig, ax = plt.subplots()
    ax.plot(t_graph, y)
    ax.axhline(0, color="black", linewidth=2)  # eixo horizontal destacado
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

if st.button("Tocar áudio"):
    y_audio = evaluate_equation(expr, t_audio)
    audio_int16 = to_int16_no_normalization(y_audio)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    st.audio(tmp.name, format="audio/wav")
