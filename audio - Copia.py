# app_equacao_audio.py
# Requisitos: Python 3.x, bibliotecas padrão (tkinter, matplotlib, numpy)
# Instale dependências: pip install numpy matplotlib

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tempfile
import wave
import winsound
import time

# Configurações de áudio e tempo
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
            try:
                value = eval(rhs, safe_globals, safe_locals)
                safe_globals[name.strip()] = value
            except Exception as e:
                raise ValueError(f"Erro ao avaliar variável '{name}': {e}")

    try:
        y = eval(final_expr, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(f"Erro ao avaliar a expressão: {e}")

    y = np.asarray(y)
    if y.shape != t.shape:
        try:
            y = np.broadcast_to(y, t.shape)
        except Exception:
            raise ValueError("A expressão não resultou em vetor compatível.")
    return y

def to_int16_no_normalization(y: np.ndarray) -> np.ndarray:
    """
    Converte diretamente o sinal para int16 sem normalizar.
    Assim, a amplitude digitada influencia diretamente o volume.
    """
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_clipped = np.clip(y, -1.0, 1.0)  # evita estouro
    return (y_clipped * 32767.0).astype(np.int16)

def write_wav_temp(audio_int16: np.ndarray, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return tmp_path

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Equação → Gráfico (15s) → Áudio (3s)")
        self.geometry("950x650")

        # Escalas iniciais (expoentes de 10)
        self.y_scale_exp = 0   # eixo Y começa em 10^0 = 1
        self.x_scale_exp = 1   # eixo X começa em 10^1 = 10

        # Estado de animação
        self.anim_running = False
        self.play_start_time = None
        self.last_y_graph = None

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        lbl = ttk.Label(top, text="Equação (função de t):")
        lbl.pack(side=tk.LEFT)

        self.expr_var = tk.StringVar()
        self.expr_entry = ttk.Entry(top, textvariable=self.expr_var, width=60)
        self.expr_entry.pack(side=tk.LEFT, padx=8)
        self.expr_entry.insert(0, "sin(2*pi*220*t)")

        self.btn_plot = ttk.Button(top, text="Plotar", command=self.on_plot)
        self.btn_plot.pack(side=tk.LEFT, padx=5)

        self.btn_play = ttk.Button(top, text="Tocar 3s", command=self.on_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        # Botões de escala
        scale_frame = ttk.Frame(self)
        scale_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(scale_frame, text="Escala Y:").pack(side=tk.LEFT)
        ttk.Button(scale_frame, text="-", command=self.decrease_y_scale).pack(side=tk.LEFT)
        ttk.Button(scale_frame, text="+", command=self.increase_y_scale).pack(side=tk.LEFT, padx=5)

        ttk.Label(scale_frame, text="Escala X:").pack(side=tk.LEFT, padx=20)
        ttk.Button(scale_frame, text="-", command=self.decrease_x_scale).pack(side=tk.LEFT)
        ttk.Button(scale_frame, text="+", command=self.increase_x_scale).pack(side=tk.LEFT, padx=5)

        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Sinal no tempo (0 a 15 s)")
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Vetores de tempo
        self.t_graph = np.linspace(0.0, DURATION_GRAPH, int(SAMPLE_RATE * DURATION_GRAPH), endpoint=False)
        self.t_audio = np.linspace(0.0, DURATION_AUDIO, int(SAMPLE_RATE * DURATION_AUDIO), endpoint=False)

        self.line = None
        self.marker = None
        self.plot_current_expr()

    def plot_current_expr(self):
        expr = self.expr_var.get()
        try:
            y = evaluate_equation(expr, self.t_graph)
        except Exception as e:
            messagebox.showerror("Erro de avaliação", str(e))
            return

        self.last_y_graph = y  # cache do sinal para animação

        self.ax.clear()
        self.ax.set_title("Sinal no tempo (0 a 15 s)")
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)

        # Eixo horizontal destacado em y=0
        self.ax.axhline(0, color="black", linewidth=2)

        # Escalas (podem ser maiores ou menores que 1)
        y_lim = 10 ** self.y_scale_exp
        x_lim = 10 ** self.x_scale_exp
        self.ax.set_ylim(-y_lim, y_lim)
        self.ax.set_xlim(0, x_lim)

        self.line, = self.ax.plot(self.t_graph, y, color="tab:blue", linewidth=1.0)
        # marcador inicial
        self.marker, = self.ax.plot([0], [y[0]], marker="o", markersize=10, color="red")
        self.canvas.draw()

    def on_plot(self):
        self.plot_current_expr()

    def on_play(self):
        if self.anim_running:
            return  # evita iniciar outra animação simultânea

        expr = self.expr_var.get()
        try:
            # preparar áudio
            y_audio = evaluate_equation(expr, self.t_audio)
            audio_int16 = to_int16_no_normalization(y_audio)
            wav_path = write_wav_temp(audio_int16, SAMPLE_RATE)

            # reproduzir áudio de forma assíncrona (não bloqueia)
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

            # iniciar animação sincronizada
            self.play_start_time = time.perf_counter()
            self.anim_running = True
            if self.last_y_graph is None:
                self.last_y_graph = evaluate_equation(expr, self.t_graph)
            self.animate_marker()

        except Exception as e:
            messagebox.showerror("Erro ao tocar áudio", str(e))

    def animate_marker(self):
        if not self.anim_running:
            return

        elapsed = time.perf_counter() - self.play_start_time
        if elapsed >= DURATION_AUDIO:
            # posiciona no final do trecho tocado (3s) e encerra
            end_idx = min(int(DURATION_AUDIO * SAMPLE_RATE), len(self.t_graph) - 1)
            self.marker.set_data([self.t_graph[end_idx]], [self.last_y_graph[end_idx]])
            self.canvas.draw()
            self.anim_running = False
            # para o som se ainda tocando (opcional)
            winsound.PlaySound(None, winsound.SND_PURGE)
            return

        # índice correspondente ao tempo decorrido (no domínio do gráfico)
        idx = int(elapsed * SAMPLE_RATE)
        if idx >= len(self.t_graph):
            idx = len(self.t_graph) - 1

        self.marker.set_data([self.t_graph[idx]], [self.last_y_graph[idx]])
        self.canvas.draw()

        # agenda próximo passo (~60 FPS)
        self.after(16, self.animate_marker)

    # Funções para alterar escala (permitem valores negativos: < 1)
    def increase_y_scale(self):
        self.y_scale_exp += 1
        self.plot_current_expr()

    def decrease_y_scale(self):
        self.y_scale_exp -= 1
        self.plot_current_expr()

    def increase_x_scale(self):
        self.x_scale_exp += 1
        self.plot_current_expr()

    def decrease_x_scale(self):
        self.x_scale_exp -= 1
        self.plot_current_expr()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()