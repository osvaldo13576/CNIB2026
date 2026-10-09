#!/usr/bin/env python3
"""
calibracion_qt.py
Herramienta de calibración con adquisición serial en tiempo real.

Flujo de uso:
  1. Seleccionar puerto y conectar.
  2. Configurar la etiqueta del eje X, el valor X conocido,
     el número de muestras y el tiempo de adquisición.
  3. Pulsar "Agregar muestras al CSV" para capturar un punto.
  4. Repetir con distintos valores X.
  5. Pulsar "Finalizar y ver calibración" para mostrar la curva.

Dependencias:
    pip install PyQt5 pyserial matplotlib numpy scipy
"""

import csv
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import serial
import serial.tools.list_ports
from scipy import stats

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# Límites físicos del ADS1115 (16 bits con signo, GAIN_ONE = ±4.096 V)
_ADC_RAW_MIN  =      0
_ADC_RAW_MAX  =  32767
_VREF         =  4.096
_V_MARGIN     =   0.05   # tolerancia del 5% sobre Vref


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers de estilo
# ─────────────────────────────────────────────────────────────────────────────

def _btn_style(bg: str, disabled: str = "#90A4AE") -> str:
    return (
        f"QPushButton{{background:{bg};color:white;font-size:13px;"
        f"padding:8px;border-radius:4px;}}"
        f"QPushButton:disabled{{background:{disabled};color:#ddd;}}"
        f"QPushButton:hover:enabled{{background:{bg}CC;}}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SerialReader — hilo daemon que lee el puerto serie
# ─────────────────────────────────────────────────────────────────────────────

class SerialReader:
    """
    Lee líneas del puerto serial en un hilo daemon.
    Expone `drain()` para que el hilo principal (via QTimer) consuma los datos.
    Protocolo esperado: "raw,voltage\\n"  (solo se usa el campo voltage).
    """

    def __init__(self) -> None:
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._buf: deque[tuple[float, int, float]] = deque()
        self._t0 = 0.0
        self.connected = False

    # ── API pública ────────────────────────────────────────────────────────

    def connect(self, port: str, baud: int) -> tuple[bool, str | None]:
        try:
            self._ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(0.5)               # espera reset + estabilización
            self._ser.reset_input_buffer()
            time.sleep(0.1)               # descarta bytes tardíos post-reset
            self._t0 = time.time()
            self._running = True
            self.connected = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True, None
        except serial.SerialException as exc:
            return False, str(exc)

    def disconnect(self) -> None:
        self._running = False
        self.connected = False
        ser = self._ser
        if ser and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass

    def drain(self) -> list[tuple[float, int, float]]:
        """Retorna todos los puntos (t, raw, V) pendientes y vacía el buffer."""
        with self._lock:
            pts = list(self._buf)
            self._buf.clear()
        return pts

    # ── bucle interno ──────────────────────────────────────────────────────

    def _loop(self) -> None:
        ser = self._ser
        while self._running:
            try:
                raw_bytes = ser.readline()
                line = raw_bytes.decode("utf-8", errors="replace").strip()
                if not line or line.upper() == "READY" or line.startswith("ERROR"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    raw_val = int(float(parts[0]))
                    voltage = float(parts[1])
                    # Descartar lecturas fuera del rango físico del ADC
                    if not (_ADC_RAW_MIN <= raw_val <= _ADC_RAW_MAX):
                        continue
                    if not (0.0 <= voltage <= _VREF + _V_MARGIN):
                        continue
                    t = time.time() - self._t0
                    with self._lock:
                        self._buf.append((t, raw_val, voltage))
            except (ValueError, UnicodeDecodeError):
                pass
            except Exception:
                self._running = False
                self.connected = False
                break


# ─────────────────────────────────────────────────────────────────────────────
#  RealtimePlot — gráfica de señal con ventana deslizante de 5 s
# ─────────────────────────────────────────────────────────────────────────────

class RealtimePlot(FigureCanvas):
    WINDOW_S = 5.0   # segundos visibles en la ventana deslizante

    def __init__(self) -> None:
        fig = Figure(figsize=(7, 3.8), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Voltaje (V)")
        self.ax.set_title("Señal en tiempo real")
        self.ax.grid(True, alpha=0.3, ls="--")
        self.ax.set_facecolor("#F5F5F5")
        fig.patch.set_facecolor("#F5F5F5")
        self._ts: deque[float] = deque()
        self._vs: deque[float] = deque()
        (self._line,) = self.ax.plot([], [], color="#1565C0", lw=1.2)

    def add(self, t: float, raw: int, v: float) -> None:
        self._ts.append(t)
        self._vs.append(v)
        cutoff = t - self.WINDOW_S
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()
            self._vs.popleft()

    def set_collecting(self, state: bool) -> None:
        color = "#E65100" if state else "#1565C0"
        title = "ADQUIRIENDO MUESTRAS..." if state else "Señal en tiempo real"
        self._line.set_color(color)
        self.ax.set_title(
            title,
            color=color,
            fontweight="bold" if state else "normal",
        )

    def refresh(self) -> None:
        if not self._ts:
            return
        ta = np.asarray(self._ts)
        va = np.asarray(self._vs)
        self._line.set_data(ta, va)
        t_end = ta[-1]
        self.ax.set_xlim(t_end - self.WINDOW_S, t_end + 0.2)
        if len(va) > 1:
            lo, hi = va.min(), va.max()
            mg = max((hi - lo) * 0.15, 0.05)
            self.ax.set_ylim(lo - mg, hi + mg)
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  CalibrationDialog — curva de calibración + estadísticas
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationDialog(QDialog):
    def __init__(
        self,
        x_data: list,
        y_voltage: list,
        y_dbm: list,
        x_label: str = "X",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Curva de Calibración")
        self.resize(900, 720)
        lay = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(
            self._build_tab(x_data, y_dbm, x_label, "dBm", "#2E7D32"),
            "Calibracion - dBm",
        )
        tabs.addTab(
            self._build_tab(x_data, y_voltage, x_label, "Voltaje (V)", "#1565C0"),
            "Calibracion - Voltaje",
        )
        lay.addWidget(tabs)

        btn = QPushButton("Cerrar")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)

    @staticmethod
    def _linreg_stats(x: np.ndarray, y: np.ndarray):
        n = len(x)
        slope, intercept, r, p_val, se_m = stats.linregress(x, y)
        r2 = r ** 2
        y_hat = slope * x + intercept
        res = y - y_hat
        ss_res = float(np.sum(res ** 2))
        s_res = float(np.sqrt(ss_res / max(n - 2, 1)))
        sxx = float(np.sum((x - x.mean()) ** 2))
        se_b = (
            s_res * float(np.sqrt(1.0 / n + x.mean() ** 2 / sxx))
            if sxx > 0 else float("nan")
        )
        rmse = float(np.sqrt(np.mean(res ** 2)))
        return slope, intercept, r, r2, p_val, se_m, se_b, rmse, n

    def _build_tab(
        self,
        x_data: list,
        y_data: list,
        x_label: str,
        y_label: str,
        color: str,
    ) -> QWidget:
        w = QWidget()
        wl = QVBoxLayout(w)

        x = np.asarray(x_data, dtype=float)
        y = np.asarray(y_data, dtype=float)

        slope, intercept, r, r2, p_val, se_m, se_b, rmse, n = self._linreg_stats(x, y)
        sign = "+" if intercept >= 0 else "−"
        eq_str = f"y = {slope:.5f} · x {sign} {abs(intercept):.5f}"

        # Gráfica
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, s=70, color=color, zorder=5, label="Puntos adquiridos")
        xf = np.linspace(x.min(), x.max(), 500)
        ax.plot(
            xf, slope * xf + intercept,
            color="#C62828", lw=2,
            label=f"Ajuste lineal\n{eq_str}",
        )
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f"Calibración — {y_label}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.35, ls="--")
        ax.set_facecolor("#F5F5F5")

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        wl.addWidget(toolbar)
        wl.addWidget(canvas)

        # Estadísticas
        sg = QGroupBox("Parámetros Estadísticos")
        sf = QFormLayout(sg)
        for lbl, val in [
            ("Ecuación",                    eq_str),
            ("Pendiente  m",                f"{slope:.6f}"),
            ("Error estándar  SE(m)",       f"{se_m:.6f}"),
            ("Intercepto  b",               f"{intercept:.6f}"),
            ("Error estándar  SE(b)",       f"{se_b:.6f}"),
            ("Coef. determinación  R²",     f"{r2:.6f}"),
            ("Coef. correlación  R",        f"{r:.6f}"),
            ("RMSE residual",               f"{rmse:.6f}"),
            ("p-valor (pendiente ≠ 0)",     f"{p_val:.3e}"),
            ("N puntos de calibración",     str(n)),
        ]:
            v_lbl = QLabel(val)
            v_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            sf.addRow(QLabel(f"<b>{lbl}:</b>"), v_lbl)
        wl.addWidget(sg)

        return w


# ─────────────────────────────────────────────────────────────────────────────
#  TimeSeriesDialog — serie de tiempo de la última adquisición
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesDialog(QDialog):
    def __init__(
        self,
        voltages: list,
        title: str = "Serie de Tiempo",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Serie de Tiempo")
        self.resize(860, 560)
        lay = QVBoxLayout(self)

        va = np.asarray(voltages, dtype=float)
        n = len(va)
        idx = np.arange(n)
        mean_v = float(va.mean())
        std_v = float(va.std())

        fig = Figure(figsize=(8, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(idx, va, color="#1565C0", lw=0.9, alpha=0.85, label="Voltaje")
        ax.axhline(mean_v, color="#C62828", lw=1.5, ls="--",
                   label=f"Media: {mean_v:.5f} V")
        ax.axhline(mean_v + std_v, color="#E65100", lw=1.0, ls=":")
        ax.axhline(mean_v - std_v, color="#E65100", lw=1.0, ls=":",
                   label=f"±σ: {std_v:.5f} V")
        ax.fill_between(idx, mean_v - std_v, mean_v + std_v,
                        color="#E65100", alpha=0.07)
        ax.set_xlabel("Muestra", fontsize=11)
        ax.set_ylabel("Voltaje (V)", fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, ls="--")
        ax.set_facecolor("#F5F5F5")
        fig.patch.set_facecolor("#F5F5F5")

        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        lay.addWidget(toolbar)
        lay.addWidget(canvas)

        sg = QGroupBox("Estadísticas de la serie")
        sf = QFormLayout(sg)
        for lbl, val in [
            ("N muestras",      str(n)),
            ("Media",           f"{mean_v:.6f} V"),
            ("Desv. estándar",  f"{std_v:.6f} V"),
            ("Mínimo",          f"{float(va.min()):.6f} V"),
            ("Máximo",          f"{float(va.max()):.6f} V"),
        ]:
            v_lbl = QLabel(val)
            v_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            sf.addRow(QLabel(f"<b>{lbl}:</b>"), v_lbl)
        lay.addWidget(sg)

        btn = QPushButton("Cerrar")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)


# ─────────────────────────────────────────────────────────────────────────────
#  MainWindow
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Herramienta de Calibración Serial")
        self.resize(1200, 720)

        self._reader = SerialReader()

        # Estado de colección
        self._collecting = False
        self._collect_done = False        # flag: N alcanzado o timeout
        self._collect_buf: list[tuple[float, int, float]] = []  # (t, raw, v)
        self._collect_t0 = 0.0
        self._collect_dur = 1.0
        self._collect_n = 1              # muestras deseadas
        self._collect_interval = 0.0    # segundos entre muestras aceptadas
        self._collect_last_t = 0.0      # tiempo de la última muestra aceptada

        # Adquisición continua
        self._continuous = False
        self._cont_series: list[float] = []  # voltajes acumulados en modo continuo

        # Datos de calibración [(x, mean_voltage, mean_raw), ...]
        self._calib: list[tuple[float, float, float]] = []

        self._build_ui()

        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.setInterval(80)   # ~12 fps

    # ── construcción de la interfaz ───────────────────────────────────────────

    def _build_ui(self) -> None:
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QHBoxLayout(root_w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # ── panel izquierdo ───────────────────────────────────────────────────
        left_w = QWidget()
        left_w.setFixedWidth(350)
        lv = QVBoxLayout(left_w)
        lv.setSpacing(6)

        # Conexión serial
        cg = QGroupBox("Conexión Serial")
        cf = QFormLayout(cg)
        self.port_cb = QComboBox()
        self.baud_cb = QComboBox()
        self.baud_cb.addItems(["9600", "57600", "115200", "230400"])
        self.baud_cb.setCurrentText("115200")
        ref_btn = QPushButton("Actualizar puertos")
        ref_btn.clicked.connect(self._refresh_ports)
        self.conn_btn = QPushButton("Conectar")
        self.conn_btn.setStyleSheet(_btn_style("#388E3C"))
        self.conn_btn.clicked.connect(self._toggle_conn)
        cf.addRow("Puerto:", self.port_cb)
        cf.addRow("Baud:",   self.baud_cb)
        cf.addRow("",        ref_btn)
        cf.addRow("",        self.conn_btn)
        lv.addWidget(cg)
        self._refresh_ports()

        # Parámetros de adquisición
        ag = QGroupBox("Parámetros de Adquisición")
        af = QFormLayout(ag)

        self.x_label_edit = QLineEdit("PotenciadB")

        self.x_val_sp = QDoubleSpinBox()
        self.x_val_sp.setRange(-1e9, 1e9)
        self.x_val_sp.setDecimals(4)
        self.x_val_sp.setValue(0.0)

        self.n_sp = QSpinBox()
        self.n_sp.setRange(1, 10000)
        self.n_sp.setValue(50)

        self.t_sp = QDoubleSpinBox()
        self.t_sp.setRange(0.1, 3600.0)
        self.t_sp.setDecimals(2)
        self.t_sp.setValue(5.0)
        self.t_sp.setSuffix(" s")

        self.rate_lbl = QLabel("10.00 Hz")
        self.rate_lbl.setStyleSheet("font-weight:bold;color:#1565C0;")

        self.n_sp.valueChanged.connect(self._update_rate_lbl)
        self.t_sp.valueChanged.connect(self._update_rate_lbl)
        self._update_rate_lbl()

        af.addRow("Etiqueta eje X:", self.x_label_edit)
        af.addRow("Valor X:",        self.x_val_sp)
        af.addRow("N° muestras:",    self.n_sp)
        af.addRow("Tiempo adq.:",    self.t_sp)
        af.addRow("Tasa adq.:",      self.rate_lbl)
        lv.addWidget(ag)

        # Archivo CSV de salida
        csvg = QGroupBox("Archivo CSV de salida")
        csvl = QHBoxLayout(csvg)
        self.csv_edit = QLineEdit("calibracion.csv")
        csvl.addWidget(self.csv_edit)
        lv.addWidget(csvg)

        # Botones de acción
        self.add_btn = QPushButton("Agregar muestras al CSV")
        self.add_btn.setEnabled(False)
        self.add_btn.setStyleSheet(_btn_style("#1565C0"))
        self.add_btn.clicked.connect(self._start_collection)

        self.show_btn = QPushButton("Finalizar y ver calibracion")
        self.show_btn.setEnabled(False)
        self.show_btn.setStyleSheet(_btn_style("#BF360C"))
        self.show_btn.clicked.connect(self._show_calibration)

        self.cont_btn = QPushButton("Iniciar Adquisición Continua")
        self.cont_btn.setEnabled(False)
        self.cont_btn.setStyleSheet(_btn_style("#6A1B9A"))
        self.cont_btn.clicked.connect(self._toggle_continuous)

        lv.addWidget(self.add_btn)
        lv.addWidget(self.cont_btn)
        lv.addWidget(self.show_btn)

        # Barra de progreso (visible solo durante adquisición)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setVisible(False)
        lv.addWidget(self.pbar)

        # Tabla de puntos recopilados
        tg = QGroupBox("Puntos de calibración recopilados")
        tl = QVBoxLayout(tg)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["X", "Ȳ (V)", "dBm"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl.setMaximumHeight(200)
        del_btn = QPushButton("Eliminar fila seleccionada")
        del_btn.clicked.connect(self._delete_selected)
        tl.addWidget(self.tbl)
        tl.addWidget(del_btn)
        lv.addWidget(tg)

        lv.addStretch()

        self.status_lbl = QLabel("Desconectado")
        self.status_lbl.setStyleSheet("color:gray;font-style:italic;")
        lv.addWidget(self.status_lbl)

        root.addWidget(left_w)

        # ── panel derecho: gráfica en tiempo real ─────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(0, 0, 0, 0)
        self.rt = RealtimePlot()
        toolbar = NavigationToolbar(self.rt, self)
        rv.addWidget(toolbar)
        rv.addWidget(self.rt)
        root.addWidget(right_w, stretch=1)

    # ── utilidades ────────────────────────────────────────────────────────────

    def _refresh_ports(self) -> None:
        self.port_cb.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_cb.addItems(ports or ["(sin puertos)"])

    def _update_rate_lbl(self) -> None:
        n = self.n_sp.value()
        t = self.t_sp.value()
        self.rate_lbl.setText(f"{n / t:.2f} Hz" if t > 0 else "—")

    def _set_status(self, msg: str, color: str = "black") -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(
            f"color:{color};font-style:normal;font-weight:bold;"
        )

    # ── conexión serial ───────────────────────────────────────────────────────

    def _toggle_conn(self) -> None:
        if self._reader.connected:
            self._continuous = False
            self._tick_timer.stop()
            self._reader.disconnect()
            self.conn_btn.setText("Conectar")
            self.conn_btn.setStyleSheet(_btn_style("#388E3C"))
            self.add_btn.setEnabled(False)
            self.cont_btn.setEnabled(False)
            self.cont_btn.setText("Iniciar Adquisición Continua")
            self.cont_btn.setStyleSheet(_btn_style("#6A1B9A"))
            self._set_status("Desconectado", "gray")
        else:
            port = self.port_cb.currentText()
            baud = int(self.baud_cb.currentText())
            ok, err = self._reader.connect(port, baud)
            if ok:
                self._tick_timer.start()
                self.conn_btn.setText("Desconectar")
                self.conn_btn.setStyleSheet(_btn_style("#C62828"))
                self.add_btn.setEnabled(True)
                self.cont_btn.setEnabled(True)
                self._set_status(f"Conectado → {port} @ {baud}", "#1B5E20")
            else:
                QMessageBox.critical(self, "Error de conexión", err or "Desconocido")

    # ── ciclo principal (llamado por QTimer cada 80 ms) ───────────────────────

    def _tick(self) -> None:
        pts = self._reader.drain()
        direct_pts: list[tuple[int, float]] = []
        for t, raw_val, v in pts:
            self.rt.add(t, raw_val, v)
            if self._continuous:
                direct_pts.append((raw_val, v))
            elif self._collecting and not self._collect_done:
                # Usar el timestamp propio de cada muestra para el intervalo
                if t - self._collect_last_t >= self._collect_interval:
                    self._collect_buf.append((t, raw_val, v))
                    self._collect_last_t = t
                    if len(self._collect_buf) >= self._collect_n:
                        self._collect_done = True
        if pts:
            self.rt.refresh()
        if direct_pts:
            self._process_continuous_pts(direct_pts)

        # Actualizar barra de progreso (basada en muestras recolectadas)
        if self._collecting and self._collect_n > 0:
            pct = int(min(len(self._collect_buf) / self._collect_n * 100, 99))
            self.pbar.setValue(pct)

        # Detectar fin de adquisición — solo si la colección sigue activa
        # (evita que el timeout dispare un segundo procesado con buffer vacío)
        if self._collect_done and self._collecting:
            self._collect_done = False
            buf = list(self._collect_buf)
            self._collect_buf.clear()
            self._collecting = False
            self.rt.set_collecting(False)
            self.pbar.setVisible(False)
            self.add_btn.setEnabled(True)
            self._process_collected(buf)

    # ── adquisición de muestras ───────────────────────────────────────────────

    def _start_collection(self) -> None:
        if self._collecting:
            return
        dur = self.t_sp.value()
        n = self.n_sp.value()
        self._collect_buf.clear()
        self._collecting = True
        self._collect_done = False
        self._collect_t0 = time.time()
        self._collect_dur = dur
        self._collect_n = n
        self._collect_interval = dur / n   # segundos entre muestras
        self._collect_last_t = -float("inf")  # primera muestra siempre aceptada
        self.add_btn.setEnabled(False)
        self.pbar.setValue(0)
        self.pbar.setVisible(True)
        self.rt.set_collecting(True)
        self._set_status(
            f"Adquiriendo {n} muestras en {dur:.1f} s  ({n/dur:.1f} Hz)…",
            "#E65100",
        )

        def _timeout() -> None:
            # Timeout de seguridad: espera el doble del tiempo configurado
            # antes de forzar el cierre, por si el hardware no entrega N muestras a tiempo
            time.sleep(dur * 2)
            self._collect_done = True

        threading.Thread(target=_timeout, daemon=True).start()

    def _process_collected(self, buf: list[tuple[float, int, float]]) -> None:
        if not buf:
            QMessageBox.warning(
                self, "Sin datos",
                "No se recibieron datos durante la adquisición.\n"
                "Verifique la conexión serial.",
            )
            self._set_status("Error: sin datos recibidos", "#C62828")
            return

        x_val = self.x_val_sp.value()
        voltages = [s[2] for s in buf]
        mean_v = float(np.mean(voltages))
        mean_dbm = 39.293 * mean_v - 83.312
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._calib.append((x_val, mean_v, mean_dbm))

        # Escribir en CSV — una fila por muestra individual
        csv_path = self.csv_edit.text()
        is_new = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["x", "dbm", "voltage", "timestamp"])
            for _, _, v in buf:
                w.writerow([x_val, f"{39.293 * v - 83.312:.6f}", f"{v:.6f}", ts_str])

        # Actualizar tabla
        row = self.tbl.rowCount()
        self.tbl.insertRow(row)
        self.tbl.setItem(row, 0, QTableWidgetItem(f"{x_val:.4f}"))
        self.tbl.setItem(row, 1, QTableWidgetItem(f"{mean_v:.6f}"))
        self.tbl.setItem(row, 2, QTableWidgetItem(f"{mean_dbm:.2f}"))

        self.show_btn.setEnabled(len(self._calib) >= 2)
        self._set_status(
            f"Punto anadido: X={x_val}, Y={mean_v:.5f} V  (dBm={mean_dbm:.2f})",
            "#1B5E20",
        )

        # Mostrar serie de tiempo de esta adquisición
        dlg = TimeSeriesDialog(
            voltages,
            title=f"Serie de Tiempo — X = {x_val}  ({len(voltages)} muestras)",
            parent=self,
        )
        dlg.exec()

    def _delete_selected(self) -> None:
        row = self.tbl.currentRow()
        if row < 0:
            return
        self.tbl.removeRow(row)
        self._calib.pop(row)
        self.show_btn.setEnabled(len(self._calib) >= 2)

    # ── adquisición continua (muestra a muestra, sin promediado) ────────────

    def _toggle_continuous(self) -> None:
        if self._continuous:
            self._continuous = False
            self.rt.set_collecting(False)
            self.cont_btn.setText("Iniciar Adquisición Continua")
            self.cont_btn.setStyleSheet(_btn_style("#6A1B9A"))
            self.add_btn.setEnabled(True)
            n_total = len(self._cont_series)
            self._set_status(
                f"Continua detenida — {n_total} muestras totales.",
                "#1565C0",
            )
            if self._cont_series:
                dlg = TimeSeriesDialog(
                    self._cont_series,
                    title=f"Serie de Tiempo — adquisición continua  ({n_total} muestras)",
                    parent=self,
                )
                dlg.exec()
                self._cont_series = []
        else:
            self._cont_series = []
            self._continuous = True
            self.cont_btn.setText("Detener Adquisición Continua")
            self.cont_btn.setStyleSheet(_btn_style("#C62828"))
            self.add_btn.setEnabled(False)
            self.rt.set_collecting(True)
            self._set_status("Adquisición continua activa…", "#6A1B9A")

    def _process_continuous_pts(self, pts: list[tuple[int, float]]) -> None:
        """Escribe cada muestra directamente al CSV y a la tabla, sin promediado."""
        x_val = self.x_val_sp.value()
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_path = self.csv_edit.text()
        is_new = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            if is_new:
                wr.writerow(["x", "dbm", "voltage", "timestamp"])
            for _, v in pts:
                dbm = 39.293 * v - 83.312
                wr.writerow([x_val, f"{dbm:.6f}", f"{v:.6f}", ts_str])
                self._calib.append((x_val, v, dbm))
                self._cont_series.append(v)
        # Actualizar tabla — una fila por muestra
        for _, v in pts:
            dbm = 39.293 * v - 83.312
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            self.tbl.setItem(row, 0, QTableWidgetItem(f"{x_val:.4f}"))
            self.tbl.setItem(row, 1, QTableWidgetItem(f"{v:.6f}"))
            self.tbl.setItem(row, 2, QTableWidgetItem(f"{dbm:.2f}"))
        self.show_btn.setEnabled(len(self._calib) >= 2)
        self._set_status(
            f"Continuo: {len(self._calib)} muestras — X={x_val}  V={pts[-1][1]:.5f}",
            "#6A1B9A",
        )

    # ── curva de calibración ──────────────────────────────────────────────────

    def _show_calibration(self) -> None:
        if len(self._calib) < 2:
            QMessageBox.warning(
                self, "Datos insuficientes",
                "Se necesitan al menos 2 puntos de calibración.",
            )
            return
        xs = [d[0] for d in self._calib]
        ys_v = [d[1] for d in self._calib]
        ys_dbm = [d[2] for d in self._calib]
        dlg = CalibrationDialog(
            xs, ys_v, ys_dbm,
            x_label=self.x_label_edit.text(),
            parent=self,
        )
        dlg.exec()

    # ── cierre ────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._tick_timer.stop()
        self._reader.disconnect()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
 