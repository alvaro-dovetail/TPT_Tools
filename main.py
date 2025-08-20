import sys
import math
from typing import Dict, List, Tuple

import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QFileDialog,
    QTabWidget,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib


# Parsing ---------------------------------------------------------------------

def parse_headers(lines: List[str]) -> Tuple[List[str], Dict[str, str], int]:
    """Parse header lines and return field tokens, units map and header index."""
    field_tokens: List[str] | None = None
    units_tokens: List[str] | None = None
    header_index = -1

    for idx, line in enumerate(lines):
        if line.startswith('<TIME>,HEADER,,VAUNIT'):
            units_tokens = line.strip().split(',')[4:]
        elif (
            line.startswith('<TIME>,HEADER,,,position_pkt,')
            and 'ballistic_extrapolation_position_pkt' in line
        ):
            field_tokens = line.strip().split(',')[4:]
            header_index = idx
            break

    if field_tokens is None:
        raise ValueError('Required field header not found')

    vaunit_map: Dict[str, str] = {}
    if units_tokens and len(units_tokens) == len(field_tokens):
        i = 0
        while i < len(field_tokens):
            group = field_tokens[i]
            i += 1
            while i < len(field_tokens) and not field_tokens[i].endswith('_pkt'):
                key = f"{group}.{field_tokens[i]}"
                vaunit_map[key] = units_tokens[i]
                i += 1
    return field_tokens, vaunit_map, header_index


def parse_rows(path: str, field_header_tokens: List[str]) -> List[Dict[str, float]]:
    """Parse INFO rows returning dictionaries keyed by group.field."""
    groups: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(field_header_tokens):
        g = field_header_tokens[i]
        i += 1
        fields: List[str] = []
        while i < len(field_header_tokens) and not field_header_tokens[i].endswith('_pkt'):
            fields.append(field_header_tokens[i])
            i += 1
        groups.append((g, fields))

    rows: List[Dict[str, float]] = []
    data_started = False
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not data_started:
                if (
                    line.startswith('<TIME>,HEADER,,,position_pkt,')
                    and 'ballistic_extrapolation_position_pkt' in line
                ):
                    data_started = True
                continue
            if not line.startswith('<TIME>,INFO'):
                continue
            tokens = line.strip().split(',')
            idx = 4
            row: Dict[str, float] = {}
            for g, fields in groups:
                if idx < len(tokens) and tokens[idx] == g:
                    idx += 1
                for field in fields:
                    if idx >= len(tokens):
                        value = float('nan')
                    else:
                        tok = tokens[idx]
                        idx += 1
                        try:
                            value = float(tok)
                        except ValueError:
                            value = float('nan')
                    row[f"{g}.{field}"] = value
            rows.append(row)
    return rows


def _convert(value: float, unit: str | None) -> float:
    if math.isnan(value):
        return value
    if unit == 'rad':
        return math.degrees(value)
    if unit in {'ft', 'feet'}:
        return value * 0.3048
    return value


def build_time_and_signals(
    rows: List[Dict[str, float]], vaunit_map: Dict[str, str]
) -> Tuple[List[float], List[Tuple[float, float, float]], List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[Tuple[float, float, float]], Dict[str, List[float]]]:
    """Build time axis, track, rays, endpoints and signals."""
    exclude = {
        'position_pkt.longitude',
        'position_pkt.latitude',
        'position_pkt.height',
        'ballistic_extrapolation_position_pkt.longitude',
        'ballistic_extrapolation_position_pkt.latitude',
        'ballistic_extrapolation_position_pkt.height',
    }

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    signal_keys = sorted(k for k in all_keys if k not in exclude)
    signals: Dict[str, List[float]] = {k: [] for k in signal_keys}

    t: List[float] = []
    track_pts: List[Tuple[float, float, float]] = []
    rays: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    endpoints: List[Tuple[float, float, float]] = []

    for row in rows:
        time = None
        if 'position_pkt.sec' in row:
            sec = row.get('position_pkt.sec', float('nan'))
            usec = row.get('position_pkt.usec', 0.0)
            if not math.isnan(sec):
                time = sec + usec / 1e6
        if time is None:
            time = row.get('position_pkt.t')
        if time is None or math.isnan(time):
            continue
        t.append(time)

        lon = _convert(row.get('position_pkt.longitude', float('nan')), vaunit_map.get('position_pkt.longitude'))
        lat = _convert(row.get('position_pkt.latitude', float('nan')), vaunit_map.get('position_pkt.latitude'))
        alt = _convert(row.get('position_pkt.height', float('nan')), vaunit_map.get('position_pkt.height'))
        lonb = _convert(row.get('ballistic_extrapolation_position_pkt.longitude', float('nan')), vaunit_map.get('ballistic_extrapolation_position_pkt.longitude'))
        latb = _convert(row.get('ballistic_extrapolation_position_pkt.latitude', float('nan')), vaunit_map.get('ballistic_extrapolation_position_pkt.latitude'))
        altb = _convert(row.get('ballistic_extrapolation_position_pkt.height', float('nan')), vaunit_map.get('ballistic_extrapolation_position_pkt.height'))

        track_pts.append((lon, lat, alt))
        start = (lon, lat, alt)
        end = (lonb, latb, altb)
        rays.append((start, end))
        endpoints.append(end)

        for key in signal_keys:
            val = row.get(key, float('nan'))
            val = _convert(val, vaunit_map.get(key))
            signals[key].append(val)

    return t, track_pts, rays, endpoints, signals


# Grouping --------------------------------------------------------------------

group_specs = [
    {
        "name": "acceleration",
        "base": "acceleration_pkt",
        "axes": {
            "x": "acceleration_pkt.acc_x",
            "y": "acceleration_pkt.acc_y",
            "z": "acceleration_pkt.acc_z",
        },
        "mag": "acceleration_pkt.acc_mag",
        "axis_units_key": "acceleration_pkt.acc_x",
        "colors": {
            "x": "#d62728",
            "y": "#2ca02c",
            "z": "#ff7f0e",
            "mag": "#1f77b4",
        },
    },
    {
        "name": "velocity",
        "base": "velocity_pkt",
        "axes": {
            "n": "velocity_pkt.v_n",
            "e": "velocity_pkt.v_e",
            "d": "velocity_pkt.v_d",
        },
        "mag": "velocity_pkt.v_mag",
        "axis_units_key": "velocity_pkt.v_n",
        "colors": {
            "n": "#d62728",
            "e": "#2ca02c",
            "d": "#ff7f0e",
            "mag": "#1f77b4",
        },
    },
]


def build_grouped_signals(
    signals: Dict[str, List[float]],
    units_map: Dict[str, str],
) -> Tuple[List[Dict], List[str]]:
    """Build grouped signal specs and leftover ungrouped keys."""
    grouped: List[Dict] = []
    used: set[str] = set()
    for spec in group_specs:
        components: Dict[str, List[float]] = {}
        key_map: Dict[str, str] = {}
        for comp, key in spec["axes"].items():
            if key in signals:
                components[comp] = signals[key]
                key_map[comp] = key
                used.add(key)
        mag_key = spec.get("mag")
        if mag_key and mag_key in signals:
            components["mag"] = signals[mag_key]
            key_map["mag"] = mag_key
            used.add(mag_key)
        if components:
            unit = units_map.get(spec.get("axis_units_key"))
            grouped.append(
                {
                    "name": spec["name"],
                    "components": components,
                    "keys": key_map,
                    "unit": unit,
                    "colors": spec["colors"],
                }
            )
    leftover = [k for k in signals.keys() if k not in used]
    return grouped, leftover


# Rendering -------------------------------------------------------------------

def render_map(
    ax,
    track_pts: List[Tuple[float, float, float]],
    rays: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    endpoints: List[Tuple[float, float, float]],
    rays_limit: int,
) -> None:
    ax.clear()
    if not track_pts:
        ax.figure.canvas.draw()
        return

    lons = [p[0] for p in track_pts]
    lats = [p[1] for p in track_pts]
    ax.plot(lons, lats, color='yellow', linewidth=3, label='Track')

    if rays:
        step = max(1, len(rays) // max(1, rays_limit))
        for a, b in rays[::step]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=(1, 1, 0, 0.3), linewidth=1)

    if endpoints:
        lons_e = [p[0] for p in endpoints]
        lats_e = [p[1] for p in endpoints]
        ax.plot(lons_e, lats_e, color='orange', linewidth=3, label='Envelope')

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='best')
    ax.figure.tight_layout()
    ax.figure.canvas.draw()


def render_timeseries(
    canvas: FigureCanvas,
    t: List[float],
    signals: Dict[str, List[float]],
    units_map: Dict[str, str],
    filter_text: str = '',
) -> Tuple[List[List[str]], List, List[List[List[float]]]]:
    """Render stacked time series plots and return keys, axes and data."""
    fig = canvas.figure
    fig.clear()

    grouped, leftover = build_grouped_signals(signals, units_map)

    ft = filter_text.lower()
    if ft:
        grouped = [
            g
            for g in grouped
            if ft in g["name"].lower()
            or any(ft in k.lower() for k in g["keys"].values())
        ]
        leftover = sorted([k for k in leftover if ft in k.lower()])
    else:
        leftover = sorted(leftover)

    n = len(grouped) + len(leftover)
    if n == 0:
        fig.canvas.draw()
        return [], [], []

    axes = fig.subplots(
        n,
        1,
        sharex=True,
        gridspec_kw={"hspace": 0.35, "left": 0.07, "right": 0.98, "top": 0.98, "bottom": 0.05},
    )
    if n == 1:
        axes = [axes]

    axis_keys: List[List[str]] = []
    axis_data: List[List[List[float]]] = []
    idx = 0

    for grp in grouped:
        ax = axes[idx]
        idx += 1
        keys_in_axis: List[str] = []
        data_in_axis: List[List[float]] = []
        for comp, key in grp["keys"].items():
            y = grp["components"][comp]
            ax.plot(
                t,
                y,
                linewidth=1.2,
                label=comp,
                color=grp["colors"][comp],
            )
            keys_in_axis.append(key)
            data_in_axis.append(y)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
        label = f"{grp['name']}{f' ({grp['unit']})' if grp['unit'] else ''}"
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#ffffff",
                alpha=0.75,
                edgecolor="#cccccc",
            ),
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        axis_keys.append(keys_in_axis)
        axis_data.append(data_in_axis)

    for key in leftover:
        ax = axes[idx]
        idx += 1
        y = signals[key]
        ax.plot(t, y, linewidth=1.0)
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
        unit = units_map.get(key)
        label = f"{key}{f' ({unit})' if unit else ''}"
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#ffffff",
                alpha=0.75,
                edgecolor="#cccccc",
            ),
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        axis_keys.append([key])
        axis_data.append([y])

    axes[-1].set_xlabel("Time (s)")

    for a, b in zip(axes[:-1], axes[1:]):
        ymid = (a.get_position().y0 + b.get_position().y1) / 2.0
        fig.lines.append(
            matplotlib.lines.Line2D(
                [0.05, 0.98],
                [ymid, ymid],
                transform=fig.transFigure,
                color="#dddddd",
                linewidth=1,
            )
        )

    fig.set_figheight(max(1, n) * 1.2)
    if hasattr(canvas, "setMinimumHeight"):
        canvas.setMinimumHeight(int(fig.get_figheight() * fig.dpi))
    fig.canvas.draw()
    return axis_keys, axes, axis_data


class SharedCrosshair:
    """Synchronized crosshair across multiple axes with blitting."""

    def __init__(
        self,
        canvas: FigureCanvas,
        axes: List,
        t: List[float],
        data: List[List[List[float]]],
        keys: List[List[str]],
        status_bar=None,
    ) -> None:
        self.canvas = canvas
        self.axes = axes
        self.t = t
        self.data = data
        self.keys = keys
        self.status = status_bar
        self.vlines = []
        self.markers: List[List] = []
        for ax, series in zip(axes, data):
            lines = list(ax.get_lines())
            v = ax.axvline(0, color="red", linewidth=0.8, alpha=0.7, visible=False)
            marker_list = []
            for line in lines:
                m, = ax.plot([], [], "o", color=line.get_color(), markersize=4, visible=False)
                marker_list.append(m)
            self.vlines.append(v)
            self.markers.append(marker_list)
        self.backgrounds: Dict = {}
        self.cids: List[int] = []
        self.enabled = False

    # ------------------------------------------------------------------
    def enable(self) -> None:
        if not self.enabled:
            self.cids.append(self.canvas.mpl_connect("draw_event", self._on_draw))
            self.cids.append(
                self.canvas.mpl_connect("motion_notify_event", self._on_move)
            )
            self.cids.append(
                self.canvas.mpl_connect("axes_leave_event", self._on_leave)
            )
            self.enabled = True

    def disable(self) -> None:
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids.clear()
        self.enabled = False
        self._hide()

    # ------------------------------------------------------------------
    def _on_draw(self, _event) -> None:
        for ax in self.axes:
            self.backgrounds[ax] = self.canvas.copy_from_bbox(ax.bbox)

    def _on_move(self, event) -> None:
        if event.xdata is None:
            self._hide()
            return
        idx = np.searchsorted(self.t, event.xdata)
        idx = max(0, min(idx, len(self.t) - 1))
        time = self.t[idx]
        for ax, v, marks, series in zip(
            self.axes, self.vlines, self.markers, self.data
        ):
            if ax in self.backgrounds:
                self.canvas.restore_region(self.backgrounds[ax])
            v.set_xdata([time, time])
            v.set_ydata(ax.get_ylim())
            v.set_visible(True)
            for m, y in zip(marks, series):
                if y:
                    j = min(idx, len(y) - 1)
                    m.set_data([time], [y[j]])
                    m.set_visible(True)
                ax.draw_artist(m)
            ax.draw_artist(v)
            self.canvas.blit(ax.bbox)
        if self.status and event.inaxes in self.axes:
            a_idx = self.axes.index(event.inaxes)
            msgs = []
            for key, y in zip(self.keys[a_idx], self.data[a_idx]):
                val = y[min(idx, len(y) - 1)] if y else float("nan")
                msgs.append(f"{key}={val:.3f}")
            self.status.showMessage(f"t={time:.3f}s  " + "  ".join(msgs))

    def _on_leave(self, _event) -> None:
        self._hide()

    def _hide(self) -> None:
        for ax, v, marks in zip(self.axes, self.vlines, self.markers):
            if ax in self.backgrounds:
                self.canvas.restore_region(self.backgrounds[ax])
            v.set_visible(False)
            ax.draw_artist(v)
            for m in marks:
                m.set_visible(False)
                ax.draw_artist(m)
            self.canvas.blit(ax.bbox)
        if self.status:
            self.status.clearMessage()


# KML -------------------------------------------------------------------------

def make_kml(
    track: List[Tuple[float, float, float]],
    rays: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    endpoints: List[Tuple[float, float, float]],
    rays_limit: int,
) -> str:
    def fmt(pt: Tuple[float, float, float]) -> str:
        return f"{pt[0]:.9f},{pt[1]:.9f},{pt[2]:.3f}"

    track_coords = ' '.join(fmt(p) for p in track)
    step = max(1, len(rays) // max(1, rays_limit)) if rays else 1
    ray_strings = []
    for a, b in rays[::step]:
        coords = f"{fmt(a)} {fmt(b)}"
        ray_strings.append(
            f"<LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>{coords}</coordinates></LineString>"
        )
    envelope_coords = ' '.join(fmt(p) for p in endpoints)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Track Export</name>
  <Style id="track"><LineStyle><color>ff00ffff</color><width>3</width></LineStyle></Style>
  <Style id="rays"><LineStyle><color>9900ffff</color><width>1</width></LineStyle></Style>
  <Style id="envelope"><LineStyle><color>ff00a5ff</color><width>3</width></LineStyle></Style>
  <Placemark><name>Track</name><styleUrl>#track</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>{track_coords}</coordinates></LineString>
  </Placemark>
  <Placemark><name>Rays</name><styleUrl>#rays</styleUrl>
    <MultiGeometry>{''.join(ray_strings)}</MultiGeometry>
  </Placemark>
  <Placemark><name>Envelope</name><styleUrl>#envelope</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>{envelope_coords}</coordinates></LineString>
  </Placemark>
</Document>
</kml>"""


# Application -----------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('TPT Tools')

        self.t: List[float] = []
        self.track: List[Tuple[float, float, float]] = []
        self.rays: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        self.endpoints: List[Tuple[float, float, float]] = []
        self.signals: Dict[str, List[float]] = {}
        self.units_map: Dict[str, str] = {}
        self.crosshair: SharedCrosshair | None = None

        open_btn = QPushButton('Open…')
        open_btn.clicked.connect(self.open_file)

        self.rays_spin = QSpinBox()
        self.rays_spin.setRange(1, 1_000_000)
        self.rays_spin.setValue(1500)
        self.rays_spin.setPrefix('Rays limit: ')
        self.rays_spin.valueChanged.connect(self.update_map)

        self.export_btn = QPushButton('Export KML')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_kml)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.rays_spin)
        top.addWidget(self.export_btn)
        top_widget = QWidget()
        top_widget.setLayout(top)

        self.tabs = QTabWidget()

        # Map tab
        self.map_fig = Figure()
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_ax = self.map_fig.add_subplot(111)
        map_layout = QVBoxLayout()
        map_layout.addWidget(self.map_canvas)
        map_widget = QWidget()
        map_widget.setLayout(map_layout)
        self.tabs.addTab(map_widget, 'Map')

        # Time series tab
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText('Filter signals')
        self.filter_edit.textChanged.connect(self.update_timeseries)

        self.ts_fig = Figure()
        self.ts_canvas = FigureCanvas(self.ts_fig)
        self.ts_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ts_toolbar = NavigationToolbar2QT(self.ts_canvas, self)

        self.ts_plot_container = QWidget()
        pc_layout = QVBoxLayout()
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.addWidget(self.ts_toolbar)
        pc_layout.addWidget(self.ts_canvas)
        self.ts_plot_container.setLayout(pc_layout)

        self.ts_scroll = QScrollArea()
        self.ts_scroll.setWidget(self.ts_plot_container)
        self.ts_scroll.setWidgetResizable(True)

        ts_layout = QVBoxLayout()
        ts_layout.addWidget(self.filter_edit)
        ts_layout.addWidget(self.ts_scroll)
        ts_widget = QWidget()
        ts_widget.setLayout(ts_layout)
        self.ts_tab = ts_widget
        self.tabs.addTab(ts_widget, 'Time Series')

        layout = QVBoxLayout()
        layout.addWidget(top_widget)
        layout.addWidget(self.tabs)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar()
        self.tabs.currentChanged.connect(self.on_tab_changed)

    # ------------------------------------------------------------------
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open file',
            '',
            'TPT/CSV files (*.tpt *.csv *.txt);;All files (*)',
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            header_tokens, vaunit_map, _ = parse_headers(lines)
            rows = parse_rows(path, header_tokens)
            self.t, self.track, self.rays, self.endpoints, self.signals = build_time_and_signals(
                rows, vaunit_map
            )
            self.units_map = vaunit_map
        except Exception as exc:  # pragma: no cover - display error in UI
            self.statusBar().showMessage(f'Error: {exc}')
            return
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage(
            f'Loaded {len(self.track)} points, {len(self.rays)} rays, {len(self.signals)} signals'
        )
        self.update_map()
        self.update_timeseries()

    def update_map(self) -> None:
        render_map(
            self.map_ax, self.track, self.rays, self.endpoints, self.rays_spin.value()
        )

    def update_timeseries(self) -> None:
        self.filter_edit.setVisible(len(self.signals) > 50)
        filter_text = self.filter_edit.text() if self.filter_edit.isVisible() else ''
        if self.crosshair:
            self.crosshair.disable()
            self.crosshair = None
        axis_keys, axes, axis_data = render_timeseries(
            self.ts_canvas, self.t, self.signals, self.units_map, filter_text
        )
        if axes:
            self.crosshair = SharedCrosshair(
                self.ts_canvas, axes, self.t, axis_data, axis_keys, self.statusBar()
            )
            if self.tabs.currentWidget() is self.ts_tab:
                self.crosshair.enable()

    def on_tab_changed(self, idx: int) -> None:
        if self.crosshair:
            if self.tabs.widget(idx) is self.ts_tab:
                self.crosshair.enable()
            else:
                self.crosshair.disable()

    def export_kml(self) -> None:
        if not self.track:
            self.statusBar().showMessage('No data to export')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save KML', '', 'KML files (*.kml)'
        )
        if not path:
            return
        kml = make_kml(self.track, self.rays, self.endpoints, self.rays_spin.value())
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(kml)
            self.statusBar().showMessage(f'Saved {path}')
        except Exception as exc:  # pragma: no cover - display error in UI
            self.statusBar().showMessage(f'Error: {exc}')


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec())
