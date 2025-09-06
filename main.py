import sys
import math
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import os
import numpy as np
from xml.sax.saxutils import escape as xml_escape

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
    QLabel,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib

# -----------------------------------------------------------------------------
# TUNABLES for visual smoothing / densifying
SMOOTH_WINDOW = 7        # odd >=3 recommended; 1 disables smoothing
DENSIFY_FACTOR = 6       # >=1; 1 disables densifying
# 3D gate model scale for KML exports
GATE_MODEL_SCALE = (1.0, 1.0, 1.0)
EXTRUSION_HEIGHT_M = 200.0  # height of the vertical "wall" for extruded lines

# Warnings collected during KML generation (e.g. missing assets)
GATE_MODEL_WARNINGS: List[str] = []
# -----------------------------------------------------------------------------

# Parsing ---------------------------------------------------------------------

def parse_headers(lines: List[str]) -> Tuple[List[str], Dict[str, str], int]:
    field_tokens: List[str] | None = None
    units_tokens: List[str] | None = None
    header_index = -1

    for idx, line in enumerate(lines):
        if line.startswith('<TIME>,HEADER,,VAUNIT'):
            units_tokens = line.strip().split(',')[4:]
        if (line.startswith('<TIME>,HEADER,,,position_pkt,')
                and 'ballistic_extrapolation_position_pkt' in line):
            field_tokens = line.strip().split(',')[4:]
            header_index = idx

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
) -> Tuple[
    List[float],
    List[Tuple[float, float, float]],
    List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    List[Tuple[float, float, float]],
    Dict[str, List[float]],
]:
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

        start = (lon, lat, alt)
        end = (lonb, latb, altb)

        track_pts.append(start)
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
    safety_line: Optional[List[Tuple[float, float, float]]] = None,
    crowd_line: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
    ax.clear()
    any_legend = False

    # Track: green, alpha 0.8, width 2
    if track_pts:
        lons = [p[0] for p in track_pts]
        lats = [p[1] for p in track_pts]
        ax.plot(lons, lats, color=(0.0, 1.0, 0.0, 0.8), linewidth=2, label='Track')
        any_legend = True

    # Rays: unchanged (yellow-ish, alpha 0.3, width 1)
    if rays:
        step = max(1, len(rays) // max(1, rays_limit))
        for a, b in rays[::step]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=(1, 1, 0, 0.3), linewidth=1)

    # Envelope unchanged (orange, width 3)
    if endpoints:
        lons_e = [p[0] for p in endpoints]
        lats_e = [p[1] for p in endpoints]
        ax.plot(lons_e, lats_e, color='orange', linewidth=3, label='Envelope')
        any_legend = True

    # Safety line: yellow, alpha 0.6, width 8
    if safety_line:
        lons_s = [p[0] for p in safety_line]
        lats_s = [p[1] for p in safety_line]
        ax.plot(lons_s, lats_s, color=(1.0, 1.0, 0.0, 0.6), linewidth=8, label='Safety Line')
        any_legend = True

    # Crowd line: #aa00ff, alpha 0.6, width 8
    if crowd_line:
        lons_c = [p[0] for p in crowd_line]
        lats_c = [p[1] for p in crowd_line]
        ax.plot(lons_c, lats_c, color=(170/255.0, 0.0, 1.0, 0.6), linewidth=8, label='Crowd Line')
        any_legend = True

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')
    ax.set_aspect('equal', adjustable='datalim')
    if any_legend:
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
    axes = np.atleast_1d(axes).ravel().tolist()

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
            ls = "--" if comp != "mag" else "-"   # dashed for axes, solid for magnitude
            ax.plot(
                t,
                y,
                linewidth=1.2,
                label=comp,
                color=grp["colors"][comp],
                linestyle=ls,
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
        for ax, _series in zip(axes, data):
            v = ax.axvline(0, color="red", linewidth=0.8, alpha=0.7, visible=False)
            marker_list = []
            for line in ax.get_lines():
                m, = ax.plot([], [], "o", color=line.get_color(), markersize=4, visible=False)
                marker_list.append(m)
            self.vlines.append(v)
            self.markers.append(marker_list)
        self.backgrounds: Dict = {}
        self.cids: List[int] = []
        self.enabled = False

    def enable(self) -> None:
        if not self.enabled:
            self.cids.append(self.canvas.mpl_connect("draw_event", self._on_draw))
            self.cids.append(self.canvas.mpl_connect("motion_notify_event", self._on_move))
            self.cids.append(self.canvas.mpl_connect("axes_leave_event", self._on_leave))
            self.enabled = True

    def disable(self) -> None:
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids.clear()
        self.enabled = False
        self._hide()

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
        for ax, v, marks, series in zip(self.axes, self.vlines, self.markers, self.data):
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


# Helpers for smoothing / densifying -----------------------------------------

def _moving_average_path(
    points: List[Tuple[float, float, float]], window: int
) -> List[Tuple[float, float, float]]:
    """Edge-preserving moving average on lon/lat/alt; length preserved."""
    if window is None or window <= 1 or len(points) < 3:
        return points
    if window % 2 == 0:
        window += 1
    arr = np.asarray(points, dtype=float)  # Nx3
    pad = window // 2
    arrp = np.pad(arr, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.empty_like(arr)
    for col in range(3):
        smoothed[:, col] = np.convolve(arrp[:, col], kernel, mode='valid')
    return [tuple(row) for row in smoothed.tolist()]


def _moving_average_1d(values: List[float], window: int) -> List[float]:
    """Edge-preserving moving average for 1D time series; length preserved."""
    if window is None or window <= 1 or len(values) < 3:
        return values
    if window % 2 == 0:
        window += 1
    arr = np.asarray(values, dtype=float)
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    arrp = np.pad(arr, pad, mode='edge')
    smoothed = np.convolve(arrp, kernel, mode='valid')
    nan_mask = np.isnan(arr)
    smoothed[nan_mask] = np.nan
    return smoothed.tolist()


def _densify_line(
    points: List[Tuple[float, float, float]], factor: int
) -> List[Tuple[float, float, float]]:
    """Linear interpolation between points; preserves endpoints."""
    if factor is None or factor <= 1 or len(points) < 2:
        return points
    dense: List[Tuple[float, float, float]] = []
    for a, b in zip(points, points[1:]):
        dense.append(a)
        for k in range(1, factor):
            s = k / factor
            dense.append((
                a[0] * (1 - s) + b[0] * s,
                a[1] * (1 - s) + b[1] * s,
                a[2] * (1 - s) + b[2] * s,
            ))
    dense.append(points[-1])
    return dense


# KML -------------------------------------------------------------------------

GX_TRACK_LIMIT = 50_000  # keep
KNOTS_PER_MPS = 1.9438444924574
G_PER_MPS2 = 1.0 / 9.80665

def _ffill_numeric(values: List[float], target_len: int) -> List[float]:
    out: List[float] = []
    last = 0.0
    vals = (values[:target_len] + [math.nan] * max(0, target_len - len(values)))
    for v in vals:
        if v is None or not math.isfinite(v):
            out.append(last)
        else:
            last = float(v)
            out.append(last)
    return out


def make_gx_track_kml(
    t: List[float],
    track_pts: List[Tuple[float, float, float]],
    arrays: Dict[str, List[float]],
    max_points: int = GX_TRACK_LIMIT,
    angles_deg: tuple[list[float], list[float], list[float]] | None = None,
) -> str:
    """Build a gx:Track placemark with per-sample arrays suitable for GE's Elevation Profile."""
    if not t or not track_pts:
        return ""

    # Thin uniformly
    step = max(1, len(track_pts) // max_points) if max_points and len(track_pts) > max_points else 1
    t_thin = t[::step]
    pts_thin = track_pts[::step]
    N = min(len(t_thin), len(pts_thin))
    t_thin = t_thin[:N]
    pts_thin = pts_thin[:N]

    angle_lines: List[str] = []
    if angles_deg is not None:
        hdg, pit, rol = angles_deg
        hdg_thin = hdg[::step][:N]
        pit_thin = pit[::step][:N]
        rol_thin = rol[::step][:N]

        def _nan_to_zero(x: float | None) -> float:
            return 0.0 if (x is None or not math.isfinite(x)) else float(x)

        angle_lines = [
            f"<gx:angles>{(_nan_to_zero(h) % 360.0):.3f} {_nan_to_zero(p):.3f} {_nan_to_zero(r):.3f}</gx:angles>"
            for h, p, r in zip(hdg_thin, pit_thin, rol_thin)
        ]

    # Build <when> + <gx:coord>
    t0 = datetime(1970, 1, 1, tzinfo=timezone.utc)
    when_lines: List[str] = []
    coord_lines: List[str] = []
    for ts, pt in zip(t_thin, pts_thin):
        when = (t0 + timedelta(seconds=ts)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        when_lines.append(f"<when>{when}</when>")
        coord_lines.append(f"<gx:coord>{pt[0]:.9f} {pt[1]:.9f} {pt[2]:.3f}</gx:coord>")

    # Clean arrays: numeric, N-length, forward-filled
    array_parts: List[str] = []
    for name, values in arrays.items():
        cleaned = _ffill_numeric(values[::step], N)
        vals = "".join(f"<gx:value>{v:.6g}</gx:value>" for v in cleaned)
        array_parts.append(f"<gx:SimpleArrayData name=\"{name}\">{vals}</gx:SimpleArrayData>")

    ext = f"<ExtendedData><SchemaData schemaUrl=\"#ts_schema\">{''.join(array_parts)}</SchemaData></ExtendedData>" if array_parts else ""

    return (
        "<Placemark>"
        "<name>Track (time series)</name>"
        "<styleUrl>#track</styleUrl>"
        "<gx:Track>"
        "<altitudeMode>absolute</altitudeMode>"
        f"{ext}"
        f"{''.join(when_lines)}"
        f"{''.join(coord_lines)}"
        f"{''.join(angle_lines)}"
        "</gx:Track>"
        "</Placemark>"
    )


def _find_project_from_any_root(root: object) -> Optional[dict]:
    """Accept either a dict or a list root and return the first dict with 'trackList'."""
    if isinstance(root, dict):
        return root
    if isinstance(root, list):
        for obj in root:
            if isinstance(obj, dict) and ('trackList' in obj or 'raceBoxList' in obj):
                return obj
    return None


def make_gate_models_kml(
    track_spec: dict,
    assets_dir: str = "assets",
    *,
    scale: Tuple[float, float, float] = GATE_MODEL_SCALE,
    altitude_mode: str = "absolute",
    kml_dir: Optional[str] = None,  # <-- added
) -> str:
    global GATE_MODEL_WARNINGS
    GATE_MODEL_WARNINGS = []

    if not isinstance(track_spec, dict):
        return ""

    # determine gate_list (unchanged) ...
    gate_list: Optional[List[dict]] = None
    tracks = track_spec.get("trackList") if isinstance(track_spec.get("trackList"), list) else None
    if tracks:
        # prefer the first track with gateList
        for tr in tracks:
            if isinstance(tr, dict) and tr.get("gateList"):
                gate_list = tr.get("gateList")
                break
    if not gate_list:
        return ""

    type_map = {
        "StartGate": "StartGate.dae",
        "TwinGate": "Gate.dae",
        "SingleLeft": "SinglePylonGate.dae",
        "SingleRight": "SinglePylonGate.dae",
    }

    placemarks: List[str] = []
    missing_files: set[str] = set()
    assets_dir_exists = os.path.isdir(assets_dir)

    for gate in gate_list:
        gtype = gate.get("type")
        filename = type_map.get(gtype)
        if not filename:
            GATE_MODEL_WARNINGS.append(f"Unrecognized gate type: {gtype}")
            continue

        heading = float(gate.get("headingDeg", 0.0))
        if gtype == "SingleLeft":
            heading += 180.0
        heading = heading % 360.0

        pos = gate.get("position", {}) or {}
        lat = pos.get("latitude") if pos.get("latitude") is not None else pos.get("latitudeDeg")
        lon = pos.get("longitude") if pos.get("longitude") is not None else pos.get("longitudeDeg")
        if lat is None or lon is None:
            GATE_MODEL_WARNINGS.append(f"Gate {gate.get('id')} missing lat/lon")
            continue
        elev = pos.get("elevationM", pos.get("elevation", 0.0))

        # Filesystem path to the model (next to main.py)
        model_fs_path = os.path.join(assets_dir, filename)

        # KML href relative to the KML output directory
        if kml_dir:
            href = os.path.relpath(model_fs_path, kml_dir).replace(os.sep, "/")
        else:
            href = os.path.join(os.path.basename(os.path.normpath(assets_dir)), filename).replace("\\", "/")

        if not os.path.isfile(model_fs_path):
            missing_files.add(href)

        placemarks.append(
            "<Placemark>"
            f"<name>Gate {xml_escape(str(gate.get('id')))} [{xml_escape(str(gtype))}]</name>"
            "<Model>"
            f"<altitudeMode>{altitude_mode}</altitudeMode>"
            "<Location>"
            f"<longitude>{lon}</longitude>"
            f"<latitude>{lat}</latitude>"
            f"<altitude>{elev}</altitude>"
            "</Location>"
            "<Orientation>"
            f"<heading>{heading}</heading>"
            "<tilt>0</tilt><roll>0</roll>"
            "</Orientation>"
            "<Scale>"
            f"<x>{scale[0]}</x><y>{scale[1]}</y><z>{scale[2]}</z>"
            "</Scale>"
            "<Link>"
            f"<href>{href}</href>"
            "</Link>"
            "</Model>"
            "</Placemark>"
        )

    if not assets_dir_exists:
        GATE_MODEL_WARNINGS.append(f"Missing assets folder: {os.path.basename(os.path.normpath(assets_dir)) or assets_dir}")
    for m in sorted(missing_files):
        GATE_MODEL_WARNINGS.append(f"Missing model file: {m}")

    return "".join(placemarks)


def make_kml(
    t: List[float],
    track: List[Tuple[float, float, float]],
    rays: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    endpoints: List[Tuple[float, float, float]],
    signals: Dict[str, List[float]],
    rays_limit: int,
    *,
    doc_name: str = "Track Export",
    smooth_window: int = SMOOTH_WINDOW,
    densify_factor: int = DENSIFY_FACTOR,
    safety_line: Optional[List[Tuple[float, float, float]]] = None,
    crowd_line: Optional[List[Tuple[float, float, float]]] = None,
    track_spec: dict | None = None,
    assets_dir: str = "assets",
    kml_dir: Optional[str] = None,  # <-- added
) -> str:
    def fmt(pt: Tuple[float, float, float]) -> str:
        return f"{pt[0]:.9f},{pt[1]:.9f},{pt[2]:.3f}"

    # --- Apply smoothing to track coordinates ---
    smoothed_track = _moving_average_path(track, smooth_window)

    # --- Build visual track (densified from smoothed) ---
    dense_track = _densify_line(smoothed_track, densify_factor)
    track_coords = " ".join(fmt(p) for p in dense_track)

    # --- Rays: re-anchor starts to the smoothed track ---
    rays_visual = list(zip(smoothed_track, endpoints))
    step = max(1, len(rays_visual) // max(1, rays_limit)) if rays_visual else 1
    ray_strings = []
    for a, b in rays_visual[::step]:
        coords = f"{fmt(a)} {fmt(b)}"
        ray_strings.append(
            "<LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>"
            f"<coordinates>{coords}</coordinates></LineString>"
        )

    envelope_coords = " ".join(fmt(p) for p in endpoints)

    # Arrays (converted + smoothed)
    KNOTS_PER_MPS = 1.9438444924574
    G_PER_MPS2 = 1.0 / 9.80665
    field_specs = [
        ("speed_kt",     "velocity_pkt.v_mag",          "Speed (kt)",      lambda v: v * KNOTS_PER_MPS),
        ("tas_kt",       "body_pkt.tas",                "TAS (kt)",        lambda v: v * KNOTS_PER_MPS),
        ("v_n_mps",      "velocity_pkt.v_n",            "V_N (m/s)",       None),
        ("v_e_mps",      "velocity_pkt.v_e",            "V_E (m/s)",       None),
        ("v_d_mps",      "velocity_pkt.v_d",            "V_D (m/s)",       None),
        ("acc_x_mps2",   "acceleration_pkt.acc_x",      "Acc X (m/s^2)",   None),
        ("acc_y_mps2",   "acceleration_pkt.acc_y",      "Acc Y (m/s^2)",   None),
        ("acc_z_mps2",   "acceleration_pkt.acc_z",      "Acc Z (m/s^2)",   None),
        ("acc_mag_g",    "acceleration_pkt.acc_mag",    "Acc Mag (g)",     lambda v: v * G_PER_MPS2),
        ("hdg_deg",      "position_pkt.heading",        "Heading (deg)",   None),
        ("pitch_deg",    "position_pkt.pitch",          "Pitch (deg)",     None),
        ("roll_deg",     "position_pkt.roll",           "Roll (deg)",      None),
        ("alpha_deg",    "body_pkt.alpha",              "Alpha (deg)",     None),
        ("beta_deg",     "body_pkt.beta",               "Beta (deg)",      None),
        ("aileron",      "parameter_pkt.aileron",       "Aileron",         None),
        ("elevator",     "parameter_pkt.elevator",      "Elevator",        None),
        ("rudder",       "parameter_pkt.rudder",        "Rudder",          None),
        ("thrust",       "parameter_pkt.thrust",        "Thrust",          None),
    ]

    arrays: Dict[str, List[float]] = {}
    display_names: Dict[str, str] = {}
    for name, src, disp, conv in field_specs:
        vals = signals.get(src)
        if vals:
            if conv:
                converted_vals = [float('nan') if (v is None or not math.isfinite(v)) else conv(v) for v in vals]
            else:
                converted_vals = vals
            smoothed_vals = _moving_average_1d(converted_vals, smooth_window)
            arrays[name] = smoothed_vals
            display_names[name] = disp

    schema = ""
    if arrays:
        fields = []
        for name, disp in display_names.items():
            fields.append(
                f'<gx:SimpleArrayField name="{name}" type="float"><displayName>{disp}</displayName></gx:SimpleArrayField>'
            )
        schema = f"<Schema id=\"ts_schema\">{''.join(fields)}</Schema>"

    hdg_raw = signals.get("position_pkt.heading") or []
    pit_raw = signals.get("position_pkt.pitch") or []
    rol_raw = signals.get("position_pkt.roll") or []

    hdg_s = _moving_average_1d(hdg_raw, smooth_window)
    pit_s = _moving_average_1d(pit_raw, smooth_window)
    rol_s = _moving_average_1d(rol_raw, smooth_window)
    # pit_s = [-p for p in pit_s]  # optional inversion if needed

    angles: tuple[list[float], list[float], list[float]] | None = None
    if hdg_s and pit_s and rol_s:
        angles = (hdg_s, pit_s, rol_s)

    gx_track = make_gx_track_kml(t, smoothed_track, arrays, angles_deg=angles)

    # Safety & Crowd lines (clampToGround)
    def line_to_kml(name: str, style: str, line: List[Tuple[float, float, float]]) -> str:
        coords = " ".join(f"{p[0]:.9f},{p[1]:.9f},0" for p in line)
        return f"""
    <Placemark><name>{xml_escape(name)}</name><styleUrl>#{style}</styleUrl>
      <LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>
        <coordinates>{coords}</coordinates>
      </LineString>
    </Placemark>""".rstrip()

    def line_to_kml_extruded(
        name: str,
        style: str,
        line: List[Tuple[float, float, float]],
        height_m: float,
    ) -> str:
        """
        Build an extruded LineString at a fixed relative height.
        Emits <extrude>1</extrude> with <altitudeMode>relativeToGround</altitudeMode>.
        Coordinates use altitude=height_m so GE draws a “fence” down to ground.
        """
        if not line:
            return ""
        coords = " ".join(f"{p[0]:.9f},{p[1]:.9f},{height_m:.3f}" for p in line)
        return f"""
    <Placemark><name>{xml_escape(name)} extruded</name><styleUrl>#{style}</styleUrl>
      <LineString>
        <coordinates>{coords}</coordinates>
        <extrude>1</extrude>
        <altitudeMode>relativeToGround</altitudeMode>
      </LineString>
    </Placemark>""".rstrip()

    safety_kml = line_to_kml("Safety Line", "safety", safety_line) if safety_line else ""
    crowd_kml  = line_to_kml("Crowd Line",  "crowd",  crowd_line)  if crowd_line  else ""
    safety_kml_ex = line_to_kml_extruded("safetyLine", "safety_ex", safety_line, EXTRUSION_HEIGHT_M) if safety_line else ""
    crowd_kml_ex  = line_to_kml_extruded("crowdLine",  "crowd_ex",  crowd_line,  EXTRUSION_HEIGHT_M) if crowd_line  else ""
    gates_kml = make_gate_models_kml(track_spec, assets_dir, kml_dir=kml_dir) if track_spec else ""
    gates_folder = f"<Folder>\n  <name>Gates</name>\n  {gates_kml}\n</Folder>" if gates_kml else ""

    safety_folder = ""
    if any([safety_kml, crowd_kml, safety_kml_ex, crowd_kml_ex]):
        safety_folder = f"""
    <Folder>
      <name>Safety areas</name>
      {crowd_kml}
      {crowd_kml_ex}
      {safety_kml}
      {safety_kml_ex}
    </Folder>
    """.rstrip()

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <name>{xml_escape(doc_name)}</name>

    <!-- Track: green, 80% opacity, width 2 (ABGR in KML) -->
    <Style id="track">
      <IconStyle>
        <scale>1.2</scale>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/airports.png</href></Icon>
      </IconStyle>
      <LineStyle><color>cc00ff00</color><width>2</width></LineStyle>
    </Style>
    <!-- Rays: yellow-ish, 60% opacity, width 1 -->
    <Style id="rays"><LineStyle><color>9900ffff</color><width>1</width></LineStyle></Style>
    <Style id="envelope"><LineStyle><color>ff00a5ff</color><width>3</width></LineStyle></Style>
    <!-- Safety: yellow, 60% opacity, width 8 -->
    <Style id="safety"><LineStyle><color>9900ffff</color><width>8</width></LineStyle></Style>
    <!-- Crowd: #aa00ff, 60% opacity, width 8 -->
    <Style id="crowd"><LineStyle><color>99ff00aa</color><width>8</width></LineStyle></Style>
    <!-- Extruded Safety: yellow (60%), width 8, with fill -->
    <Style id="safety_ex">
      <LineStyle><color>9900ffff</color><width>8</width></LineStyle>
      <PolyStyle><color>9900ffff</color><fill>1</fill><outline>1</outline></PolyStyle>
    </Style>
    <!-- Extruded Crowd: #aa00ff (60%), width 8, with fill -->
    <Style id="crowd_ex">
      <LineStyle><color>99ff00aa</color><width>8</width></LineStyle>
      <PolyStyle><color>99ff00aa</color><fill>1</fill><outline>1</outline></PolyStyle>
    </Style>

    {schema}

    <Placemark><name>Track</name><styleUrl>#track</styleUrl>
      <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>
        <coordinates>{track_coords}</coordinates>
      </LineString>
    </Placemark>

    <Placemark><name>Rays</name><styleUrl>#rays</styleUrl>
      <MultiGeometry>{''.join(ray_strings)}</MultiGeometry>
    </Placemark>

    <Placemark><name>Envelope</name><styleUrl>#envelope</styleUrl>
      <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode>
        <coordinates>{envelope_coords}</coordinates>
      </LineString>
    </Placemark>

    {safety_folder}

    {gx_track}

    {gates_folder}

  </Document>
</kml>"""


# ---- Track JSON parser (returns project dict + safetyLine / crowdLine) ------

def parse_track_spec_json(path: str) -> Tuple[dict,
                                              Optional[str],
                                              Optional[List[Tuple[float, float, float]]],
                                              Optional[List[Tuple[float, float, float]]]]:
    """
    Returns (project_dict, project_name, safety_line_points, crowd_line_points)
    Each line is a list of (lon, lat, 0.0).
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Find the main project object (handle list or dict roots)
    project = _find_project_from_any_root(data)
    if project is None or not isinstance(project, dict):
        raise ValueError("Project object with raceBoxList/trackList not found in track JSON")

    name = project.get('name')
    safety: Optional[List[Tuple[float, float, float]]] = None
    crowd:  Optional[List[Tuple[float, float, float]]] = None

    def get_lon_lat(corner: dict) -> Tuple[Optional[float], Optional[float]]:
        lon = corner.get('longitude')
        if lon is None:
            lon = corner.get('longitudeDeg')
        lat = corner.get('latitude')
        if lat is None:
            lat = corner.get('latitudeDeg')
        return (float(lon) if lon is not None else None,
                float(lat) if lat is not None else None)

    for rb in project.get('raceBoxList', []):
        rb_id = str(rb.get('id', '')).lower()
        pts: List[Tuple[float, float, float]] = []
        for c in rb.get('cornerList', []):
            lon, lat = get_lon_lat(c)
            if lon is None or lat is None:
                continue
            pts.append((lon, lat, 0.0))
        if not pts:
            continue
        if rb_id == 'safetyline':
            safety = pts
        elif rb_id == 'crowdline':
            crowd = pts

    return project, name, safety, crowd


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
        self.current_path: str | None = None

        # Track spec (project + safety/crowd)
        self.track_spec_path: Optional[str] = None
        self.track_spec_name: Optional[str] = None
        self.track_spec: Optional[dict] = None
        self.safety_line: Optional[List[Tuple[float, float, float]]] = None
        self.crowd_line: Optional[List[Tuple[float, float, float]]] = None

        open_btn = QPushButton('Open…')
        open_btn.clicked.connect(self.open_file)

        load_track_btn = QPushButton('Load Track JSON…')
        load_track_btn.clicked.connect(self.load_track_json)

        self.rays_spin = QSpinBox()
        self.rays_spin.setRange(1, 1_000_000)
        self.rays_spin.setValue(1500)
        self.rays_spin.setPrefix('Rays limit: ')
        self.rays_spin.valueChanged.connect(self.update_map)

        self.export_btn = QPushButton('Export KML')
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_kml)

        self.loaded_label = QLabel("Loaded flight: —")
        self.loaded_label.setStyleSheet("color: #666;")

        self.spec_label = QLabel("Track spec: —")
        self.spec_label.setStyleSheet("color: #666;")

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(load_track_btn)
        top.addWidget(self.rays_spin)
        top.addWidget(self.export_btn)
        top.addStretch(1)
        top.addWidget(self.loaded_label)
        top.addSpacing(12)
        top.addWidget(self.spec_label)

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
            self.current_path = path
            self.loaded_label.setText(f"Loaded flight: {os.path.basename(path)}")
        except Exception as exc:
            self.statusBar().showMessage(f'Error: {exc}')
            return
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage(
            f'Loaded {len(self.track)} points, {len(self.rays)} rays, {len(self.signals)} signals'
        )
        self.update_map()
        self.update_timeseries()

    def load_track_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Track JSON',
            '',
            'Track JSON (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            project, name, safety, crowd = parse_track_spec_json(path)
            self.track_spec_path = path
            self.track_spec_name = name or os.path.splitext(os.path.basename(path))[0]
            self.track_spec = project  # <-- store the project dict (has trackList/gateList)
            self.safety_line = safety
            self.crowd_line = crowd
            self.spec_label.setText(f"Track spec: {os.path.basename(path)}")
            self.statusBar().showMessage(
                f'Loaded track JSON "{os.path.basename(path)}"  '
                f'({"safety" if safety else "no safety"}/{ "crowd" if crowd else "no crowd"})'
            )
        except Exception as exc:
            self.statusBar().showMessage(f'Error loading track JSON: {exc}')
            self.track_spec_path = None
            self.track_spec_name = None
            self.track_spec = None
            self.safety_line = None
            self.crowd_line = None
        self.update_map()

    def update_map(self) -> None:
        # Apply smoothing to track for display
        smoothed_track = _moving_average_path(self.track, SMOOTH_WINDOW)
        # Update rays to use smoothed track as start points
        smoothed_rays = list(zip(smoothed_track, self.endpoints)) if len(smoothed_track) == len(self.endpoints) else self.rays
        render_map(
            self.map_ax,
            smoothed_track,
            smoothed_rays,
            self.endpoints,
            self.rays_spin.value(),
            safety_line=self.safety_line,
            crowd_line=self.crowd_line,
        )

    def update_timeseries(self) -> None:
        # Show filter only when it's useful
        self.filter_edit.setVisible(len(self.signals) > 50)
        filter_text = self.filter_edit.text() if self.filter_edit.isVisible() else ''

        # Tear down any previous crosshair
        if self.crosshair:
            self.crosshair.disable()
            self.crosshair = None

        # Smooth all signals for plotting
        smoothed_signals = {k: _moving_average_1d(v, SMOOTH_WINDOW) for k, v in self.signals.items()}

        # Re-render plots
        axis_keys, axes, axis_data = render_timeseries(
            self.ts_canvas, self.t, smoothed_signals, self.units_map, filter_text
        )

        # If nothing to show, clear figure + status
        if not isinstance(axes, list) or len(axes) == 0:
            self.statusBar().clearMessage()
            self.ts_canvas.draw_idle()
            return

        # Recreate crosshair for the new axes
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

        # Default name/path = input base name + .kml
        suggested = "export.kml"
        start_dir = ""
        doc_name = "Track Export"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            suggested = base + ".kml"
            start_dir = os.path.dirname(self.current_path)
            doc_name = base

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save KML', os.path.join(start_dir, suggested), 'KML files (*.kml)'
        )
        if not save_path:
            return

        # assets next to main.py, NOT next to the JSON and NOT next to the KML
        assets_fs_dir = os.path.join(os.path.dirname(__file__), "assets")
        # the folder where the user is saving the KML
        kml_dir = os.path.dirname(save_path)

        kml = make_kml(
            self.t,
            self.track,          # smoothing applied inside make_kml
            self.rays,
            self.endpoints,
            self.signals,
            self.rays_spin.value(),
            doc_name=doc_name,
            smooth_window=SMOOTH_WINDOW,
            densify_factor=DENSIFY_FACTOR,
            safety_line=self.safety_line,
            crowd_line=self.crowd_line,
            track_spec=self.track_spec,
            assets_dir=assets_fs_dir,   # verify existence here
            kml_dir=kml_dir,            # build href relative to here
        )
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(kml)
            self.statusBar().showMessage(f'Saved {save_path}')
            for msg in GATE_MODEL_WARNINGS:
                self.statusBar().showMessage(msg, 5000)
        except Exception as exc:
            self.statusBar().showMessage(f'Error: {exc}')


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec())
