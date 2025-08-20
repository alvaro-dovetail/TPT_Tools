import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QFileDialog,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def parse_tpt(path):
    """Parse a TPT/CSV file and extract track points and rays."""
    track_pts = []
    rays = []
    endpoints = []
    groups = []
    group_fields = {}

    def parse_header(line):
        tokens = line.strip().split(',')
        i = 4  # skip <TIME>,HEADER,,,
        while i < len(tokens):
            name = tokens[i]
            if not name:
                i += 1
                continue
            i += 1
            fields = []
            while i < len(tokens) and not tokens[i].endswith('_pkt'):
                fields.append(tokens[i])
                i += 1
            groups.append(name)
            group_fields[name] = fields

    def parse_info(tokens):
        data = {}
        i = 4
        for g in groups:
            if i < len(tokens) and tokens[i] == g:
                i += 1
            fields = group_fields[g]
            values = tokens[i:i + len(fields)]
            i += len(fields)
            data[g] = values
        return data

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('<TIME>,HEADER') and 'ballistic_extrapolation_position_pkt' in line:
                parse_header(line)
                pos_fields = group_fields.get('position_pkt', [])
                pos_lon = pos_fields.index('longitude')
                pos_lat = pos_fields.index('latitude')
                pos_h = pos_fields.index('height')
                be_fields = group_fields.get('ballistic_extrapolation_position_pkt', [])
                be_lon = be_fields.index('longitude')
                be_lat = be_fields.index('latitude')
                be_h = be_fields.index('height')
            elif line.startswith('<TIME>,INFO') and group_fields:
                tokens = line.strip().split(',')
                data = parse_info(tokens)
                try:
                    lon = float(data['position_pkt'][pos_lon])
                    lat = float(data['position_pkt'][pos_lat])
                    h = float(data['position_pkt'][pos_h])
                    lonb = float(data['ballistic_extrapolation_position_pkt'][be_lon])
                    latb = float(data['ballistic_extrapolation_position_pkt'][be_lat])
                    hb = float(data['ballistic_extrapolation_position_pkt'][be_h])
                except (KeyError, ValueError, IndexError):
                    continue
                track_pts.append((lon, lat, h))
                start = (lon, lat, h)
                end = (lonb, latb, hb)
                rays.append((start, end))
                endpoints.append(end)
    return track_pts, rays, endpoints


def make_kml(track, rays, endpoints, rays_limit):
    """Create a KML string for the provided data."""
    def fmt(pt):
        return f"{pt[0]:.9f},{pt[1]:.9f},{pt[2]:.3f}"

    track_coords = ' '.join(fmt(p) for p in track)
    step = max(1, len(rays) // max(1, rays_limit)) if rays else 1
    ray_strings = []
    for i in range(0, len(rays), step):
        a, b = rays[i]
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
  <Style id="rays"><LineStyle><color>7f00ffff</color><width>1</width></LineStyle></Style>
  <Style id="envelope"><LineStyle><color>ff00a5ff</color><width>3</width></LineStyle></Style>
  <Placemark><name>Track</name><styleUrl>#track</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>{track_coords}</coordinates></LineString>
  </Placemark>
  <Placemark><name>Rays</name><styleUrl>#rays</styleUrl>
    <MultiGeometry>
      {''.join(ray_strings)}
    </MultiGeometry>
  </Placemark>
  <Placemark><name>Envelope</name><styleUrl>#envelope</styleUrl>
    <LineString><tessellate>1</tessellate><altitudeMode>absolute</altitudeMode><coordinates>{envelope_coords}</coordinates></LineString>
  </Placemark>
</Document>
</kml>"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TPT Tools")
        self.track = []
        self.rays = []
        self.endpoints = []

        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self.open_file)
        self.rays_spin = QSpinBox()
        self.rays_spin.setRange(1, 1000000)
        self.rays_spin.setValue(1500)
        self.rays_spin.setPrefix("Rays limit: ")
        self.rays_spin.valueChanged.connect(self.update_plot)
        export_btn = QPushButton("Export KML")
        export_btn.clicked.connect(self.export_kml)

        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.rays_spin)
        top.addWidget(export_btn)
        top_widget = QWidget()
        top_widget.setLayout(top)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")

        layout = QVBoxLayout()
        layout.addWidget(top_widget)
        layout.addWidget(self.canvas)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "", "TPT/CSV files (*.tpt *.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            track, rays, endpoints = parse_tpt(path)
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")
            return
        self.track, self.rays, self.endpoints = track, rays, endpoints
        self.statusBar().showMessage(f"Loaded {len(track)} points, {len(rays)} rays")
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if not self.track:
            self.canvas.draw()
            return
        lons = [p[0] for p in self.track]
        lats = [p[1] for p in self.track]
        self.ax.plot(lons, lats, color='yellow', linewidth=3)
        limit = self.rays_spin.value()
        if self.rays:
            step = max(1, len(self.rays) // limit)
            for i in range(0, len(self.rays), step):
                (lon1, lat1, _), (lon2, lat2, _) = self.rays[i]
                self.ax.plot([lon1, lon2], [lat1, lat2], color=(1, 1, 0, 0.3), linewidth=1)
        if self.endpoints:
            lons_e = [p[0] for p in self.endpoints]
            lats_e = [p[1] for p in self.endpoints]
            self.ax.plot(lons_e, lats_e, color='orange', linewidth=3)
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.canvas.draw()

    def export_kml(self):
        if not self.track:
            self.statusBar().showMessage("No data to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save KML", "", "KML files (*.kml)")
        if not path:
            return
        kml = make_kml(self.track, self.rays, self.endpoints, self.rays_spin.value())
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(kml)
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:
            self.statusBar().showMessage(f"Error: {exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(800, 600)
    w.show()
    sys.exit(app.exec())
