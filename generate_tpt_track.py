#!/usr/bin/env python3
"""Generate a TrackPlanningTool JSON project for the Nürburgring Nordschleife."""

import json
import math
from pathlib import Path

import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString


def download_track_line(place_name: str) -> LineString:
    """Download geometries from OpenStreetMap and return the longest raceway line."""
    # Request all raceway geometries within the place boundary.
    tags = {"highway": "raceway"}
    try:
        geometries = ox.geometries_from_place(place_name, tags=tags)
    except AttributeError:
        # osmnx 2.x renamed this helper to ``features_from_place``.
        geometries = ox.features_from_place(place_name, tags=tags)

    # Extract all LineString segments, flattening MultiLineStrings when present.
    track_lines = []
    for geom in geometries.geometry:
        if geom is None:
            continue
        if isinstance(geom, LineString):
            track_lines.append(geom)
        elif isinstance(geom, MultiLineString):
            track_lines.extend(list(geom.geoms))
        else:
            geom_type = getattr(geom, "geom_type", "")
            if geom_type == "Polygon":
                track_lines.append(LineString(geom.exterior.coords))
            elif geom_type == "MultiPolygon":
                track_lines.extend(
                    LineString(poly.exterior.coords) for poly in geom.geoms
                )

    if not track_lines:
        raise ValueError("No raceway LineString geometries found for the provided place.")

    # Pick the LineString with the greatest length (assumes track is the longest feature).
    longest_line = max(track_lines, key=lambda line: line.length)
    return longest_line


def interpolate_gates(line: LineString, num_gates: int = 200) -> list:
    """Interpolate evenly spaced gates along the provided LineString."""
    # Compute equally spaced distances along the line, omitting the terminal point to
    # avoid duplicating the starting gate on a closed loop.
    distances = np.linspace(0, line.length, num=num_gates, endpoint=False)

    # Use a deterministic random generator for repeatable elevation variations.
    rng = np.random.default_rng(seed=42)

    points = [line.interpolate(distance) for distance in distances]

    gates = []
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]

        # Compute heading in degrees using atan2, converted to a 0-360 degree range.
        dy = next_point.y - point.y
        dx = next_point.x - point.x
        heading_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        # Slightly vary the elevation around a nominal 500 m reference.
        elevation = 500.0 + rng.uniform(-5.0, 5.0)

        gate_type = "TwinGate"
        if idx == 0:
            gate_type = "StartGate"
        elif idx == num_gates - 1:
            gate_type = "FinishGate"

        gate = {
            "headingDeg": round(heading_deg, 6),
            "type": gate_type,
            "id": str(idx + 1),
            "position": {
                "elevationM": round(float(elevation), 3),
                "latitudeDeg": round(point.y, 9),
                "longitudeDeg": round(point.x, 9),
            },
        }
        gates.append(gate)

    return gates


def build_sequence(gates: list) -> list:
    """Create the sequence array referencing each gate in order."""
    sequence = []
    for gate in gates:
        entry = {"gateId": gate["id"]}
        if gate["type"] in {"StartGate", "FinishGate"}:
            entry["type"] = gate["type"]
        sequence.append(entry)

    # Append an extra finish gate entry marking the finish line from the opposite direction.
    if gates:
        sequence.append(
            {
                "gateId": gates[-1]["id"],
                "type": "FinishGate",
                "opposite": True,
            }
        )
    return sequence


def build_project_structure(gates: list, sequence: list) -> dict:
    """Assemble the full TrackPlanningTool project structure."""
    project = {
        "content": "TrackPlanningTool::Project",
        "version": [2, 0],
        "name": "Nurburgring_Nordschleife",
        "navigationCoordinates": {
            "position": {
                "latitudeDeg": 50.334,
                "longitudeDeg": 6.942,
                "elevationM": 500,
            },
            "headingDeg": 0,
        },
        "trackList": [
            {
                "id": "nord_loop",
                "sequence": sequence,
                "gateList": gates,
                "raceBoxId": "safetyLine",
            }
        ],
        "raceBoxList": [],
    }
    return project


def main() -> None:
    place_name = "Nürburgring, Nürburg, Germany"

    # Download the track geometry and prepare the gate definitions.
    track_line = download_track_line(place_name)
    gates = interpolate_gates(track_line, num_gates=200)
    sequence = build_sequence(gates)

    # Assemble the project structure and export it as JSON.
    project = build_project_structure(gates, sequence)
    output_path = Path("nurburgring_tpt.json")
    output_path.write_text(json.dumps(project, indent=4))

    print(f"Generated TrackPlanningTool project at {output_path.resolve()}")


if __name__ == "__main__":
    main()
