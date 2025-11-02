#!/usr/bin/env python3
"""Generate TrackPlanningTool JSON projects and validate their structure."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from pathlib import Path
import random
import uuid

try:  # pragma: no cover - optional dependency for higher fidelity interpolation
    from shapely.geometry import LineString as _ShapelyLineString  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback implemented below
    _ShapelyLineString = None


# Target output filename for the Nordschleife project.
OUTPUT_FILENAME = "nurburgring_nordschleife_tpt.json"

# Simplified representation of the Nürburgring Nordschleife centreline.
# (longitude, latitude, elevation metres)
NORDSCHLEIFE_POINTS = [
    (6.94972, 50.33546, 470),
    (6.95380, 50.33695, 465),
    (6.95915, 50.33967, 460),
    (6.96373, 50.34212, 455),
    (6.97043, 50.34625, 450),
    (6.97496, 50.34942, 445),
    (6.98154, 50.35338, 440),
    (6.98764, 50.35767, 435),
    (6.99286, 50.36135, 430),
    (6.99893, 50.36541, 425),
    (7.00321, 50.36855, 420),
    (7.00872, 50.37281, 418),
    (7.01285, 50.37590, 416),
    (7.01831, 50.37981, 415),
    (7.02296, 50.38322, 413),
    (7.02884, 50.38760, 412),
    (7.03277, 50.39068, 414),
    (7.03525, 50.39288, 416),
    (7.03819, 50.39609, 418),
    (7.04095, 50.39945, 421),
    (7.04174, 50.40344, 425),
    (7.04068, 50.40809, 429),
    (7.03792, 50.41248, 434),
    (7.03340, 50.41707, 439),
    (7.02791, 50.42048, 443),
    (7.02231, 50.42315, 448),
    (7.01607, 50.42583, 452),
    (7.00875, 50.42817, 456),
    (7.00248, 50.43015, 460),
    (6.99618, 50.43192, 465),
    (6.98945, 50.43341, 470),
    (6.98218, 50.43481, 474),
    (6.97531, 50.43572, 478),
    (6.96814, 50.43636, 482),
    (6.96045, 50.43673, 486),
    (6.95283, 50.43691, 490),
    (6.94544, 50.43650, 494),
    (6.93856, 50.43583, 498),
    (6.93166, 50.43456, 502),
    (6.92483, 50.43270, 506),
    (6.91867, 50.43027, 510),
    (6.91282, 50.42732, 514),
    (6.90714, 50.42391, 518),
    (6.90193, 50.42008, 522),
    (6.89710, 50.41615, 526),
    (6.89280, 50.41161, 530),
    (6.88972, 50.40702, 534),
    (6.88790, 50.40254, 537),
    (6.88725, 50.39757, 540),
    (6.88783, 50.39241, 543),
    (6.88980, 50.38767, 546),
    (6.89276, 50.38346, 548),
    (6.89648, 50.37922, 550),
    (6.90160, 50.37503, 452),
    (6.90758, 50.37148, 454),
    (6.91350, 50.36827, 456),
    (6.92008, 50.36513, 458),
    (6.92712, 50.36222, 460),
    (6.93441, 50.35973, 462),
    (6.94150, 50.35758, 465),
    (6.94972, 50.35512, 470),
]


EARTH_RADIUS_M = 6_371_000.0


class _FallbackPoint:
    """Minimal point representation compatible with shapely's interface."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


class _FallbackLineString:
    """Fallback implementation for environments without shapely installed."""

    def __init__(self, coords: list[tuple[float, float]]) -> None:
        if len(coords) < 2:
            raise ValueError("LineString requires at least two coordinate pairs.")
        self._coords = [(float(x), float(y)) for x, y in coords]
        self._segment_lengths = [
            math.hypot(x1 - x0, y1 - y0)
            for (x0, y0), (x1, y1) in zip(self._coords[:-1], self._coords[1:])
        ]
        cumulative = [0.0]
        for length in self._segment_lengths:
            cumulative.append(cumulative[-1] + length)
        self._cumulative = cumulative
        self.length = cumulative[-1]

    def interpolate(self, distance: float) -> _FallbackPoint:
        if distance <= 0.0:
            x, y = self._coords[0]
            return _FallbackPoint(x, y)
        if distance >= self.length:
            x, y = self._coords[-1]
            return _FallbackPoint(x, y)

        index = bisect_right(self._cumulative, distance) - 1
        segment_length = self._segment_lengths[index]
        if segment_length == 0.0:
            x, y = self._coords[index + 1]
            return _FallbackPoint(x, y)

        ratio = (distance - self._cumulative[index]) / segment_length
        x0, y0 = self._coords[index]
        x1, y1 = self._coords[index + 1]
        x = x0 + ratio * (x1 - x0)
        y = y0 + ratio * (y1 - y0)
        return _FallbackPoint(x, y)


def _make_line_string(coords: list[tuple[float, float]]):
    """Instantiate a LineString using shapely when available, otherwise a fallback."""
    if _ShapelyLineString is not None:
        return _ShapelyLineString(coords)
    return _FallbackLineString(coords)


def _projection_functions():
    """Return helpers to convert between geographic and local metric coordinates."""
    origin_lon, origin_lat, _ = NORDSCHLEIFE_POINTS[0]
    origin_lon_rad = math.radians(origin_lon)
    origin_lat_rad = math.radians(origin_lat)
    cos_lat = math.cos(origin_lat_rad)
    if abs(cos_lat) < 1e-9:
        raise ValueError("Cannot build local projection near the geographic poles.")

    def to_local(lon: float, lat: float) -> tuple[float, float]:
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        x = EARTH_RADIUS_M * (lon_rad - origin_lon_rad) * cos_lat
        y = EARTH_RADIUS_M * (lat_rad - origin_lat_rad)
        return x, y

    def to_geo(x: float, y: float) -> tuple[float, float]:
        lon_rad = origin_lon_rad + x / (EARTH_RADIUS_M * cos_lat)
        lat_rad = origin_lat_rad + y / EARTH_RADIUS_M
        return math.degrees(lon_rad), math.degrees(lat_rad)

    return to_local, to_geo


def _compute_cumulative_distances(points_xy: list[tuple[float, float]]) -> list[float]:
    """Return cumulative distances (metres) along the projected polyline."""
    cumulative = [0.0]
    for (x0, y0), (x1, y1) in zip(points_xy[:-1], points_xy[1:]):
        segment_length = math.hypot(x1 - x0, y1 - y0)
        cumulative.append(cumulative[-1] + segment_length)
    return cumulative


def _interpolate_elevation(
    distance: float,
    cumulative: list[float],
    elevations: list[float],
) -> float:
    """Linearly interpolate the elevation at the requested distance."""
    if distance <= 0.0:
        return elevations[0]
    if distance >= cumulative[-1]:
        return elevations[-1]

    index = bisect_right(cumulative, distance) - 1
    next_index = min(index + 1, len(cumulative) - 1)
    segment_length = cumulative[next_index] - cumulative[index]
    if segment_length == 0:
        return elevations[index]

    ratio = (distance - cumulative[index]) / segment_length
    start_elev = elevations[index]
    end_elev = elevations[next_index]
    return start_elev + ratio * (end_elev - start_elev)


def generate_nordschleife_gates(num_gates: int | None = None) -> list[dict]:
    """Return an evenly spaced list of gate definitions for the Nordschleife."""
    lons, lats, elevs = zip(*NORDSCHLEIFE_POINTS)
    to_local, to_geo = _projection_functions()
    projected_points = [to_local(lon, lat) for lon, lat in zip(lons, lats)]
    line_projected = _make_line_string(projected_points)

    total_length_m = line_projected.length
    if num_gates is None:
        estimated = int(round(total_length_m / 80.0))
        num_gates = max(250, min(300, estimated))
    else:
        if num_gates < 2:
            raise ValueError("At least two gates are required to build a valid loop.")

    distances = [total_length_m * index / num_gates for index in range(num_gates)]

    cumulative_distances = _compute_cumulative_distances(projected_points)
    rng = random.Random(42)

    gates = []
    for index, distance in enumerate(distances):
        point_xy = line_projected.interpolate(distance)
        lon, lat = to_geo(point_xy.x, point_xy.y)

        base_elevation = _interpolate_elevation(distance, cumulative_distances, list(elevs))
        elevation = base_elevation + rng.uniform(-5.0, 5.0)

        # Determine heading using differences in longitude/latitude to the next gate.
        next_distance = distances[(index + 1) % len(distances)]
        next_point_xy = line_projected.interpolate(next_distance)
        next_lon, next_lat = to_geo(next_point_xy.x, next_point_xy.y)
        d_lon = next_lon - lon
        d_lat = next_lat - lat
        heading = (math.degrees(math.atan2(d_lon, d_lat)) + 360.0) % 360.0

        gate_type = "TwinGate"
        if index == 0:
            gate_type = "StartGate"
        elif index == len(distances) - 1:
            gate_type = "FinishGate"

        gate = {
            "headingDeg": round(heading, 6),
            "type": gate_type,
            "id": str(index + 1),
            "position": {
                "elevationM": round(elevation, 3),
                "latitudeDeg": round(lat, 9),
                "longitudeDeg": round(lon, 9),
            },
        }
        gates.append(gate)

    return gates


def build_sequence(gates: list[dict]) -> list[dict]:
    """Create the Firenze-compatible sequence entries for the gate list."""
    sequence: list[dict] = []

    for index, gate in enumerate(gates):
        entry: dict[str, object] = {
            "gateId": gate["id"],
            "id": uuid.uuid4().hex,
            "aboveGroundLevelConstraintType": "EnableConstraint",
            "aboveGroundLevelRange": [1, 2],
        }
        if index == 0:
            entry["type"] = "StartGate"
        elif index == len(gates) - 1:
            entry["type"] = "FinishGate"
        sequence.append(entry)

    for reverse_gate in reversed(gates):
        entry = {
            "gateId": reverse_gate["id"],
            "id": uuid.uuid4().hex,
            "opposite": True,
            "aboveGroundLevelConstraintType": "EnableConstraint",
            "aboveGroundLevelRange": [1, 2],
        }
        if reverse_gate is gates[0]:
            entry["type"] = "FinishGate"
        sequence.append(entry)

    return sequence


def build_project(gates: list[dict], sequence: list[dict]) -> list[dict]:
    """Construct the Nordschleife project payload."""
    project_header = {
        "content": "TrackPlaningTool::Project",
        "version": [2, 0],
    }

    project_body = {
        "name": "Nurburgring_Nordschleife",
        "navigationCoordinates": {
            "position": {
                "latitudeDeg": 50.33546,
                "longitudeDeg": 6.94972,
                "elevationM": 470,
            },
            "headingDeg": 0,
        },
        "trackList": [
            {
                "id": "v1.0",
                "sequence": sequence,
                "gateList": gates,
                "raceBoxId": "safetyLine",
            }
        ],
        "raceBoxList": [
            {
                "id": "safetyLine",
                "cornerList": [
                    {"longitude": 6.94, "latitude": 50.333},
                    {"longitude": 6.96, "latitude": 50.34},
                    {"longitude": 6.98, "latitude": 50.35},
                    {"longitude": 6.95, "latitude": 50.355},
                    {"longitude": 6.94, "latitude": 50.333},
                ],
            }
        ],
    }

    return [project_header, project_body]


def write_json(project: list[dict], output_path: Path) -> None:
    """Serialize the project to disk using the required formatting."""
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(project, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def validate_project(path: Path) -> bool:
    """Validate that a file conforms to the expected TrackPlanningTool schema."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return False
    except json.JSONDecodeError as exc:
        print(f"❌ Invalid JSON: {exc}")
        return False

    errors: list[str] = []
    if not isinstance(data, list) or len(data) != 2:
        errors.append("Root must be a list containing exactly two objects (header + body).")
        body = None
    else:
        header, body = data
        if not isinstance(header, dict) or header.get("content") != "TrackPlaningTool::Project":
            errors.append("First object must declare the TrackPlanningTool project header.")
        if not isinstance(body, dict):
            errors.append("Second object must contain the project body dictionary.")

    if isinstance(body, dict):
        track_list = body.get("trackList")
        if not isinstance(track_list, list) or not track_list:
            errors.append("trackList must be a non-empty list.")
        else:
            track = track_list[0]
            sequence = track.get("sequence") if isinstance(track, dict) else None
            gate_list = track.get("gateList") if isinstance(track, dict) else None

            if not isinstance(sequence, list) or not sequence:
                errors.append("trackList[0].sequence must be a non-empty list.")
            if not isinstance(gate_list, list) or not gate_list:
                errors.append("trackList[0].gateList must be a non-empty list.")
            else:
                gate_ids = {gate.get("id") for gate in gate_list if isinstance(gate, dict)}
                for idx, gate in enumerate(gate_list, start=1):
                    if not isinstance(gate, dict):
                        errors.append(f"gateList entry {idx} must be an object.")
                        continue
                    position = gate.get("position")
                    if not isinstance(position, dict):
                        errors.append(f"gateList entry {idx} missing position dictionary.")
                        continue
                    if "latitudeDeg" not in position or "longitudeDeg" not in position:
                        errors.append(
                            f"gateList entry {idx} position must contain latitudeDeg and longitudeDeg."
                        )

                for idx, entry in enumerate(sequence, start=1):
                    if not isinstance(entry, dict):
                        errors.append(f"sequence entry {idx} must be an object.")
                        continue
                    gate_id = entry.get("gateId")
                    if gate_id not in gate_ids:
                        errors.append(f"sequence entry {idx} references unknown gateId '{gate_id}'.")

    if errors:
        print("❌ Validation errors detected:")
        for issue in errors:
            print(f"  - {issue}")
        return False

    print("✅ valid TPT schema")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate TPT tracks.")
    parser.add_argument(
        "--nordschleife",
        action="store_true",
        help="Generate the Nürburgring Nordschleife Firenze-compatible project.",
    )
    parser.add_argument(
        "--gates",
        type=int,
        default=None,
        help="Override the number of evenly spaced gates to generate (250-300 recommended).",
    )
    parser.add_argument(
        "--validate",
        metavar="FILE",
        type=str,
        help="Validate an existing TrackPlanningTool JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help=f"Destination filename for generated projects (default: {OUTPUT_FILENAME}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    performed_action = False

    if args.nordschleife:
        gates = generate_nordschleife_gates(args.gates)
        sequence = build_sequence(gates)
        project = build_project(gates, sequence)
        output_path = Path(args.output)
        write_json(project, output_path)
        print(f"Generated Nordschleife project with {len(gates)} gates → {output_path.resolve()}")
        performed_action = True

    if args.validate:
        validate_project(Path(args.validate))
        performed_action = True

    if not performed_action:
        raise SystemExit(
            "No action requested. Use --nordschleife to generate or --validate <file> to validate."
        )


if __name__ == "__main__":
    main()
