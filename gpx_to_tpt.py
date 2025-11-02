#!/usr/bin/env python3
"""Convert GPX tracks to Firenze TrackPlanningTool JSON projects."""

import argparse
import json
import math
import os
import sys
import uuid
import xml.etree.ElementTree as ET

try:
    from shapely.geometry import LineString, Point
except ImportError:  # pragma: no cover - handled at runtime
    print("Please install shapely: pip install shapely")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a GPX track into a Firenze TrackPlanningTool JSON file.",
    )
    parser.add_argument("input", help="Input GPX file path")
    parser.add_argument(
        "--gates",
        type=int,
        default=200,
        help="Number of gates to generate (default: 200)",
    )
    return parser.parse_args()


def parse_gpx(file_path: str):
    """Extract latitude, longitude and elevation tuples from a GPX file."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    points = []

    for trkpt in root.findall(".//{*}trkpt"):
        lat = trkpt.get("lat")
        lon = trkpt.get("lon")
        ele_elem = trkpt.find("{*}ele")
        if lat is None or lon is None or ele_elem is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            ele_f = float(ele_elem.text.strip()) if ele_elem.text else 0.0
        except (TypeError, ValueError):
            continue
        points.append((lat_f, lon_f, ele_f))

    if not points:
        raise ValueError("No valid track points found in GPX file.")

    return points


def interpolate_points(points, target_count):
    """Interpolate or return the original list of points based on target count."""
    if len(points) <= target_count or target_count <= 1:
        return points[:target_count]

    line = LineString([(lon, lat) for lat, lon, _ in points])
    total_length = line.length
    if total_length == 0:
        return [points[0]] * target_count

    projected = []
    for lat, lon, ele in points:
        distance = line.project(Point(lon, lat))
        projected.append((distance, ele))

    distances = [d for d, _ in projected]

    resampled = []
    step = total_length / (target_count - 1)

    for i in range(target_count):
        distance_along = min(step * i, total_length)
        point = line.interpolate(distance_along)
        elevation = interpolate_elevation(distance_along, projected, distances)
        resampled.append((point.y, point.x, elevation))

    return resampled


def interpolate_elevation(distance, projected, distances):
    """Linearly interpolate elevation for a given distance along the path."""
    if not projected:
        return 0.0

    if distance <= distances[0]:
        return projected[0][1]
    if distance >= distances[-1]:
        return projected[-1][1]

    # Binary search for surrounding points
    low, high = 0, len(distances) - 1
    while low <= high:
        mid = (low + high) // 2
        if distances[mid] < distance:
            low = mid + 1
        else:
            high = mid - 1

    idx = max(low, 1)
    d0, e0 = projected[idx - 1]
    d1, e1 = projected[idx]
    if d1 == d0:
        return e1
    ratio = (distance - d0) / (d1 - d0)
    return e0 + ratio * (e1 - e0)


def bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees between two coordinates."""
    d_lon = math.radians(lon2 - lon1)
    lat1_rad, lat2_rad = map(math.radians, [lat1, lat2])
    y = math.sin(d_lon) * math.cos(lat2_rad)
    x = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def compute_headings(points):
    """Compute heading for each point."""
    headings = []
    for idx, (lat, lon, _) in enumerate(points):
        if idx < len(points) - 1:
            next_lat, next_lon, _ = points[idx + 1]
            heading = bearing(lat, lon, next_lat, next_lon)
        else:
            heading = headings[-1] if headings else 0.0
        headings.append(heading)
    return headings


def build_gates(points, headings):
    """Create gate dictionaries for the TrackPlanningTool format."""
    gates = []
    total = len(points)
    for idx, ((lat, lon, ele), heading) in enumerate(zip(points, headings), start=1):
        gate_type = "TwinGate"
        if idx == 1:
            gate_type = "StartGate"
        elif idx == total:
            gate_type = "FinishGate"
        gate = {
            "headingDeg": heading,
            "type": gate_type,
            "id": str(idx),
            "position": {
                "elevationM": ele,
                "latitudeDeg": lat,
                "longitudeDeg": lon,
            },
        }
        gates.append(gate)
    return gates


def build_sequence(gates):
    """Build the sequence list including forward and reverse passes."""
    sequence = []
    for gate in gates:
        sequence.append(
            {
                "gateId": gate["id"],
                "id": str(uuid.uuid4()),
                "aboveGroundLevelConstraintType": "EnableConstraint",
                "aboveGroundLevelRange": [1, 2],
            }
        )

    reversed_entries = []
    for idx, gate in enumerate(reversed(gates)):
        entry_type = "TwinGate"
        if idx == 0:
            entry_type = "FinishGate"
        elif idx == len(gates) - 1:
            entry_type = "StartGate"
        reversed_entries.append(
            {
                "gateId": gate["id"],
                "id": str(uuid.uuid4()),
                "aboveGroundLevelConstraintType": "EnableConstraint",
                "aboveGroundLevelRange": [1, 2],
                "opposite": True,
                "type": entry_type,
            }
        )

    sequence.extend(reversed_entries)
    return sequence


def build_project(basename, points, gates, sequence):
    """Construct the TrackPlanningTool project JSON structure."""
    first_lat, first_lon, first_ele = points[0]
    last_lat, last_lon, _ = points[-1]

    project = [
        {"content": "TrackPlaningTool::Project", "version": [2, 0]},
        {
            "name": basename,
            "navigationCoordinates": {
                "position": {
                    "latitudeDeg": first_lat,
                    "longitudeDeg": first_lon,
                    "elevationM": first_ele,
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
                        {"longitude": first_lon, "latitude": first_lat},
                        {"longitude": last_lon, "latitude": last_lat},
                    ],
                }
            ],
        },
    ]
    return project


def validate_output(data):
    """Validate the generated JSON structure and report the result."""
    valid = True
    errors = []

    if not isinstance(data, list) or len(data) != 2:
        valid = False
        errors.append("Root must be a list with two objects.")
    else:
        project = data[1]
        track_list = project.get("trackList") if isinstance(project, dict) else None
        if not track_list:
            valid = False
            errors.append("Missing trackList entries.")
        else:
            track = track_list[0]
            if "sequence" not in track:
                valid = False
                errors.append("Missing sequence in track list entry.")
            if "gateList" not in track:
                valid = False
                errors.append("Missing gateList in track list entry.")
            else:
                for gate in track["gateList"]:
                    position = gate.get("position", {})
                    if "latitudeDeg" not in position or "longitudeDeg" not in position:
                        valid = False
                        errors.append("Gate missing latitudeDeg/longitudeDeg.")
                        break

    if valid:
        print("✅ JSON structure valid")
    else:
        print("❌ JSON structure invalid:")
        for message in errors:
            print(f"  - {message}")


def main():
    args = parse_arguments()
    input_path = args.input
    gate_target = args.gates

    if gate_target <= 0:
        print("--gates must be a positive integer")
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        points = parse_gpx(input_path)
    except (ET.ParseError, ValueError) as exc:
        print(f"Failed to parse GPX: {exc}")
        sys.exit(1)

    processed_points = (
        interpolate_points(points, gate_target)
        if len(points) > gate_target
        else points
    )

    headings = compute_headings(processed_points)
    gates = build_gates(processed_points, headings)
    sequence = build_sequence(gates)

    basename = os.path.splitext(os.path.basename(input_path))[0]
    project = build_project(basename, processed_points, gates, sequence)

    output_dir = os.path.dirname(os.path.abspath(input_path))
    output_path = os.path.join(output_dir, f"{basename}_tpt.json")

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(project, file_obj, indent=2, ensure_ascii=False)

    # Re-open file for validation and summary
    with open(output_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    validate_output(data)

    start_lat, start_lon, start_ele = processed_points[0]
    end_lat, end_lon, end_ele = processed_points[-1]
    print(f"✅ Generated {len(processed_points)} gates")
    print(
        "From: (" +
        f"{start_lat:.6f}, {start_lon:.6f}, {start_ele:.3f}" +
        ")"
    )
    print(
        "To:   (" +
        f"{end_lat:.6f}, {end_lon:.6f}, {end_ele:.3f}" +
        ")"
    )
    print(f"Output: {os.path.basename(output_path)}")


if __name__ == "__main__":
    main()

