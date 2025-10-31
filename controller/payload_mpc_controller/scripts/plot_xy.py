#!/usr/bin/env python3
import sys
import os
import csv
import math

WIDTH = 860
HEIGHT = 720
MARGIN = 60
LINE_WIDTH = 2.5
COLORS = {
    "quad_ref": "#FFD43B",
    "quad_actual": "#1F77B4",
    "payload_ref": "#FF6F61",
    "payload_actual": "#2CA02C",
}
STYLES = {
    "quad_ref": "8,5",
    "quad_actual": None,
    "payload_ref": "8,5",
    "payload_actual": None,
}

if len(sys.argv) < 5:
    print("Usage: plot_xy.py <csv> <quad_rmse> <payload_rmse> <output_svg>")
    sys.exit(1)

csv_path = sys.argv[1]
quad_rmse = float(sys.argv[2])
payload_rmse = float(sys.argv[3])
output_path = sys.argv[4]

series_names = [
    ("quad_ref_x", "quad_ref_y", "quad_ref"),
    ("quad_actual_x", "quad_actual_y", "quad_actual"),
    ("payload_ref_x", "payload_ref_y", "payload_ref"),
    ("payload_actual_x", "payload_actual_y", "payload_actual"),
]

rows = []
if not os.path.isfile(csv_path):
    print(f"CSV file not found: {csv_path}")
    sys.exit(1)

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({k: float(row[k]) if row[k] else float('nan') for k in row})

series = {}
min_x = float('inf')
max_x = float('-inf')
min_y = float('inf')
max_y = float('-inf')

for x_key, y_key, name in series_names:
    pts = []
    for row in rows:
        x = row.get(x_key, float('nan'))
        y = row.get(y_key, float('nan'))
        if math.isfinite(x) and math.isfinite(y):
            pts.append((x, y))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    series[name] = pts

valid_series = any(len(v) > 1 for v in series.values())
if not valid_series:
    print("No valid data to plot")
    sys.exit(0)

if max_x - min_x < 1e-6:
    max_x += 0.5
    min_x -= 0.5
if max_y - min_y < 1e-6:
    max_y += 0.5
    min_y -= 0.5

scale_x = (WIDTH - 2 * MARGIN) / (max_x - min_x)
scale_y = (HEIGHT - 2 * MARGIN) / (max_y - min_y)

def to_svg(x, y):
    sx = MARGIN + (x - min_x) * scale_x
    sy = HEIGHT - (MARGIN + (y - min_y) * scale_y)
    return sx, sy

def build_polyline(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)

svg = []
svg.append('<?xml version="1.0" encoding="UTF-8"?>')
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#FFFFFF"/>')
svg.append(f'<rect x="{MARGIN}" y="{MARGIN}" width="{WIDTH - 2*MARGIN}" height="{HEIGHT - 2*MARGIN}" fill="none" stroke="#E0E0E0" stroke-width="1"/>')

for name, pts in series.items():
    if len(pts) < 2:
        continue
    transformed = [to_svg(x, y) for x, y in pts]
    dash = STYLES.get(name)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    color = COLORS.get(name, '#000000')
    polyline = build_polyline(transformed)
    svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="{LINE_WIDTH}"{dash_attr} points="{polyline}"/>')

legend_width = 260
legend_height = 110
legend_x = WIDTH - MARGIN - legend_width
legend_y = HEIGHT - MARGIN - legend_height
svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" fill="#FFFFFF" stroke="#B0B0B0" stroke-width="1" opacity="0.9"/>')

legend_items = [
    ("Quad Ref", "quad_ref"),
    ("Quad Actual", "quad_actual"),
    ("Payload Ref", "payload_ref"),
    ("Payload Actual", "payload_actual"),
]
legend_line_y = legend_y + 20
legend_line_x1 = legend_x + 15
legend_line_x2 = legend_line_x1 + 50

for label, name in legend_items:
    color = COLORS.get(name, '#000000')
    dash = STYLES.get(name)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    svg.append(f'<line x1="{legend_line_x1}" y1="{legend_line_y}" x2="{legend_line_x2}" y2="{legend_line_y}" stroke="{color}" stroke-width="{LINE_WIDTH}"{dash_attr}/>' )
    svg.append(f'<text x="{legend_line_x2 + 10}" y="{legend_line_y + 5}" font-size="16" fill="#333333">{label}</text>')
    legend_line_y += 22

rmse_text = f"Quad RMSE: {quad_rmse:.3f} m\nPayload RMSE: {payload_rmse:.3f} m"
svg.append(f'<text x="{legend_x + 15}" y="{legend_y + legend_height - 35}" font-size="16" fill="#333333">Quad RMSE: {quad_rmse:.3f} m</text>')
svg.append(f'<text x="{legend_x + 15}" y="{legend_y + legend_height - 15}" font-size="16" fill="#333333">Payload RMSE: {payload_rmse:.3f} m</text>')

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for line in svg:
        f.write(line + '\n')
print(f"Saved XY plot to {output_path}")
