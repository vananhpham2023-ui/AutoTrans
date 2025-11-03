#!/usr/bin/env python3
import sys
import os
import csv
import math

WIDTH = 1000
HEIGHT = 720
MARGIN = 60
LEGEND_SPACE = 220
LINE_WIDTH = 2.5
FRAME_PADDING = 20
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

plot_width = WIDTH - 2 * MARGIN - LEGEND_SPACE
plot_height = HEIGHT - 2 * MARGIN
scale_x = plot_width / (max_x - min_x)
scale_y = plot_height / (max_y - min_y)

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

for name, pts in series.items():
    if len(pts) < 2:
        continue
    transformed = [to_svg(x, y) for x, y in pts]
    dash = STYLES.get(name)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    color = COLORS.get(name, '#000000')
    polyline = build_polyline(transformed)
    svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="{LINE_WIDTH}"{dash_attr} points="{polyline}"/>')

legend_width = LEGEND_SPACE - 5
legend_x = MARGIN + plot_width + 20
legend_y = MARGIN

legend_items = [
    ("Quad Ref", "quad_ref"),
    ("Quad Actual", "quad_actual"),
    ("Payload Ref", "payload_ref"),
    ("Payload Actual", "payload_actual"),
]
legend_padding_top = 20
legend_padding_bottom = 16
legend_item_spacing = 26
legend_metric_gap = 12
legend_metric_spacing = 22
legend_line_y = legend_y + legend_padding_top
legend_line_x1 = legend_x + 15
legend_line_x2 = legend_line_x1 + 55
legend_svg_elements = []

for label, name in legend_items:
    color = COLORS.get(name, '#000000')
    dash = STYLES.get(name)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    legend_svg_elements.append(f'<line x1="{legend_line_x1}" y1="{legend_line_y}" x2="{legend_line_x2}" y2="{legend_line_y}" stroke="{color}" stroke-width="{LINE_WIDTH}"{dash_attr}/>')
    legend_svg_elements.append(f'<text x="{legend_line_x2 + 10}" y="{legend_line_y + 5}" font-size="16" fill="#333333">{label}</text>')
    legend_line_y += legend_item_spacing

rmse_y1 = legend_line_y + legend_metric_gap
legend_svg_elements.append(f'<text x="{legend_x + 15}" y="{rmse_y1}" font-size="16" fill="#333333">Quad RMSE: {quad_rmse:.3f} m</text>')
rmse_y2 = rmse_y1 + legend_metric_spacing
legend_svg_elements.append(f'<text x="{legend_x + 15}" y="{rmse_y2}" font-size="16" fill="#333333">Payload RMSE: {payload_rmse:.3f} m</text>')

legend_bottom = rmse_y2 + legend_padding_bottom
legend_height = legend_bottom - legend_y
svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" fill="#FFFFFF" stroke="#B0B0B0" stroke-width="1" opacity="0.9"/>')
svg.extend(legend_svg_elements)

content_left = MARGIN
content_top = MARGIN
content_right = max(MARGIN + plot_width, legend_x + legend_width)
content_bottom = max(MARGIN + plot_height, legend_bottom)
frame_x = content_left - FRAME_PADDING
frame_y = content_top - FRAME_PADDING
frame_width = (content_right - content_left) + 2 * FRAME_PADDING
frame_height = (content_bottom - content_top) + 2 * FRAME_PADDING
svg.insert(3, f'<rect x="{frame_x:.1f}" y="{frame_y:.1f}" width="{frame_width:.1f}" height="{frame_height:.1f}" fill="#FFFFFF" stroke="#D0D0D0" stroke-width="1.2"/>')

svg.append('</svg>')

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for line in svg:
        f.write(line + '\n')
print(f"Saved XY plot to {output_path}")
