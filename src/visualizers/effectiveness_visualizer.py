#!/usr/bin/env python3
"""
Effectiveness Matrix Visualization Module

Generates interactive HTML visualizations for thruster configurations:
- 3D thruster layout with thrust vectors
- Effectiveness matrix heatmap
- Control authority bar chart

Requires: plotly>=5.0.0, numpy

Author: Generated for UUV Reconbot project
"""

import numpy as np
from typing import Optional
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Thruster color scheme by function
THRUSTER_COLORS = {
    'heave': '#3498db',      # Blue - vertical thrusters
    'surge': '#e74c3c',      # Red - forward thrusters
    'sway': '#2ecc71',       # Green - lateral thrusters
    'default': '#9b59b6',    # Purple - unclassified
}


def classify_thruster(rotor: dict) -> str:
    """
    Classify thruster by its primary function based on thrust axis.

    Args:
        rotor: Rotor configuration dict with ax, ay, az

    Returns:
        Classification string: 'heave', 'surge', 'sway', or 'default'
    """
    ax = abs(rotor.get('ax', 0))
    ay = abs(rotor.get('ay', 0))
    az = abs(rotor.get('az', 0))

    # Dominant axis determines function
    if ax > ay and ax > az:
        return 'surge'  # X-axis dominant = forward/backward
    elif ay > ax and ay > az:
        return 'sway'   # Y-axis dominant = left/right
    elif az > 0.3:
        return 'heave'  # Z-axis component = vertical
    else:
        return 'default'


def create_3d_thruster_plot(rotors: list[dict],
                            effectiveness: Optional[np.ndarray] = None) -> go.Figure:
    """
    Create interactive 3D plot of thruster positions and thrust vectors.

    Args:
        rotors: List of rotor configuration dicts
        effectiveness: Optional effectiveness matrix for annotations

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Vehicle body (semi-transparent box approximation)
    # Estimate vehicle dimensions from thruster positions
    positions = np.array([[r.get('px', 0), r.get('py', 0), r.get('pz', 0)]
                          for r in rotors])

    x_range = [positions[:, 0].min() - 0.1, positions[:, 0].max() + 0.1]
    y_range = [positions[:, 1].min() - 0.05, positions[:, 1].max() + 0.05]
    z_range = [positions[:, 2].min() - 0.05, positions[:, 2].max() + 0.05]

    # Add vehicle body as mesh
    # Create a simple rectangular box
    x_body = [x_range[0], x_range[1], x_range[1], x_range[0],
              x_range[0], x_range[1], x_range[1], x_range[0]]
    y_body = [y_range[0], y_range[0], y_range[1], y_range[1],
              y_range[0], y_range[0], y_range[1], y_range[1]]
    z_body = [z_range[0], z_range[0], z_range[0], z_range[0],
              z_range[1], z_range[1], z_range[1], z_range[1]]

    fig.add_trace(go.Mesh3d(
        x=x_body, y=y_body, z=z_body,
        i=[0, 0, 0, 1, 2, 3, 4, 4, 4, 5, 6, 7],
        j=[1, 2, 4, 5, 6, 7, 5, 6, 0, 1, 2, 3],
        k=[2, 3, 5, 6, 7, 4, 6, 7, 1, 2, 3, 0],
        color='lightgray',
        opacity=0.3,
        name='Vehicle Body',
        hoverinfo='name'
    ))

    # Add thrusters and thrust vectors
    for i, rotor in enumerate(rotors):
        px = rotor.get('px', 0)
        py = rotor.get('py', 0)
        pz = rotor.get('pz', 0)
        ax = rotor.get('ax', 0)
        ay = rotor.get('ay', 0)
        az = rotor.get('az', 0)

        # Normalize axis
        axis_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if axis_norm > 0:
            ax, ay, az = ax/axis_norm, ay/axis_norm, az/axis_norm

        # Classify and color
        thruster_type = classify_thruster(rotor)
        color = THRUSTER_COLORS[thruster_type]

        # Thruster position marker
        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[pz],
            mode='markers+text',
            marker=dict(size=10, color=color, symbol='diamond'),
            text=[f'R{i}'],
            textposition='top center',
            name=f'Rotor {i} ({thruster_type})',
            hovertemplate=(
                f'<b>Rotor {i}</b><br>'
                f'Position: ({px:.3f}, {py:.3f}, {pz:.3f})<br>'
                f'Axis: ({ax:.3f}, {ay:.3f}, {az:.3f})<br>'
                f'Type: {thruster_type}<extra></extra>'
            )
        ))

        # Thrust vector as cone
        scale = 0.15  # Vector length
        fig.add_trace(go.Cone(
            x=[px], y=[py], z=[pz],
            u=[ax * scale], v=[ay * scale], w=[az * scale],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=0.08,
            anchor='tail',
            name=f'Thrust {i}',
            hoverinfo='skip'
        ))

    # Add coordinate frame at origin
    axis_len = 0.2
    for axis, color, label in [([1,0,0], 'red', 'X (Forward)'),
                                ([0,1,0], 'green', 'Y (Port)'),
                                ([0,0,1], 'blue', 'Z (Up)')]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]*axis_len],
            y=[0, axis[1]*axis_len],
            z=[0, axis[2]*axis_len],
            mode='lines+text',
            line=dict(color=color, width=4),
            text=['', label],
            textposition='top center',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text='<b>3D Thruster Configuration</b>',
            x=0.5, xanchor='center'
        ),
        scene=dict(
            xaxis_title='X (Forward) [m]',
            yaxis_title='Y (Port) [m]',
            zaxis_title='Z (Up) [m]',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=0.01
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )

    return fig


def create_effectiveness_heatmap(effectiveness: np.ndarray,
                                  rotor_names: Optional[list[str]] = None) -> go.Figure:
    """
    Create heatmap visualization of effectiveness matrix.

    Args:
        effectiveness: 6xN effectiveness matrix
        rotor_names: Optional list of rotor names

    Returns:
        Plotly Figure object
    """
    n_rotors = effectiveness.shape[1]

    if rotor_names is None:
        rotor_names = [f'R{i}' for i in range(n_rotors)]

    axis_names = ['Roll', 'Pitch', 'Yaw', 'Fx', 'Fy', 'Fz']

    # Find max absolute value for symmetric colorscale
    max_val = np.abs(effectiveness).max()

    # Create annotation text
    annotations = []
    for i in range(6):
        for j in range(n_rotors):
            val = effectiveness[i, j]
            annotations.append(
                dict(
                    x=rotor_names[j],
                    y=axis_names[i],
                    text=f'{val:.3f}' if abs(val) > 0.001 else '0',
                    showarrow=False,
                    font=dict(
                        color='white' if abs(val) > max_val * 0.5 else 'black',
                        size=11
                    )
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=effectiveness,
        x=rotor_names,
        y=axis_names,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-max_val,
        zmax=max_val,
        colorbar=dict(
            title=dict(text='Effect', side='right')
        ),
        hovertemplate=(
            '<b>%{y}</b> from <b>%{x}</b><br>'
            'Value: %{z:.4f}<extra></extra>'
        )
    ))

    fig.update_layout(
        title=dict(
            text='<b>Effectiveness Matrix</b><br><sup>Motor contribution to each DOF</sup>',
            x=0.5, xanchor='center'
        ),
        xaxis_title='Rotor',
        yaxis_title='Degree of Freedom',
        annotations=annotations,
        height=400
    )

    return fig


def create_control_authority_chart(controllability_result: dict) -> go.Figure:
    """
    Create bar chart showing control authority (singular values) per axis.

    Args:
        controllability_result: Dict from analyze_controllability() containing
            singular_values, rank, controllable, condition_number

    Returns:
        Plotly Figure object
    """
    singular_values = controllability_result['singular_values']
    axis_names = ['Roll', 'Pitch', 'Yaw', 'Fx', 'Fy', 'Fz']

    # Only show first 6 singular values
    sv = singular_values[:6]
    max_sv = sv[0] if len(sv) > 0 else 1

    # Color by strength relative to max
    colors = []
    for s in sv:
        ratio = s / max_sv if max_sv > 0 else 0
        if ratio > 0.3:
            colors.append('#2ecc71')  # Green - strong
        elif ratio > 0.1:
            colors.append('#f39c12')  # Yellow - moderate
        else:
            colors.append('#e74c3c')  # Red - weak

    fig = go.Figure()

    # Horizontal bar chart
    fig.add_trace(go.Bar(
        y=axis_names[:len(sv)],
        x=sv,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.3f}' for s in sv],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Singular value: %{x:.4f}<extra></extra>'
    ))

    # Add threshold line at 10% of max
    threshold = max_sv * 0.1
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='red',
        annotation_text='10% threshold',
        annotation_position='top right'
    )

    # Add summary text
    rank = controllability_result.get('rank', 0)
    controllable = controllability_result.get('controllable', False)
    cond_num = controllability_result.get('condition_number', float('inf'))

    status_color = '#2ecc71' if controllable else '#e74c3c'
    status_text = 'CONTROLLABLE' if controllable else 'NOT CONTROLLABLE'

    fig.update_layout(
        title=dict(
            text=(f'<b>Control Authority (Singular Values)</b><br>'
                  f'<sup>Rank: {rank}/6 | '
                  f'<span style="color:{status_color}">{status_text}</span> | '
                  f'Condition: {cond_num:.1f}</sup>'),
            x=0.5, xanchor='center'
        ),
        xaxis_title='Singular Value',
        yaxis_title='DOF',
        showlegend=False,
        height=400,
        xaxis=dict(range=[0, max_sv * 1.2])
    )

    return fig


def generate_visualization_report(rotors: list[dict],
                                   effectiveness: np.ndarray,
                                   controllability_result: dict,
                                   issues: list[str],
                                   output_path: str = 'effectiveness_report.html',
                                   title: str = 'Thruster Configuration Report') -> str:
    """
    Generate complete HTML visualization report.

    Args:
        rotors: List of rotor configuration dicts
        effectiveness: 6xN effectiveness matrix
        controllability_result: Dict from analyze_controllability()
        issues: List of issue strings from detect_issues()
        output_path: Output HTML file path
        title: Report title

    Returns:
        Path to generated HTML file
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly>=5.0.0"
        )

    # Create individual figures
    fig_3d = create_3d_thruster_plot(rotors, effectiveness)
    fig_heatmap = create_effectiveness_heatmap(effectiveness)
    fig_authority = create_control_authority_chart(controllability_result)

    # Build HTML report
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 15px;
        }}
        .row {{
            display: flex;
            gap: 20px;
        }}
        .col {{
            flex: 1;
        }}
        .issues-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .issues-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .issue {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .no-issues {{
            padding: 10px;
            background: #d4edda;
            border-left: 4px solid #28a745;
            border-radius: 4px;
        }}
        .config-summary {{
            background: #e8f4fd;
            border-radius: 4px;
            padding: 10px 15px;
            margin-bottom: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Generated by Effectiveness Calculator</p>

        <div class="config-summary">
            <strong>Configuration:</strong> {len(rotors)} rotors |
            <strong>Rank:</strong> {controllability_result.get('rank', 'N/A')}/6 |
            <strong>Condition Number:</strong> {controllability_result.get('condition_number', 0):.2f}
        </div>

        <div class="plot-container">
            <div id="plot3d"></div>
        </div>

        <div class="row">
            <div class="col plot-container">
                <div id="heatmap"></div>
            </div>
            <div class="col plot-container">
                <div id="authority"></div>
            </div>
        </div>

        <div class="issues-container">
            <div class="issues-title">Issue Detection</div>
'''

    if issues:
        for issue in issues:
            html_content += f'            <div class="issue">{issue}</div>\n'
    else:
        html_content += '            <div class="no-issues">No issues detected</div>\n'

    html_content += f'''        </div>
    </div>

    <script>
        var plot3d = {fig_3d.to_json()};
        var heatmap = {fig_heatmap.to_json()};
        var authority = {fig_authority.to_json()};

        Plotly.newPlot('plot3d', plot3d.data, plot3d.layout, {{responsive: true}});
        Plotly.newPlot('heatmap', heatmap.data, heatmap.layout, {{responsive: true}});
        Plotly.newPlot('authority', authority.data, authority.layout, {{responsive: true}});
    </script>
</body>
</html>'''

    # Write to file
    output_path = Path(output_path)
    output_path.write_text(html_content)

    return str(output_path.absolute())


def check_plotly_available() -> bool:
    """Check if plotly is available for import."""
    return PLOTLY_AVAILABLE
