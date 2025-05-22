import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors

def df_grouping(df: pd.DataFrame):
    # Map Trackman columns
    df = df.rename(columns={
        'TaggedPitchType': 'pitch_type',
        'RelSpeed': 'release_speed',
        'HorzBreak': 'pfx_x',
        'InducedVertBreak': 'pfx_z',
        'SpinRate': 'release_spin_rate',
        'RelSide': 'release_pos_x',
        'RelHeight': 'release_pos_z',
        'Extension': 'release_extension',
        'PlateLocSide': 'plate_x',
        'PlateLocHeight': 'plate_z'
    })

    # Define pitch type mapping
    pitch_mapping = {
        'Fastball': '4-SEAM FASTBALL',
        'Sinker': 'SINKER',
        'Curveball': 'CURVEBALL',
        'ChangeUp': 'CHANGEUP',
        'Sweeper': 'SWEEPER',
        'Slider': 'SLIDER',
        'Cutter': 'CUTTER'
    }
    df['pitch_type'] = df['pitch_type'].map(pitch_mapping).fillna(df['pitch_type'])

    # Map pitch types to colors
    dict_colour = {
        '4-SEAM FASTBALL': 'pink',
        'SINKER': 'purple',
        'CURVEBALL': 'blue',
        'CHANGEUP': 'orange',
        'SWEEPER': 'red',
        'SLIDER': 'green',
        'CUTTER': 'yellow'
    }
    df['colour'] = df['pitch_type'].map(dict_colour).fillna('gray')

    # Convert numeric columns to float, handling errors
    numeric_cols = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate',
                    'release_pos_x', 'release_pos_z', 'release_extension',
                    'plate_x', 'plate_z']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def plot_3d_pitch_track(df: pd.DataFrame, pitcher_name: str):
    # Group by pitch type and compute averages
    df_avg = df.groupby('pitch_type').agg({
        'release_pos_x': 'mean',
        'release_pos_z': 'mean',
        'release_extension': 'mean',
        'plate_x': 'mean',
        'plate_z': 'mean',
        'release_speed': 'mean',
        'colour': 'first'  # Take the first color (consistent per pitch type)
    }).reset_index()

    # Convert release speed from mph to ft/s
    df_avg.loc[:, 'release_speed_fps'] = df_avg['release_speed'] * 1.467

    # Calculate release y-position (60.5 ft minus extension)
    df_avg.loc[:, 'release_y'] = 60.5 - df_avg['release_extension'].fillna(6)  # Default: 6 ft

    # Initialize figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot home plate (strike zone: -1 to 1 ft x, 1 to 3 ft z)
    plate_x = [-1, 1, 1, -1, -1]
    plate_z = [1.5, 1.5, 3.5, 3.5, 1.5]
    plate_y = [0, 0, 0, 0, 0]
    ax.plot(plate_x, plate_y, plate_z, 'k-', label='Strike Zone')

    # Plot average trajectory for each pitch type
    for _, row in df_avg.iterrows():
        pitch_type = row['pitch_type']
        color = row['colour']

        # Release point (average)
        x0 = row['release_pos_x']
        y0 = row['release_y']
        z0 = row['release_pos_z']

        # Plate location (average)
        x1 = row['plate_x']
        y1 = 0
        z1 = row['plate_z']

        # Skip if any coordinates are NaN
        if any(pd.isna([x0, y0, z0, x1, y1, z1])):
            print(f"Skipping {pitch_type} due to missing coordinates")
            continue

        # Simulate trajectory (linear for visualization)
        t = np.linspace(0, 1, 100)
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        z = z0 + t * (z1 - z0)

        # Plot trajectory
        ax.plot(x, y, z, color=color, alpha=0.7, label=pitch_type,linewidth=3)

    # Set labels and title
    ax.set_xlabel('Horizontal (ft)')
    ax.set_ylabel('Distance to Plate (ft)')
    ax.set_zlabel('Height (ft)')
    ax.set_title(f'Average 3D Pitch Tracks for {pitcher_name}')

    # Set axis limits
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 7)

    # Adjust view angle
    ax.view_init(elev=8, azim=-75)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Save and close
    plt.savefig(f"{pitcher_name.replace(', ', '_')}_3d_avg_pitch_track.png", bbox_inches='tight', dpi=500)
    plt.close()

# Main execution
dtypes = {
    'Pitcher': str,
    'PitcherTeam': str,
    'TaggedPitchType': str,
    'RelSpeed': float,
    'HorzBreak': float,
    'InducedVertBreak': float,
    'SpinRate': float,
    'RelSide': float,
    'RelHeight': float,
    'Extension': float,
    'PlateLocSide': float,
    'PlateLocHeight': float
}

try:
    df = pd.read_csv('2025.csv', dtype=dtypes, low_memory=False)
except ValueError as e:
    print(f"Error reading CSV: {e}")
    df = pd.read_csv('2025.csv', low_memory=False)

pitcher_name = 'Chadwick, Tyrelle'
pitcher_data = df[(df['PitcherTeam'] == 'ILL_RED') & (df['Pitcher'] == pitcher_name)]

if pitcher_data.empty:
    print(f"No data found for {pitcher_name} from ILL_RED")
else:
    # Process data
    pitcher_data = df_grouping(pitcher_data)
    
    # Generate 3D pitch track with averages
    plot_3d_pitch_track(pitcher_data, pitcher_name)