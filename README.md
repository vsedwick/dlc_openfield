# Open Field DLC Post-Processing Script

This script post-processes open-field videos labeled with DeepLabCut. It calculates metrics like distance traveled, time spent in specific zones, and generates tracklet visualizations within the arena. Additionally, it smooths aberrant tracklets and sets up designated arena zones.

## Features

- **Distance Calculation**: Computes the distance traveled by the subject within the arena.
- **Zone Timing**: Tracks the time spent in specific user-defined zones.
- **Tracklet Plotting**: Plots movement tracklets within the arena for visual inspection.
- **Data Smoothing**: Cleans and smooths tracklets to correct noisy data points.
- **Configurable Arena Zones**: Establishes custom zones within the arena via a configuration file.

## Requirements

- **Dependencies**: The script relies on the following libraries:
  - `pandas`, `numpy`, `matplotlib`, `shapely`, `yaml`, `math`, `datetime`
- **Configuration**: Uses a `config.yaml` file for customized parameters.

## Setup

1. **Clone Repository**: Clone this repository to your local machine.
2. **Install Requirements**: Install the required Python libraries using:
   ```bash
   pip install pandas numpy matplotlib shapely pyyaml
   ```
3. **Prepare Configuration**: Customize the `config.yaml` file with specific settings for your experiment.

## Usage

1. Ensure that the video data labeled by DeepLabCut is accessible and that you have configured the zones and arena parameters in `config.yaml`.
2. Run the script:
   ```bash
   python open_field_v2.py
   ```

## Troubleshooting

For issues or questions, contact Victoria Sedwick at sedwick.victoria@gmail.com.
