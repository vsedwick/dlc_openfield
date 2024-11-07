
# Open Field DLC Post-Processing Script

This script post-processes open-field videos labeled with DeepLabCut using a custom OpenField template. It calculates metrics like distance traveled, time spent in specific zones, and generates tracklet visualizations within the arena. The script can be run in **calibration mode** to fine-tune zone creation, particularly useful for first-time setups.

Example data files can be found on [Google Drive](https://drive.google.com/drive/folders/1pjuHoSbZbApui3a1_4NhBAjQv24lvuGl?usp=drive_link).

## Table of Contents
1. [Experiment Description](#experiment-description)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Configuration File (`config.yaml`)](#configuration-file-configyaml)
   - [Calibration Mode](#calibration-mode)
6. [Usage](#usage)
7. [Example Output](#example-output)
8. [Troubleshooting](#troubleshooting)

---

## Experiment Description

The **open field test** is a behavioral experiment used to assess exploration, anxiety, and locomotion in rodents. In an open arena, researchers track:

- **Exploration**: Time spent in various zones (center vs. border).
- **Anxiety**: Avoidance of the center suggests higher anxiety.
- **Locomotion**: Measures like distance and speed provide insight into activity levels.

This test helps evaluate effects of neurological and pharmacological interventions on behavior.


## Features

- **Distance Calculation**: Computes the total distance traveled by the subject within the arena.
- **Zone Timing**: Tracks time spent in specific user-defined zones (center, border, etc.).
- **Tracklet Plotting**: Plots movement paths within the arena for visual inspection.
- **Data Smoothing**: Cleans and smooths tracklets to correct noisy data points.
- **Configurable Arena Zones**: Establishes custom zones within the arena, configurable in `config.yaml`.

## Requirements

- **Dependencies**: The script relies on the following libraries:
  - `pandas`, `numpy`, `matplotlib`, `shapely`, `yaml`, `math`, `datetime`
- **Configuration**: Uses a [`config.yaml`](config.yaml) file for customized parameters.

## Setup

1. **Clone Repository**: Clone this repository to your local machine.
2. **Install Requirements**: Install the required Python libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Configuration**: Customize the [`config.yaml`](config.yaml) file with specific settings for your experiment.

## Configuration File (`config.yaml`)

The [`config.yaml`](config.yaml) file contains all necessary settings. The project information and Video Parameter settings must be provided to run the script. If values are missing in "Arena_parameters" or user is uncertain, the program will prompt the user for values. These settings can be further adjusted during calibration. Here is an example configuration file structure:

```yaml
rootdir: directory/path_to_project_folder
example_DLC_file_h5: directory/path_to_project_folder/dlc-output-file.h5
project_name: "Super Awesome Experiment"
scoretype: batch  # Options: single, batch (for analyzing an entire folder)
calibration_mode: on

Video Parameters:
  real_sizecm: 65
  fps: 30
  length_s: 300  # Length of video analysis in seconds

Arena_parameters:
  movement_correction_factor: 0.58
  scale_for_center_zone: 0.28
  plikelihood: 0.9  # Likelihood cutoff for tracking accuracy
  save_tracklets: yes
  bodyparts_to_determine_trial_start:
    - nose
    - Neck
  bodyparts_to_analyze:
    - Neck
```

The configuration file includes:

- **Root Directory and Project Info**: Paths to data and project settings.
- **Video Parameters**:
  - `real_sizecm`: Real-world size of the arena in centimeters.
  - `fps`: Frames per second for the video.
  - `length_s`: Duration of video analysis in seconds.
- **Arena Parameters**:
  - `movement_correction_factor`: Adjusts smoothing to correct noisy tracklets.
  - `scale_for_center_zone`: Sets the size of the center zone.
  - `plikelihood`: Sets the likelihood threshold for valid tracking.
  - `save_tracklets`: Option to save tracking data.
  - `bodyparts_to_determine_trial_start`: Specifies body parts that signal the start of the trial.
  - `bodyparts_to_analyze`: Defines body parts to analyze.

> **Note**: In **calibration mode**, the script will prompt for any missing or non-crucial values in the configuration file. These values are saved back to `config.yaml` for future use. 

### Calibration Mode

Calibration mode is recommended for first-time use, as it allows fine-tuning of the arena setup. This mode will generate two images:
1. **Arena Template**: The basic layout of the arena zones, saved as `Images/Arena1.png`.

   ![Arena Template](/Images/Arena.png)

   - Center and borders represent the actual arena as per DeepLabCut's open field template.
   - The wall extends around the arena to capture edge movements.
   - Bodymarker coordinates identified outside of the wall boundary will be considered an error and will be cleaned in the dataset.

2. **Preliminary Tracking Output**: A sample tracklet visualization is produced with all of the body markers defined in the configuration file. Example images are provided below with one and three tracklets.

   ![1 Point Tracklet](/Images/1point_tracklet.png)
   
   ![3 Point Tracklet](/Images/3point_tracklet.png)

   - Adjust `plikelihood` and `movement_correction_factor` if the visualization appears inaccurate.

### Adjusting Parameters in Calibration Mode

You can adjust parameters like `plikelihood` and `movement_correction_factor` in `config.yaml` to refine the tracking. For example:
```yaml
Arena_parameters:
  plikelihood: 0.85  # Adjust threshold to reduce tracking errors
  movement_correction_factor: 0.6  # Increase smoothing for noisy tracklets
```

## Usage

Run the script with a configuration file as an argument:
```bash
python open_field_v2.py <path_to_config.yaml>
```

Alternatively, specify the configuration file path directly in the script (instructions in code comments).

## Example Output

The primary output is a dataframe with the following metrics:

| Video Name             | Body Marker | Distance (cm) | Velocity (cm/s) | Center Entries | Border Entries | Time Immobile (s) | Time Mobile (s) | Time in Center (s) | Time in Border (s) | Time in Arena (s) |
|------------------------|-------------|----------------|------------------|----------------|----------------|--------------------|-----------------|---------------------|---------------------|--------------------|
| Sample_Video1          | nose        | 15.2           | 0.5              | 5              | 8              | 30                | 120             | 50                  | 100                 | 300                |
| Sample_Video1          | neck        | 14.8           | 0.45             | 5              | 7              | 32                | 118             | 52                  | 98                  | 300                |
| Sample_Video1          | tail_base   | 16.0           | 0.55             | 4              | 9              | 29                | 121             | 48                  | 102                 | 300                |
| Sample_Video2          | nose        | 13.6           | 0.52             | 6              | 10             | 35                | 125             | 54                  | 96                  | 300                |

This output includes key metrics such as distance traveled, velocity, center and border entries, time spent immobile and mobile, and time spent in each zone (center, border, and arena).

### Loading and Inspecting Output Data

To load and inspect the output data in Python, you can use the following code snippet:

```python
import pandas as pd

# Load and inspect the output data
data = pd.read_excel('Super Awesome Experiment-2024-11-06_18-27-00.xlsx')
print(data.head())
```

## Troubleshooting

For issues or questions, contact Victoria Sedwick at sedwick.victoria@gmail.com.
