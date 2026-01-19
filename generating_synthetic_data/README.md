# Generating Synthetic Training Data

This folder contains scripts and documentation for generating synthetic marker trajectories to train the automatic marker labeling algorithm.

## Overview

Synthetic data generation creates artificial motion capture marker trajectories by simulating where markers would be positioned during real human movements. This eliminates the need for large amounts of manually labeled mocap data.

## Prerequisites

### 1. Body Kinematics Data

Download the body kinematics HDF5 file from Zenodo:
- **DOI:** [10.5281/zenodo.4293999](https://doi.org/10.5281/zenodo.4293999)
- **File:** `bodykinematics.hdf5` (several GB)
- **Contains:** Body segment kinematics from 100 participants performing athletic movements

**Download Instructions:**
```bash
# Option 1: Direct download (if you have the direct link)
wget https://zenodo.org/record/4293999/files/bodykinematics.hdf5 -P ../data/

# Option 2: Manual download
# Visit https://doi.org/10.5281/zenodo.4293999
# Download bodykinematics.hdf5
# Place it in the ../data/ folder
```

### 2. OpenSim Software (Optional but Recommended)

Install OpenSim to create/edit marker sets:
- **Website:** https://simtk.org/projects/opensim
- **Download:** Get the latest version for your OS
- **Purpose:** Create and visualize marker set definitions

### 3. Python Dependencies

Ensure you have all required packages:
```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `h5py` - Read HDF5 body kinematics file
- `torch` - PyTorch for tensor operations
- `scipy` - Signal processing and interpolation
- `numpy` - Numerical operations

## What You Need to Provide

### 1. Marker Set XML File

You need an OpenSim marker set XML that defines:
- **Marker names** - Labels for each marker (e.g., "LFHD", "RAC", "T10")
- **Body segments** - Which bone/segment each marker attaches to
- **Local coordinates** - Position relative to the segment frame

**Creating from Vicon VST:**
If you have a Vicon VST file (like `ASD_FullSpine_TO_TL_T1cluster.txt`), you can:
1. Use the `convert_vst_to_opensim.py` script (see below)
2. Manually create in OpenSim GUI using your VST as reference

**Example marker set structure:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="30000">
  <MarkerSet>
    <objects>
      <Marker name="LFHD">
        <socket_parent_frame>/bodyset/torso</socket_parent_frame>
        <location>0.065 0.580 0.085</location>
      </Marker>
      <Marker name="RAC">
        <socket_parent_frame>/bodyset/torso</socket_parent_frame>
        <location>-0.045 0.420 0.170</location>
      </Marker>
      <!-- More markers... -->
    </objects>
  </MarkerSet>
</OpenSimDocument>
```

### 2. Configuration Parameters

Specify these parameters for your marker set:
- **Sampling frequency (fs):** Hz of your mocap system (e.g., 120, 240)
- **Alignment markers:** Left/right bilateral markers for orientation
  - Example: Right acromion (RAC), Left acromion (LAC)
  - Or: Right ASIS (RASI), Left ASIS (LASI)
- **Number of participants:** 1-100 (more = larger training set)
- **Window size:** Frames per segment (default: 240)

## Step-by-Step Guide

### Step 1: Prepare Your Marker Set

**Option A: Use existing OpenSim marker set**
```bash
# If you already have a MarkerSet.xml
cp /path/to/your/MarkerSet.xml ../data/MarkerSet.xml
```

**Option B: Convert from Vicon VST**
```bash
# Use the conversion script (if you upload your VST file)
python convert_vst_to_opensim.py \
    --vst ASD_FullSpine_TO_TL_T1cluster.txt \
    --output ../data/MarkerSet_ASD.xml \
    --model ../data/Rajagopal2015_mod.osim
```

**Option C: Create in OpenSim GUI**
1. Open OpenSim
2. Load `../data/Rajagopal2015_mod.osim`
3. Add markers using the Markers panel
4. Assign each marker to a body segment
5. Set local coordinates
6. Save marker set: File → Save → MarkerSet

### Step 2: Verify Body Kinematics File

Check that the HDF5 file is properly downloaded:
```bash
python check_hdf5_structure.py ../data/bodykinematics.hdf5
```

This will show:
- Number of participants available
- Trial names for each participant
- Body segments included
- Data dimensions

### Step 3: Configure Generation Script

Edit `generate_synthetic_data.py` or use command-line arguments:

```python
# Configuration
bodykin_path = '../data/bodykinematics.hdf5'
markerset_path = '../data/MarkerSet_ASD.xml'
output_path = '../data/simulatedTrajectories_ASD.pickle'

# Alignment markers (must exist in your marker set)
align_marker_right = 'RAC'  # Right acromion
align_marker_left = 'LAC'   # Left acromion

# Parameters
sampling_frequency = 240  # Hz
num_participants = 100    # 1-100
max_segment_length = 240  # frames
```

### Step 4: Generate Synthetic Data

Run the generation script:
```bash
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet_ASD.xml \
    --output ../data/simulatedTrajectories_ASD.pickle \
    --align-right RAC \
    --align-left LAC \
    --fs 240 \
    --num-participants 100 \
    --max-len 240
```

**Expected output:**
```
Loaded marker set: 43 markers across 8 body segments
Generating trajectories...
10/100 complete
20/100 complete
...
100/100 complete
Training data saved to ../data/simulatedTrajectories_ASD.pickle
Generated 3847 trajectory segments
```

**Time estimate:** 10-30 minutes depending on number of participants and system specs

### Step 5: Verify Generated Data

Check the output:
```bash
python inspect_synthetic_data.py ../data/simulatedTrajectories_ASD.pickle
```

This shows:
- Number of trajectory segments
- Marker count per segment
- Segment length statistics
- Data quality metrics (NaN counts, coordinate ranges)

## Understanding the Output

### Output File Format

**File:** `simulatedTrajectories_ASD.pickle`

**Contents:** Python list of PyTorch tensors
- **List length:** Number of trajectory segments (typically 3000-4000)
- **Tensor shape:** `(num_frames, num_markers, 3)`
  - `num_frames`: Temporal dimension (varies, max 240)
  - `num_markers`: Number of markers in your set
  - `3`: X, Y, Z coordinates in millimeters

**Example:**
```python
import pickle
with open('../data/simulatedTrajectories_ASD.pickle', 'rb') as f:
    data = pickle.load(f)

print(f"Number of segments: {len(data)}")
print(f"First segment shape: {data[0].shape}")
# Output: Number of segments: 3847
#         First segment shape: torch.Size([180, 43, 3])
```

### Data Characteristics

**Coordinate System:**
- **Z-axis:** Vertical (up)
- **X-axis:** Forward (subject facing direction)
- **Y-axis:** Lateral (right-to-left)
- **Units:** Millimeters

**Alignment:**
- All segments are rotated so the subject faces +X direction
- Calculated using the bilateral alignment markers you specified

**Temporal:**
- Resampled to match your target sampling frequency
- Low-pass filtered at 6 Hz (removes noise)
- Split into segments ≤ max_segment_length frames

## Next Steps: Training

Once you have generated synthetic data, proceed to training:

```bash
cd ..
python trainAlgorithm.py
```

Configure `trainAlgorithm.py` to use your synthetic data:
```python
datapath = './data/simulatedTrajectories_ASD.pickle'
markersetpath = './data/MarkerSet_ASD.xml'
fld = './data/'
fs = 240
num_epochs = 10
```

**Training outputs:**
- `model_YYYY-MM-DD.ckpt` - Trained neural network
- `trainingvals_YYYY-MM-DD.pickle` - Normalization parameters
- `training_stats_YYYY-MM-DD.pickle` - Training loss history

## Troubleshooting

### Problem: "Body kinematics file not found"
**Solution:** Download from Zenodo (see Prerequisites)

### Problem: "Marker X not found in body kinematics"
**Solution:** Your marker set includes markers on body segments not in the kinematics data. Either:
- Remove those markers from your marker set
- Map them to available segments in the HDF5 file

### Problem: "Alignment markers not in marker set"
**Solution:** Check marker names match exactly (case-sensitive). Common names:
- Acromions: RAC/LAC, RSHO/LSHO
- ASIS: RASI/LASI, RASIS/LASIS
- Update `align_marker_right` and `align_marker_left` to match your naming

### Problem: "Out of memory during generation"
**Solution:** Reduce `num_participants` to 50 or 25, or generate in batches

### Problem: "Generated data has too many NaNs"
**Solution:** Check that marker local coordinates are reasonable (within ±1 meter of segment origin)

## File Structure

```
generating_synthetic_data/
├── README.md                        # This file
├── generate_synthetic_data.py       # Main generation script
├── convert_vst_to_opensim.py        # VST → OpenSim converter
├── check_hdf5_structure.py          # Inspect body kinematics
├── inspect_synthetic_data.py        # Verify generated data
└── example_markerset/               # Example marker set files
    ├── example_simple.xml           # Simple 23-marker set
    └── example_fullbody.xml         # Full body 43-marker set
```

## Technical Details

### How It Works

1. **Load marker set:** Parse XML to get marker names, segments, local coordinates
2. **Load body kinematics:** Read transformation matrices for each body segment over time
3. **For each participant and trial:**
   - For each marker:
     - Scale local position by participant body dimensions
     - Transform to global coordinates using segment kinematics
     - Convert from Y-up to Z-up coordinate system
   - Resample to target frequency using cubic spline
   - Align subject to face +X direction
   - Split long trials into manageable segments
4. **Save:** Output as pickle file of PyTorch tensors

### Mathematical Formula

Global marker position at frame `i`:
```
p_global = R * T[i] * (scale * p_local - com)
```

Where:
- `p_local`: Marker position in segment frame (from XML)
- `scale`: Participant body segment scale factors
- `com`: Segment center of mass offset
- `T[i]`: 4×4 transformation matrix at frame i
- `R`: Rotation from Y-up to Z-up

### Body Segments Available

The HDF5 file contains kinematics for:
- torso
- pelvis
- femur_r / femur_l
- tibia_r / tibia_l
- talus_r / talus_l
- calcn_r / calcn_l
- toes_r / toes_l
- humerus_r / humerus_l
- ulna_r / ulna_l
- radius_r / radius_l
- hand_r / hand_l

If your marker set uses different segment names, you'll need to map them.

## Additional Resources

- **OpenSim Documentation:** https://simtk-confluence.stanford.edu/display/OpenSim/Markers
- **Body Kinematics Paper:** Fukuchi et al. (2018) DOI: 10.1038/sdata.2018.23
- **Main Paper:** Clouthier et al. (2021) DOI: 10.1109/ACCESS.2021.3062748

## Support

For questions about synthetic data generation:
1. Check this README
2. Review `../CLAUDE.md` for technical details
3. Inspect existing example data in `../data/`
4. Contact: aclouthi@uottawa.ca
