# Complete Workflow Example

This document walks through a complete example of generating synthetic data and training a model for a custom marker set.

## Scenario

You have:
- A custom marker set with 43 markers
- Vicon mocap system running at 240 Hz
- Alignment markers: RSHO (right shoulder), LSHO (left shoulder)
- Want to train a model for automatic marker labeling

## Directory Structure at Each Step

### Initial State
```
auto-marker-label/
├── automarkerlabel.py
├── trainAlgorithm.py
├── generateSimTrajectories.py
├── markerLabelGUI.py
├── requirements.txt
├── CLAUDE.md
├── README.md
├── data/
│   ├── MarkerSet.xml                    # Existing example
│   └── Rajagopal2015_mod.osim
└── generating_synthetic_data/
    ├── README.md
    ├── QUICKSTART.md
    ├── generate_synthetic_data.py
    ├── check_hdf5_structure.py
    ├── inspect_synthetic_data.py
    └── config_template.py
```

### After Downloading Body Kinematics
```
data/
├── bodykinematics.hdf5                   # ← Downloaded (3-5 GB)
├── MarkerSet.xml
└── Rajagopal2015_mod.osim
```

**Command used:**
```bash
cd data/
wget https://zenodo.org/record/4293999/files/bodykinematics.hdf5
```

**Verification:**
```bash
cd ../generating_synthetic_data/
python check_hdf5_structure.py ../data/bodykinematics.hdf5
```

**Expected output:**
```
==================================================================
BODY KINEMATICS HDF5 FILE INSPECTION
==================================================================
File: /path/to/data/bodykinematics.hdf5
Size: 4.23 GB
==================================================================

📦 TOP-LEVEL GROUPS:
------------------------------------------------------------------
  • kin/
  • com/
  • scale/

🏃 KINEMATICS DATA:
------------------------------------------------------------------
Number of participants: 100
Participant IDs: S01, S02, S03, S04, S05...

✓ File appears valid and properly formatted
==================================================================
```

### After Creating Your Marker Set

**Option 1: You created MarkerSet_Custom.xml in OpenSim**
```
data/
├── bodykinematics.hdf5
├── MarkerSet.xml                         # Original example
├── MarkerSet_Custom.xml                  # ← Your custom marker set
└── Rajagopal2015_mod.osim
```

**Option 2: You converted from VST file**
```
generating_synthetic_data/
├── ASD_FullSpine_TO_TL_T1cluster.txt     # ← Your uploaded VST
├── convert_vst_to_opensim.py             # ← Conversion script
└── ...

data/
├── bodykinematics.hdf5
├── MarkerSet.xml
├── MarkerSet_ASD.xml                     # ← Converted from VST
└── Rajagopal2015_mod.osim
```

### After Generating Synthetic Data

```
data/
├── bodykinematics.hdf5
├── MarkerSet.xml
├── MarkerSet_Custom.xml
├── simulatedTrajectories_Custom_240Hz.pickle  # ← Generated synthetic data
└── Rajagopal2015_mod.osim
```

**Command used:**
```bash
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet_Custom.xml \
    --output ../data/simulatedTrajectories_Custom_240Hz.pickle \
    --align-right RSHO \
    --align-left LSHO \
    --fs 240 \
    --num-participants 100
```

**Expected output:**
```
==================================================================
SYNTHETIC TRAJECTORY GENERATION
==================================================================
Body kinematics:    ../data/bodykinematics.hdf5
Marker set:         ../data/MarkerSet_Custom.xml
Output file:        ../data/simulatedTrajectories_Custom_240Hz.pickle
Alignment markers:  RSHO (R) / LSHO (L)
Sampling frequency: 240 Hz
Participants:       100 / 100
Max segment length: 240 frames
==================================================================

Starting generation...
10/100 complete
20/100 complete
...
100/100 complete
Training data saved to ../data/simulatedTrajectories_Custom_240Hz.pickle

==================================================================
GENERATION COMPLETE
==================================================================
Total segments:     3847
Markers per segment: 43
Segment lengths:    60 - 240 frames
Time elapsed:       453.2 seconds (7.6 minutes)
Output saved to:    /path/to/data/simulatedTrajectories_Custom_240Hz.pickle
==================================================================
```

**Verification:**
```bash
python inspect_synthetic_data.py ../data/simulatedTrajectories_Custom_240Hz.pickle
```

**Expected output:**
```
==================================================================
SYNTHETIC TRAJECTORY DATA INSPECTION
==================================================================
File: /path/to/data/simulatedTrajectories_Custom_240Hz.pickle
Size: 156.32 MB
==================================================================

✓ Loaded successfully

📊 DATASET STATISTICS:
------------------------------------------------------------------
Number of segments:  3847
Data type:           <class 'torch.Tensor'>
Number of markers:   43

📏 SEGMENT LENGTH STATISTICS:
------------------------------------------------------------------
  Minimum:           60 frames
  Maximum:           240 frames
  Mean:              182.4 frames
  Median:            210 frames
  Std deviation:     47.3 frames

  Length distribution:
      0-60 frames:     0 segments (0.0%)
     60-120 frames:  412 segments (10.7%)
    120-180 frames:  789 segments (20.5%)
    180-240 frames: 2646 segments (68.8%)

🗺️  COORDINATE STATISTICS:
------------------------------------------------------------------
  X-axis (mm):
    Min:     -1847.3
    Max:      2134.6
    Mean:       412.8
    Std:        523.4
  Y-axis (mm):
    Min:     -1245.8
    Max:      1389.2
    Mean:         1.3
    Std:        401.7
  Z-axis (mm):
    Min:      -234.5
    Max:      1876.4
    Mean:       834.2
    Std:        356.8

💾 MEMORY USAGE:
------------------------------------------------------------------
  In-memory size:    156.32 MB

==================================================================
✓ Data appears valid and ready for training
==================================================================
```

### After Training the Model

```
data/
├── bodykinematics.hdf5
├── MarkerSet_Custom.xml
├── simulatedTrajectories_Custom_240Hz.pickle
├── model_2026-01-19.ckpt                      # ← Trained model
├── trainingvals_2026-01-19.pickle             # ← Training values
└── training_stats_2026-01-19.pickle           # ← Loss history
```

**Command used:**
```bash
cd ..
python trainAlgorithm.py
```

With `trainAlgorithm.py` configured as:
```python
# Training script configuration
fld = './data/'
datapath = './data/simulatedTrajectories_Custom_240Hz.pickle'
markersetpath = './data/MarkerSet_Custom.xml'
fs = 240
num_epochs = 10
prevModel = None
windowSize = 120
```

**Expected output:**
```
Loaded simulated trajectory training data
Training epoch 1/10...
  Batch 100/385 - Loss: 1.234
  Batch 200/385 - Loss: 0.876
  ...
Epoch 1/10 complete - Avg loss: 0.543
...
Epoch 10/10 complete - Avg loss: 0.087
Model saved to /path/to/data/
Algorithm trained in 3847.2 seconds (64.1 minutes)
```

### Final Setup for GUI

Edit `markerLabelGUI.py` (lines 27-29):
```python
modelpath = './data/model_2026-01-19.ckpt'
trainvalpath = './data/trainingvals_2026-01-19.pickle'
markersetpath = './data/MarkerSet_Custom.xml'
```

Run GUI:
```bash
python markerLabelGUI.py
```

Open browser to: http://localhost:8050

## Complete Command Timeline

Here's the exact sequence of commands for this example:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download body kinematics
cd data/
wget https://zenodo.org/record/4293999/files/bodykinematics.hdf5
cd ..

# 3. Verify download
cd generating_synthetic_data/
python check_hdf5_structure.py ../data/bodykinematics.hdf5

# 4. Create/convert marker set (one of these options)
# Option A: Create in OpenSim GUI, save as MarkerSet_Custom.xml
# Option B: Convert from VST (if conversion script available)

# 5. Generate synthetic data
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet_Custom.xml \
    --output ../data/simulatedTrajectories_Custom_240Hz.pickle \
    --align-right RSHO \
    --align-left LSHO \
    --fs 240 \
    --num-participants 100

# 6. Verify generated data
python inspect_synthetic_data.py ../data/simulatedTrajectories_Custom_240Hz.pickle --verbose

# 7. Train model
cd ..
# Edit trainAlgorithm.py to set paths
python trainAlgorithm.py

# 8. Use model in GUI
# Edit markerLabelGUI.py to set model paths
python markerLabelGUI.py
```

## Time Estimates

| Step | Time (approx) | Can run overnight? |
|------|---------------|-------------------|
| Download bodykinematics.hdf5 | 10-30 min | No (needs monitoring) |
| Create marker set in OpenSim | 30-60 min | No (interactive) |
| Generate synthetic data (100p) | 10-30 min | No (quick enough) |
| Train model (10 epochs, GPU) | 1-3 hours | Yes |
| Train model (10 epochs, CPU) | 10-30 hours | Yes |

## File Size Reference

| File | Typical Size | Notes |
|------|-------------|-------|
| bodykinematics.hdf5 | 3-5 GB | One-time download |
| MarkerSet_Custom.xml | 5-20 KB | Small text file |
| simulatedTrajectories_*.pickle | 100-200 MB | Depends on marker count |
| model_*.ckpt | 30-50 MB | Neural network weights |
| trainingvals_*.pickle | <1 MB | Normalization parameters |
| training_stats_*.pickle | <1 MB | Loss history |

## Troubleshooting Your Workflow

### My marker set has 43 markers but generation only uses 38

**Likely cause:** 5 markers are attached to body segments not in the HDF5 file (e.g., head markers on a "skull" segment)

**Solution:**
1. Check HDF5 segments: `python check_hdf5_structure.py ../data/bodykinematics.hdf5`
2. Edit marker set XML to use available segments
3. Common mappings:
   - head/skull → torso
   - hand markers → hand_r / hand_l (not fingers)

### Generation completes but inspect shows many NaNs

**Likely cause:** Marker local coordinates are unrealistic (e.g., 10 meters from segment origin)

**Solution:**
1. Open marker set in OpenSim
2. Visualize markers on body model
3. Adjust local coordinates to be within ±0.5 m of segment origin

### Training loss not decreasing

**Likely cause:** Data quality issue or hyperparameter problem

**Solution:**
1. Verify data: `python inspect_synthetic_data.py --check-nans`
2. Start with fewer epochs to test: `num_epochs = 2`
3. Check that marker set matches training data
4. Consider using pretrained model for transfer learning

### GUI predictions are poor

**Likely causes:**
1. Model undertrained (need more epochs)
2. Real data very different from synthetic
3. Marker set mismatch

**Solutions:**
1. Train for more epochs (20-30)
2. Use transfer learning with labeled real data
3. Ensure marker set exactly matches C3D files

## Success Criteria

At each step, you should see:

✅ **After HDF5 download:** File size ~4 GB, check_hdf5_structure shows 100 participants

✅ **After marker set creation:** XML file exists, contains all your markers with correct names

✅ **After synthetic generation:** ~3000-4000 segments, correct marker count, no/few NaNs

✅ **After training:** Loss decreases from ~2.0 to <0.2 over 10 epochs

✅ **In GUI:** Model correctly labels most markers (>80% accuracy) on test data

## Next Steps After Success

1. **Label real data:** Use GUI to label your actual mocap C3D files
2. **Transfer learning:** Improve model with real labeled data
3. **Batch processing:** Label multiple trials efficiently
4. **Quality control:** Review low-confidence predictions
5. **Export:** Save labeled C3D files for analysis

## Getting Help

If stuck at any step:
1. Check the error message carefully
2. Review README.md and QUICKSTART.md
3. Verify all file paths are correct
4. Check marker names match exactly (case-sensitive)
5. Ask for help with specific error messages
