# Quick Start Guide

Get synthetic training data generated in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Required packages installed: `pip install -r ../requirements.txt`
- [ ] Body kinematics HDF5 file downloaded (see below)
- [ ] Marker set XML file ready (OpenSim format)

## 1. Download Body Kinematics Data

**Option A: Direct Download (Recommended)**
```bash
cd ../data/
# If wget is available:
wget https://zenodo.org/record/4293999/files/bodykinematics.hdf5

# If curl is available:
curl -L -o bodykinematics.hdf5 https://zenodo.org/record/4293999/files/bodykinematics.hdf5
```

**Option B: Manual Download**
1. Visit: https://doi.org/10.5281/zenodo.4293999
2. Download `bodykinematics.hdf5` (several GB)
3. Place in `../data/` folder

**Verify download:**
```bash
python check_hdf5_structure.py ../data/bodykinematics.hdf5
```

## 2. Prepare Your Marker Set

**If you have a Vicon VST file:**

Upload your VST file (e.g., `ASD_FullSpine_TO_TL_T1cluster.txt`) to this folder, then we can create a conversion script together.

**If you have an OpenSim XML:**

Copy it to the data folder:
```bash
cp /path/to/your/MarkerSet.xml ../data/MarkerSet_Custom.xml
```

**If you want to use the example:**

The repository includes `../data/MarkerSet.xml` - this is a pre-defined marker set you can use for testing.

## 3. Identify Your Alignment Markers

You need two bilateral markers (left/right pair) for aligning the subject. Common choices:

| Body Region | Right Marker | Left Marker | Description |
|------------|--------------|-------------|-------------|
| Shoulders  | RAC / RSHO   | LAC / LSHO  | Acromions (most common) |
| Pelvis     | RASI / RASIS | LASI / LASIS | Anterior superior iliac spine |
| Torso      | RCLA         | LCLA        | Clavicle markers |

**Check your marker set:**
```bash
# View marker names in your XML file
grep 'Marker name' ../data/MarkerSet.xml
```

Choose one left/right pair that exists in your marker set.

## 4. Generate Synthetic Data

**Basic usage (recommended for first run):**
```bash
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet.xml \
    --output ../data/simulatedTrajectories.pickle \
    --align-right RAC \
    --align-left LAC \
    --fs 240 \
    --num-participants 100
```

**Quick test (10 participants, ~2 minutes):**
```bash
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet.xml \
    --output ../data/test_synthetic.pickle \
    --align-right RAC \
    --align-left LAC \
    --fs 240 \
    --num-participants 10
```

**For 120 Hz mocap systems:**
```bash
python generate_synthetic_data.py \
    --bodykin ../data/bodykinematics.hdf5 \
    --markerset ../data/MarkerSet.xml \
    --output ../data/simulatedTrajectories_120Hz.pickle \
    --align-right RAC \
    --align-left LAC \
    --fs 120 \
    --num-participants 100
```

## 5. Verify Generated Data

```bash
python inspect_synthetic_data.py ../data/simulatedTrajectories.pickle --verbose
```

Look for:
- ✓ Number of segments (should be ~3000-4000 for 100 participants)
- ✓ Correct number of markers
- ✓ No NaN values (or very few)
- ✓ Reasonable coordinate ranges (typically -2000 to 2000 mm)

## 6. Train Your Model

```bash
cd ..
python trainAlgorithm.py
```

Make sure `trainAlgorithm.py` is configured with:
```python
datapath = './data/simulatedTrajectories.pickle'
markersetpath = './data/MarkerSet.xml'
fld = './data/'
fs = 240  # Match what you used in generation
num_epochs = 10
```

**Expected training time:**
- With GPU: 1-3 hours for 10 epochs
- Without GPU: 10-30 hours for 10 epochs

## Common Issues and Solutions

### Issue: "File not found: bodykinematics.hdf5"
**Solution:** Download the file from Zenodo (see step 1)

### Issue: "Marker RAC not found in marker set"
**Solution:**
```bash
# Check available markers:
grep 'Marker name' ../data/MarkerSet.xml

# Update --align-right and --align-left to match your marker names
```

### Issue: "KeyError: 'torso'" or similar segment error
**Solution:** Your marker set references a body segment not in the HDF5 file. Available segments are:
- torso, pelvis
- femur_r, femur_l, tibia_r, tibia_l
- talus_r, talus_l, calcn_r, calcn_l, toes_r, toes_l
- humerus_r, humerus_l, ulna_r, ulna_l, radius_r, radius_l, hand_r, hand_l

Edit your marker set XML to use these segment names.

### Issue: Generation is very slow
**Solution:**
- Start with fewer participants: `--num-participants 25`
- Generate in batches: Run script 4 times with 25 participants each
- Note: This is normal - expect 10-30 minutes for 100 participants

### Issue: "Out of memory"
**Solution:** Reduce participants to 50 or 25

## Parameter Reference

| Parameter | Description | Typical Values | Required |
|-----------|-------------|----------------|----------|
| `--bodykin` | Path to HDF5 file | `../data/bodykinematics.hdf5` | Yes |
| `--markerset` | Path to marker set XML | `../data/MarkerSet.xml` | Yes |
| `--output` | Output pickle path | `../data/simulatedTrajectories.pickle` | Yes |
| `--align-right` | Right alignment marker | `RAC`, `RSHO`, `RASI` | Yes |
| `--align-left` | Left alignment marker | `LAC`, `LSHO`, `LASI` | Yes |
| `--fs` | Sampling frequency (Hz) | `120`, `240` | Yes |
| `--num-participants` | How many subjects | `10`-`100` | No (default: 100) |
| `--max-len` | Max segment length | `180`, `240` | No (default: 240) |

## Next Steps After Generation

1. **Verify data quality:**
   ```bash
   python inspect_synthetic_data.py ../data/simulatedTrajectories.pickle --check-nans
   ```

2. **Train the model:**
   ```bash
   cd ..
   python trainAlgorithm.py
   ```

3. **Test predictions:**
   ```bash
   python markerLabelGUI.py
   ```
   Then load a test C3D file and see how well the model labels it.

4. **Improve with transfer learning:**
   - Label real mocap data using the GUI
   - Correct any errors
   - Use `transferLearning.py` to improve the model

## Getting Help

- **README.md** - Comprehensive documentation
- **CLAUDE.md** - Technical details for AI assistants
- **GitHub Issues** - Report bugs or ask questions
- **Email** - aclouthi@uottawa.ca

## Tips for Best Results

1. **Use all 100 participants** - More data = better model
2. **Match sampling frequency** - Use the same fs as your real mocap data
3. **Choose good alignment markers** - Bilateral markers on rigid segments (shoulders/pelvis)
4. **Validate marker set** - Ensure segment names match HDF5 file structure
5. **Check generated data** - Always run `inspect_synthetic_data.py` before training
