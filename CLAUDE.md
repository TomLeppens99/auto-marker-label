# CLAUDE.md - AI Assistant Guide for Auto-Marker-Label

## Project Overview

This repository contains an open-source platform for automatic labelling of motion capture markers using deep learning. The system uses a multi-layer LSTM neural network to predict marker labels from 3D trajectory data.

**Academic Citation:**
A. L. Clouthier, G. B. Ross, M. P. Mavor, I. Coll, A. Boyle and R. B. Graham, "Development and Validation of a Deep Learning Algorithm and Open-Source Platform for the Automatic Labelling of Motion Capture Markers," in IEEE Access, doi: [10.1109/ACCESS.2021.3062748](https://doi.org/10.1109/ACCESS.2021.3062748).

**Key Capabilities:**
- Train deep learning models on existing or simulated marker trajectory data
- Automatically label optical motion capture markers in C3D files
- Interactive GUI for visualization, correction, and export
- Transfer learning for continuous model improvement

## Repository Structure

```
auto-marker-label/
├── automarkerlabel.py          # Core library with all ML functions
├── trainAlgorithm.py           # Training script for new models
├── generateSimTrajectories.py  # Generate simulated training data
├── transferLearning.py         # Transfer learning script
├── markerLabelGUI.py          # Dash-based web GUI
├── requirements.txt            # Python dependencies
├── README.md                   # User documentation
├── LICENSE                     # License file
├── data/                       # Data directory
│   ├── MarkerSet.xml          # OpenSim marker set definition
│   ├── Rajagopal2015_mod.osim # OpenSim model file
│   ├── *.c3d                  # Sample C3D files
│   ├── *.ckpt                 # Trained model checkpoints
│   ├── *.pickle               # Training values and data
│   └── transfer_learning_data/ # Transfer learning datasets
├── assets/                     # GUI assets (CSS)
└── images/                     # Documentation images
```

## Core Components

### 1. automarkerlabel.py (67KB, ~1500 lines)

The main library containing all functionality. Key components:

#### Main Functions
- `generateSimTrajectories()` - Generate synthetic marker trajectories from body kinematics
- `trainAlgorithm()` - Train the neural network on labeled or simulated data
- `transferLearning()` - Perform transfer learning with new data
- `marker_label()` - Predict labels for unlabeled marker data

#### Data Import/Export Functions
- `import_markerSet()` - Parse OpenSim XML marker set files (line 356)
- `import_labelled_c3ds()` - Load labeled C3D files for training (line 498)
- `import_raw_c3d()` - Load unlabeled C3D files for prediction (line 573)
- `export_labelled_c3d()` - Export labeled data to C3D format (line 1493)

#### Data Processing Functions
- `align()` - Rotate data so subject faces +X direction (line 401)
- `window_data()` - Segment continuous data into analysis windows (line 445)
- `get_trainingVals()` - Calculate scaling values for neural network inputs (line 685)
- `procrustes()` - Procrustes alignment for trajectory matching (line 1389)

#### Neural Network Components
- `Net` class (line 859) - PyTorch neural network architecture
- `markerdata` class (line 779) - Custom PyTorch Dataset
- `train_nn()` - Neural network training loop (line 875)
- `predict_nn()` - Neural network inference (line 977)

### 2. trainAlgorithm.py

Entry point script for training new models. Key parameters:
- `fld` - Save directory for trained models
- `datapath` - Path to training data (.pickle for simulated, folder for C3D files)
- `markersetpath` - Path to OpenSim marker set XML
- `fs` - Sampling frequency (default: 240 Hz)
- `num_epochs` - Training epochs (default: 10)
- `prevModel` - Path to pretrained model for transfer learning (optional)
- `windowSize` - Data window size in frames (default: 120)
- `alignMkR` / `alignMkL` - Right/left alignment markers (e.g., 'RAC', 'LAC')

### 3. generateSimTrajectories.py

Generates synthetic marker trajectories from body segment kinematics. Requires:
- Body kinematics HDF5 file (download from Zenodo: 10.5281/zenodo.4293999)
- OpenSim marker set XML defining marker positions
- Outputs pickle file with simulated trajectories

### 4. markerLabelGUI.py

Dash-based web GUI for interactive marker labeling (line 1-34 contains key parameters):
- `modelpath` - Trained model checkpoint path
- `trainvalpath` - Training values pickle path
- `markersetpath` - Marker set XML path
- `gapfillsize` - Max gap size for interpolation (default: 24 frames)
- `windowSize` - Analysis window size (default: 120 frames)

### 5. transferLearning.py

Script for transfer learning to improve models with new labeled data.

## Neural Network Architecture

### Model: Multi-Layer LSTM with Fully Connected Head

**Architecture Parameters (automarkerlabel.py:36-46):**
```python
batch_size = 100
nLSTMcells = 256        # LSTM hidden size
nLSTMlayers = 3         # Number of LSTM layers
LSTMdropout = 0.17      # Dropout rate
FCnodes = 128           # Fully connected layer nodes
lr = 0.078              # Learning rate
momentum = 0.65         # SGD momentum
filtfreq = 6            # Low-pass filter cutoff (Hz)
```

**Network Structure (automarkerlabel.py:859-873):**
1. **Input:** For each marker, relative distances, velocities, and accelerations to all other markers
   - Shape: `(num_markers-1) * 5` features per marker
2. **LSTM Layers:** 3 layers with 256 cells each, 17% dropout
3. **Fully Connected Layers:**
   - Linear: `max_len * nLSTMcells → 128`
   - BatchNorm1d
   - ReLU activation
   - Linear: `128 → num_markers` (classification output)

**Loss Function:** Cross-entropy loss
**Optimizer:** SGD with momentum

## Data Formats

### C3D Files
- Industry-standard motion capture format
- Imported/exported using `ezc3d` library
- Contains 3D marker trajectories and metadata
- Sampling frequency typically 240 Hz

### Marker Set XML (OpenSim Format)
- Defines marker names and body segment attachments
- Required for training and prediction
- Created/edited using OpenSim software
- Located in `data/MarkerSet.xml`

### Pickle Files
- **Training data:** `simulatedTrajectories.pickle` - List of PyTorch tensors (num_frames × num_markers × 3)
- **Training values:** `trainingvals_*.pickle` - Dict with `segdists`, `scaleVals`, `max_len`
- **Model checkpoints:** `*.ckpt` - PyTorch state dicts

### HDF5 Files
- Body kinematics data for trajectory simulation
- Hierarchical structure: `/kin/sid/trial/segment`
- Also contains `/com/` and `/scale/` data for body segments

## Development Workflows

### Initial Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Tested with Python 3.12

2. **Download Body Kinematics Data (if generating simulated data):**
   - Get `bodykinematics.hdf5` from Zenodo (DOI: 10.5281/zenodo.4293999)
   - Place in `data/` folder

3. **Define Marker Set:**
   - Install OpenSim
   - Open `data/Rajagopal2015_mod.osim`
   - Add/modify markers in OpenSim GUI
   - Save marker set to `data/MarkerSet.xml`

### Training a New Model

**Option A: Using Simulated Data**
1. Configure `generateSimTrajectories.py` parameters
2. Run: `python generateSimTrajectories.py`
3. Configure `trainAlgorithm.py` to use generated pickle file
4. Run: `python trainAlgorithm.py`
5. Training outputs:
   - `model_YYYY-MM-DD.ckpt` - Model checkpoint
   - `trainingvals_YYYY-MM-DD.pickle` - Training values
   - `training_stats_YYYY-MM-DD.pickle` - Loss history

**Option B: Using Existing Labeled C3D Files**
1. Place labeled C3D files in a folder
2. Configure `trainAlgorithm.py`:
   - Set `datapath` to folder containing C3Ds
   - Set `alignMkR` and `alignMkL` to alignment markers
3. Run: `python trainAlgorithm.py`

**Training Notes:**
- Uses GPU if available (CUDA)
- Can take hours to days depending on dataset size
- Saves model with current date in filename
- Monitor training loss in `training_stats_*.pickle`

### Using the GUI

1. **Setup:**
   - Update paths in `markerLabelGUI.py` (lines 27-29)
   - Run: `python markerLabelGUI.py`
   - Opens automatically in browser at `http://localhost:8050`

2. **Labeling Workflow:**
   - Enter folder path with C3D files
   - Select file from dropdown
   - Click "Load Data"
   - Adjust rotation angle if needed (align to +X)
   - Click "Label Markers"
   - Review predictions:
     - Hover for marker info
     - Use slider/arrow keys for frames
     - Check "Error Detection" section
   - Correct errors using "Modify Labels" section
   - Split trajectories if needed
   - Export to C3D

3. **Visualization Modes:**
   - **Confidence:** Color by prediction confidence
   - **Unlabelled:** Highlight unlabeled markers in red
   - **Segments:** Color by body segment

### Transfer Learning

1. Label and correct data using GUI
2. Export corrected C3D files
3. Configure `transferLearning.py`:
   - Set `modelpath` to existing model
   - Set `trainvalpath` to matching training values
   - Set `datapath` to folder with new labeled C3Ds
4. Run: `python transferLearning.py`
5. Outputs new model with current date

## Key Conventions and Best Practices

### Code Style
- **Author:** aclouthi@uottawa.ca
- **Encoding:** UTF-8
- **Docstrings:** NumPy style with Parameters/Returns sections
- **Comments:** Inline for complex algorithms
- **Line Length:** Generally <100 characters, flexible for readability

### File Naming Conventions
- Models: `model_YYYY-MM-DD.ckpt`
- Training values: `trainingvals_YYYY-MM-DD.pickle`
- Training stats: `training_stats_YYYY-MM-DD.pickle`
- Simulated data: `simulatedTrajectories.pickle`

### Data Conventions
- **Coordinate System:** Z-up (vertical), subject faces +X after alignment
- **Units:** Millimeters for positions
- **Alignment Markers:** Typically acromions (RAC/LAC) or pelvis markers
- **Window Size:** 120 frames default (0.5 seconds at 240 Hz)
- **Gap Filling:** Cubic spline interpolation for gaps < `gapfillsize`

### Neural Network Conventions
- **Input Normalization:** Scale by mean distances/velocities/accelerations from training set
- **Missing Data:** Handled as NaN, excluded from calculations
- **Batch Processing:** Batch size 100
- **Sequence Padding:** Pack/unpack sequences for variable-length inputs
- **Label Assignment:** Hungarian algorithm (linear_sum_assignment) for optimal matching

### Important Technical Details

1. **Data Alignment:** Critical for consistent predictions
   - Always align data so subject faces +X direction
   - Use bilateral markers (left/right acromions) for rotation calculation
   - Rotation applied about Z-axis (vertical)

2. **Trajectory Splitting:** Handles marker swaps/disappearances
   - Automatically splits trajectories at large gaps (>60 frames)
   - Checks distance jumps vs. nearby markers
   - Creates new trajectories when markers likely swapped

3. **Label Verification:** Post-prediction validation
   - Checks inter-marker distances within body segments
   - Uses mean ± std from training data
   - Flags duplicates and unlabeled markers

4. **Filter Settings:** Low-pass Butterworth filter
   - 2nd order, 6 Hz cutoff
   - Applied to training and prediction data
   - Removes high-frequency noise from marker trajectories

## Common Tasks for AI Assistants

### Debugging Training Issues
- Check paths in training scripts (absolute vs. relative)
- Verify marker set XML matches data
- Ensure alignment markers exist in dataset
- Check CUDA availability for GPU training
- Monitor loss in `training_stats_*.pickle`

### Modifying Neural Network
- Architecture defined in automarkerlabel.py:859-873
- Hyperparameters at lines 36-46
- Must retrain if architecture changes
- Update both training and prediction code

### Adding New Markers
1. Edit marker set in OpenSim
2. Save new MarkerSet.xml
3. Regenerate simulated data or relabel C3D files
4. Retrain model with new marker set
5. Update GUI with new model/marker set

### Extending Functionality
- Core library is automarkerlabel.py - import functions
- GUI is modular Dash app - add callbacks for new features
- Training/prediction logic separated - modify independently
- C3D I/O uses ezc3d - check documentation for format details

### Troubleshooting GUI
- Check console for Dash errors
- Verify file paths are absolute or relative to script location
- Ensure C3D files have expected structure
- Check browser console for JavaScript errors
- Port 8050 must be available

### Version Compatibility
- **Python:** 3.12 (tested)
- **PyTorch:** 2.5.1 - CPU/CUDA compatible
- **Dash:** 2.18.2 - breaking changes between major versions
- **ezc3d:** 1.5.18 - C3D format library
- **NumPy:** 2.2.2 - updated for recent Python versions
- **OpenSim:** Any recent version for marker set editing

## Testing and Validation

### Sample Data
Located in `data/`:
- `S01_DropJump_mksrem.c3d` - Drop jump trial
- `S01_Pronetokneel01_mksrem.c3d` - Prone to kneel
- `S09_KneeltoRun02.c3d` - Kneel to run transition
- `S12_Tbalance_L.c3d` - Balance task

### Pre-trained Model
- `model_sim_add9_2020-09-15.ckpt` - Trained on simulated data
- `trainingvals_sim_add9_2020-09-15.pickle` - Corresponding training values

### Manual Validation
- Visual inspection in GUI critical for quality assurance
- Check "Error Detection" output for duplicates/unlabeled
- Compare confidence scores - low confidence indicates uncertainty
- Verify anatomically plausible marker positions

## Performance Considerations

### Training Performance
- **GPU strongly recommended** - 10-100x faster than CPU
- Training time: Hours to days depending on dataset size
- Reduce `num_participants` in simulation to speed up initial testing
- Use `tempCkpt` parameter to save progress incrementally

### Prediction Performance
- Real-time labeling not feasible - preprocessing required
- Window-based prediction allows parallel processing
- GUI updates can be slow for long trials (>10,000 frames)

### Memory Usage
- Large datasets may require >16GB RAM
- Models are ~38MB (checkpoint size)
- C3D files vary widely (1-40MB typical)

## Git Workflow

### Branch Naming
- Feature branches: `claude/feature-description-<sessionId>`
- Bug fixes: `claude/fix-description-<sessionId>`
- Documentation: `claude/docs-description-<sessionId>`

### Commit Messages
- Imperative mood: "Add feature" not "Added feature"
- First line: brief summary (<50 chars)
- Blank line, then detailed description if needed
- Reference issues if applicable

### Files to Avoid Committing
- `.ckpt` files (models are large, version separately)
- `.pickle` files with training data (large datasets)
- `bodykinematics.hdf5` (available on Zenodo)
- User-specific C3D files
- `__pycache__/` directories
- Virtual environment folders

## External Resources

- **OpenSim:** https://simtk.org/projects/opensim
- **Body Kinematics Data:** https://doi.org/10.5281/zenodo.4293999
- **Paper:** https://doi.org/10.1109/ACCESS.2021.3062748
- **PyTorch Docs:** https://pytorch.org/docs/stable/
- **Dash Docs:** https://dash.plotly.com/
- **ezc3d:** https://github.com/pyomeca/ezc3d

## Contact and Citation

**Author:** aclouthi@uottawa.ca

**Citation:**
```
@article{clouthier2021development,
  title={Development and Validation of a Deep Learning Algorithm and Open-Source Platform for the Automatic Labelling of Motion Capture Markers},
  author={Clouthier, A. L. and Ross, G. B. and Mavor, M. P. and Coll, I. and Boyle, A. and Graham, R. B.},
  journal={IEEE Access},
  year={2021},
  doi={10.1109/ACCESS.2021.3062748}
}
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Maintained for:** AI assistants working with this codebase
