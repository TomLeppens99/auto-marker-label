#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration template for synthetic data generation.

Copy this file, edit the parameters below, then run:
    python your_config.py

Author: aclouthi@uottawa.ca
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import automarkerlabel as aml

# ============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# ============================================================================

# Input files
BODYKIN_PATH = '../data/bodykinematics.hdf5'
MARKERSET_PATH = '../data/MarkerSet.xml'  # Change to your marker set XML

# Output file
OUTPUT_PATH = '../data/simulatedTrajectories.pickle'

# Alignment markers (must exist in your marker set)
# These should be bilateral markers on a rigid segment (shoulders or pelvis)
ALIGN_MARKER_RIGHT = 'RAC'  # Right acromion (or RSHO, RASI, etc.)
ALIGN_MARKER_LEFT = 'LAC'   # Left acromion (or LSHO, LASI, etc.)

# Data parameters
SAMPLING_FREQUENCY = 240  # Hz - match your mocap system (120, 240, etc.)
NUM_PARTICIPANTS = 100    # 1-100, more = better model but slower generation
MAX_SEGMENT_LENGTH = 240  # frames, segments longer than this will be split

# ============================================================================
# NOTES AND TIPS
# ============================================================================
"""
ALIGNMENT MARKERS:
- Choose bilateral markers (left/right pair) on a rigid segment
- Common choices:
  * Shoulders: RAC/LAC, RSHO/LSHO, RCLA/LCLA
  * Pelvis: RASI/LASI, RASIS/LASIS, RPSI/LPSI
- Check marker names in your XML: grep 'Marker name' YOUR_MARKERSET.xml

SAMPLING FREQUENCY:
- Must match the frequency of the mocap data you will label
- Common values: 120 Hz, 240 Hz, 100 Hz
- If unsure, check your C3D files in Vicon Nexus or similar software

NUM_PARTICIPANTS:
- More participants = larger training set = better model
- But also longer generation time (10-30 minutes for 100 participants)
- For quick testing: Use 10 participants (~2 minutes)
- For final model: Use 100 participants (~20 minutes)

OUTPUT PATH:
- Can be anywhere, but recommend ../data/ folder
- Use descriptive names if you generate multiple versions:
  * simulatedTrajectories_ASD_240Hz.pickle
  * simulatedTrajectories_FullSpine_120Hz.pickle
"""

# ============================================================================
# VALIDATION - CHECK BEFORE RUNNING
# ============================================================================

def validate_config():
    """Validate configuration before running generation."""
    errors = []
    warnings = []

    # Check files exist
    if not os.path.exists(BODYKIN_PATH):
        errors.append(f"Body kinematics file not found: {BODYKIN_PATH}")
        errors.append("  Download from: https://doi.org/10.5281/zenodo.4293999")

    if not os.path.exists(MARKERSET_PATH):
        errors.append(f"Marker set file not found: {MARKERSET_PATH}")

    # Check parameters
    if NUM_PARTICIPANTS < 1 or NUM_PARTICIPANTS > 100:
        errors.append(f"NUM_PARTICIPANTS must be 1-100, got {NUM_PARTICIPANTS}")

    if SAMPLING_FREQUENCY < 50 or SAMPLING_FREQUENCY > 500:
        warnings.append(f"Unusual sampling frequency: {SAMPLING_FREQUENCY} Hz")
        warnings.append("  Common values: 120, 240 Hz")

    if MAX_SEGMENT_LENGTH < 60 or MAX_SEGMENT_LENGTH > 500:
        warnings.append(f"Unusual max segment length: {MAX_SEGMENT_LENGTH} frames")
        warnings.append("  Typical range: 120-240 frames")

    # Check output directory exists
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        warnings.append(f"Output directory will be created: {output_dir}")

    return errors, warnings


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("SYNTHETIC TRAJECTORY GENERATION - CONFIGURATION")
    print("=" * 70)
    print()

    # Display configuration
    print("CONFIGURATION:")
    print("-" * 70)
    print(f"Body kinematics:    {BODYKIN_PATH}")
    print(f"Marker set:         {MARKERSET_PATH}")
    print(f"Output file:        {OUTPUT_PATH}")
    print(f"Alignment markers:  {ALIGN_MARKER_RIGHT} (R) / {ALIGN_MARKER_LEFT} (L)")
    print(f"Sampling frequency: {SAMPLING_FREQUENCY} Hz")
    print(f"Participants:       {NUM_PARTICIPANTS} / 100")
    print(f"Max segment length: {MAX_SEGMENT_LENGTH} frames")
    print()

    # Validate
    print("VALIDATION:")
    print("-" * 70)
    errors, warnings = validate_config()

    if warnings:
        print("⚠️  WARNINGS:")
        for w in warnings:
            print(f"    {w}")
        print()

    if errors:
        print("❌ ERRORS:")
        for e in errors:
            print(f"    {e}")
        print()
        print("Please fix errors and run again.")
        sys.exit(1)

    print("✓ Configuration valid")
    print()

    # Confirm
    response = input("Proceed with generation? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)

    print()
    print("=" * 70)
    print("STARTING GENERATION")
    print("=" * 70)
    print()

    # Create output directory if needed
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Generate
    import time
    start_time = time.time()

    data = aml.generateSimTrajectories(
        bodykinpath=BODYKIN_PATH,
        markersetpath=MARKERSET_PATH,
        outputfile=OUTPUT_PATH,
        alignMkR=ALIGN_MARKER_RIGHT,
        alignMkL=ALIGN_MARKER_LEFT,
        fs=SAMPLING_FREQUENCY,
        num_participants=NUM_PARTICIPANTS,
        max_len=MAX_SEGMENT_LENGTH
    )

    elapsed_time = time.time() - start_time

    # Summary
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total segments:      {len(data)}")
    print(f"Markers per segment: {data[0].shape[1]}")
    print(f"Time elapsed:        {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} min)")
    print(f"Output saved to:     {os.path.abspath(OUTPUT_PATH)}")
    print()
    print("Next steps:")
    print(f"  1. Inspect data: python inspect_synthetic_data.py {OUTPUT_PATH}")
    print(f"  2. Train model:  python ../trainAlgorithm.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
