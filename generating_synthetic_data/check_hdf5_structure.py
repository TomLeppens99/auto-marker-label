#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect the structure and contents of the body kinematics HDF5 file.

This script helps verify that the bodykinematics.hdf5 file is properly
downloaded and contains the expected data structure.

Usage:
    python check_hdf5_structure.py ../data/bodykinematics.hdf5

Author: aclouthi@uottawa.ca
"""

import sys
import os
import argparse
import h5py
import numpy as np


def print_group_structure(group, indent=0):
    """Recursively print HDF5 group structure."""
    for key in group.keys():
        item = group[key]
        prefix = "  " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}📁 {key}/")
            if indent < 3:  # Limit recursion depth
                print_group_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print(f"{prefix}📄 {key} - shape: {item.shape}, dtype: {item.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect body kinematics HDF5 file structure'
    )
    parser.add_argument('hdf5_file', type=str,
                        help='Path to bodykinematics.hdf5')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed information')

    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file):
        print(f"ERROR: File not found: {args.hdf5_file}")
        print()
        print("Download the body kinematics data from:")
        print("https://doi.org/10.5281/zenodo.4293999")
        sys.exit(1)

    print("=" * 70)
    print("BODY KINEMATICS HDF5 FILE INSPECTION")
    print("=" * 70)
    print(f"File: {os.path.abspath(args.hdf5_file)}")
    print(f"Size: {os.path.getsize(args.hdf5_file) / (1024**3):.2f} GB")
    print("=" * 70)
    print()

    try:
        with h5py.File(args.hdf5_file, 'r') as hf:
            # Top-level groups
            print("📦 TOP-LEVEL GROUPS:")
            print("-" * 70)
            for key in hf.keys():
                print(f"  • {key}/")
            print()

            # Kinematics data
            if 'kin' in hf:
                print("🏃 KINEMATICS DATA:")
                print("-" * 70)
                sids = list(hf['kin'].keys())
                print(f"Number of participants: {len(sids)}")
                print(f"Participant IDs: {', '.join(sids[:5])}{'...' if len(sids) > 5 else ''}")
                print()

                # Sample one participant
                sample_sid = sids[0]
                trials = list(hf['kin'][sample_sid].keys())
                print(f"Example participant: {sample_sid}")
                print(f"  Trials: {len(trials)}")
                print(f"  Trial names: {', '.join(trials[:5])}{'...' if len(trials) > 5 else ''}")
                print()

                # Sample one trial
                sample_trial = trials[0]
                segments = list(hf['kin'][sample_sid][sample_trial].keys())
                print(f"  Example trial: {sample_trial}")
                print(f"    Body segments: {len(segments)}")
                for seg in segments:
                    data = hf['kin'][sample_sid][sample_trial][seg]
                    print(f"      • {seg:15s} - shape: {data.shape}")
                print()

            # Center of mass data
            if 'com' in hf:
                print("📍 CENTER OF MASS DATA:")
                print("-" * 70)
                com_segments = list(hf['com'].keys())
                print(f"Body segments: {len(com_segments)}")
                for seg in com_segments:
                    data = hf['com'][seg]
                    print(f"  • {seg:15s} - shape: {data.shape} (participants × 3)")
                print()

            # Scale data
            if 'scale' in hf:
                print("📏 SCALE DATA (Body Dimensions):")
                print("-" * 70)
                scale_segments = list(hf['scale'].keys())
                print(f"Body segments: {len(scale_segments)}")
                for seg in scale_segments:
                    data = hf['scale'][seg]
                    print(f"  • {seg:15s} - shape: {data.shape} (participants × 3)")
                print()

            # Verbose mode: show sample data
            if args.verbose:
                print("🔍 SAMPLE DATA:")
                print("-" * 70)
                if 'kin' in hf:
                    sample_data = np.array(hf['kin'][sids[0]][trials[0]]['torso'])
                    print(f"Sample transformation matrix (torso, frame 0):")
                    print(sample_data[:, :, 0])
                    print()

                if 'scale' in hf:
                    sample_scale = np.array(hf['scale']['torso'])
                    print(f"Sample scale factors (torso, first 3 participants):")
                    print(sample_scale[:3, :])
                    print()

            # Full structure
            if args.verbose:
                print("🗂️  FULL FILE STRUCTURE:")
                print("-" * 70)
                print_group_structure(hf)
                print()

        print("=" * 70)
        print("✓ File appears valid and properly formatted")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR reading file: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
