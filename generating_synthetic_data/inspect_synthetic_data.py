#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect generated synthetic marker trajectory data.

This script verifies that the generated pickle file contains valid data
and displays statistics about the synthetic dataset.

Usage:
    python inspect_synthetic_data.py ../data/simulatedTrajectories.pickle

Author: aclouthi@uottawa.ca
"""

import sys
import os
import argparse
import pickle
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Inspect generated synthetic trajectory data'
    )
    parser.add_argument('pickle_file', type=str,
                        help='Path to .pickle file containing synthetic trajectories')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed statistics')
    parser.add_argument('--check-nans', action='store_true',
                        help='Check for NaN values in data')

    args = parser.parse_args()

    if not os.path.exists(args.pickle_file):
        print(f"ERROR: File not found: {args.pickle_file}")
        sys.exit(1)

    print("=" * 70)
    print("SYNTHETIC TRAJECTORY DATA INSPECTION")
    print("=" * 70)
    print(f"File: {os.path.abspath(args.pickle_file)}")
    print(f"Size: {os.path.getsize(args.pickle_file) / (1024**2):.2f} MB")
    print("=" * 70)
    print()

    try:
        # Load data
        print("Loading data...")
        with open(args.pickle_file, 'rb') as f:
            data = pickle.load(f)

        print(f"✓ Loaded successfully")
        print()

        # Basic statistics
        print("📊 DATASET STATISTICS:")
        print("-" * 70)
        print(f"Number of segments:  {len(data)}")
        print(f"Data type:           {type(data[0])}")

        if len(data) > 0:
            num_markers = data[0].shape[1]
            print(f"Number of markers:   {num_markers}")
            print()

            # Segment length statistics
            lengths = [seg.shape[0] for seg in data]
            print("📏 SEGMENT LENGTH STATISTICS:")
            print("-" * 70)
            print(f"  Minimum:           {min(lengths)} frames")
            print(f"  Maximum:           {max(lengths)} frames")
            print(f"  Mean:              {np.mean(lengths):.1f} frames")
            print(f"  Median:            {np.median(lengths):.0f} frames")
            print(f"  Std deviation:     {np.std(lengths):.1f} frames")
            print()

            # Length distribution
            bins = [0, 60, 120, 180, 240, 300]
            hist, _ = np.histogram(lengths, bins=bins)
            print("  Length distribution:")
            for i in range(len(bins) - 1):
                print(f"    {bins[i]:3d}-{bins[i+1]:3d} frames: {hist[i]:5d} segments ({100*hist[i]/len(data):.1f}%)")
            if max(lengths) > bins[-1]:
                count = sum(1 for l in lengths if l > bins[-1])
                print(f"    >{bins[-1]:3d} frames:       {count:5d} segments ({100*count/len(data):.1f}%)")
            print()

            # Coordinate statistics
            print("🗺️  COORDINATE STATISTICS:")
            print("-" * 70)

            # Flatten all data
            all_coords = torch.cat([seg.reshape(-1, 3) for seg in data], dim=0)
            all_coords_np = all_coords.numpy()

            for axis, name in enumerate(['X', 'Y', 'Z']):
                valid_coords = all_coords_np[~np.isnan(all_coords_np[:, axis]), axis]
                print(f"  {name}-axis (mm):")
                print(f"    Min:    {np.min(valid_coords):8.1f}")
                print(f"    Max:    {np.max(valid_coords):8.1f}")
                print(f"    Mean:   {np.mean(valid_coords):8.1f}")
                print(f"    Std:    {np.std(valid_coords):8.1f}")

            print()

            # Check for NaNs
            if args.check_nans:
                print("🔍 NaN ANALYSIS:")
                print("-" * 70)
                total_values = sum(seg.numel() for seg in data)
                total_nans = sum(torch.isnan(seg).sum().item() for seg in data)
                print(f"  Total values:      {total_values:,}")
                print(f"  NaN values:        {total_nans:,} ({100*total_nans/total_values:.3f}%)")

                if total_nans > 0:
                    print(f"  ⚠️  Warning: Data contains NaNs")
                    print(f"     This is unusual for synthetic data and may indicate an issue")
                else:
                    print(f"  ✓ No NaNs detected (expected for synthetic data)")
                print()

            # Verbose mode: show sample data
            if args.verbose:
                print("🔬 SAMPLE DATA (First segment, first 3 markers, first frame):")
                print("-" * 70)
                sample = data[0][0, :min(3, num_markers), :]
                for i, coords in enumerate(sample):
                    print(f"  Marker {i}: X={coords[0]:8.2f}, Y={coords[1]:8.2f}, Z={coords[2]:8.2f} mm")
                print()

                print("📈 DETAILED SEGMENT INFORMATION:")
                print("-" * 70)
                print(f"{'Segment':<10} {'Frames':<8} {'Shape':<20} {'Data Range (mm)':<25}")
                print("-" * 70)
                for i in range(min(10, len(data))):
                    seg = data[i]
                    seg_np = seg.numpy()
                    valid_data = seg_np[~np.isnan(seg_np)]
                    if len(valid_data) > 0:
                        data_range = f"{np.min(valid_data):.0f} to {np.max(valid_data):.0f}"
                    else:
                        data_range = "N/A"
                    print(f"{i:<10} {seg.shape[0]:<8} {str(seg.shape):<20} {data_range:<25}")
                if len(data) > 10:
                    print(f"... and {len(data) - 10} more segments")
                print()

        # Memory usage
        memory_mb = sum(seg.element_size() * seg.numel() for seg in data) / (1024**2)
        print("💾 MEMORY USAGE:")
        print("-" * 70)
        print(f"  In-memory size:    {memory_mb:.2f} MB")
        print()

        print("=" * 70)
        print("✓ Data appears valid and ready for training")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Use this data to train the model with trainAlgorithm.py")
        print(f"  2. Specify datapath='{args.pickle_file}' in training script")

    except Exception as e:
        print(f"\n❌ ERROR reading file: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
