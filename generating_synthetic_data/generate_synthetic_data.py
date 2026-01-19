#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line wrapper for generating synthetic marker trajectory data.

This script provides an easy-to-use interface for the generateSimTrajectories()
function from automarkerlabel.py.

Usage:
    python generate_synthetic_data.py --bodykin ../data/bodykinematics.hdf5 \
                                       --markerset ../data/MarkerSet.xml \
                                       --output ../data/simulatedTrajectories.pickle \
                                       --align-right RAC --align-left LAC \
                                       --fs 240 --num-participants 100

Author: aclouthi@uottawa.ca
"""

import sys
import os
import argparse
import time

# Add parent directory to path to import automarkerlabel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import automarkerlabel as aml


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic marker trajectories for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data using all 100 participants
  python generate_synthetic_data.py \\
      --bodykin ../data/bodykinematics.hdf5 \\
      --markerset ../data/MarkerSet.xml \\
      --output ../data/simulatedTrajectories.pickle \\
      --align-right RAC --align-left LAC \\
      --fs 240 --num-participants 100

  # Quick test with 10 participants
  python generate_synthetic_data.py \\
      --bodykin ../data/bodykinematics.hdf5 \\
      --markerset ../data/MarkerSet_ASD.xml \\
      --output ../data/test_synthetic.pickle \\
      --align-right RSHO --align-left LSHO \\
      --fs 120 --num-participants 10 --max-len 180
        """
    )

    # Required arguments
    parser.add_argument('--bodykin', type=str, required=True,
                        help='Path to bodykinematics.hdf5 file')
    parser.add_argument('--markerset', type=str, required=True,
                        help='Path to OpenSim marker set XML file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for .pickle file')
    parser.add_argument('--align-right', type=str, required=True,
                        help='Right-side alignment marker name (e.g., RAC, RSHO, RASI)')
    parser.add_argument('--align-left', type=str, required=True,
                        help='Left-side alignment marker name (e.g., LAC, LSHO, LASI)')
    parser.add_argument('--fs', type=int, required=True,
                        help='Sampling frequency in Hz (e.g., 120, 240)')

    # Optional arguments
    parser.add_argument('--num-participants', type=int, default=100,
                        help='Number of participants to use (1-100, default: 100)')
    parser.add_argument('--max-len', type=int, default=240,
                        help='Maximum segment length in frames (default: 240)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.bodykin):
        print(f"ERROR: Body kinematics file not found: {args.bodykin}")
        print("Download from: https://doi.org/10.5281/zenodo.4293999")
        sys.exit(1)

    if not os.path.exists(args.markerset):
        print(f"ERROR: Marker set file not found: {args.markerset}")
        sys.exit(1)

    if args.num_participants < 1 or args.num_participants > 100:
        print(f"ERROR: num_participants must be between 1 and 100")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Display configuration
    print("=" * 70)
    print("SYNTHETIC TRAJECTORY GENERATION")
    print("=" * 70)
    print(f"Body kinematics:    {args.bodykin}")
    print(f"Marker set:         {args.markerset}")
    print(f"Output file:        {args.output}")
    print(f"Alignment markers:  {args.align_right} (R) / {args.align_left} (L)")
    print(f"Sampling frequency: {args.fs} Hz")
    print(f"Participants:       {args.num_participants} / 100")
    print(f"Max segment length: {args.max_len} frames")
    print("=" * 70)
    print()

    # Generate trajectories
    print("Starting generation...")
    start_time = time.time()

    try:
        data = aml.generateSimTrajectories(
            bodykinpath=args.bodykin,
            markersetpath=args.markerset,
            outputfile=args.output,
            alignMkR=args.align_right,
            alignMkL=args.align_left,
            fs=args.fs,
            num_participants=args.num_participants,
            max_len=args.max_len
        )

        elapsed_time = time.time() - start_time

        # Summary statistics
        print()
        print("=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total segments:     {len(data)}")
        print(f"Markers per segment: {data[0].shape[1]}")
        print(f"Segment lengths:    {min([x.shape[0] for x in data])} - {max([x.shape[0] for x in data])} frames")
        print(f"Time elapsed:       {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"Output saved to:    {os.path.abspath(args.output)}")
        print("=" * 70)
        print()
        print("Next step: Train the model using trainAlgorithm.py")

    except Exception as e:
        print(f"\nERROR during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
