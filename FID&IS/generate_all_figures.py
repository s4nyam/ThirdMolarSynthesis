#!/usr/bin/env python3
"""
Master Figure Generation Script for MolarSynTh Paper

This script runs both mosaic and individual figure generation with options
for selective generation based on user needs.

Usage:
    python generate_all_figures.py                    # Generate everything
    python generate_all_figures.py --mosaics-only     # Only mosaics
    python generate_all_figures.py --individual-only  # Only individual figures
    python generate_all_figures.py --quick            # Quick mode (subset)

Author: Generated for MolarSynTh Paper
Date: September 18, 2025
"""

import os
import sys
import argparse
from pathlib import Path

def run_mosaic_generation(base_dir):
    """Run the mosaic generation script"""
    print("=" * 60)
    print("RUNNING MOSAIC GENERATION")
    print("=" * 60)
    
    # Import and run mosaic generator
    sys.path.append(str(base_dir))
    
    try:
        from create_paper_mosaics import MosaicGenerator
        generator = MosaicGenerator(base_dir)
        generator.create_all_mosaics()
        print("✅ Mosaic generation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in mosaic generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_individual_generation(base_dir):
    """Run the individual figures generation script"""
    print("=" * 60)
    print("RUNNING INDIVIDUAL FIGURE GENERATION")
    print("=" * 60)
    
    # Import and run individual figure generator
    sys.path.append(str(base_dir))
    
    try:
        from create_individual_figures import IndividualFigureGenerator
        generator = IndividualFigureGenerator(base_dir)
        generator.create_all_individual_figures()
        print("✅ Individual figure generation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in individual figure generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_generation(base_dir):
    """Run a quick subset of important figures"""
    print("=" * 60)
    print("RUNNING QUICK GENERATION (SUBSET)")
    print("=" * 60)
    
    sys.path.append(str(base_dir))
    
    try:
        # Generate key mosaics
        from create_paper_mosaics import MosaicGenerator
        mosaic_gen = MosaicGenerator(base_dir)
        
        print("Creating key mosaics...")
        mosaic_gen.create_method_comparison_summary()
        mosaic_gen.create_model_comparison_grid()
        mosaic_gen.create_quality_progression_mosaic()
        
        # Generate key individual figures
        from create_individual_figures import IndividualFigureGenerator
        individual_gen = IndividualFigureGenerator(base_dir)
        
        print("Creating key individual figures...")
        individual_gen.create_performance_charts()
        individual_gen.create_method_comparison_grids()
        
        print("✅ Quick generation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in quick generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(base_dir):
    """Print a summary of generated files"""
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    
    base_path = Path(base_dir)
    
    # Check mosaic outputs
    mosaic_dir = base_path / "mosaic_outputs"
    if mosaic_dir.exists():
        mosaic_files = list(mosaic_dir.glob("*.png"))
        print(f"\n📊 MOSAICS ({len(mosaic_files)} files):")
        print(f"   Location: {mosaic_dir}")
        for file in sorted(mosaic_files):
            print(f"   • {file.name}")
    
    # Check individual figure outputs
    individual_dir = base_path / "individual_figures"
    if individual_dir.exists():
        print(f"\n🔍 INDIVIDUAL FIGURES:")
        print(f"   Location: {individual_dir}")
        
        subdirs = {
            'class_samples': 'Class-specific comparisons',
            'model_comparisons': 'Method comparisons',
            'performance_charts': 'Quantitative charts',
            'complexity_analysis': 'Complexity analysis',
            'method_showcases': 'Model showcases'
        }
        
        total_individual = 0
        for subdir_name, description in subdirs.items():
            subdir_path = individual_dir / subdir_name
            if subdir_path.exists():
                files = list(subdir_path.glob("*.png"))
                total_individual += len(files)
                print(f"   📁 {subdir_name}/ ({len(files)} files) - {description}")
                for file in sorted(files)[:3]:  # Show first 3 files
                    print(f"      • {file.name}")
                if len(files) > 3:
                    print(f"      • ... and {len(files) - 3} more")
        
        print(f"\n   Total individual figures: {total_individual}")
    
    print(f"\n📖 DOCUMENTATION:")
    readme_mosaic = mosaic_dir / "README_VISUALIZATIONS.md"
    readme_individual = individual_dir / "README_INDIVIDUAL_FIGURES.md"
    
    if readme_mosaic.exists():
        print(f"   • {readme_mosaic} - Mosaic usage guide")
    if readme_individual.exists():
        print(f"   • {readme_individual} - Individual figures guide")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("   • Use mosaics for comprehensive overviews")
    print("   • Use individual figures for specific analyses")
    print("   • Check README files for usage guidelines")
    print("   • All figures are 300 DPI publication-ready")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Generate all figures for MolarSynTh research paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_figures.py                    # Generate everything
  python generate_all_figures.py --mosaics-only     # Only mosaics  
  python generate_all_figures.py --individual-only  # Only individual figures
  python generate_all_figures.py --quick            # Quick subset generation
        """
    )
    
    parser.add_argument('--base_dir', type=str, default='/home/sanyam/playground/3m-FIDandIS',
                        help='Base directory containing the dataset folders')
    parser.add_argument('--mosaics-only', action='store_true',
                        help='Generate only mosaic visualizations')
    parser.add_argument('--individual-only', action='store_true',
                        help='Generate only individual figures')
    parser.add_argument('--quick', action='store_true',
                        help='Generate only key figures (faster)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Skip the summary output')
    
    args = parser.parse_args()
    
    # Validate base directory
    if not os.path.exists(args.base_dir):
        print(f"❌ Error: Base directory {args.base_dir} does not exist")
        return 1
    
    print("🎨 MolarSynTh Figure Generation Suite")
    print(f"📁 Base directory: {args.base_dir}")
    
    success = True
    
    # Determine what to generate
    if args.quick:
        success = run_quick_generation(args.base_dir)
    elif args.mosaics_only:
        success = run_mosaic_generation(args.base_dir)
    elif args.individual_only:
        success = run_individual_generation(args.base_dir)
    else:
        # Generate everything
        success_mosaic = run_mosaic_generation(args.base_dir)
        success_individual = run_individual_generation(args.base_dir)
        success = success_mosaic and success_individual
    
    # Print summary unless disabled
    if not args.no_summary:
        print_summary(args.base_dir)
    
    if success:
        print("\n🎉 All figure generation completed successfully!")
        print("   Ready for paper submission! 📄✨")
        return 0
    else:
        print("\n❌ Some errors occurred during generation.")
        print("   Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())