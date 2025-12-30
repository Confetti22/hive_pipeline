#!/usr/bin/env python3
"""
Temporary script to randomly delete 50% of files in the z_slices directory.
This script will:
1. Find all .tiff files in the target directory
2. Randomly select 50% of them for deletion
3. Delete the selected files
4. Provide a summary of the operation
"""

import os
import random
import glob
from pathlib import Path

def main():
    # Target directory
    target_dir = "/home/confetti/data/rm009/boundary_seg/new_boundary_seg_data/z_slices"
    
    # Check if directory exists
    if not os.path.exists(target_dir):
        print(f"Error: Directory {target_dir} does not exist!")
        return
    
    # Find all .tiff files
    pattern = os.path.join(target_dir, "*.tiff")
    all_files = glob.glob(pattern)
    
    print(f"Found {len(all_files)} .tiff files in {target_dir}")
    
    if len(all_files) == 0:
        print("No .tiff files found to delete.")
        return
    
    # Calculate 50% of files to delete
    files_to_delete_count = len(all_files) // 2
    print(f"Will randomly delete {files_to_delete_count} files (50%)")
    
    # Randomly select files to delete
    files_to_delete = random.sample(all_files, files_to_delete_count)
    
    # Confirm before deletion
    response = input(f"Are you sure you want to delete {files_to_delete_count} files? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # Delete the selected files
    deleted_count = 0
    failed_count = 0
    
    print("Deleting files...")
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            if deleted_count % 1000 == 0:  # Progress indicator
                print(f"Deleted {deleted_count} files...")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            failed_count += 1
    
    # Summary
    print(f"\nOperation completed!")
    print(f"Successfully deleted: {deleted_count} files")
    print(f"Failed to delete: {failed_count} files")
    print(f"Remaining files: {len(all_files) - deleted_count}")
    
    # Verify final count
    remaining_files = glob.glob(pattern)
    print(f"Verification - Current file count: {len(remaining_files)}")

if __name__ == "__main__":
    main()
