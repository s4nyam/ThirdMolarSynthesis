import os
import pandas as pd
from tqdm import tqdm
import shutil
from datetime import datetime


# Dataset configuration
DATASET_CONFIG = {
    'ADLD': {
        'excel_path': '/home/sj/working_dir/third_molar_project/1.adld/file_list_split.xlsx',
        'original_path': '/home/sj/working_dir/third_molar_project/1.adld/adld_cropped',
        'renamed_path': '/home/sj/working_dir/third_molar_project/1.adld/adld_cropped_renamed'
    },
    'Dentex': {
        'excel_path': '/home/sj/working_dir/third_molar_project/2.dentex/file_list.xlsx',
        'original_path': '/home/sj/working_dir/third_molar_project/2.dentex/dentex_cropped',
        'renamed_path': '/home/sj/working_dir/third_molar_project/2.dentex/dentex_cropped_renamed'
    },
    'tsxk': {
        'excel_path': '/home/sj/working_dir/third_molar_project/3.tsxk/file_list.xlsx',
        'original_path': '/home/sj/working_dir/third_molar_project/3.tsxk/tsxk_cropped',
        'renamed_path': '/home/sj/working_dir/third_molar_project/3.tsxk/tsxk_cropped_renamed'
    },
    'tufts': {
        'excel_path': '/home/sj/working_dir/third_molar_project/4.tufts/file_list.xlsx',
        'original_path': '/home/sj/working_dir/third_molar_project/4.tufts/tufts_cropped',
        'renamed_path': '/home/sj/working_dir/third_molar_project/4.tufts/tufts_cropped_renamed'
    },
    'uspforp': {
        'excel_path': '/home/sj/working_dir/third_molar_project/5.uspforp/file_list.xlsx',
        'original_path': '/home/sj/working_dir/third_molar_project/5.uspforp/uspforp_cropped',
        'renamed_path': '/home/sj/working_dir/third_molar_project/5.uspforp/uspforp_cropped_renamed'
    }
}

# File list with dataset mapping
FILE_LIST = [
    ('1000_38.png', 'ADLD'),
    ('1000_48_flipped.png', 'ADLD'),
    ('1001_48_flipped.png', 'ADLD'),
    ('1002_38.png', 'ADLD'),
    ('1009_38.png', 'ADLD'),
    ('1014_48_flipped.png', 'ADLD'),
    ('1021_38.png', 'ADLD'),
    ('1025_48_flipped.png', 'ADLD'),
    ('1033_48_flipped.png', 'ADLD'),
    ('1079_48_flipped.png', 'ADLD'),
    ('train_16_37_flipped.png', 'Dentex'),
    ('train_25_37_flipped.png', 'Dentex'),
    ('train_27_37_flipped.png', 'Dentex'),
    ('train_31_37_flipped.png', 'Dentex'),
    ('train_32_27.png', 'Dentex'),
    ('train_32_37_flipped.png', 'Dentex'),
    ('train_36_37_flipped.png', 'Dentex'),
    ('train_37_27.png', 'Dentex'),
    ('train_39_37_flipped.png', 'Dentex'),
    ('train_74_37_flipped.png', 'Dentex'),
    ('37_17.png', 'tsxk'),
    ('39_32_flipped.png', 'tsxk'),
    ('43_17.png', 'tsxk'),
    ('73_17.png', 'tsxk'),
    ('95_32_flipped.png', 'tsxk'),
    ('133_17.png', 'tsxk'),
    ('145_32_flipped.png', 'tsxk'),
    ('160_17.png', 'tsxk'),
    ('259_32_flipped.png', 'tsxk'),
    ('264_17.png', 'tsxk'),
    ('126-M-32_48_flipped.png', 'uspforp'),
    ('131-M-21_48_flipped.png', 'uspforp'),
    ('141-M-30_38.png', 'uspforp'),
    ('144-M-48_38.png', 'uspforp'),
    ('147-M-28_48_flipped.png', 'uspforp'),
    ('156-F-28_48_flipped.png', 'uspforp'),
    ('180-F-55_38.png', 'uspforp'),
    ('259-F-86_38.png', 'uspforp'),
    ('268-F-40_48_flipped.png', 'uspforp'),
    ('270-F-16_38.png', 'uspforp'),
    ('50_17.png', 'tufts'),
    ('66_17.png', 'tufts'),
    ('74_17.png', 'tufts'),
    ('93_17.png', 'tufts'),
    ('106_17.png', 'tufts'),
    ('117_17.png', 'tufts'),
    ('126_17.png', 'tufts'),
    ('163_32_flipped.png', 'tufts'),
    ('165_32_flipped.png', 'tufts'),
    ('208_32_flipped.png', 'tufts')
]

# Output directories
OUTPUT_ORIGINAL = 'selected_samples_original'
OUTPUT_RENAMED = 'selected_samples_renamed'

def setup_directories():
    """Create output directories if they don't exist"""
    os.makedirs(OUTPUT_ORIGINAL, exist_ok=True)
    os.makedirs(OUTPUT_RENAMED, exist_ok=True)
    
    # Create subdirectories for each dataset
    for dataset in DATASET_CONFIG.keys():
        os.makedirs(os.path.join(OUTPUT_ORIGINAL, dataset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_RENAMED, dataset), exist_ok=True)

def read_excel_data(excel_path, dataset_name):
    """Read Excel file and return combined DataFrame"""
    try:
        if dataset_name == 'ADLD':
            # For ADLD, read all 3 sheets and combine
            xls = pd.ExcelFile(excel_path)
            dfs = []
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                dfs.append(df)
            combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        else:
            # For other datasets, read single sheet
            return pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file {excel_path}: {str(e)}")
        return None

def find_file_in_excel(filename, dataset_name):
    """Search for filename in the dataset's Excel file"""
    excel_path = DATASET_CONFIG[dataset_name]['excel_path']
    df = read_excel_data(excel_path, dataset_name)
    
    if df is None:
        return None
    
    # Search for the filename in the first column
    first_col = df.columns[0]
    result = df[df[first_col] == filename]
    
    if not result.empty:
        return result.iloc[0].to_dict()
    return None


def copy_files():
    """Copy original and renamed files based on Excel lookup and collect data for report"""
    report_data = []
    missing_files = []
    excel_errors = []
    
    for filename, dataset in tqdm(FILE_LIST, desc="Processing files"):
        record = {
            'Requested_Filename': filename,
            'Dataset': dataset,
            'Status': 'Pending',
            'Original_Filename': '',
            'New_Filename': '',
            'Original_Source_Path': '',
            'Renamed_Source_Path': '',
            'Original_Destination_Path': '',
            'Renamed_Destination_Path': '',
            'Original_Copied': False,
            'Renamed_Copied': False,
            'Error': '',
            'Excel_Source': DATASET_CONFIG[dataset]['excel_path']
        }
        
        try:
            # Look up file in Excel
            file_info = find_file_in_excel(filename, dataset)
            
            if file_info is None:
                record.update({
                    'Status': 'Not Found',
                    'Error': 'File not found in Excel'
                })
                missing_files.append(record)
                continue
            
            # Get original and new filenames
            original_filename = file_info.get('Original Filename', filename)
            new_filename = file_info.get('New Filename', filename)
            
            record.update({
                'Original_Filename': original_filename,
                'New_Filename': new_filename
            })
            
            # Prepare paths
            original_src = os.path.join(DATASET_CONFIG[dataset]['original_path'], original_filename)
            renamed_src = os.path.join(DATASET_CONFIG[dataset]['renamed_path'], new_filename)
            original_dst = os.path.join(OUTPUT_ORIGINAL, dataset, original_filename)
            renamed_dst = os.path.join(OUTPUT_RENAMED, dataset, new_filename)
            
            record.update({
                'Original_Source_Path': original_src,
                'Renamed_Source_Path': renamed_src,
                'Original_Destination_Path': original_dst,
                'Renamed_Destination_Path': renamed_dst
            })
            
            # Copy files
            original_copied = False
            renamed_copied = False
            
            if os.path.exists(original_src):
                shutil.copy2(original_src, original_dst)
                original_copied = True
            else:
                record['Error'] = 'Original source file missing'
            
            if os.path.exists(renamed_src):
                shutil.copy2(renamed_src, renamed_dst)
                renamed_copied = True
            elif original_copied:
                record['Error'] = 'Renamed source file missing'
                # Remove original if renamed is missing for consistency
                if os.path.exists(original_dst):
                    os.remove(original_dst)
                    original_copied = False
            
            record.update({
                'Original_Copied': original_copied,
                'Renamed_Copied': renamed_copied,
                'Status': 'Success' if original_copied and renamed_copied else 'Partial' if original_copied or renamed_copied else 'Failed'
            })
            
            report_data.append(record)
            
        except Exception as e:
            record.update({
                'Status': 'Error',
                'Error': str(e)
            })
            excel_errors.append(record)
    
    # Combine all records
    all_records = report_data + missing_files + excel_errors
    return pd.DataFrame(all_records)

def generate_report(report_df):
    """Generate Excel report with multiple sheets"""
    report_file = 'file_processing_report.xlsx'
    
    with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
        # Summary sheet
        summary = report_df[['Requested_Filename', 'Dataset', 'Status', 'Error']]
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Successful copies
        success = report_df[report_df['Status'] == 'Success']
        if not success.empty:
            success.to_excel(writer, sheet_name='Successful', index=False)
        
        # Partial copies (only original or renamed)
        partial = report_df[report_df['Status'] == 'Partial']
        if not partial.empty:
            partial.to_excel(writer, sheet_name='Partial', index=False)
        
        # Missing files
        missing = report_df[report_df['Status'].isin(['Not Found', 'Failed'])]
        if not missing.empty:
            missing.to_excel(writer, sheet_name='Missing_Files', index=False)
        
        # Errors
        errors = report_df[report_df['Status'] == 'Error']
        if not errors.empty:
            errors.to_excel(writer, sheet_name='Errors', index=False)
        
        # Full details
        report_df.to_excel(writer, sheet_name='All_Details', index=False)
        
        # Add metadata sheet
        metadata = pd.DataFrame({
            'Report_Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Total_Files_Processed': [len(report_df)],
            'Successful_Copies': [len(success)],
            'Partial_Copies': [len(partial)],
            'Missing_Files': [len(missing)],
            'Errors': [len(errors)]
        })
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    return report_file

def main():
    print("Setting up directories...")
    setup_directories()
    
    print("Processing files and generating report...")
    report_df = copy_files()
    
    print("Generating Excel report...")
    report_file = generate_report(report_df)
    
    print("\nOperation complete!")
    print(f"Report generated: {report_file}")
    print(f"Files copied to: {OUTPUT_ORIGINAL} and {OUTPUT_RENAMED}")
    
    # Print summary
    success_count = len(report_df[report_df['Status'] == 'Success'])
    partial_count = len(report_df[report_df['Status'] == 'Partial'])
    error_count = len(report_df[report_df['Status'] == 'Error'])
    missing_count = len(report_df[report_df['Status'].isin(['Not Found', 'Failed'])])
    
    print(f"\nSummary:")
    print(f"- Successfully copied: {success_count}")
    print(f"- Partially copied: {partial_count}")
    print(f"- Missing/not found: {missing_count}")
    print(f"- Errors: {error_count}")

if __name__ == "__main__":
    main()