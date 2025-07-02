import os
import pandas as pd
from tqdm import tqdm
import shutil
import traceback
from datetime import datetime

def process_dataset():
    # Define paths
    csv_folder = "workers_data/completed_labels"
    photos_dir = "workers_data/all_in_one"
    output_dir = "thirdmolar_classwise"
    log_file = "processing_report.csv"
    
    # Clean output directory
    os.system("rm -rf thirdmolar_classwise")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logging metrics
    log_data = {
        'timestamp': [],
        'csv_file': [],
        'operation': [],
        'filename': [],
        'class_label': [],
        'status': [],
        'message': [],
        'exception': []
    }
    
    # Class mapping (folder_name: (display_name, numeric_suffix))
    class_mapping = {
        '1': ('1', 0),
        '2A': ('2A', 1),
        '2B': ('2B', 2),
        '2C': ('2C', 3),
        '3A': ('3A', 4),
        '3B': ('3B', 5),
        '3C': ('3C', 6)
    }
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    total_files_processed = 0
    total_files_skipped = 0
    exceptions_occurred = 0
    
    # Process each CSV file
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        csv_path = os.path.join(csv_folder, csv_file)
        
        try:
            df = pd.read_csv(csv_path)
            rows_processed = 0
            rows_skipped = 0
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}", leave=False):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                filename = row['Filename']
                craniocaudal = str(row['Craniocaudal (based on the guide above, you should put 1,2, or 3)']).strip()
                
                # Handle float values (e.g., "2.0" becomes "2")
                try:
                    craniocaudal = str(int(float(craniocaudal))) if craniocaudal else ''
                except ValueError:
                    craniocaudal = ''
                
                if not craniocaudal:
                    log_skip(log_data, timestamp, csv_file, filename, '', 'Missing craniocaudal value')
                    rows_skipped += 1
                    continue
                
                # Determine class label
                if craniocaudal == '1':
                    class_label = '1'
                else:
                    tooth_position = str(row['Tooth position (based on the guide above, you should put a, b, or c)']).strip().upper()
                    if not tooth_position:
                        log_skip(log_data, timestamp, csv_file, filename, '', 'Missing tooth position for class 2/3')
                        rows_skipped += 1
                        continue
                    class_label = f"{craniocaudal}{tooth_position}"
                
                if class_label not in class_mapping:
                    log_skip(log_data, timestamp, csv_file, filename, class_label, 'Unexpected class label')
                    rows_skipped += 1
                    continue
                
                folder_name, numeric_suffix = class_mapping[class_label]
                dest_dir = os.path.join(output_dir, folder_name)
                os.makedirs(dest_dir, exist_ok=True)
                
                src_path = os.path.join(photos_dir, filename)
                
                if not os.path.exists(src_path):
                    log_skip(log_data, timestamp, csv_file, filename, class_label, 'Source file not found')
                    rows_skipped += 1
                    continue
                
                # Generate new filename with numeric suffix
                file_base = os.path.splitext(filename)[0]
                file_ext = os.path.splitext(filename)[1]
                new_filename = f"{file_base}_{folder_name}_{numeric_suffix}{file_ext}"
                dest_path = os.path.join(dest_dir, new_filename)
                
                try:
                    shutil.copy2(src_path, dest_path)
                    log_data['timestamp'].append(timestamp)
                    log_data['csv_file'].append(csv_file)
                    log_data['operation'].append('file_copy')
                    log_data['filename'].append(filename)
                    log_data['class_label'].append(class_label)
                    log_data['status'].append('success')
                    log_data['message'].append(f'Copied to {dest_path}')
                    log_data['exception'].append('')
                    rows_processed += 1
                except Exception as e:
                    log_skip(log_data, timestamp, csv_file, filename, class_label, 'File copy failed', str(e))
                    rows_skipped += 1
            
            total_files_processed += rows_processed
            total_files_skipped += rows_skipped
            
        except Exception as e:
            exceptions_occurred += 1
            log_data['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            log_data['csv_file'].append(csv_file)
            log_data['operation'].append('csv_processing')
            log_data['filename'].append('')
            log_data['class_label'].append('')
            log_data['status'].append('failed')
            log_data['message'].append('CSV processing failed')
            log_data['exception'].append(traceback.format_exc())
    
    # Save log data to CSV
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_file, index=False)
    
    # Print summary
    print(f"\nProcessing complete. Summary:")
    print(f"- Total CSV files processed: {len(csv_files)}")
    print(f"- Total files successfully copied: {total_files_processed}")
    print(f"- Total files skipped: {total_files_skipped}")
    print(f"- Total exceptions occurred: {exceptions_occurred}")
    print(f"- Detailed log saved to: {log_file}")

def log_skip(log_data, timestamp, csv_file, filename, class_label, message, exception=''):
    """Helper function to log skipped files"""
    log_data['timestamp'].append(timestamp)
    log_data['csv_file'].append(csv_file)
    log_data['operation'].append('file_copy')
    log_data['filename'].append(filename)
    log_data['class_label'].append(class_label)
    log_data['status'].append('skipped')
    log_data['message'].append(message)
    log_data['exception'].append(exception)

if __name__ == "__main__":
    process_dataset()


# import os
# import pandas as pd
# from tqdm import tqdm
# import shutil
# import traceback
# from datetime import datetime
# os.system("rm -rf thirdmolar_classwise")

# def process_dataset():
#     # Define paths
#     csv_folder = "workers_data/completed_labels"
#     photos_dir = "workers_data/all_in_one"
#     output_dir = "thirdmolar_classwise"
#     log_file = "processing_report.csv"
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize logging metrics
#     log_data = {
#         'timestamp': [],
#         'csv_file': [],
#         'operation': [],
#         'filename': [],
#         'class_label': [],
#         'status': [],
#         'message': [],
#         'exception': []
#     }
    
#     # Class mapping
#     class_mapping = {
#         '1': ('1', 0),
#         '2A': ('2A', 1),
#         '2B': ('2B', 2),
#         '2C': ('2C', 3),
#         '3A': ('3A', 4),
#         '3B': ('3B', 5),
#         '3C': ('3C', 6)
#     }
    
#     # Get all CSV files
#     csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
#     total_files_processed = 0
#     total_files_skipped = 0
#     exceptions_occurred = 0
    
#     # Process each CSV file
#     for csv_file in tqdm(csv_files, desc="Processing CSV files"):
#         csv_path = os.path.join(csv_folder, csv_file)
        
#         try:
#             df = pd.read_csv(csv_path)
#             rows_processed = 0
#             rows_skipped = 0
            
#             for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}", leave=False):
#                 timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 filename = row['Filename']
#                 craniocaudal = str(row['Craniocaudal (based on the guide above, you should put 1,2, or 3)']).strip()
                
#                 # Handle float values (e.g., "2.0" becomes "2")
#                 try:
#                     craniocaudal = str(int(float(craniocaudal))) if craniocaudal else ''
#                 except ValueError:
#                     craniocaudal = ''
                
#                 if not craniocaudal:
#                     log_data['timestamp'].append(timestamp)
#                     log_data['csv_file'].append(csv_file)
#                     log_data['operation'].append('row_processing')
#                     log_data['filename'].append(filename)
#                     log_data['class_label'].append('')
#                     log_data['status'].append('skipped')
#                     log_data['message'].append('Missing craniocaudal value')
#                     log_data['exception'].append('')
#                     rows_skipped += 1
#                     continue
                
#                 if craniocaudal == '1':
#                     class_label = '1'
#                     tooth_position = ''
#                 else:
#                     tooth_position = str(row['Tooth position (based on the guide above, you should put a, b, or c)']).strip().upper()
#                     if not tooth_position:
#                         log_data['timestamp'].append(timestamp)
#                         log_data['csv_file'].append(csv_file)
#                         log_data['operation'].append('row_processing')
#                         log_data['filename'].append(filename)
#                         log_data['class_label'].append('')
#                         log_data['status'].append('skipped')
#                         log_data['message'].append('Missing tooth position for class 2/3')
#                         log_data['exception'].append('')
#                         rows_skipped += 1
#                         continue
#                     class_label = f"{craniocaudal}{tooth_position}"
                
#                 if class_label not in class_mapping:
#                     log_data['timestamp'].append(timestamp)
#                     log_data['csv_file'].append(csv_file)
#                     log_data['operation'].append('row_processing')
#                     log_data['filename'].append(filename)
#                     log_data['class_label'].append(class_label)
#                     log_data['status'].append('skipped')
#                     log_data['message'].append('Unexpected class label')
#                     log_data['exception'].append('')
#                     rows_skipped += 1
#                     continue
                
#                 folder_name, class_index = class_mapping[class_label]
#                 dest_dir = os.path.join(output_dir, folder_name)
#                 os.makedirs(dest_dir, exist_ok=True)
                
#                 src_path = os.path.join(photos_dir, filename)
                
#                 if not os.path.exists(src_path):
#                     log_data['timestamp'].append(timestamp)
#                     log_data['csv_file'].append(csv_file)
#                     log_data['operation'].append('file_copy')
#                     log_data['filename'].append(filename)
#                     log_data['class_label'].append(class_label)
#                     log_data['status'].append('skipped')
#                     log_data['message'].append('Source file not found')
#                     log_data['exception'].append('')
#                     rows_skipped += 1
#                     continue
                
#                 existing_files = [f for f in os.listdir(dest_dir) if f.startswith(f"{os.path.splitext(filename)[0]}_{folder_name}_")]
#                 next_index = max([int(f.split('_')[-1].split('.')[0]) for f in existing_files]) + 1 if existing_files else 0
                
#                 file_ext = os.path.splitext(filename)[1]
#                 new_filename = f"{os.path.splitext(filename)[0]}_{folder_name}_{next_index}{file_ext}"
#                 dest_path = os.path.join(dest_dir, new_filename)
                
#                 try:
#                     shutil.copy2(src_path, dest_path)
#                     log_data['timestamp'].append(timestamp)
#                     log_data['csv_file'].append(csv_file)
#                     log_data['operation'].append('file_copy')
#                     log_data['filename'].append(filename)
#                     log_data['class_label'].append(class_label)
#                     log_data['status'].append('success')
#                     log_data['message'].append(f'Copied to {dest_path}')
#                     log_data['exception'].append('')
#                     rows_processed += 1
#                 except Exception as e:
#                     log_data['timestamp'].append(timestamp)
#                     log_data['csv_file'].append(csv_file)
#                     log_data['operation'].append('file_copy')
#                     log_data['filename'].append(filename)
#                     log_data['class_label'].append(class_label)
#                     log_data['status'].append('failed')
#                     log_data['message'].append('File copy failed')
#                     log_data['exception'].append(str(e))
#                     rows_skipped += 1
            
#             total_files_processed += rows_processed
#             total_files_skipped += rows_skipped
            
#         except Exception as e:
#             exceptions_occurred += 1
#             log_data['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#             log_data['csv_file'].append(csv_file)
#             log_data['operation'].append('csv_processing')
#             log_data['filename'].append('')
#             log_data['class_label'].append('')
#             log_data['status'].append('failed')
#             log_data['message'].append('CSV processing failed')
#             log_data['exception'].append(traceback.format_exc())
    
#     # Save log data to CSV
#     log_df = pd.DataFrame(log_data)
#     log_df.to_csv(log_file, index=False)
    
#     # Print summary
#     print(f"\nProcessing complete. Summary:")
#     print(f"- Total CSV files processed: {len(csv_files)}")
#     print(f"- Total files successfully copied: {total_files_processed}")
#     print(f"- Total files skipped: {total_files_skipped}")
#     print(f"- Total exceptions occurred: {exceptions_occurred}")
#     print(f"- Detailed log saved to: {log_file}")

# if __name__ == "__main__":
#     process_dataset()