import os
import shutil
import pandas as pd

df = pd.read_csv('Data_Entry_2017.csv')

mass_nodule_mask = df['Finding Labels'].apply(lambda labels: any(label in {'Mass', 'Nodule'} for label in labels.split('|')))
no_finding_mask = df['Finding Labels'] == 'No Finding'

mass_nodule_df = df[mass_nodule_mask]
no_finding_df = df[no_finding_mask]

num_mass_nodule = len(mass_nodule_df)

sampled_no_finding = no_finding_df.sample(n=num_mass_nodule, 
                                         replace=False, 
                                         random_state=42)

balanced_df = pd.concat([mass_nodule_df, sampled_no_finding])

folder_ranges = [
    (1335, 6, 'images_001'),
    (3923, 13, 'images_002'),
    (6585, 6, 'images_003'),
    (9232, 3, 'images_004'),
    (11558, 7, 'images_005'),
    (13774, 25, 'images_006'),
    (16051, 9, 'images_007'),
    (18387, 34, 'images_008'),
    (20945, 49, 'images_009'),
    (24717, 0, 'images_010'),
    (28173, 2, 'images_011'),
    (30805, 0, 'images_012')
]

source_base_dir = ''
main_dest_dir = ''

dest_dirs = {
    'mass_nodule': os.path.join(main_dest_dir, 'mass_nodule_finding'),
    'no_finding': os.path.join(main_dest_dir, 'no_finding')
}

for dir_path in dest_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

for index, row in balanced_df.iterrows():
    try:
        image_filename = row['Image Index']
        labels = row['Finding Labels']
        
        if labels == 'No Finding':
            category = 'no_finding'
        else:
            category = 'mass_nodule' if any(label in ['Mass', 'Nodule'] for label in labels.split('|')) else 'unknown'
        
        if category == 'unknown':
            print(f"Skipping unknown category for {image_filename}")
            continue

        base_name = os.path.splitext(image_filename)[0]
        part1_str, part2_str = base_name.split('_')
        part1 = int(part1_str)
        part2 = int(part2_str)

        source_folder = None
        for f_part1, f_part2, f_name in folder_ranges:
            if part1 < f_part1 or (part1 == f_part1 and part2 <= f_part2):
                source_folder = f_name
                break
        
        if not source_folder:
            print(f"No folder found for {image_filename}")
            continue
            
        source_path = os.path.join(source_base_dir, source_folder, 'images', image_filename)
        dest_path = os.path.join(dest_dirs[category], image_filename)
        
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"Copied {image_filename} to {category} folder")
        else:
            print(f"Missing: {source_path}")
            
    except Exception as e:
        print(f"Error processing {image_filename}: {str(e)}")

print("Image copy operation completed.")