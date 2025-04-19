import pandas as pd
import re
import numpy as np
import os
import shutil
import zipfile

# Function to format questions and options into a readable string
def format_question(row):
    # Retrieve the question and options
    question = row['question']
    options = row['options']
    # Convert options to a single string with newline separation
    choices = '\n'.join(options)
    # Return the formatted question string
    return f"Question: {question}\nChoices:\n{choices}"


# Function to extract scene_name from video_id
def extract_scene_name(video_id):
    # Check the prefix of video_id and extract scene_name accordingly
    if video_id.startswith('EmbodiedCity'):
        if video_id.count('_') >= 2:
            # Extract the substring before the second underscore
            second_underscore_index = video_id.find('_', video_id.find('_') + 1)
            return video_id[:second_underscore_index]
        elif video_id.count('_') == 1:
            # Extract the substring before the dot
            return video_id.split('.')[0]

    elif video_id.startswith('AerialVLN'):
        if video_id.count('_') >= 3:
            # Extract the substring before the third underscore
            third_underscore_index = video_id.find('_', video_id.find('_', video_id.find('_') + 1) + 1)
            return video_id[:third_underscore_index]
        elif video_id.count('_') == 2:
            # Extract the substring before the dot
            return video_id.split('.')[0]

    elif video_id.startswith('RealWorld'):
        if video_id.count('_') == 1:
            # Extract the substring before the dot
            return video_id.split('.')[0]
        else:
            # Extract the first numeric value and format it as RealWorld_x
            match = re.search(r'_(\d+)', video_id)
            if match:
                number = match.group(1)
                return f'RealWorld_{number}'

    # Return None if no matching pattern is found
    return None

# Create target directories
complete_dir = os.path.join('dataset', 'complete')  # Path to dataset/complete
videos_dir = os.path.join(complete_dir, 'videos')  # Path to dataset/complete/videos

# Create directories if they don't exist
os.makedirs(videos_dir, exist_ok=True)

# Load the VSI-Bench dataset
vsi_data = pd.read_parquet('dataset/VSI-Bench/test-00000-of-00001.parquet')

# Load the UrbanVideo-Bench dataset
uvb_data = pd.read_parquet('dataset/UrbanVideo-Bench/MCQ.parquet')

# Map video_id in VSI-Bench to scene_name
vsi_data['video_id'] = vsi_data['scene_name']

# 1. Rename columns for consistency
vsi_data = vsi_data.rename(columns={
    'id': 'Question_id',
    'question_type': 'question_category',
    'ground_truth': 'answer'
})

# 2. Append '.mp4' suffix to all strings in the 'video_id' column
vsi_data['video_id'] = vsi_data['video_id'].astype(str) + '.mp4'

# 3. Filter rows based on specific question categories
categories_to_keep = [
    'object_rel_direction_hard', 'object_rel_direction_medium',
    'object_rel_direction_easy', 'object_rel_distance',
    'obj_appearance_order', 'route_planning'
]
vsi_data = vsi_data[vsi_data['question_category'].isin(categories_to_keep)]

# Merge specific categories into a single unified category
vsi_data['question_category'] = vsi_data['question_category'].replace(
    ['object_rel_direction_hard', 'object_rel_direction_medium', 'object_rel_direction_easy'],
    'object_rel_direction'
)

# 4. Format the 'question' column
vsi_data['question'] = vsi_data.apply(format_question, axis=1)

# Retain only the relevant columns
vsi_data = vsi_data[['Question_id', 'video_id', 'question_category', 'question', 'answer', 'scene_name']]

# Extract scene_name column for UrbanVideo-Bench based on video_id
uvb_data['scene_name'] = uvb_data['video_id'].apply(extract_scene_name)

# Filter rows based on selected tasks
select_task = ['Counterfactual', 'Landmark Position', 'Action Generation', 'object_rel_direction',
               'object_rel_distance', 'Progress Evaluation', 'obj_appearance_order', 'route_planning']
uvb_data = uvb_data[uvb_data['question_category'].isin(select_task)]
vsi_data = vsi_data[vsi_data['question_category'].isin(select_task)]

# Combine the two datasets
complete_data = pd.concat([uvb_data, vsi_data], axis=0)

# Reset the index of the combined dataset
complete_data.index = range(complete_data.shape[0])

# Assign Question_id based on the index
complete_data['Question_id'] = complete_data.index

# Retrieve all unique scene_name values
unique_scene_names = complete_data['scene_name'].unique()

# Shuffle the scene_name array randomly
np.random.shuffle(unique_scene_names)

# Divide scene_name values into train, validation, and test sets based on proportion
total_count = len(unique_scene_names)
train_size = int(total_count * 7.5 / 10)  # Train set: 75%
val_size = int(total_count * 0.5 / 10)   # Validation set: 5%

# Assign scene_name values to train, validation, and test sets
train_scene_names = unique_scene_names[:train_size]
val_scene_names = unique_scene_names[train_size:train_size + val_size]
test_scene_names = unique_scene_names[train_size + val_size:]

# Filter the dataset into train, validation, and test sets based on scene_name
train_data = complete_data[complete_data['scene_name'].isin(train_scene_names)]
val_data = complete_data[complete_data['scene_name'].isin(val_scene_names)]
test_data = complete_data[complete_data['scene_name'].isin(test_scene_names)]

# Save train_data to JSON file
train_data_path = os.path.join(complete_dir, 'train_data.json')
train_data.to_json(train_data_path, orient='records', indent=4)
print(f"Train data saved to {train_data_path}")

# Save val_data to JSON file
val_data_path = os.path.join(complete_dir, 'val_data.json')
val_data.to_json(val_data_path, orient='records', indent=4)
print(f"Validation data saved to {val_data_path}")

# Save test_data to JSON file
test_data_path = os.path.join(complete_dir, 'test_data.json')
test_data.to_json(test_data_path, orient='records', indent=4)
print(f"Test data saved to {test_data_path}")


# Move all files from UrbanVideo-Bench/videos to complete/videos
urban_videos_dir = os.path.join('dataset', 'UrbanVideo-Bench', 'videos')
if os.path.exists(urban_videos_dir):
    for file_name in os.listdir(urban_videos_dir):
        file_path = os.path.join(urban_videos_dir, file_name)
        shutil.move(file_path, videos_dir)  # Move files to the target directory

# Extract all zip files in VSI-Bench directory
vsi_bench_dir = os.path.join('dataset', 'VSI-Bench')
if os.path.exists(vsi_bench_dir):
    for file_name in os.listdir(vsi_bench_dir):
        file_path = os.path.join(vsi_bench_dir, file_name)
        # Check if the file is a zip archive
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(vsi_bench_dir)  # Extract contents to VSI-Bench directory

# Move all .mp4 files from VSI-Bench and its subdirectories to complete/videos
for root, _, files in os.walk(vsi_bench_dir):
    for file_name in files:
        if file_name.endswith('.mp4'):  # Check if the file is an mp4 video
            file_path = os.path.join(root, file_name)
            destination_path = os.path.join(videos_dir, file_name)

            # Check if the file already exists at the destination
            if os.path.exists(destination_path):
                print(f"File {destination_path} already exists. Overwriting.")

            # Use shutil.copy to overwrite the file at the destination
            shutil.copy(file_path, destination_path)  # Copy the file (overwrite if exists)

            # Remove the original file after copying
            os.remove(file_path)

