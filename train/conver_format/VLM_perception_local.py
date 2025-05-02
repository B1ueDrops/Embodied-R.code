import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
import re
import pandas as pd
import numpy as np
import warnings
import math
from natsort import natsorted
import copy
import argparse  # Import argparse module
import torch
import json
import datetime
from pathlib import Path
from utils import *

# Import ms-swift framework related modules
from swift.llm import (
    get_model_tokenizer,
    InferRequest,
    PtEngine,
    RequestConfig,
    get_template
)
from swift.llm.template import load_image, load_file
from swift.tuners import Swift

warnings.filterwarnings("ignore")

def extract_qa(text):
    index = text.find("Question:")
    if index != -1:
        result = text[index + len("Question:"):].strip()
        return result
    else:
        return text

def split_points(df):
    """
    Separate the string in a single-column DataFrame into two parts.
    If '1.' and '2.' are not found, put the original string in 'Point_2'
    and set 'Point_1' to None.

    Parameters:
        df (pd.DataFrame): A DataFrame containing only one column of strings.

    Returns:
        pd.DataFrame: A DataFrame with two new columns corresponding to 'Point_1' and 'Point_2'.
    """

    def extract_points(text):
        # Check if the string contains '1.' and '2.'
        try:
            if '1.' in text and '2.' in text and '3.' in text:
                # Use regex to match the content of each point
                point_1_match = re.search(r'(?:###\s*1\.|1\.)(.*?)(?=\n\s*(?:###\s*2\.|2\.))', text, re.DOTALL)
                point_2_match = re.search(r'(?:###\s*2\.|2\.)(.*?)(?=\n\s*(?:###\s*3\.|3\.))', text, re.DOTALL)
                point_3_match = re.search(r'(?:###\s*3\.|3\.)(.*)', text, re.DOTALL)

                # Extract matching results, set to None if not matched
                point_1 = point_1_match.group(1).strip() if point_1_match else None
                point_2 = point_2_match.group(1).strip() if point_2_match else None
                point_3 = point_3_match.group(1).strip() if point_3_match else None

                if point_3 == None:
                    view_content = point_2
                else:
                    view_content = point_2 + ' ' + point_3
                return point_1, view_content
            else:
                # If '1.' and '2.' are not found, put the entire string in 'Point_2'
                return None, text
        except:
            return None, text

    # Apply the function to each row of strings
    df[['action', 'view']] = df.iloc[1:, 0].apply(lambda x: pd.Series(extract_points(x)))
    df.loc[0, 'view'] = df.iloc[0, 0]

    return df

def derive_video_content_str(df):
    video_content_str = "The content of the video will be described in the form of text. \
    The 0th frame is the initial frame, which includes the scene that agent observe at the initial position. \
    The agent keeps moving, thus constantly obtaining new visual observations (frames). \
    The last frame is the visual observation at the current position."
    video_content_str += '\nFrame 0: Observe: %s' % df['view'].iloc[0]
    for i in range(1, df.shape[0]):
        video_content_str += '\nFrame %d: After %s, Observe: %s' % (i, df['action'].iloc[i], df['view'].iloc[i])

    return video_content_str

def question_fil(text):
    # Find the first occurrence of 'Question: '
    start_index = text.find("Question:")

    # If 'Question: ' is found
    if start_index != -1:
        # Extract the content after 'Question: '
        result = text[start_index + len("Question:"):]
        return result
    else:
        return text


BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
ANSWER_PATTERN = r"<answer>\s*([A-H])"
SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."  # Pattern with dot, no restriction on content after the dot
SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"  # Pattern without dot
VALID_OPTIONS = set('ABCDEFGH')
def extract_option_letter(answer):
    answer = answer.strip()

    # First, try to find the answer in standard format ($\boxed{X}$)
    # boxed_matches = list(re.finditer(self.BOXED_PATTERN, answer, re.IGNORECASE))
    # if boxed_matches:
    #     # Use the last matched answer
    #     return boxed_matches[-1].group(1).upper()

    # Then look for the answer pattern
    answer_matches = list(re.finditer(ANSWER_PATTERN, answer, re.IGNORECASE))
    if answer_matches:
        # Use the first matched answer
        return answer_matches[0].group(1).upper()

    # Finally, look for single letters
    # First, find those with dots
    dot_matches = list(re.finditer(SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
    if dot_matches:
        # Use the last matched with dot
        return dot_matches[-1].group(1).upper()

    # # Lastly, find those without dots
    # simple_matches = list(re.finditer(SIMPLE_PATTERN, answer, re.IGNORECASE))
    # if simple_matches:
    #     # Use the last matched without dot
    #     return simple_matches[-1].group(1).upper()

    # If nothing is found, return the original text (will return 0 points in subsequent comparison)
    return answer.upper()

def calculate_acc(file_path):
    try:
        df = pd.read_json(file_path)
    except:
        df = pd.read_json(file_path, lines=True)

    df['Extracted_Option'] = df['content'].apply(extract_option_letter)

    # Remove rows where no Option was matched
    df = df.dropna(subset=['Extracted_Option'])

    # Calculate accuracy by category
    accuracy_data = []

    try:
        for category, group in df.groupby('question_category'):
            correct_count = (group['Extracted_Option'] == group['answer']).sum()
            accuracy = correct_count / len(group)
            accuracy_data.append({
                'Category': category,
                'Accuracy': accuracy,
                'Num': len(group)
            })
    except:
        pass

    # Overall accuracy
    total_correct = (df['Extracted_Option'] == df['answer']).sum()
    total_accuracy = total_correct / len(df)
    accuracy_data.append({
        'Category': 'Total',
        'Accuracy': total_accuracy
    })

    # Save results to Excel
    accuracy_df = pd.DataFrame(accuracy_data)
    print('Acc: \n', accuracy_df)

    # Construct output file path, keep the same name as csv, with .xlsx extension
    output_file_name = file_path.replace('.json', '.xlsx')
    output_excel_path = os.path.join('results', 'acc_'+os.path.basename(output_file_name))

    # Use ExcelWriter to save results
    with pd.ExcelWriter(output_excel_path) as writer:
        accuracy_df.to_excel(writer, index=False, sheet_name='Accuracy Results')



class VideoProcessor:
    """Video processing class, responsible for extracting video frames and using visual models for analysis"""

    def __init__(self, vision_model_path: str = "/data1/GRPO/model/Qwen2.5-Vl-7B-instruct"):
        """
        Initialize video processor

        Args:
            vision_model_path: Path to the visual model
        """
        print(f"Loading visual parsing model: {vision_model_path}")
        self.model, self.tokenizer = get_model_tokenizer(
            vision_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.template = get_template("qwen2_5_vl", self.tokenizer)
        self.engine = PtEngine.from_model_template(self.model, self.template)
        print("Visual parsing model loaded")

    def extract_frames(self, video_path: str):
        """
        Args:
            video_path: Video file path

        Returns:
            List of extracted frames and base64 encoded frames
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        base64_frames = []

        # Read all frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

        cap.release()
        print(len(base64_frames), "frames read.")

        return frames, base64_frames

    def analyze_video(self, video_path: str, qa: str = None):
        """
        Analyze video content, consistent with VLM_perception.py

        Args:
            video_path: Video file path
            qa: Question, used to guide the visual model to focus on relevant content

        Returns:
            DataFrame format results
        """
        # Extract video frames
        _, base64_frames = self.extract_frames(video_path)
        print(len(base64_frames), "frames extracted.")

        # Create result DataFrame
        res = pd.DataFrame(columns=['description'])

        # Process each frame
        for frame_idx in range(len(base64_frames)):
            try:
                if frame_idx > 0:
                    prompt = "I provide you with the agent's first-person perspective. Two images represent the field of view before and after the action. Please output: \n\
                                    1. Based on the relative positional changes of objects in the field of view, determine the action performed (only output one of the following options without redundant content): Move forward, move backward, move left, move right, move upward, move downward, rotate left, rotate right, tilt downward, or tilt upward. \n\
                                    2. Analyze the post-action field of view compared to the pre-action view, identifying any newly captured spatial information. This includes objects that were previously invisible or unclear. Note that many objects of the same category may appear repetitive, such as buildings in a city, but they might differ in color or shape. When describing these, include features such as color and shape. Additionally, focus on the relationship between the agent and its surrounding environment. To do so, first imagine the three-dimensional scene around the agent. When describing relative positions, use terms such as 'to the left,' 'in the front-left,' or 'below' rather than simply referring to their positions in the field of view. \n\
                                    3. If the objects mentioned in the following question appear in the images, please make sure to describe them: '%s'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n\
                                    Ensure responses are concise and clear." % qa
                    content = [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frames[frame_idx-1]}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frames[frame_idx]}",
                            },
                        }
                    ]
                else:
                    prompt = "I provide you with the first-person perspective of an intelligent agent. \
                        Please output a description of the current observed scene. \
                        When describing the scene, focus on using features such as color and shape to characterize the objects. \
                        Additionally, emphasize the spatial relationship between the agent itself and the surrounding environment. \
                        You should first visualize the three-dimensional space around the agent. \
                        When describing relative positions, use terms such as 'to the left,' 'ahead-left,' or 'below,' rather than merely stating their positions within the field of view. \
                        More important, if the objects mentioned in the following question appear in the images, please make sure to describe them: '%s'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n\
                        The response should be concise and clear." % qa

                    content = [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_frames[frame_idx]}",
                            },
                        }
                    ]

                # Build message
                messages = [{"role": "user", "content": content}]

                # Create inference request
                infer_request = InferRequest(messages=messages)

                # Execute inference
                response = self.engine.infer([infer_request])[0]

                # Get result
                frame_description = response.choices[0].message.content
                res.loc[frame_idx, 'description'] = frame_description

                # Print current frame processing result
                print(frame_description)

            except Exception as e:
                print(f"An error occurred: {e}")
                res['description'].loc[frame_idx] = None
                time.sleep(10)  # Wait briefly when an error occurs

        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process videos with local vision model.')
    parser.add_argument('--data_paths', type=str, nargs='+', default=['dataset/complete/test_data.json', 'dataset/complete/train_data.json', 'dataset/complete/val_data.json'], help='JSON files containing data')
    parser.add_argument('--folder_path', type=str, default='dataset/complete/videos', help='Folder containing videos')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-Vl-72B-instruct', help='Path to the vision model')
    parser.add_argument('--save_path', type=str, default='results/inter', help='Path to save results')
    args = parser.parse_args()

    # Initialize video processor
    video_processor = VideoProcessor(args.model_path)

    # Key-frame folder
    video_keyframe_folder = os.path.join(args.save_path, 'video_keyframe')

    for data_path in args.data_paths:
        # Data reading
        folder_path = args.folder_path
        QA_df = pd.read_json(data_path)
        QA_df['response'] = None

        # Use the save path from command line arguments
        save_dir = getattr(args, 'save_path', 'results/inter')
        save_path = os.path.join(save_dir, f'processed_{os.path.basename(data_path)}')

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(QA_df.shape[0]):
            print('idx: %d / %d '% (idx, QA_df.shape[0]))
            video_name = QA_df['video_id'].iloc[idx]
            qa = extract_qa(QA_df['question'].iloc[idx])

            # Read video. If key-frames do not exist, extract and save
            video_path = os.path.join(video_keyframe_folder, video_name)
            if not os.path.exists(video_path):
                original_video_path = os.path.join(folder_path, video_name)
                if not os.path.exists(original_video_path):
                    print(f"Video file not found: {video_path}")
                    continue
                extract_keyframes_from_video(original_video_path, video_keyframe_folder)

            # Analyze video
            res = video_processor.analyze_video(video_path, qa)

            # Based on video text description content for inference
            video_content = res
            video_content = split_points(video_content)
            video_content_str = derive_video_content_str(video_content)

            original_qa = question_fil(qa)

            qa_w_content = "Please assume the role of an agent. The video represents your egocentric observations from the past to the present. \
            Video content: \n<\n%s\n>\n\
            Question: %s" % (video_content_str, original_qa)

            # Update question
            QA_df['question'].iloc[idx] = qa_w_content

            # Save intermediate results
            QA_df.to_json(save_path, orient='records', indent=4)
            print(f"Results saved to {save_path}")
