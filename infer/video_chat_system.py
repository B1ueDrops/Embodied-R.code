#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video chat system
Uses Qwen2.5-VL-72B-Instruct model to parse video content,
then uses Qwen2.5-3B-Instruct model for inference and answering
"""

import os
import cv2
import base64
import re
import time
import torch
import numpy as np
import json
import datetime
from typing import List, Dict, Any, Optional, Union
import tempfile
import warnings
from pathlib import Path

# Set environment variable, set maximum model length to 16K
os.environ['MAX_MODEL_LEN'] = '16384'

# Import ms-swift framework modules
from swift.llm import (
    get_model_tokenizer,
    InferRequest,
    PtEngine,
    RequestConfig,
    get_template
)
from swift.llm.template import load_image, load_file
from swift.tuners import Swift

# Ignore warnings
warnings.filterwarnings("ignore")

class VideoProcessor:
    """Video processing class, responsible for extracting video frames and using visual model for parsing"""

    def __init__(self, vision_model_path: str = "Qwen/Qwen2.5-Vl-72B-Instruct",
                 vision_max_tokens: int = 6144, vision_temperature: float = 0.1):
        """
        Initialize video processor

        Args:
            vision_model_path: Path to visual model
            vision_max_tokens: Maximum output tokens for visual model
            vision_temperature: Temperature for visual model
        """
        print(f"Loading visual parsing model: {vision_model_path}")
        self.model, self.tokenizer = get_model_tokenizer(
            vision_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.template = get_template("qwen2_5_vl", self.tokenizer)

        # Set maximum model length for engine
        self.model.config.max_model_len = 16384  # Set to 16K
        self.engine = PtEngine.from_model_template(self.model, self.template)

        # Set video parsing parameters
        self.vision_max_tokens = vision_max_tokens
        self.vision_temperature = vision_temperature

        print("Visual parsing model loaded")

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract all frames from video, consistent with video_description_each_question_8tasks.py main function

        Args:
            video_path: Path to video file

        Returns:
            List of extracted frames
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        # Read all frames, consistent with video_description_each_question_8tasks.py main function
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)

        cap.release()
        print(len(frames), "frames read.")
        return frames

    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """
        Convert frames to base64 encoding

        Args:
            frames: List of frames

        Returns:
            List of base64 encoded frames
        """
        base64_frames = []
        for frame in frames:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64_frames

    def analyze_video(self, video_path: str, question: str = None) -> str:
        """
        Args:
            video_path: Path to video file
            question: User question, used to guide the visual model to focus on relevant content

        Returns:
            Text description of video content
        """
        # Extract video frames
        frames = self.extract_frames(video_path)
        base64_frames = self.frames_to_base64(frames)

        # Build result string
        all_descriptions = []

        # Process each frame, consistent with video_description_each_question_8tasks.py
        for frame_idx in range(len(base64_frames)):
            if frame_idx > 0 and frame_idx < len(base64_frames):
                # For non-first frame, use comparative prompt
                prompt = "I provide you with the agent's first-person perspective. Two images represent the field of view before and after the action. Please output: \n\
                            1. Based on the relative positional changes of objects in the field of view, determine the action performed (only output one of the following options without redundant content): Move forward, move backward, move left, move right, move upward, move downward, rotate left, rotate right, tilt downward, or tilt upward. \n\
                            2. Analyze the post-action field of view compared to the pre-action view, identifying any newly captured spatial information. This includes objects that were previously invisible or unclear. Note that many objects of the same category may appear repetitive, such as buildings in a city, but they might differ in color or shape. When describing these, include features such as color and shape. Additionally, focus on the relationship between the agent and its surrounding environment. To do so, first imagine the three-dimensional scene around the agent. When describing relative positions, use terms such as 'to the left,' 'in the front-left,' or 'below' rather than simply referring to their positions in the field of view. \n\
                            3. If the objects mentioned in the following question appear in the images, please make sure to describe them: '%s'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n\
                            Ensure responses are concise and clear." % (question if question else "")

                # Build message
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
                # For first frame, use descriptive prompt
                prompt = "I provide you with the first-person perspective of an intelligent agent. \
                Please output a description of the current observed scene. \
                When describing the scene, focus on using features such as color and shape to characterize the objects. \
                Additionally, emphasize the spatial relationship between the agent itself and the surrounding environment. \
                You should first visualize the three-dimensional space around the agent. \
                When describing relative positions, use terms such as 'to the left,' 'ahead-left,' or 'below,' rather than merely stating their positions within the field of view. \
                More important, if the objects mentioned in the following question appear in the images, please make sure to describe them: '%s'. To reiterate, don't answer this question and don't give any option, you just need to observe whether the image contains the objects mentioned in the question/option. \n\
                The response should be concise and clear." % (question if question else "")

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

            # Create request configuration, using video parsing module parameters
            request_config = RequestConfig(max_tokens=self.vision_max_tokens, temperature=self.vision_temperature)

            # Execute inference
            response = self.engine.infer([infer_request], request_config)[0]

            # Add to result
            frame_description = response.choices[0].message.content
            all_descriptions.append(f"Frame {frame_idx+1}:\n{frame_description}\n")

            # Print current frame processing result
            print(f"\nProcessed frame {frame_idx+1}/{len(base64_frames)}")
            print(frame_description)

        # Merge all frame descriptions
        return "\n\n".join(all_descriptions)

class ReasoningProcessor:
    """Reasoning processing class, responsible for using language model for inference and answering"""

    def __init__(self, reasoning_model_path: str = "Qwen/Qwen2.5-3B-Instruct",
                 reasoning_max_tokens: int = 4096, reasoning_temperature: float = 0.7):
        """
        Initialize reasoning processor

        Args:
            reasoning_model_path: Path to reasoning model
            reasoning_max_tokens: Maximum output tokens for reasoning
            reasoning_temperature: Temperature for reasoning
        """
        print(f"Loading reasoning model: {reasoning_model_path}")
        self.model, self.tokenizer = get_model_tokenizer(
            reasoning_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.template = get_template("qwen2_5_vl", self.tokenizer)

        # Set maximum model length for engine
        self.model.config.max_model_len = 16384  # Set to 16K
        self.engine = PtEngine.from_model_template(self.model, self.template)

        # Set reasoning parameters
        self.reasoning_max_tokens = reasoning_max_tokens
        self.reasoning_temperature = reasoning_temperature

        print("Reasoning model loaded")

    def reason(self, video_description: str, question: str, history: List[Dict[str, Any]] = None) -> str:
        """
        Reasoning based on video description and question

        Args:
            video_description: Text description of video content
            question: User question
            history: Conversation history

        Returns:
            Reasoning result
        """
        try:
            # Build system prompt
            system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \n<answer> answer here </answer>. Ensure that your answer is consistent with and directly derived from your thinking process, maintaining logical coherence between the two sections."

            # Build user prompt
            user_prompt = f"Please assume the role of an agent. The video represents your egocentric observations from the past to the present. \
        Video content: \n<\n{video_description}\n\nBased on this video description, please answer the following question: {question}"

            # Build messages
            messages = [{"role": "system", "content": system_prompt}]

            # Add history conversation, ensure history is not empty
            if history and len(history) > 0:
                # Print history conversation information for debugging
                print(f"History conversation entries: {len(history)}")
                for i, msg in enumerate(history):
                    print(f"  History[{i}]: {msg['role']} - {msg['content'][:30]}...")
                messages.extend(history)

            # Add current question
            messages.append({"role": "user", "content": user_prompt})

            # Create inference request
            infer_request = InferRequest(messages=messages)

            # Create request configuration, using reasoning module parameters
            request_config = RequestConfig(max_tokens=self.reasoning_max_tokens, temperature=self.reasoning_temperature)

            # Execute inference
            print(f"Sending inference request, message count: {len(messages)}")
            response = self.engine.infer([infer_request], request_config)[0]

            # Check response
            if not response or not response.choices or len(response.choices) == 0:
                print("Warning: Model returned empty response")
                return "Model returned empty response, please retry"

            content = response.choices[0].message.content
            print(f"Received model response, length: {len(content) if content else 0}")

            # Return result
            return content
        except Exception as e:
            error_msg = f"Error during reasoning: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return error_msg

class VideoChatSystem:
    """Video chat system, integrating video processing and reasoning functionality"""

    def __init__(self, vision_model_path: str = "Qwen/Qwen2.5-Vl-72B-Instruct",
                 reasoning_model_path: str = "Qwen/Qwen2.5-3B-Instruct"):
        """Initialize video chat system

        Args:
            vision_model_path: Path to vision model
            reasoning_model_path: Path to reasoning model
        """
        # Initialize parameters
        # Video parsing module parameters
        self.vision_max_tokens = 6144  # Video parsing needs more tokens to describe video content
        self.vision_temperature = 0.1  # Video parsing needs more deterministic description
        self.vision_model_path = vision_model_path

        # Reasoning module parameters
        self.reasoning_max_tokens = 4096
        self.reasoning_temperature = 0.7
        self.reasoning_model_path = reasoning_model_path

        # Initialize processors, pass parameters
        self.video_processor = VideoProcessor(
            vision_model_path=self.vision_model_path,
            vision_max_tokens=self.vision_max_tokens,
            vision_temperature=self.vision_temperature
        )
        self.reasoning_processor = ReasoningProcessor(
            reasoning_model_path=self.reasoning_model_path,
            reasoning_max_tokens=self.reasoning_max_tokens,
            reasoning_temperature=self.reasoning_temperature
        )

        self.conversation_history = {}  # Store conversation history for different sessions
        self.video_descriptions = {}  # Store video descriptions for different sessions

        # Create directory for saving chat history
        self.chat_history_dir = os.path.join(os.getcwd(), "chat_history")
        os.makedirs(self.chat_history_dir, exist_ok=True)

    def set_inference_params(self, max_tokens: int = 4096, temperature: float = 0.7,
                           vision_max_tokens: Optional[int] = None, vision_temperature: Optional[float] = None,
                           vision_model_path: Optional[str] = None, reasoning_model_path: Optional[str] = None):
        """
        Set inference parameters

        Args:
            max_tokens: Reasoning module maximum output tokens
            temperature: Reasoning module temperature, controls randomness
            vision_max_tokens: Video parsing module maximum output tokens, if None, do not modify
            vision_temperature: Video parsing module temperature, if None, do not modify
            vision_model_path: Path to vision model, if None, do not modify
            reasoning_model_path: Path to reasoning model, if None, do not modify
        """
        # Set reasoning module parameters
        self.reasoning_max_tokens = max_tokens
        self.reasoning_temperature = temperature
        self.reasoning_processor.reasoning_max_tokens = max_tokens
        self.reasoning_processor.reasoning_temperature = temperature

        # If video parsing module parameters are provided, set them
        if vision_max_tokens is not None:
            self.vision_max_tokens = vision_max_tokens
            self.video_processor.vision_max_tokens = vision_max_tokens
        if vision_temperature is not None:
            self.vision_temperature = vision_temperature
            self.video_processor.vision_temperature = vision_temperature

        # If model paths are provided, reinitialize processors
        need_reinit_vision = False
        need_reinit_reasoning = False

        if vision_model_path is not None and vision_model_path != self.vision_model_path:
            self.vision_model_path = vision_model_path
            need_reinit_vision = True

        if reasoning_model_path is not None and reasoning_model_path != self.reasoning_model_path:
            self.reasoning_model_path = reasoning_model_path
            need_reinit_reasoning = True

        # Reinitialize processors if needed
        if need_reinit_vision:
            self.video_processor = VideoProcessor(
                vision_model_path=self.vision_model_path,
                vision_max_tokens=self.vision_max_tokens,
                vision_temperature=self.vision_temperature
            )

        if need_reinit_reasoning:
            self.reasoning_processor = ReasoningProcessor(
                reasoning_model_path=self.reasoning_model_path,
                reasoning_max_tokens=self.reasoning_max_tokens,
                reasoning_temperature=self.reasoning_temperature
            )

    def process_video(self, video_path: str, session_id: str, question: str = None) -> str:
        """
        Process video and store description

        Args:
            video_path: Video file path
            session_id: Session ID
            question: User question, used to guide the visual model to focus on relevant content

        Returns:
            Video processing status message
        """
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                return f"Error: Video file {video_path} does not exist"

            print(f"Processing video: {video_path}")

            # Analyze video
            video_description = self.video_processor.analyze_video(video_path, question)

            # Store video description and path
            self.video_descriptions[session_id] = video_description

            # Store video path
            if not hasattr(self, 'video_paths'):
                self.video_paths = {}
            self.video_paths[session_id] = os.path.abspath(video_path)

            # Initialize session history
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Print video processing result
            print("\n=== Video Processing Result ===")
            print(video_description)
            print("=== End of Result ===\n")

            return f"Video processing completed."
        except Exception as e:
            return f"Error processing video: {str(e)}"

    def chat(self, question: str, session_id: str) -> str:
        """
        Chat with the system

        Args:
            question: User question
            session_id: Session ID

        Returns:
            System answer
        """
        # Check if video has been processed
        if session_id not in self.video_descriptions:
            return "Please upload and process a video before asking questions."

        # Get video description
        video_description = self.video_descriptions[session_id]

        # Get conversation history
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        history = self.conversation_history[session_id]

        try:
            # Perform reasoning
            print(f"\nProcessing question: {question}")
            print(f"Current conversation history length: {len(history)}")

            response = self.reasoning_processor.reason(video_description, question, history)

            # Check if response is empty
            if not response or len(response.strip()) == 0:
                print("Warning: Model returned empty answer")
                return "Model returned empty answer, please retry"

            print(f"Received model response, length: {len(response)}")

            # Update conversation history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
            self.conversation_history[session_id] = history

            # Do not save history after each conversation, save it when user exits
            return response

        except Exception as e:
            error_msg = f"Error during reasoning: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return error_msg

    def reset_session(self, session_id: str) -> str:
        """
        Reset session

        Args:
            session_id: Session ID

        Returns:
            Reset status message
        """
        # Save chat history
        if session_id in self.conversation_history and self.conversation_history[session_id]:
            self.save_chat_history(session_id)
            print(f"Chat history saved before reset, session ID: {session_id}")
            del self.conversation_history[session_id]

        if session_id in self.video_descriptions:
            del self.video_descriptions[session_id]

        if hasattr(self, 'video_paths') and session_id in self.video_paths:
            del self.video_paths[session_id]

        return "Session has been reset."

    def save_chat_history(self, session_id: str) -> None:
        """
        Save chat history

        Args:
            session_id: Session ID
        """
        if session_id not in self.conversation_history or not self.conversation_history[session_id]:
            return

        # Create filename, use date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{session_id}_{timestamp}.json"
        filepath = os.path.join(self.chat_history_dir, filename)

        # Get video path
        video_path = ""
        if hasattr(self, 'video_paths') and session_id in self.video_paths:
            video_path = self.video_paths[session_id]

        # Prepare data to save
        data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "video_path": video_path,
            "video_description": self.video_descriptions.get(session_id, ""),
            "conversation": self.conversation_history[session_id]
        }

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Chat history saved to: {filepath}")

def save_uploaded_file(file_content: bytes) -> str:
    """
    Save uploaded file

    Args:
        file_content: File content

    Returns:
        Saved file path
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    return temp_path

def main():
    """Main function, used to demonstrate system functionality, modified to only require video when session starts"""
    print("Initializing video chat system...")
    system = VideoChatSystem()
    print("System initialization complete")

    session_id = "demo"
    video_processed = False
    has_conversation = False

    while True:
        # If video has not been processed, get question and video first
        if not video_processed:
            print("\nPlease enter your question (enter 'q' to quit):")
            question = input().strip()

            if question.lower() == 'q':
                break

            print(f"Question received: {question}")
            print("Now please provide the video file path.")

            video_path = input().strip()

            if video_path.lower() == 'q':
                break

            if not os.path.exists(video_path):
                print(f"Error: File {video_path} does not exist")
                continue

            print(f"Processing video with question: {question}")
            print("Please wait...")
            result = system.process_video(video_path, session_id, question)
            print(result)
            video_processed = True

            # Automatically answer current question
            print("\nAnswering your question, please wait...")
            response = system.chat(question, session_id)
            print("Answer:")
            print(response)
            has_conversation = True

        # If video has been processed, only get question
        else:
            print("\nPlease enter your question (enter 'q' to quit, 'reset' to reset the session):")
            question = input().strip()

            if question.lower() == 'q':
                if video_processed and has_conversation:
                    # Save chat history and exit
                    system.save_chat_history(session_id)
                    print(f"Chat history saved, session ID: {session_id}")
                break

            if question.lower() == 'reset':
                if video_processed:
                    result = system.reset_session(session_id)
                    print(result)
                    video_processed = False
                    has_conversation = False
                continue

            # Directly answer question, no need to process video again
            print(f"Question received: {question}")
            print("\nAnswering your question, please wait...")
            response = system.chat(question, session_id)
            print("Answer:")
            print(response)

if __name__ == "__main__":
    main()
