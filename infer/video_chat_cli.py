#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video chat system command line interface
Provides a command line interface for interacting with video chat systems
"""

import os
import argparse
import datetime
from typing import Optional

from video_chat_system import VideoChatSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Video Chat System Command Line Interface")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--question", type=str, help="Question to ask")

    # Model paths
    parser.add_argument("--vision_model_path", type=str, default="Qwen/Qwen2.5-Vl-72B-Instruct",
                        help="Path to vision model (default: Qwen/Qwen2.5-Vl-72B-Instruct)")
    parser.add_argument("--reasoning_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Path to reasoning model (default: Qwen/Qwen2.5-VL-3B-Instruct)")

    # Reasoning module parameters
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Maximum number of tokens for reasoning module (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for reasoning module (default: 0.7, 0=deterministic, higher=more random)")

    # Video parsing module parameters
    parser.add_argument("--vision_max_tokens", type=int, default=6144,
                        help="Maximum number of tokens for vision module (default: 6144)")
    parser.add_argument("--vision_temperature", type=float, default=0.1,
                        help="Temperature for vision module (default: 0.1)")
    parser.add_argument("--use_keyframes", action="store_true", default=True,
                        help="Use keyframe extraction for video processing (default: True)")
    parser.add_argument("--no_keyframes", action="store_false", dest="use_keyframes",
                        help="Disable keyframe extraction for video processing")

    return parser.parse_args()

def interactive_mode(system: VideoChatSystem, video_path: Optional[str] = None,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   vision_max_tokens: int = 6144, vision_temperature: float = 0.1,
                   vision_model_path: str = "Qwen/Qwen2.5-Vl-72B-Instruct",
                   reasoning_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                   use_keyframes: bool = True):
    """
    Interactive mode, modified to only require video at session start

    Args:
        system: Video chat system instance
        video_path: Optional video file path
        max_tokens: Maximum output tokens for reasoning module
        temperature: Temperature for reasoning module
        vision_max_tokens: Maximum output tokens for vision module
        vision_temperature: Temperature for vision module
        vision_model_path: Path to vision model
        reasoning_model_path: Path to reasoning model
        use_keyframes: Whether to use keyframe extraction
    """
    # Set inference parameters
    system.set_inference_params(
        max_tokens=max_tokens,
        temperature=temperature,
        vision_max_tokens=vision_max_tokens,
        vision_temperature=vision_temperature,
        vision_model_path=vision_model_path,
        reasoning_model_path=reasoning_model_path,
        use_keyframes=use_keyframes
    )

    # Use date and time as session ID for easy tracking
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_processed = False
    has_conversation = False

    print("Welcome to the Video Chat System!")
    print(f"Current session ID: {session_id}")
    print("Enter 'q' to quit, 'reset' to reset the session")

    # If video path is provided, ask question first
    if video_path and os.path.exists(video_path):
        print("\nPlease enter your question before processing the video:")
        question = input().strip()
        if question.lower() != 'q':
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
        else:
            return
    elif video_path:
        print(f"Error: Video file {video_path} does not exist")

    while True:
        # If video has not been processed, first get question and video
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
                    # Create new session ID
                    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"New session ID: {session_id}")
                    video_processed = False
                    has_conversation = False
                continue

            # Directly answer question, no need to process video again
            print(f"Question received: {question}")
            print("\nAnswering your question, please wait...")
            response = system.chat(question, session_id)
            print("Answer:")
            print(response)
            has_conversation = True

def main():
    """Main function"""
    args = parse_args()

    print("Initializing video chat system...")
    system = VideoChatSystem(
        vision_model_path=args.vision_model_path,
        reasoning_model_path=args.reasoning_model_path,
        use_keyframes=args.use_keyframes
    )
    print("System initialization complete")

    if args.interactive or (not args.video and not args.question):
        # Interactive mode
        interactive_mode(system, args.video, args.max_tokens, args.temperature,
                         args.vision_max_tokens, args.vision_temperature,
                         args.vision_model_path, args.reasoning_model_path,
                         args.use_keyframes)
    elif args.video and args.question:
        # Single Q&A mode
        if not os.path.exists(args.video):
            print(f"Error: Video file {args.video} does not exist")
            return

        # Use date and time as session ID
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Session ID: {session_id}")

        print(f"Question: {args.question}")
        print(f"Processing video with question: {args.question}")
        print("Please wait...")
        # Set inference parameters
        system.set_inference_params(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            vision_max_tokens=args.vision_max_tokens,
            vision_temperature=args.vision_temperature,
            vision_model_path=args.vision_model_path,
            reasoning_model_path=args.reasoning_model_path,
            use_keyframes=args.use_keyframes
        )
        result = system.process_video(args.video, session_id, args.question)
        print(result)

        print("Answering your question, please wait...")
        response = system.chat(args.question, session_id)
        print(f"Answer:\n{response}")

        # Save chat history
        system.save_chat_history(session_id)
        print(f"Chat history saved, session ID: {session_id}")
    else:
        print("Error: Please provide both video file path and question, or use interactive mode")
        print("Use --interactive parameter to start interactive mode")
        print("Or use both --video and --question parameters for single Q&A")

if __name__ == "__main__":
    main()
