import re
import json
import random
import time
import os
import requests
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from swift.plugin import ORM, orms

# Local API client class, used to communicate with the locally running model API service
class LocalAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        """Initialize the local API client

        Args:
            base_url: The base URL of the local API service, default is http://localhost:8000
        """
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"
        self.health_endpoint = f"{base_url}/health"

        # Check if the API service is available
        self._check_api_available()

    def _check_api_available(self):
        """Check if the API service is available"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                print("Local API service is running normally")
            else:
                print(f"Warning: Local API service returned status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Warning: Unable to connect to the local API service: {str(e)}")
            print("Please make sure the start_consistency_model_server.py script is running")

    def generate_completion(self, prompt, system_prompt="You are a professional question answering assistant.", max_tokens=10, temperature=0.1):
        """Generate text completion

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum number of generated tokens
            temperature: Temperature parameter

        Returns:
            Generated text, returns an empty string if failed
        """
        try:
            # Build request data
            request_data = {
                "model": "qwen2.5-vl-3b-instruct",  # Model name, actually the server will not use this value
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Send request
            response = requests.post(self.api_endpoint, json=request_data, timeout=30)

            # Check response
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    print("API response format is incorrect")
                    return ""
            else:
                print(f"API request failed, status code: {response.status_code}, response: {response.text}")
                return ""

        except Exception as e:
            print(f"Error occurred when calling local API: {str(e)}")
            return ""


class ConsistencyReward(ORM):
    """
    Reward function, used to evaluate whether the model's answer is consistent with the training model's answer.

    Extracts the content after the last Question in the content field and the content inside the <think></think> tag,
    lets the model provide an option based on the Question and think process, then compares the model's option with the training model's option.
    Returns 1 if consistent, 0 if not.

    Uses the local qwen3B model for consistency checking, not through an API call.
    """

    # Define class constants for answer recognition
    BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
    ANSWER_PATTERN = r"answer\s+([A-H])\.?"  # answer pattern
    SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."  # Pattern with dot
    SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"  # Pattern without dot
    VALID_OPTIONS = set('ABCDEFGH')

    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the reward function

        Args:
            api_url: The URL of the local API service, default is http://localhost:8000
        """
        # Create local API client
        self.api_client = LocalAPIClient(api_url)

    @staticmethod
    def extract_question(text):
        """Extract question content from text, supports multiple formats"""
        if not text:
            return ""

        # First, try to find the "Question:" pattern
        if "Question:" in text:
            question_parts = text.split("Question:")
            return "Question:" + question_parts[-1].strip()

        # Try to find a pattern with options (A. B. C. D. format)
        options_pattern = r"(.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)"
        match = re.search(options_pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()

        # Try to find "Choices:"
        if "Choices:" in text:
            # Extract the complete content containing the question and options
            return text.strip()

        # If it's messages format, may contain video description and question
        # Try to extract the entire content as the question
        return text.strip()

    @staticmethod
    def extract_think_content(text):
        """Extract content inside <think></think> tag from text"""
        if not text:
            return ""
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_answer(self, answer: str) -> str:
        """Standardize answer format, extract answer from text"""
        answer = answer.strip()

        # First try to find the standard format answer ($\boxed{X}$)
        boxed_matches = list(re.finditer(self.BOXED_PATTERN, answer, re.IGNORECASE))
        if boxed_matches:
            # Use the last matched answer
            return boxed_matches[-1].group(1).upper()

        # Next, look for answer pattern
        answer_matches = list(re.finditer(self.ANSWER_PATTERN, answer, re.IGNORECASE))
        if answer_matches:
            # Use the first matched answer
            return answer_matches[0].group(1).upper()

        # Finally, look for a single letter
        # First, find those with dot
        dot_matches = list(re.finditer(self.SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
        if dot_matches:
            # Use the last match with dot
            return dot_matches[-1].group(1).upper()

        # Finally, find those without dot
        simple_matches = list(re.finditer(self.SIMPLE_PATTERN, answer, re.IGNORECASE))
        if simple_matches:
            # Use the last match without dot
            return simple_matches[-1].group(1).upper()

        # Extract from <answer> tag
        answer_tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_tag_pattern, answer, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # Recursively call itself to process the extracted content
            return self.normalize_answer(answer_text)

        # If none found, return original text (will result in 0 score in later comparison)
        return answer.upper()

    def call_local_model_for_inference(self, question, think_content):
        """Use local API service for inference"""
        if not question or not think_content:
            print("Question content or thinking process is empty, unable to perform inference")
            return ""

        try:
            # Build prompt - simplify prompt to reduce complexity
            prompt = f"""This is a multiple-choice question. Based on the following question and thinking process, select the most appropriate option (one letter from A-H).

Please answer with only one letter, without any explanation or additional content.

Question: {question}

Thinking process: {think_content}

Please respond with only one letter (A-H):"""

            # Call API service
            model_answer = self.api_client.generate_completion(
                prompt=prompt,
                system_prompt="You are a professional question-answering assistant.",
                max_tokens=10,
                temperature=0.1
            )

            if model_answer:
                # Extract answer
                normalized_answer = self.normalize_answer(model_answer)
                # print(f"Local API inference result: raw={model_answer}, normalized={normalized_answer}")
                return normalized_answer
            else:
                # If API call fails, try to extract answer directly from thinking process
                print("API call failed, trying backup method")
                think_answer = self.extract_answer_from_think(think_content)
                if think_answer:
                    print(f"Extracted answer from thinking process: {think_answer}")
                    return think_answer
                return ""

        except Exception as e:
            print(f"Local API inference failed: {str(e)}")
            # If all methods fail, try to extract answer from thinking process
            think_answer = self.extract_answer_from_think(think_content)
            if think_answer:
                print(f"Extracted answer from thinking process in exception handling: {think_answer}")
                return think_answer
            return ""

    def extract_answer_from_think(self, think_content):
        """Directly extract answer from thinking process"""
        try:
            # Try to find the last mentioned option
            for pattern in [self.BOXED_PATTERN, self.ANSWER_PATTERN, self.SIMPLE_DOT_PATTERN, self.SIMPLE_PATTERN]:
                matches = list(re.finditer(pattern, think_content, re.IGNORECASE))
                if matches:
                    # Return the last matched answer
                    return matches[-1].group(1).upper()

            # Try to find conclusion part
            conclusion_patterns = [
                r"(?:so|thus|in conclusion|the answer is|choose|select|option|answer)[^A-H]*([A-H])\b",
                r"(?:the answer is|choose|select|option|answer)[^A-H]*([A-H])\b",
                r"\b([A-H])\s+(?:is correct)\b"
            ]

            for pattern in conclusion_patterns:
                matches = list(re.finditer(pattern, think_content, re.IGNORECASE))
                if matches:
                    return matches[-1].group(1).upper()

            return ""
        except Exception as e:
            print(f"Failed to extract answer from thinking process: {str(e)}")
            return ""

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        """
        Evaluate the consistency between the model answer and the training model answer

        Args:
            completions: List of answers generated by the model
            solution: List of correct answers, used to pre-judge whether the model answer is correct
            kwargs: Other parameters, including other fields in the dataset
                - content: field containing the thinking process
                - question: question text
                - query: field that may contain the question
                - prompt: field that may contain the question
                - input: field that may contain the question

        Returns:
            List of consistency rewards, 1.0 for consistent, 0.0 for inconsistent
        """
        rewards = []

        # Ensure completions and solution are both lists
        if not isinstance(completions, list):
            completions = [completions]

        if solution is not None and not isinstance(solution, list):
            solution = [solution] * len(completions)

        # Print all available fields for debugging
        # print(f"Available fields: {list(kwargs.keys())}")

        # Extract question content from 'messages' field
        question_field = None

        # If there is a messages field, extract the user question from it
        if 'messages' in kwargs and kwargs['messages'] is not None:
            messages = kwargs['messages']
            # print(f"Found messages field, trying to extract question content")

            # If messages is a list of lists (batch situation)
            if isinstance(messages, list) and all(isinstance(item, list) for item in messages):
                question_field = []
                for msg_list in messages:
                    # Find user message
                    user_content = ""
                    for msg in msg_list:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            user_content = msg.get('content', "")
                    question_field.append(user_content)
                # print(f"Extracted {len(question_field)} user questions from message list")

            # If messages is a single list
            elif isinstance(messages, list):
                # Find user message
                user_content = ""
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        user_content = msg.get('content', "")
                question_field = user_content
                # print(f"Extracted user question from message list")

        # If still not found, try other fields
        if question_field is None:
            question_field_names = ['question', 'query', 'prompt', 'input', 'text']
            for field_name in question_field_names:
                if field_name in kwargs and kwargs[field_name] is not None:
                    question_field = kwargs[field_name]
                    # print(f"Using field '{field_name}' as question content")
                    break

        # Directly extract thinking process and answer from completions
        for i, completion in enumerate(completions):
            try:
                # Extract thinking process from model answer
                think_content = self.extract_think_content(completion)
                if not think_content:
                    # print(f"Unable to extract thinking process from model answer of sample {i}")
                    rewards.append(0.0)
                    continue

                # Normalize model answer
                normalized_completion = self.normalize_answer(completion)

                # If a correct answer is provided, first check whether the model answer is correct
                if solution is not None:
                    # Ensure index is within range
                    sol = solution[i] if i < len(solution) else solution[-1]
                    normalized_solution = self.normalize_answer(sol)

                    # If the model answer is incorrect, return 0.0 directly, do not call API
                    if normalized_completion != normalized_solution:
                        # print(f"Model answer={normalized_completion}, correct answer={normalized_solution}, incorrect, skipping API call")
                        rewards.append(0.0)
                        continue

                # Get question content
                if question_field is not None:
                    # Ensure question is accessible
                    if isinstance(question_field, list) and i < len(question_field):
                        ques = question_field[i]
                    elif isinstance(question_field, str):
                        ques = question_field
                    else:
                        print(f"Unable to get question for sample {i}")
                        rewards.append(0.0)
                        continue

                    # Print part of the question content for debugging
                    # if ques:
                    #     print(f"First 100 characters of question content: {ques[:100]}...")
                    # else:
                    #     print("Question content is empty")

                    # If question content is empty, try to extract from completions
                    if not ques:
                        print("Question content is empty, trying to extract from completions")
                        question_content = self.extract_question(completion)
                    else:
                        # Extract question content
                        question_content = self.extract_question(ques)

                    if not question_content:
                        print(f"Unable to extract content from question of sample {i}")
                        rewards.append(0.0)
                        continue
                else:
                    # If no question field is found, try to use completions directly
                    print(f"No question field found, trying to use completions directly")
                    # Assume model answer contains question content
                    question_content = completion
                    if not question_content:
                        print(f"Unable to extract question content from completions")
                        rewards.append(0.0)
                        continue

                # Print the complete question content for user verification
                # print(f"\nExtracted full question content:\n{question_content}\n")

                # Use local API service for inference
                model_answer = self.call_local_model_for_inference(question_content, think_content)

                # If the API returns a valid answer, compare it with the model answer
                if model_answer and model_answer in self.VALID_OPTIONS:
                    # Compare whether the local API inference result is consistent with the model answer
                    if model_answer == normalized_completion:
                        print(f"Consistency check: Local API inference result={model_answer}, model answer={normalized_completion}, consistent")
                        rewards.append(1.0)
                    else:
                        print(f"Consistency check: Local API inference result={model_answer}, model answer={normalized_completion}, inconsistent")
                        rewards.append(0.0)
                else:
                    # API did not return a valid answer, default to inconsistent
                    print(f"Local API did not return a valid answer, model answer={normalized_completion}, default to inconsistent")
                    rewards.append(0.0)
            except Exception as e:
                print(f"Error occurred when processing sample {i}: {e}")
                rewards.append(0.0)  # Return 0 score when error occurs

        return rewards

# Register reward function
orms['consistency'] = ConsistencyReward
