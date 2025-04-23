import re
import json
import random
import time
import os
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
from swift.plugin import ORM, orms

class APIPool:
    """
    API Pool class for managing multiple API keys, implementing load balancing and fault tolerance.
    When an API key encounters issues (such as insufficient balance), it will be marked.
    If a specified number of issues occur, the API key will be flagged in the API log and no longer used.
    """
    
    def __init__(self, api_keys: List[str], max_error_count: int = 10, log_dir: str = None, retry_interval_minutes: int = 30):
        """
        Initialize the API pool
        
        Args:
            api_keys: List of API keys
            max_error_count: Maximum number of errors allowed for an API key, beyond which it will be marked as unavailable
            log_dir: Log directory, defaults to api_logs in the current directory
            retry_interval_minutes: API restart check interval (minutes), default is 30 minutes
        """
        self.api_keys = api_keys
        self.max_error_count = max_error_count
        self.error_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self.available_keys = set(api_keys)
        self.unavailable_timestamps: Dict[str, float] = {}  # Record timestamps when API keys are marked as unavailable
        self.retry_interval_seconds = retry_interval_minutes * 60  # Convert to seconds
        
        # Set log directory
        if log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), 'api_logs')
        else:
            self.log_dir = log_dir
            
        # Create log directory (if it doesn't exist)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Log file path
        self.log_file = os.path.join(self.log_dir, f'api_pool_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Initialize log
        self._log(f"API pool initialized with {len(api_keys)} API keys")
        for key in api_keys:
            self._log(f"Added API key: {self._mask_api_key(key)}")
    
    def _mask_api_key(self, api_key: str) -> str:
        """
        Mask API key, showing only the first 4 and last 4 characters
        
        Args:
            api_key: Original API key
            
        Returns:
            Masked API key
        """
        if len(api_key) <= 8:
            return api_key
        return f"{api_key[:4]}...{api_key[-4:]}"
    
    def _log(self, message: str, is_error: bool = False):
        """
        Record log
        
        Args:
            message: Log message
            is_error: Whether it's an error log, only error logs are recorded
        """
        # Only record error logs
        if is_error:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            
            # Print to console
            print(log_message)
            
            # Write to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def get_api_key(self) -> Optional[str]:
        """
        Get an available API key and check if unavailable API keys can be re-enabled
        
        Returns:
            Available API key, or None if no API key is available
        """
        # Check if any unavailable API keys can be re-enabled
        current_time = time.time()
        keys_to_check = []
        
        # Collect API keys that need to be rechecked
        for api_key, timestamp in list(self.unavailable_timestamps.items()):
            if current_time - timestamp >= self.retry_interval_seconds:
                keys_to_check.append(api_key)
                
        # Recheck these API keys
        for api_key in keys_to_check:
            self._log(f"Rechecking API key: {self._mask_api_key(api_key)}, {self.retry_interval_seconds//60} minutes have passed")
            # Reset error count and add API key back to available list
            self.reset_error_count(api_key)
            # Remove from unavailable timestamps dictionary
            self.unavailable_timestamps.pop(api_key, None)
            
        if not self.available_keys:
            self._log("Warning: No available API keys", is_error=True)
            return None
        
        # Randomly select an available API key
        api_key = random.choice(list(self.available_keys))
        return api_key
    
    def mark_error(self, api_key: str, error_message: str):
        """
        Mark an API key as having an error
        
        Args:
            api_key: API key with error
            error_message: Error information
        """
        if api_key not in self.api_keys:
            self._log(f"Warning: Attempting to mark unknown API key: {self._mask_api_key(api_key)}", is_error=True)
            return
        
        # Increase error count
        self.error_counts[api_key] += 1
        current_count = self.error_counts[api_key]
        
        self._log(f"API key {self._mask_api_key(api_key)} encountered an error ({current_count}/{self.max_error_count}): {error_message}", is_error=True)
        
        # If error count exceeds threshold, mark as unavailable
        if current_count >= self.max_error_count and api_key in self.available_keys:
            self.available_keys.remove(api_key)
            # Record timestamp when API key is marked as unavailable
            self.unavailable_timestamps[api_key] = time.time()
            self._log(f"API key {self._mask_api_key(api_key)} has been marked as unavailable, reached maximum error count {self.max_error_count}, will be rechecked in {self.retry_interval_seconds//60} minutes", is_error=True)
    
    def reset_error_count(self, api_key: str):
        """
        Reset the error count for an API key
        
        Args:
            api_key: API key to reset
        """
        if api_key not in self.api_keys:
            self._log(f"Warning: Attempting to reset unknown API key: {self._mask_api_key(api_key)}", is_error=True)
            return
        
        # Reset error count
        self.error_counts[api_key] = 0
        
        # If API key is not in the available list, add it back
        if api_key not in self.available_keys:
            self.available_keys.add(api_key)
    
    def get_status(self) -> Dict:
        """
        Get API pool status
        
        Returns:
            Dictionary containing API pool status information
        """
        current_time = time.time()
        unavailable_info = {}
        
        for api_key, timestamp in self.unavailable_timestamps.items():
            elapsed_seconds = current_time - timestamp
            remaining_seconds = max(0, self.retry_interval_seconds - elapsed_seconds)
            unavailable_info[self._mask_api_key(api_key)] = {
                "elapsed_minutes": round(elapsed_seconds / 60, 1),
                "remaining_minutes": round(remaining_seconds / 60, 1)
            }
            
        return {
            "total_keys": len(self.api_keys),
            "available_keys": len(self.available_keys),
            "unavailable_keys": len(self.api_keys) - len(self.available_keys),
            "error_counts": {self._mask_api_key(k): v for k, v in self.error_counts.items()},
            "unavailable_info": unavailable_info
        }


class ConsistencyReward(ORM):
    """
    Reward function, used to evaluate whether the model's answer is consistent with the training model's answer.
    
    Extracts the content after the last Question and the content between <think></think> tags from the content field,
    makes the model give options based on the Question and think process, and then compares the model's options with the training model's options.
    If consistent, returns 1, otherwise returns 0.
    """
    
    # Define class constants for answer recognition
    BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
    ANSWER_PATTERN = r"answer\s+([A-H])\.?"  # answer pattern
    SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."  # pattern with dots
    SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"  # pattern without dots
    VALID_OPTIONS = set('ABCDEFGH')
    
    def __init__(self, api_keys=None):
        """
        Initialize the reward function
        
        Args:
            api_keys: List of API keys for calling external models for inference
        """
        # Default API key list
        # input your api keys here
        default_api_keys = [
           
        ]
        
        # Use the provided API key list or default list
        if api_keys is None:
            api_keys = default_api_keys
        elif isinstance(api_keys, str):
            api_keys = [api_keys]  # If a single API key is provided, convert it to a list
            
        # Create API pool
        self.api_pool = APIPool(api_keys, max_error_count=10)
        
        # Ali Baidu platform API base URL
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    @staticmethod
    def extract_question(text):
        """
        Extract question content from text, supporting multiple formats
        
        Args:
            text: Text to extract question from
            
        Returns:
            Extracted question content
        """
        if not text:
            return ""
            
        # First try to find the "Question:" pattern
        if "Question:" in text:
            question_parts = text.split("Question:")
            return "Question:" + question_parts[-1].strip()
            
        # Try to find the pattern containing options (A. B. C. D. format)
        options_pattern = r"(.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)\s*([A-D]\s*\.\s*.*?)"
        match = re.search(options_pattern, text, re.DOTALL)
        if match:
            return match.group(0).strip()
            
        # Try to find the "Choices:" pattern
        if "Choices:" in text:
            # Extract the complete content containing the question and options
            return text.strip()
            
        # If it's a messages format, may contain video description and question
        # Try to extract the entire content as the question
        return text.strip()
    
    @staticmethod
    def extract_think_content(text):
        """
        Extract content between <think></think> tags
        
        Args:
            text: Text to extract content from
            
        Returns:
            Extracted content between <think></think> tags
        """
        if not text:
            return ""
        think_pattern = r"<think>(.*?)</think>"
        match = re.search(think_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer format, extract answer from text
        
        Args:
            answer: Text containing the answer
            
        Returns:
            Normalized answer (a letter from A-H)
        """
        answer = answer.strip()
        
        # First try to find standard format answer ($\boxed{X}$)
        boxed_matches = list(re.finditer(self.BOXED_PATTERN, answer, re.IGNORECASE))
        if boxed_matches:
            # Use the last matched answer
            return boxed_matches[-1].group(1).upper()
            
        # Next look for answer pattern
        answer_matches = list(re.finditer(self.ANSWER_PATTERN, answer, re.IGNORECASE))
        if answer_matches:
            # Use the first matched answer
            return answer_matches[0].group(1).upper()
            
        # Finally find single letters
        # First find those with dots
        dot_matches = list(re.finditer(self.SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
        if dot_matches:
            # Use the last match with dots
            return dot_matches[-1].group(1).upper()
            
        # Finally find those without dots
        simple_matches = list(re.finditer(self.SIMPLE_PATTERN, answer, re.IGNORECASE))
        if simple_matches:
            # Use the last match without dots
            return simple_matches[-1].group(1).upper()
            
        # Extract from <answer> tag
        answer_tag_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_tag_pattern, answer, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # Recursively call self to process extracted content
            return self.normalize_answer(answer_text)
            
        # If nothing is found, return the original text (will result in 0 points in subsequent comparison)
        return answer.upper()
    
    def call_api_for_inference(self, question, think_content):
        """
        Call Ali Baidu API for inference
        
        Args:
            question: Question content
            think_content: Thinking process content
            
        Returns:
            Answer from the API
        """
        if not question or not think_content:
            print("Question content or thinking process is empty, cannot call API")
            return ""
        
        # Maximum retries
        max_retries = min(3, len(self.api_pool.available_keys))
        retry_count = 0
        
        while retry_count < max_retries:
            # Get API key from API pool
            api_key = self.api_pool.get_api_key()
            if api_key is None:
                print("No available API key, cannot call API")
                return ""
            
            try:
                prompt = f"""This is a multiple-choice question. Based on the following question and thinking process, select the most appropriate option (one letter from A-H).

Please answer with only one letter, without any explanation or additional content.

Question: {question}

Thinking process: {think_content}

Please respond with only one letter (A-H):"""
                
                # Create OpenAI client
                client = OpenAI(
                    api_key=api_key,
                    base_url=self.base_url,
                )
                
                # Call API
                completion = client.chat.completions.create(
                    model="qwen2.5-3b-instruct",  # Use Ali Baidu platform model
                    messages=[
                        {'role': 'system', 'content': 'You are a professional question-answering assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                # Extract answer
                api_answer = completion.choices[0].message.content.strip()
                normalized_answer = self.normalize_answer(api_answer)
                # print(f"API inference result: original={api_answer}, normalized={normalized_answer}")
                return normalized_answer
                
            except Exception as e:
                error_message = str(e)
                # Mark API key as error
                self.api_pool.mark_error(api_key, error_message)
                
                # Check error type, determine if retry is needed
                if "余额不足" in error_message or "insufficient_quota" in error_message:
                    print(f"API key balance insufficient: {error_message}")
                    # Continue to try the next API key
                    retry_count += 1
                    continue
                elif "rate_limit_exceeded" in error_message:
                    print(f"API call frequency limit exceeded: {error_message}")
                    # Wait for a while and retry
                    time.sleep(2)
                    retry_count += 1
                    continue
                else:
                    print(f"API call failed: {error_message}")
                    # Other errors, also try to retry
                    retry_count += 1
                    continue
        
        print(f"Tried {max_retries} times API call, all failed")
        return ""
    
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        """
        Evaluate the consistency between model answers and training model answers
        
        Args:
            completions: List of model-generated answers
            solution: List of correct answers, used to pre-judge model answer correctness
        
        Returns:
            List of consistency rewards, 1.0 for correct, 0.0 for incorrect
        """
        rewards = []
        
        # Ensure completions and solution are lists
        if not isinstance(completions, list):
            completions = [completions]
            
        if solution is not None and not isinstance(solution, list):
            solution = [solution] * len(completions)
        
        # Print all available fields for debugging
        # print(f"Available fields: {list(kwargs.keys())}")
        
        # Extract question content from 'messages' field
        question_field = None
        
        # If messages field exists, extract user question
        if 'messages' in kwargs and kwargs['messages'] is not None:
            messages = kwargs['messages']
            # print(f"Found messages field, trying to extract question content")·
            
            # If messages is a list of lists (batch processing)
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
        
        # Directly extract think content and answer from completions
        for i, completion in enumerate(completions):
            try:
                # Extract think content from model answer
                think_content = self.extract_think_content(completion)
                if not think_content:
                    # print(f"Unable to extract think content from sample {i} model answer")
                    rewards.append(0.0)
                    continue
                
                # Standardize model answer
                normalized_completion = self.normalize_answer(completion)
                
                # If correct answer is provided, first check if model answer is correct
                if solution is not None:
                    # Ensure index is within range
                    sol = solution[i] if i < len(solution) else solution[-1]
                    normalized_solution = self.normalize_answer(sol)
                    
                    # If model answer is incorrect, directly return 0.0, no API call
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
                        print(f"Unable to get question content from sample {i}")
                        rewards.append(0.0)
                        continue
                    
                    # Print part of question content for debugging
                    # if ques:
                    #     print(f"问题内容前100个字符: {ques[:100]}...")
                    # else:
                    #     print("问题内容为空")
                    
                    # If question content is empty, try to extract from completions
                    if not ques:
                        print("Question content is empty, trying to extract from completions")
                        question_content = self.extract_question(completion)
                    else:
                        # Extract question content
                        question_content = self.extract_question(ques)
                    
                    if not question_content:
                        print(f"Unable to extract question content from sample {i}")
                        rewards.append(0.0)
                        continue
                else:
                    # If no question field found, try using completions directly
                    print(f"No question field found, trying to use completions directly")
                    # Assume model answer contains question content
                    question_content = completion
                    if not question_content:
                        print(f"Unable to extract question content from completions")
                        rewards.append(0.0)
                        continue
                
                # Print complete question content for user reference
                # print(f"\nExtracted complete question content:\n{question_content}\n")
                
                # Call API for inference
                api_answer = self.call_api_for_inference(question_content, think_content)
                
                # If API returns a valid answer, compare with model answer
                if api_answer and api_answer in self.VALID_OPTIONS:
                    # Compare API inference result with model answer
                    if api_answer == normalized_completion:
                        # print(f"一致性检查: API推理结果={api_answer}, 模型答案={normalized_completion}, 一致")
                        rewards.append(1.0)
                    else:
                        # print(f"一致性检查: API推理结果={api_answer}, 模型答案={normalized_completion}, 不一致")
                        rewards.append(0.0)
                else:
                    # API did not return a valid answer, default to 0
                    print(f"API did not return a valid answer, model answer={normalized_completion}, default to 0")
                    rewards.append(0.0)
            except Exception as e:
                print(f"Processing sample {i} when error occurred: {e}")
                rewards.append(0.0)  # Error occurred, return 0
        
        return rewards
    
# Register reward function
orms['consistency'] = ConsistencyReward
