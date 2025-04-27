import os
import re
import warnings
from typing import Dict, List, Union, Optional

from swift.plugin.orm import ORM

# Global dictionary for registering reward functions
orms = {}


class ChoiceAccuracy(ORM):
    """
    Choice accuracy reward function, used to evaluate whether the model's option answer matches the correct answer.

    Extracts the option (A-H) from the model's answer and compares it with the correct answer.
    Returns 1 if they match, 0 if they don't.
    """

    # Class constants for identifying answers
    BOXED_PATTERN = r"\$\\boxed\{([A-H])\}\$"
    ANSWER_PATTERN = r"answer\s+([A-H])\.?"  # answer pattern
    SIMPLE_DOT_PATTERN = r"(?:^|[^A-Za-z])([A-H])\s*\."  # pattern with dots, no restriction on content after the dot
    SIMPLE_PATTERN = r"(?:^|[^A-Za-z])([A-H])(?:$|[^A-Za-z])"  # pattern without dots
    VALID_OPTIONS = set('ABCDEFGH')

    def __init__(self):
        """
        Initialize the choice accuracy reward function
        """
        # Check if math_verify package is installed
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

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

        # Finally look for single letters
        # First find those with dots
        dot_matches = list(re.finditer(self.SIMPLE_DOT_PATTERN, answer, re.IGNORECASE))
        if dot_matches:
            # Use the last match with dots
            return dot_matches[-1].group(1).upper()

        # Lastly find those without dots
        simple_matches = list(re.finditer(self.SIMPLE_PATTERN, answer, re.IGNORECASE))
        if simple_matches:
            # Use the last match without dots
            return simple_matches[-1].group(1).upper()

        # If nothing is found, return the original text (will result in 0 points in subsequent comparison)
        return answer.upper()

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluate the consistency between model answers and correct answers

        Args:
            completions: List of model-generated answers
            solution: List of correct answers
            kwargs: Other parameters

        Returns:
            List of accuracy rewards, 1.0 for correct, 0.0 for incorrect
        """
        rewards = []

        # Ensure both completions and solution are lists
        if not isinstance(completions, list):
            completions = [completions]

        if not isinstance(solution, list):
            solution = [solution] * len(completions)

        for content, sol in zip(completions, solution):
            try:
                # Normalize answers and compare
                normalized_content = self.normalize_answer(content)
                normalized_solution = self.normalize_answer(sol)

                # If model answer matches correct answer, return 1.0, otherwise return 0.0
                reward = float(normalized_content == normalized_solution)

                # Optional: Print debug information
                # print(f"Model answer={normalized_content}, Correct answer={normalized_solution}, Reward={reward}")
            except Exception as e:
                print(f"Error processing answer: {e}")
                reward = 0.0  # Return 0 points when an error occurs

            rewards.append(reward)

        return rewards

# Register reward function
orms['choice_accuracy'] = ChoiceAccuracy

# For compatibility, also register under the name 'choice_accuracy'
from swift.plugin.orm import orms
orms['choice_accuracy'] = ChoiceAccuracy
