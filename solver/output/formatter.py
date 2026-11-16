"""
Solution formatting for submission to Optimise platform.

This module handles formatting solutions in the required JSON format.
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
from loguru import logger


class SolutionFormatter:
    """
    Formats solutions for submission to Optimise platform.
    """
    
    def __init__(
        self,
        challenge: str = "spoc-2-wormhole-transportation-network",
        problem: str = "wormhole-transportation-network"
    ):
        """
        Initialize solution formatter.
        
        Args:
            challenge: Challenge identifier
            problem: Problem identifier
        """
        self.challenge = challenge
        self.problem = problem
    
    def format_submission(
        self,
        chromosome: np.ndarray,
        submission_name: Optional[str] = None,
        submission_description: Optional[str] = None
    ) -> Dict:
        """
        Format chromosome as submission dictionary.
        
        Args:
            chromosome: Solution chromosome
            submission_name: Optional submission name
            submission_description: Optional submission description
            
        Returns:
            Submission dictionary
        """
        # Convert chromosome to list (JSON requires native types)
        decision_vector = chromosome.tolist()
        
        submission = {
            "decisionVector": decision_vector,
            "problem": self.problem,
            "challenge": self.challenge
        }
        
        if submission_name:
            submission["name"] = submission_name
        
        if submission_description:
            submission["description"] = submission_description
        
        return submission
    
    def create_submission_json(
        self,
        chromosome: np.ndarray,
        filepath: str,
        submission_name: Optional[str] = None,
        submission_description: Optional[str] = None
    ) -> None:
        """
        Create JSON submission file.
        
        Args:
            chromosome: Solution chromosome
            filepath: Path to output JSON file
            submission_name: Optional submission name
            submission_description: Optional submission description
        """
        submission = self.format_submission(
            chromosome, submission_name, submission_description
        )
        
        # Wrap in array (Optimise expects array of submissions)
        submission_array = [submission]
        
        # Write to file
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(submission_array, f, indent=2)
        
        logger.info(f"Submission file created: {path.absolute()}")
    
    def validate_submission(self, submission: Dict) -> Tuple[bool, List[str]]:
        """
        Validate submission format.
        
        Args:
            submission: Submission dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        if 'decisionVector' not in submission:
            errors.append("Missing 'decisionVector' field")
        else:
            decision_vector = submission['decisionVector']
            if not isinstance(decision_vector, list):
                errors.append("'decisionVector' must be a list")
            elif len(decision_vector) != 6000:
                errors.append(
                    f"'decisionVector' length {len(decision_vector)} != 6000"
                )
            else:
                # Check all integers
                if not all(isinstance(x, (int, np.integer)) for x in decision_vector):
                    errors.append("'decisionVector' must contain only integers")
        
        if 'problem' not in submission:
            errors.append("Missing 'problem' field")
        
        if 'challenge' not in submission:
            errors.append("Missing 'challenge' field")
        
        is_valid = len(errors) == 0
        return is_valid, errors


def create_submission_json(
    chromosome: np.ndarray,
    filepath: str,
    challenge: str = "spoc-2-wormhole-transportation-network",
    problem: str = "wormhole-transportation-network",
    submission_name: Optional[str] = None,
    submission_description: Optional[str] = None
) -> None:
    """
    Convenience function to create submission JSON file.
    
    Args:
        chromosome: Solution chromosome
        filepath: Path to output JSON file
        challenge: Challenge identifier
        problem: Problem identifier
        submission_name: Optional submission name
        submission_description: Optional submission description
    """
    formatter = SolutionFormatter(challenge, problem)
    formatter.create_submission_json(
        chromosome, filepath, submission_name, submission_description
    )

