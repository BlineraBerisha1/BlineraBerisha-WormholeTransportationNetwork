"""
Configuration settings for the genetic algorithm solver.

This module defines default parameters and configuration management.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class SolverConfig:
    """
    Configuration parameters for the genetic algorithm solver.
    """
    # Population parameters
    population_size: int = 100
    n_ships: int = 12
    jump_limit: int = 500
    
    # Evolution parameters
    max_generations: int = 1000
    max_evaluations: Optional[int] = None
    max_time_seconds: Optional[int] = None
    
    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    crossover_type: str = 'single_point'  # 'single_point', 'uniform', 'multi_point'
    mutation_type: str = 'node_replacement'  # 'node_replacement', 'path_extension', 'path_truncation'
    
    # Selection parameters
    selection_type: str = 'tournament'  # 'tournament', 'rank_based', 'roulette'
    tournament_size: int = 3
    elitism_rate: float = 0.1  # Fraction of population to keep as elite
    
    # Replacement strategy
    replacement_strategy: str = 'generational'  # 'generational', 'steady_state', 'elitism'
    
    # Constraint handling
    origin_penalty: float = 1e6
    path_penalty: float = 1e6
    window_penalty: float = 1000.0
    adaptive_penalty: bool = True
    
    # Termination criteria
    target_fitness: Optional[float] = None
    stagnation_generations: int = 100
    min_feasible_rate: float = 0.0
    
    # Logging and output
    log_level: str = 'INFO'
    log_interval: int = 10  # Log every N generations
    save_interval: int = 50  # Save best solution every N generations
    output_dir: str = "./results"
    
    # Initialization
    initialization_type: str = 'mixed'  # 'random', 'heuristic', 'mixed'
    heuristic_ratio: float = 0.3  # Fraction of population initialized heuristically
    
    def validate(self) -> list[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if self.population_size < 2:
            errors.append("population_size must be >= 2")
        
        if not 0.0 <= self.crossover_rate <= 1.0:
            errors.append("crossover_rate must be in [0, 1]")
        
        if not 0.0 <= self.mutation_rate <= 1.0:
            errors.append("mutation_rate must be in [0, 1]")
        
        if not 0.0 <= self.elitism_rate <= 1.0:
            errors.append("elitism_rate must be in [0, 1]")
        
        if self.tournament_size < 2:
            errors.append("tournament_size must be >= 2")
        
        if self.max_generations < 1:
            errors.append("max_generations must be >= 1")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'population_size': self.population_size,
            'n_ships': self.n_ships,
            'jump_limit': self.jump_limit,
            'max_generations': self.max_generations,
            'max_evaluations': self.max_evaluations,
            'max_time_seconds': self.max_time_seconds,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'crossover_type': self.crossover_type,
            'mutation_type': self.mutation_type,
            'selection_type': self.selection_type,
            'tournament_size': self.tournament_size,
            'elitism_rate': self.elitism_rate,
            'replacement_strategy': self.replacement_strategy,
            'origin_penalty': self.origin_penalty,
            'path_penalty': self.path_penalty,
            'window_penalty': self.window_penalty,
            'adaptive_penalty': self.adaptive_penalty,
            'target_fitness': self.target_fitness,
            'stagnation_generations': self.stagnation_generations,
            'min_feasible_rate': self.min_feasible_rate,
            'log_level': self.log_level,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'output_dir': self.output_dir,
            'initialization_type': self.initialization_type,
            'heuristic_ratio': self.heuristic_ratio,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SolverConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SolverConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def load_config(filepath: Optional[str] = None) -> SolverConfig:
    """
    Load configuration from file or return default.
    
    Args:
        filepath: Path to config file. If None, returns default config.
        
    Returns:
        SolverConfig instance
    """
    if filepath is None:
        return SolverConfig()
    
    return SolverConfig.load(filepath)

