"""
Selection operators for genetic algorithm.

This module implements various selection strategies for choosing parents.
"""

from typing import List, Optional
import numpy as np
from loguru import logger

from .population import Population


class SelectionOperator:
    """
    Base class for selection operators.
    """
    
    def select(self, population: Population, n: int) -> List[int]:
        """
        Select parent indices from population.
        
        Args:
            population: Population instance
            n: Number of parents to select
            
        Returns:
            List of chromosome indices
        """
        raise NotImplementedError


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    
    Selects parents by running tournaments among random individuals.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: Population, n: int) -> List[int]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Population instance
            n: Number of parents to select
            
        Returns:
            List of parent indices
        """
        if population.size() == 0:
            return []
        
        parents = []
        
        for _ in range(n):
            # Select random tournament participants
            tournament_indices = np.random.choice(
                population.size(),
                size=min(self.tournament_size, population.size()),
                replace=False
            )
            
            # Find winner (best fitness)
            winner_idx = None
            best_obj = float('inf')
            
            for idx in tournament_indices:
                obj = population.get_objective(idx)
                if obj is not None and obj < best_obj:
                    best_obj = obj
                    winner_idx = idx
            
            # If no fitness available, pick random
            if winner_idx is None:
                winner_idx = np.random.choice(tournament_indices)
            
            parents.append(winner_idx)
        
        return parents


class RankBasedSelection(SelectionOperator):
    """
    Rank-based selection operator.
    
    Selects parents based on rank, with higher probability for better ranks.
    """
    
    def __init__(self, selection_pressure: float = 2.0):
        """
        Initialize rank-based selection.
        
        Args:
            selection_pressure: Selection pressure (higher = more selective)
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population: Population, n: int) -> List[int]:
        """
        Select parents using rank-based selection.
        
        Args:
            population: Population instance
            n: Number of parents to select
            
        Returns:
            List of parent indices
        """
        if population.size() == 0:
            return []
        
        # Get sorted indices (best first)
        sorted_indices = population.sort_by_fitness(reverse=False)
        
        # Compute selection probabilities based on rank
        pop_size = population.size()
        probabilities = np.zeros(pop_size)
        
        for rank, idx in enumerate(sorted_indices):
            # Linear ranking: prob = (2 - s) / pop_size + 2 * rank * (s - 1) / (pop_size * (pop_size - 1))
            # where s is selection pressure
            prob = (
                (2.0 - self.selection_pressure) / pop_size
                + 2.0 * rank * (self.selection_pressure - 1.0)
                / (pop_size * (pop_size - 1))
            )
            probabilities[sorted_indices.index(idx)] = prob
        
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum()
        
        # Select parents
        parents = np.random.choice(
            pop_size,
            size=n,
            p=probabilities,
            replace=True
        )
        
        return parents.tolist()


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel (fitness proportionate) selection.
    
    Note: Requires fitness values to be positive. For minimization problems,
    fitness values should be inverted or transformed.
    """
    
    def __init__(self, fitness_transform: str = 'inverse'):
        """
        Initialize roulette wheel selection.
        
        Args:
            fitness_transform: How to transform fitness ('inverse', 'negate', 'none')
        """
        self.fitness_transform = fitness_transform
    
    def _transform_fitness(self, fitness: float) -> float:
        """Transform fitness for selection."""
        if self.fitness_transform == 'inverse':
            return 1.0 / (1.0 + fitness) if fitness >= 0 else 1.0 / (1.0 - fitness)
        elif self.fitness_transform == 'negate':
            return -fitness
        else:
            return fitness
    
    def select(self, population: Population, n: int) -> List[int]:
        """
        Select parents using roulette wheel selection.
        
        Args:
            population: Population instance
            n: Number of parents to select
            
        Returns:
            List of parent indices
        """
        if population.size() == 0:
            return []
        
        # Get transformed fitness values
        fitness_values = []
        for idx in range(population.size()):
            obj = population.get_objective(idx)
            if obj is not None:
                transformed = self._transform_fitness(obj)
            else:
                transformed = 1.0  # Default for unevaluated
            fitness_values.append(transformed)
        
        # Convert to probabilities
        fitness_array = np.array(fitness_values)
        fitness_array = np.maximum(fitness_array, 1e-10)  # Avoid zeros
        probabilities = fitness_array / fitness_array.sum()
        
        # Select parents
        parents = np.random.choice(
            population.size(),
            size=n,
            p=probabilities,
            replace=True
        )
        
        return parents.tolist()

