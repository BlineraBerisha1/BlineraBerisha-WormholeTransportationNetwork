"""
Crossover operators for genetic algorithm.

This module implements various crossover strategies for combining parent chromosomes.
"""

from typing import Tuple, List
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder, ChromosomeEncoder


class CrossoverOperator:
    """
    Base class for crossover operators.
    """
    
    def __init__(self, n_ships: int = 12, jump_limit: int = 500):
        """
        Initialize crossover operator.
        
        Args:
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
        self.encoder = ChromosomeEncoder(n_ships, jump_limit)
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        raise NotImplementedError


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover at ship boundary.
    
    Splits chromosomes at a ship boundary and swaps segments.
    """
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform single-point crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        # Choose crossover point at ship boundary
        crossover_ship = np.random.randint(1, self.n_ships)
        crossover_point = crossover_ship * self.jump_limit
        
        # Create children
        child1 = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        
        child2 = np.concatenate([
            parent2[:crossover_point],
            parent1[crossover_point:]
        ])
        
        return child1, child2


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover with gene-level mixing.
    """
    
    def __init__(self, mixing_prob: float = 0.5, n_ships: int = 12, jump_limit: int = 500):
        """
        Initialize uniform crossover.
        
        Args:
            mixing_prob: Probability of swapping each gene
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        super().__init__(n_ships, jump_limit)
        self.mixing_prob = mixing_prob
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform uniform crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        # Create mask for swapping
        mask = np.random.random(len(parent1)) < self.mixing_prob
        
        # Create children
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Swap genes based on mask
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        return child1, child2


class MultiPointCrossover(CrossoverOperator):
    """
    Multi-point crossover with multiple cut points.
    """
    
    def __init__(self, n_points: int = 2, n_ships: int = 12, jump_limit: int = 500):
        """
        Initialize multi-point crossover.
        
        Args:
            n_points: Number of crossover points
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        super().__init__(n_ships, jump_limit)
        self.n_points = n_points
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform multi-point crossover.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Tuple of (child1, child2)
        """
        # Generate crossover points
        points = sorted(np.random.choice(
            len(parent1),
            size=min(self.n_points, len(parent1) - 1),
            replace=False
        ))
        points = [0] + points + [len(parent1)]
        
        # Create children by alternating segments
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        use_parent1 = True
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            if use_parent1:
                child1[start:end] = parent1[start:end]
                child2[start:end] = parent2[start:end]
            else:
                child1[start:end] = parent2[start:end]
                child2[start:end] = parent1[start:end]
            
            use_parent1 = not use_parent1
        
        return child1, child2

