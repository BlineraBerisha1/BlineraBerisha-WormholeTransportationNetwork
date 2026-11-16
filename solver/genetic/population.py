"""
Population management for genetic algorithm.

This module handles population initialization, management, and statistics.
"""

from typing import List, Optional, Dict
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeEncoder, ChromosomeDecoder
from ..evaluation.fitness import FitnessEvaluator


class Population:
    """
    Manages a population of chromosomes for genetic algorithm.
    
    Provides:
    - Population storage and access
    - Fitness tracking
    - Statistics computation
    - Best solution tracking
    """
    
    def __init__(
        self,
        chromosomes: Optional[List[np.ndarray]] = None,
        evaluator: Optional[FitnessEvaluator] = None,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize population.
        
        Args:
            chromosomes: Initial list of chromosomes. If None, creates empty population.
            evaluator: Fitness evaluator instance
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.evaluator = evaluator
        
        if chromosomes is None:
            self.chromosomes: List[np.ndarray] = []
        else:
            self.chromosomes = chromosomes.copy()
        
        # Fitness cache
        self._fitness_cache: Dict[int, List[float]] = {}
        self._best_index: Optional[int] = None
        self._best_fitness: Optional[float] = None
    
    def add(self, chromosome: np.ndarray) -> None:
        """
        Add chromosome to population.
        
        Args:
            chromosome: Chromosome array
        """
        self.chromosomes.append(chromosome.copy())
        self._invalidate_cache()
    
    def remove(self, index: int) -> None:
        """
        Remove chromosome at index.
        
        Args:
            index: Index of chromosome to remove
        """
        if 0 <= index < len(self.chromosomes):
            del self.chromosomes[index]
            self._invalidate_cache()
    
    def replace(self, index: int, chromosome: np.ndarray) -> None:
        """
        Replace chromosome at index.
        
        Args:
            index: Index to replace
            chromosome: New chromosome
        """
        if 0 <= index < len(self.chromosomes):
            self.chromosomes[index] = chromosome.copy()
            self._invalidate_cache()
    
    def get(self, index: int) -> np.ndarray:
        """
        Get chromosome at index.
        
        Args:
            index: Index of chromosome
            
        Returns:
            Chromosome array
        """
        return self.chromosomes[index].copy()
    
    def size(self) -> int:
        """Get population size."""
        return len(self.chromosomes)
    
    def evaluate_all(self, use_cache: bool = True) -> List[List[float]]:
        """
        Evaluate fitness for all chromosomes.
        
        Args:
            use_cache: Whether to use fitness cache
            
        Returns:
            List of fitness vectors
        """
        if self.evaluator is None:
            raise RuntimeError("Fitness evaluator not set")
        
        fitness_values = []
        for idx, chromosome in enumerate(self.chromosomes):
            fitness = self.evaluator.evaluate(chromosome, use_cache=use_cache)
            fitness_values.append(fitness)
            self._fitness_cache[idx] = fitness
        
        # Update best
        self._update_best()
        
        return fitness_values
    
    def get_fitness(self, index: int) -> Optional[List[float]]:
        """
        Get fitness for chromosome at index.
        
        Args:
            index: Chromosome index
            
        Returns:
            Fitness vector or None if not evaluated
        """
        if index in self._fitness_cache:
            return self._fitness_cache[index].copy()
        return None
    
    def get_objective(self, index: int) -> Optional[float]:
        """
        Get objective value for chromosome at index.
        
        Args:
            index: Chromosome index
            
        Returns:
            Objective value or None
        """
        fitness = self.get_fitness(index)
        if fitness is not None:
            return fitness[0]
        return None
    
    def get_best(self) -> Optional[tuple[int, np.ndarray, float]]:
        """
        Get best chromosome in population.
        
        Returns:
            Tuple of (index, chromosome, fitness) or None if no evaluations
        """
        if self._best_index is None:
            return None
        
        return (
            self._best_index,
            self.chromosomes[self._best_index].copy(),
            self._best_fitness
        )
    
    def get_best_chromosome(self) -> Optional[np.ndarray]:
        """Get best chromosome (without index)."""
        best = self.get_best()
        if best is not None:
            return best[1]
        return None
    
    def _update_best(self) -> None:
        """Update best chromosome index."""
        if not self._fitness_cache:
            return
        
        best_idx = None
        best_obj = float('inf')
        
        for idx, fitness in self._fitness_cache.items():
            objective = fitness[0]
            if objective < best_obj:
                best_obj = objective
                best_idx = idx
        
        self._best_index = best_idx
        self._best_fitness = best_obj
    
    def _invalidate_cache(self) -> None:
        """Invalidate fitness cache."""
        self._fitness_cache.clear()
        self._best_index = None
        self._best_fitness = None
    
    def get_statistics(self) -> Dict:
        """
        Compute population statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.chromosomes:
            return {'size': 0}
        
        stats = {'size': len(self.chromosomes)}
        
        if self.evaluator is not None and self._fitness_cache:
            # Get fitness statistics
            fitness_stats = self.evaluator.get_statistics(self.chromosomes)
            stats.update(fitness_stats)
        
        # Best solution info
        best = self.get_best()
        if best is not None:
            stats['best_objective'] = best[2]
            stats['best_index'] = best[0]
        
        return stats
    
    def sort_by_fitness(self, reverse: bool = False) -> List[int]:
        """
        Get indices sorted by fitness (objective value).
        
        Args:
            reverse: If True, sort descending (worst first)
            
        Returns:
            List of indices sorted by fitness
        """
        if not self._fitness_cache:
            return list(range(len(self.chromosomes)))
        
        # Create list of (index, objective) tuples
        indexed_fitness = [
            (idx, self.get_objective(idx) or float('inf'))
            for idx in range(len(self.chromosomes))
        ]
        
        # Sort by objective
        indexed_fitness.sort(key=lambda x: x[1], reverse=reverse)
        
        return [idx for idx, _ in indexed_fitness]
    
    def get_elite(self, n: int) -> List[np.ndarray]:
        """
        Get top N chromosomes (elite).
        
        Args:
            n: Number of elite chromosomes
            
        Returns:
            List of elite chromosomes
        """
        sorted_indices = self.sort_by_fitness(reverse=False)
        elite_indices = sorted_indices[:min(n, len(sorted_indices))]
        
        return [self.chromosomes[idx].copy() for idx in elite_indices]
    
    def clear(self) -> None:
        """Clear population."""
        self.chromosomes.clear()
        self._invalidate_cache()
    
    def copy(self) -> 'Population':
        """Create a copy of the population."""
        new_pop = Population(
            chromosomes=self.chromosomes.copy(),
            evaluator=self.evaluator,
            n_ships=self.n_ships,
            jump_limit=self.jump_limit
        )
        new_pop._fitness_cache = self._fitness_cache.copy()
        new_pop._best_index = self._best_index
        new_pop._best_fitness = self._best_fitness
        return new_pop

