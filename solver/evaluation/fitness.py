"""
Fitness evaluation module for Wormhole Transportation Network.

This module provides a wrapper around the UDP fitness function with
caching and batch evaluation capabilities.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
from loguru import logger

# Import the UDP class from the parent directory
import sys
from pathlib import Path

# Add parent directory to path to import wormhole_udp
udp_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(udp_path))

try:
    from wormhole_udp import wormhole_traversal_udp
except ImportError:
    # Fallback: try importing from current directory
    import importlib.util
    udp_file = udp_path / "wormhole_udp.py"
    if udp_file.exists():
        spec = importlib.util.spec_from_file_location("wormhole_udp", udp_file)
        wormhole_udp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wormhole_udp_module)
        wormhole_traversal_udp = wormhole_udp_module.wormhole_traversal_udp
    else:
        raise ImportError("Could not import wormhole_udp")


class FitnessEvaluator:
    """
    Wrapper for fitness evaluation with caching and batch processing.
    
    Provides:
    - Fitness computation using UDP
    - Caching of fitness values
    - Batch evaluation
    - Fitness statistics
    """
    
    def __init__(self, udp: Optional[wormhole_traversal_udp] = None, database_path: Optional[str] = None):
        """
        Initialize fitness evaluator.
        
        Args:
            udp: UDP instance. If None, creates a new one.
            database_path: Path to database.npz file (required if udp is None)
        """
        if udp is None:
            if database_path is None:
                database_path = "./data/database.npz"
            self.udp = wormhole_traversal_udp(database=database_path)
        else:
            self.udp = udp
        
        # Cache for fitness values
        self._cache: Dict[bytes, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluation_count = 0
    
    def evaluate(
        self,
        chromosome: np.ndarray,
        use_cache: bool = True
    ) -> List[float]:
        """
        Evaluate fitness of a chromosome.
        
        Args:
            chromosome: Chromosome array of shape (6000,)
            use_cache: Whether to use caching
            
        Returns:
            Fitness vector: [objective, eq_constraint_1, eq_constraint_2, ineq_constraint_1]
        """
        self._evaluation_count += 1
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(chromosome)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].copy()
            self._cache_misses += 1
        
        # Evaluate using UDP
        try:
            fitness_vector = self.udp.fitness(chromosome)
        except Exception as e:
            logger.error(f"Error evaluating fitness: {e}")
            # Return worst possible fitness
            fitness_vector = [1e10, 12, 12, 1e10]
        
        # Store in cache
        if use_cache:
            self._cache[cache_key] = fitness_vector.copy()
        
        return fitness_vector
    
    def evaluate_batch(
        self,
        chromosomes: List[np.ndarray],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Evaluate fitness for multiple chromosomes.
        
        Args:
            chromosomes: List of chromosome arrays
            use_cache: Whether to use caching
            
        Returns:
            List of fitness vectors
        """
        results = []
        for chromosome in chromosomes:
            fitness = self.evaluate(chromosome, use_cache=use_cache)
            results.append(fitness)
        return results
    
    def get_objective(self, chromosome: np.ndarray) -> float:
        """
        Get only the objective value (fitness).
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Objective value (to minimize)
        """
        fitness_vector = self.evaluate(chromosome)
        return fitness_vector[0]
    
    def get_constraints(self, chromosome: np.ndarray) -> Tuple[int, int, float]:
        """
        Get constraint violation values.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Tuple of (origin_constraint, path_constraint, window_constraint)
        """
        fitness_vector = self.evaluate(chromosome)
        return (
            int(fitness_vector[1]),  # Origin constraint
            int(fitness_vector[2]),  # Path constraint
            float(fitness_vector[3])  # Window constraint
        )
    
    def is_feasible(self, chromosome: np.ndarray) -> bool:
        """
        Check if chromosome satisfies all constraints.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            True if all constraints satisfied, False otherwise
        """
        fitness_vector = self.evaluate(chromosome)
        
        # Equality constraints must be 0
        if fitness_vector[1] != 0 or fitness_vector[2] != 0:
            return False
        
        # Inequality constraint must be <= 0
        if fitness_vector[3] > 0:
            return False
        
        return True
    
    def get_penalized_fitness(
        self,
        chromosome: np.ndarray,
        origin_penalty: float = 1e6,
        path_penalty: float = 1e6,
        window_penalty: float = 1000.0
    ) -> float:
        """
        Get fitness with penalty for constraint violations.
        
        Args:
            chromosome: Chromosome array
            origin_penalty: Penalty weight for origin constraint violations
            path_penalty: Penalty weight for path constraint violations
            window_penalty: Penalty weight for window constraint violations
            
        Returns:
            Penalized fitness value
        """
        fitness_vector = self.evaluate(chromosome)
        
        objective = fitness_vector[0]
        origin_violation = fitness_vector[1]
        path_violation = fitness_vector[2]
        window_violation = fitness_vector[3]
        
        penalty = 0.0
        
        # Hard constraints: large penalty
        if origin_violation > 0:
            penalty += origin_violation * origin_penalty
        
        if path_violation > 0:
            penalty += path_violation * path_penalty
        
        # Soft constraint: adaptive penalty
        if window_violation > 0:
            penalty += window_violation * window_penalty
        
        return objective + penalty
    
    def _get_cache_key(self, chromosome: np.ndarray) -> bytes:
        """
        Generate cache key from chromosome.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Cache key (bytes)
        """
        return chromosome.tobytes()
    
    def clear_cache(self) -> None:
        """Clear the fitness cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Fitness cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_evaluations': self._evaluation_count
        }
    
    def get_statistics(
        self,
        chromosomes: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute fitness statistics for a population.
        
        Args:
            chromosomes: List of chromosome arrays
            
        Returns:
            Dictionary with fitness statistics
        """
        if not chromosomes:
            return {}
        
        objectives = []
        origin_violations = []
        path_violations = []
        window_violations = []
        feasible_count = 0
        
        for chromosome in chromosomes:
            fitness = self.evaluate(chromosome)
            objectives.append(fitness[0])
            origin_violations.append(fitness[1])
            path_violations.append(fitness[2])
            window_violations.append(fitness[3])
            
            if self.is_feasible(chromosome):
                feasible_count += 1
        
        objectives = np.array(objectives)
        origin_violations = np.array(origin_violations)
        path_violations = np.array(path_violations)
        window_violations = np.array(window_violations)
        
        stats = {
            'population_size': len(chromosomes),
            'feasible_count': feasible_count,
            'feasible_rate': feasible_count / len(chromosomes),
            'objective_min': float(np.min(objectives)),
            'objective_max': float(np.max(objectives)),
            'objective_mean': float(np.mean(objectives)),
            'objective_std': float(np.std(objectives)),
            'origin_violation_mean': float(np.mean(origin_violations)),
            'path_violation_mean': float(np.mean(path_violations)),
            'window_violation_mean': float(np.mean(window_violations)),
        }
        
        return stats

