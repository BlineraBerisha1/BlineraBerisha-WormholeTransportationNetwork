"""
Constraint checking and penalty computation module.

This module provides detailed constraint checking and penalty calculation
for the Wormhole Transportation Network problem.
"""

from typing import List, Tuple, Dict
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder
from ..model.network import WormholeNetwork


class ConstraintViolations:
    """
    Container for constraint violation information.
    """
    
    def __init__(
        self,
        origin_violations: int = 0,
        path_violations: int = 0,
        window_violation: float = 0.0
    ):
        """
        Initialize constraint violations.
        
        Args:
            origin_violations: Number of ships with invalid origins
            path_violations: Number of ships with invalid paths
            window_violation: Maximum arrival time difference minus window
        """
        self.origin_violations = origin_violations
        self.path_violations = path_violations
        self.window_violation = window_violation
    
    @property
    def total_hard_violations(self) -> int:
        """Total number of hard constraint violations."""
        return self.origin_violations + self.path_violations
    
    @property
    def is_feasible(self) -> bool:
        """Check if all constraints are satisfied."""
        return (
            self.origin_violations == 0
            and self.path_violations == 0
            and self.window_violation <= 0.0
        )
    
    def __repr__(self) -> str:
        return (
            f"ConstraintViolations("
            f"origin={self.origin_violations}, "
            f"path={self.path_violations}, "
            f"window={self.window_violation:.6f})"
        )


class ConstraintChecker:
    """
    Detailed constraint checking for chromosomes.
    
    Provides methods to check each constraint individually and compute
    detailed violation information.
    """
    
    def __init__(
        self,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        delays: np.ndarray,
        window: float,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize constraint checker.
        
        Args:
            network: WormholeNetwork instance
            origins: List of origin sets for each ship
            destination: Destination node ID
            delays: Initial delays for each ship
            window: Arrival time window
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.network = network
        self.origins = origins
        self.destination = destination
        self.delays = delays
        self.window = window
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
    
    def check_all(
        self,
        chromosome: np.ndarray
    ) -> ConstraintViolations:
        """
        Check all constraints for a chromosome.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            ConstraintViolations object
        """
        origin_violations = self.check_origin_constraint(chromosome)
        path_violations = self.check_path_constraint(chromosome)
        window_violation = self.check_window_constraint(chromosome)
        
        return ConstraintViolations(
            origin_violations=origin_violations,
            path_violations=path_violations,
            window_violation=window_violation
        )
    
    def check_origin_constraint(self, chromosome: np.ndarray) -> int:
        """
        Check origin node constraint.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Number of ships with invalid origins
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        violations = 0
        for ship_idx, path in enumerate(paths):
            if not path:
                violations += 1
                continue
            
            if path[0] not in self.origins[ship_idx]:
                violations += 1
        
        return violations
    
    def check_path_constraint(self, chromosome: np.ndarray) -> int:
        """
        Check path validity constraint.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Number of ships with invalid paths
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        violations = 0
        for ship_idx, path in enumerate(paths):
            if not path:
                violations += 1
                continue
            
            # Check if path is valid
            if not self.network.is_valid_path(path):
                violations += 1
                continue
            
            # Check if last node can reach destination
            last_node = path[-1]
            if not self.network.has_edge(last_node, self.destination):
                violations += 1
        
        return violations
    
    def check_window_constraint(self, chromosome: np.ndarray) -> float:
        """
        Check arrival window constraint.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Window violation (max_time_diff - window, should be <= 0)
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        arrival_times = []
        
        for ship_idx, path in enumerate(paths):
            if not path:
                # Invalid path - use delay as arrival time
                arrival_times.append(self.delays[ship_idx])
                continue
            
            # Compute path mean
            try:
                path_mean = self.network.compute_path_mean(path)
            except Exception:
                # Invalid path
                path_mean = 0.0
            
            arrival_time = self.delays[ship_idx] + path_mean
            arrival_times.append(arrival_time)
        
        if not arrival_times:
            return float('inf')
        
        arrival_times = np.array(arrival_times)
        max_time_diff = float(np.max(arrival_times) - np.min(arrival_times))
        violation = max_time_diff - self.window
        
        return violation
    
    def get_violation_details(
        self,
        chromosome: np.ndarray
    ) -> Dict[str, List]:
        """
        Get detailed violation information for each ship.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            Dictionary with violation details per ship
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        details = {
            'origin_violations': [],
            'path_violations': [],
            'arrival_times': [],
            'invalid_edges': []
        }
        
        for ship_idx, path in enumerate(paths):
            # Origin violation
            if not path or path[0] not in self.origins[ship_idx]:
                details['origin_violations'].append(ship_idx)
            
            # Path violation
            if not path:
                details['path_violations'].append(ship_idx)
                details['invalid_edges'].append([])
            else:
                invalid_edges = []
                for i in range(len(path) - 1):
                    if not self.network.has_edge(path[i], path[i + 1]):
                        invalid_edges.append((path[i], path[i + 1]))
                
                if invalid_edges:
                    details['path_violations'].append(ship_idx)
                    details['invalid_edges'].append(invalid_edges)
                else:
                    # Check destination connection
                    if not self.network.has_edge(path[-1], self.destination):
                        details['path_violations'].append(ship_idx)
                        details['invalid_edges'].append([(path[-1], self.destination)])
                    else:
                        details['invalid_edges'].append([])
            
            # Arrival time
            if path:
                try:
                    path_mean = self.network.compute_path_mean(path)
                    arrival_time = self.delays[ship_idx] + path_mean
                except Exception:
                    arrival_time = self.delays[ship_idx]
            else:
                arrival_time = self.delays[ship_idx]
            
            details['arrival_times'].append(arrival_time)
        
        return details
    
    def compute_penalty(
        self,
        chromosome: np.ndarray,
        origin_penalty: float = 1e6,
        path_penalty: float = 1e6,
        window_penalty: float = 1000.0
    ) -> float:
        """
        Compute total penalty for constraint violations.
        
        Args:
            chromosome: Chromosome array
            origin_penalty: Penalty per origin violation
            path_penalty: Penalty per path violation
            window_penalty: Penalty weight for window violation
            
        Returns:
            Total penalty value
        """
        violations = self.check_all(chromosome)
        
        penalty = 0.0
        
        # Hard constraints
        penalty += violations.origin_violations * origin_penalty
        penalty += violations.path_violations * path_penalty
        
        # Soft constraint
        if violations.window_violation > 0:
            penalty += violations.window_violation * window_penalty
        
        return penalty
    
    def get_feasibility_rate(
        self,
        chromosomes: List[np.ndarray]
    ) -> float:
        """
        Compute feasibility rate for a population.
        
        Args:
            chromosomes: List of chromosome arrays
            
        Returns:
            Fraction of feasible solutions
        """
        if not chromosomes:
            return 0.0
        
        feasible_count = 0
        for chromosome in chromosomes:
            violations = self.check_all(chromosome)
            if violations.is_feasible:
                feasible_count += 1
        
        return feasible_count / len(chromosomes)

