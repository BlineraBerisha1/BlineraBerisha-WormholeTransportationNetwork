"""
Arrival window repair operator.

This module provides operators to adjust paths to meet the arrival window constraint.
"""

from typing import List
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder, ChromosomeEncoder
from ..model.network import WormholeNetwork


class WindowRepair:
    """
    Repairs arrival window constraint violations by adjusting paths.
    """
    
    def __init__(
        self,
        network: WormholeNetwork,
        destination: int,
        delays: np.ndarray,
        window: float,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize window repair operator.
        
        Args:
            network: WormholeNetwork instance
            destination: Destination node ID
            delays: Initial delays for each ship
            window: Arrival time window
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.network = network
        self.destination = destination
        self.delays = delays
        self.window = window
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
        self.encoder = ChromosomeEncoder(n_ships, jump_limit)
    
    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Repair window constraint violations.
        
        Args:
            chromosome: Chromosome to repair
            
        Returns:
            Repaired chromosome (may not fully satisfy window if impossible)
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        # Compute arrival times
        arrival_times = []
        for ship_idx, path in enumerate(paths):
            if path:
                try:
                    path_mean = self.network.compute_path_mean(path)
                    arrival_time = self.delays[ship_idx] + path_mean
                except Exception:
                    arrival_time = self.delays[ship_idx]
            else:
                arrival_time = self.delays[ship_idx]
            arrival_times.append(arrival_time)
        
        arrival_times = np.array(arrival_times)
        min_time = np.min(arrival_times)
        max_time = np.max(arrival_times)
        time_diff = max_time - min_time
        
        # If within window, no repair needed
        if time_diff <= self.window:
            return chromosome
        
        # Try to adjust paths to meet window
        # Strategy: adjust paths that are too early or too late
        target_time = (min_time + max_time) / 2.0
        
        for ship_idx, path in enumerate(paths):
            if not path:
                continue
            
            current_time = arrival_times[ship_idx]
            time_offset = current_time - target_time
            
            # If ship is too early, try to add positive-mean edges
            # If ship is too late, try to add negative-mean edges
            if abs(time_offset) > self.window / 2:
                adjusted_path = self._adjust_path_time(
                    path, time_offset, target_time - self.delays[ship_idx]
                )
                paths[ship_idx] = adjusted_path
        
        # Re-encode
        repaired = self.encoder.encode(paths)
        return repaired
    
    def _adjust_path_time(
        self,
        path: List[int],
        time_offset: float,
        target_path_mean: float
    ) -> List[int]:
        """
        Adjust path to change arrival time.
        
        Args:
            path: Current path
            time_offset: How much to adjust (positive = too early, negative = too late)
            target_path_mean: Target path mean
            
        Returns:
            Adjusted path
        """
        if not path:
            return path
        
        # Simple strategy: try to extend path with edges that help
        adjusted = path.copy()
        current = adjusted[-1]
        
        # Try to find edges that move us towards target
        max_adjustments = min(10, self.jump_limit - len(adjusted))
        
        for _ in range(max_adjustments):
            neighbors = self.network.get_neighbors(current)
            if not neighbors:
                break
            
            if self.destination in neighbors:
                adjusted.append(self.destination)
                break
            
            # Choose neighbor based on time offset
            best_neighbor = None
            best_score = float('inf')
            
            for neighbor in neighbors:
                edge_mean = self.network.get_edge_mean(current, neighbor)
                
                # Score based on how well it moves us towards target
                if time_offset > 0:
                    # Too early - prefer positive means
                    score = -edge_mean
                else:
                    # Too late - prefer negative means
                    score = edge_mean
                
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
            
            if best_neighbor:
                adjusted.append(best_neighbor)
                current = best_neighbor
            else:
                # Fallback to random
                adjusted.append(np.random.choice(neighbors))
                current = adjusted[-1]
        
        # Ensure destination
        if adjusted[-1] != self.destination:
            if self.network.has_edge(adjusted[-1], self.destination):
                adjusted.append(self.destination)
        
        return adjusted
    
    def repair_all(self, chromosomes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Repair multiple chromosomes.
        
        Args:
            chromosomes: List of chromosomes to repair
            
        Returns:
            List of repaired chromosomes
        """
        return [self.repair(chr) for chr in chromosomes]

