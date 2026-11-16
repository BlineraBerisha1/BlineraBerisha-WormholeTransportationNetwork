"""
Path validity repair operator.

This module provides operators to fix invalid path edges.
"""

from typing import List, Optional
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder, ChromosomeEncoder
from ..model.network import WormholeNetwork


class PathRepair:
    """
    Repairs path validity constraint violations.
    """
    
    def __init__(
        self,
        network: WormholeNetwork,
        destination: int,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize path repair operator.
        
        Args:
            network: WormholeNetwork instance
            destination: Destination node ID
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.network = network
        self.destination = destination
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
        self.encoder = ChromosomeEncoder(n_ships, jump_limit)
    
    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Repair path validity violations.
        
        Args:
            chromosome: Chromosome to repair
            
        Returns:
            Repaired chromosome
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        for ship_idx, path in enumerate(paths):
            if not path:
                continue
            
            repaired_path = self._repair_path(path)
            paths[ship_idx] = repaired_path
        
        # Re-encode
        repaired = self.encoder.encode(paths)
        return repaired
    
    def _repair_path(self, path: List[int]) -> List[int]:
        """
        Repair a single path.
        
        Args:
            path: Path to repair
            
        Returns:
            Repaired path
        """
        if not path:
            return path
        
        repaired = [path[0]]  # Keep origin
        
        for i in range(1, len(path)):
            current = repaired[-1]
            target = path[i]
            
            # Check if edge is valid
            if self.network.has_edge(current, target):
                repaired.append(target)
            else:
                # Invalid edge - try to find alternative
                alternative = self._find_alternative_path(current, target)
                if alternative:
                    repaired.extend(alternative[1:])  # Skip first (already in repaired)
                else:
                    # Try to reach destination directly
                    if self.network.has_edge(current, self.destination):
                        repaired.append(self.destination)
                        break
                    else:
                        # Use shortest path to destination
                        shortest = self.network.shortest_path(current, self.destination)
                        if shortest:
                            repaired.extend(shortest[1:])
                            break
                        else:
                            # Dead end - stop here
                            break
            
            # Check if we've reached destination
            if repaired[-1] == self.destination:
                break
            
            # Check length limit
            if len(repaired) >= self.jump_limit:
                break
        
        # Ensure destination is reachable
        if repaired[-1] != self.destination:
            if self.network.has_edge(repaired[-1], self.destination):
                repaired.append(self.destination)
            else:
                # Try shortest path
                shortest = self.network.shortest_path(repaired[-1], self.destination)
                if shortest:
                    repaired.extend(shortest[1:])
        
        return repaired
    
    def _find_alternative_path(
        self,
        source: int,
        target: int,
        max_hops: int = 5
    ) -> Optional[List[int]]:
        """
        Find alternative path between two nodes.
        
        Args:
            source: Source node
            target: Target node
            max_hops: Maximum hops to search
            
        Returns:
            Alternative path or None
        """
        # Try shortest path
        shortest = self.network.shortest_path(source, target)
        if shortest and len(shortest) <= max_hops + 1:
            return shortest
        
        # Try random walk
        current = source
        path = [current]
        visited = {current}
        
        for _ in range(max_hops):
            neighbors = self.network.get_neighbors(current)
            if not neighbors:
                break
            
            # Filter unvisited neighbors
            unvisited = [n for n in neighbors if n not in visited]
            if not unvisited:
                unvisited = neighbors  # Allow revisits if necessary
            
            next_node = np.random.choice(unvisited)
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
            if current == target:
                return path
        
        return None
    
    def repair_all(self, chromosomes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Repair multiple chromosomes.
        
        Args:
            chromosomes: List of chromosomes to repair
            
        Returns:
            List of repaired chromosomes
        """
        return [self.repair(chr) for chr in chromosomes]

