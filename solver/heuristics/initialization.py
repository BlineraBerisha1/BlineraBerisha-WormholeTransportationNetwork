"""
Population initialization strategies.

This module provides various strategies for generating initial populations.
"""

from typing import List, Optional
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeEncoder
from ..model.network import WormholeNetwork
from .pathfinding import PathFinder, DijkstraPathFinder


class PopulationInitializer:
    """
    Base class for population initialization.
    """
    
    def initialize(
        self,
        size: int,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        jump_limit: int = 500
    ) -> List[np.ndarray]:
        """
        Initialize population of chromosomes.
        
        Args:
            size: Population size
            network: WormholeNetwork instance
            origins: List of origin sets for each ship
            destination: Destination node ID
            jump_limit: Maximum jumps per ship
            
        Returns:
            List of chromosomes
        """
        raise NotImplementedError


class RandomInitializer(PopulationInitializer):
    """
    Random initialization with valid paths.
    """
    
    def initialize(
        self,
        size: int,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        jump_limit: int = 500
    ) -> List[np.ndarray]:
        """
        Initialize population with random valid paths.
        
        Args:
            size: Population size
            network: WormholeNetwork instance
            origins: List of origin sets
            destination: Destination node ID
            jump_limit: Maximum jumps per ship
            
        Returns:
            List of chromosomes
        """
        encoder = ChromosomeEncoder(n_ships=12, jump_limit=jump_limit)
        chromosomes = []
        
        for _ in range(size):
            paths = []
            
            for ship_idx in range(12):
                # Choose random origin
                if origins[ship_idx]:
                    origin = np.random.choice(list(origins[ship_idx]))
                    path = [origin]
                    
                    # Random walk until destination or jump limit
                    current = origin
                    for _ in range(jump_limit - 1):
                        neighbors = network.get_neighbors(current)
                        if not neighbors:
                            break
                        
                        if destination in neighbors:
                            path.append(destination)
                            break
                        
                        current = np.random.choice(neighbors)
                        path.append(current)
                    
                    # Ensure destination is reachable
                    if path[-1] != destination:
                        if network.has_edge(path[-1], destination):
                            path.append(destination)
                        else:
                            # Try shortest path
                            shortest = network.shortest_path(path[-1], destination)
                            if shortest:
                                path.extend(shortest[1:])
                    
                    paths.append(path)
                else:
                    paths.append([])
            
            chromosome = encoder.encode(paths)
            chromosomes.append(chromosome)
        
        return chromosomes


class HeuristicInitializer(PopulationInitializer):
    """
    Heuristic initialization using shortest paths.
    """
    
    def __init__(self, pathfinder: Optional[PathFinder] = None):
        """
        Initialize heuristic initializer.
        
        Args:
            pathfinder: PathFinder instance. If None, uses DijkstraPathFinder.
        """
        self.pathfinder = pathfinder or DijkstraPathFinder(weight='v')
    
    def initialize(
        self,
        size: int,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        jump_limit: int = 500
    ) -> List[np.ndarray]:
        """
        Initialize population with heuristic paths (shortest paths).
        
        Args:
            size: Population size
            network: WormholeNetwork instance
            origins: List of origin sets
            destination: Destination node ID
            jump_limit: Maximum jumps per ship
            
        Returns:
            List of chromosomes
        """
        encoder = ChromosomeEncoder(n_ships=12, jump_limit=jump_limit)
        chromosomes = []
        
        for _ in range(size):
            paths = []
            
            for ship_idx in range(12):
                if origins[ship_idx]:
                    # Choose random origin
                    origin = np.random.choice(list(origins[ship_idx]))
                    
                    # Find shortest path
                    path = self.pathfinder.find_path(network, origin, destination)
                    
                    if path:
                        # Truncate if too long
                        if len(path) > jump_limit:
                            path = path[:jump_limit]
                            # Ensure destination connection
                            if network.has_edge(path[-1], destination):
                                pass  # Destination will be added during evaluation
                    else:
                        # Fallback to random walk
                        path = [origin]
                        current = origin
                        for _ in range(min(10, jump_limit - 1)):
                            neighbors = network.get_neighbors(current)
                            if not neighbors:
                                break
                            if destination in neighbors:
                                path.append(destination)
                                break
                            current = np.random.choice(neighbors)
                            path.append(current)
                    
                    paths.append(path)
                else:
                    paths.append([])
            
            chromosome = encoder.encode(paths)
            chromosomes.append(chromosome)
        
        return chromosomes


class MixedInitializer(PopulationInitializer):
    """
    Mixed initialization combining random and heuristic strategies.
    """
    
    def __init__(
        self,
        heuristic_ratio: float = 0.3,
        pathfinder: Optional[PathFinder] = None
    ):
        """
        Initialize mixed initializer.
        
        Args:
            heuristic_ratio: Fraction of population initialized heuristically
            pathfinder: PathFinder instance
        """
        self.heuristic_ratio = heuristic_ratio
        self.random_init = RandomInitializer()
        self.heuristic_init = HeuristicInitializer(pathfinder)
    
    def initialize(
        self,
        size: int,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        jump_limit: int = 500
    ) -> List[np.ndarray]:
        """
        Initialize population with mixed strategy.
        
        Args:
            size: Population size
            network: WormholeNetwork instance
            origins: List of origin sets
            destination: Destination node ID
            jump_limit: Maximum jumps per ship
            
        Returns:
            List of chromosomes
        """
        n_heuristic = int(size * self.heuristic_ratio)
        n_random = size - n_heuristic
        
        chromosomes = []
        
        # Heuristic initialization
        if n_heuristic > 0:
            heuristic_chromosomes = self.heuristic_init.initialize(
                n_heuristic, network, origins, destination, jump_limit
            )
            chromosomes.extend(heuristic_chromosomes)
        
        # Random initialization
        if n_random > 0:
            random_chromosomes = self.random_init.initialize(
                n_random, network, origins, destination, jump_limit
            )
            chromosomes.extend(random_chromosomes)
        
        return chromosomes

