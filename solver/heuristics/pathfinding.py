"""
Pathfinding utilities for heuristic initialization.

This module provides shortest path algorithms for generating initial solutions.
"""

from typing import List, Optional
import numpy as np
from loguru import logger

from ..model.network import WormholeNetwork


class PathFinder:
    """
    Base class for pathfinding algorithms.
    """
    
    def find_path(
        self,
        network: WormholeNetwork,
        source: int,
        target: int
    ) -> Optional[List[int]]:
        """
        Find path from source to target.
        
        Args:
            network: WormholeNetwork instance
            source: Source node ID
            target: Target node ID
            
        Returns:
            Path as list of node IDs, or None if no path exists
        """
        raise NotImplementedError


class DijkstraPathFinder(PathFinder):
    """
    Dijkstra's algorithm for finding shortest paths.
    """
    
    def __init__(self, weight: str = 'v'):
        """
        Initialize Dijkstra pathfinder.
        
        Args:
            weight: Edge weight to minimize ('v' for variance, 'm' for mean, None for hops)
        """
        self.weight = weight
    
    def find_path(
        self,
        network: WormholeNetwork,
        source: int,
        target: int
    ) -> Optional[List[int]]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Args:
            network: WormholeNetwork instance
            source: Source node ID
            target: Target node ID
            
        Returns:
            Shortest path or None
        """
        return network.shortest_path(source, target, weight=self.weight)


class AStarPathFinder(PathFinder):
    """
    A* algorithm for finding shortest paths.
    
    Note: Simplified implementation using NetworkX's A* if available.
    """
    
    def __init__(self, weight: str = 'v'):
        """
        Initialize A* pathfinder.
        
        Args:
            weight: Edge weight to minimize
        """
        self.weight = weight
    
    def find_path(
        self,
        network: WormholeNetwork,
        source: int,
        target: int
    ) -> Optional[List[int]]:
        """
        Find shortest path using A* algorithm.
        
        Args:
            network: WormholeNetwork instance
            source: Source node ID
            target: Target node ID
            
        Returns:
            Shortest path or None
        """
        # Use NetworkX A* if available, otherwise fall back to Dijkstra
        try:
            import networkx as nx
            if hasattr(nx, 'astar_path'):
                path = nx.astar_path(
                    network.graph,
                    source,
                    target,
                    weight=self.weight
                )
                return path
        except Exception:
            pass
        
        # Fallback to Dijkstra
        return network.shortest_path(source, target, weight=self.weight)

