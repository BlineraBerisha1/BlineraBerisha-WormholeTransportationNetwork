"""
Network model for Wormhole Transportation Network.

This module provides a wrapper around NetworkX DiGraph with additional
utilities for wormhole network operations.
"""

from typing import List, Set, Optional, Tuple
import numpy as np
import networkx as nx
from loguru import logger


class WormholeNetwork:
    """
    Wrapper for NetworkX DiGraph with wormhole-specific utilities.
    
    Provides convenient methods for:
    - Getting neighbors and valid edges
    - Computing path weights (mean, variance)
    - Finding shortest paths
    - Network statistics
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize wormhole network.
        
        Args:
            graph: NetworkX DiGraph with edge attributes 'm' (mean) and 'v' (variance)
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Graph must be a NetworkX DiGraph")
        
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.n_edges = graph.number_of_edges()
        
        # Validate edge attributes
        self._validate_edge_attributes()
    
    def _validate_edge_attributes(self) -> None:
        """Validate that all edges have required attributes."""
        missing_attrs = []
        for source, target in self.graph.edges():
            if 'm' not in self.graph[source][target]:
                missing_attrs.append((source, target, 'm'))
            if 'v' not in self.graph[source][target]:
                missing_attrs.append((source, target, 'v'))
        
        if missing_attrs:
            raise ValueError(
                f"Edges missing required attributes: {missing_attrs[:10]}"
            )
    
    def has_edge(self, source: int, target: int) -> bool:
        """Check if edge exists."""
        return self.graph.has_edge(source, target)
    
    def get_neighbors(self, node: int) -> List[int]:
        """
        Get outgoing neighbors of a node.
        
        Args:
            node: Source node ID
            
        Returns:
            List of target node IDs
        """
        if node not in self.graph.nodes:
            return []
        
        return list(self.graph.successors(node))
    
    def get_incoming_neighbors(self, node: int) -> List[int]:
        """
        Get incoming neighbors of a node.
        
        Args:
            node: Target node ID
            
        Returns:
            List of source node IDs
        """
        if node not in self.graph.nodes:
            return []
        
        return list(self.graph.predecessors(node))
    
    def get_edge_mean(self, source: int, target: int) -> float:
        """
        Get mean temporal offset for an edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Mean temporal offset
        """
        if not self.has_edge(source, target):
            raise ValueError(f"Edge ({source}, {target}) does not exist")
        return float(self.graph[source][target]['m'])
    
    def get_edge_variance(self, source: int, target: int) -> float:
        """
        Get variance for an edge.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Variance value
        """
        if not self.has_edge(source, target):
            raise ValueError(f"Edge ({source}, {target}) does not exist")
        return float(self.graph[source][target]['v'])
    
    def is_valid_path(self, path: List[int]) -> bool:
        """
        Check if a path is valid (all consecutive edges exist).
        
        Args:
            path: List of node IDs
            
        Returns:
            True if path is valid, False otherwise
        """
        if len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            if not self.has_edge(path[i], path[i + 1]):
                return False
        
        return True
    
    def compute_path_mean(self, path: List[int]) -> float:
        """
        Compute total mean temporal offset for a path.
        
        Args:
            path: List of node IDs
            
        Returns:
            Sum of mean values along path
        """
        if not self.is_valid_path(path):
            raise ValueError("Invalid path")
        
        total_mean = 0.0
        for i in range(len(path) - 1):
            total_mean += self.get_edge_mean(path[i], path[i + 1])
        
        return total_mean
    
    def compute_path_variance(self, path: List[int]) -> float:
        """
        Compute total base variance for a path (without revisit penalties).
        
        Args:
            path: List of node IDs
            
        Returns:
            Sum of base variance values along path
        """
        if not self.is_valid_path(path):
            raise ValueError("Invalid path")
        
        total_variance = 0.0
        for i in range(len(path) - 1):
            total_variance += self.get_edge_variance(path[i], path[i + 1])
        
        return total_variance
    
    def shortest_path(
        self,
        source: int,
        target: int,
        weight: str = 'v',
        method: str = 'dijkstra'
    ) -> Optional[List[int]]:
        """
        Find shortest path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight to minimize ('v' for variance, 'm' for mean, None for hops)
            method: Algorithm to use ('dijkstra' or 'bellman-ford')
            
        Returns:
            List of node IDs representing shortest path, or None if no path exists
        """
        if source not in self.graph.nodes:
            return None
        if target not in self.graph.nodes:
            return None
        
        try:
            if weight is None:
                # Unweighted shortest path (minimum hops)
                path = nx.shortest_path(self.graph, source, target)
            else:
                if method == 'dijkstra':
                    path = nx.dijkstra_path(self.graph, source, target, weight=weight)
                elif method == 'bellman-ford':
                    path = nx.bellman_ford_path(self.graph, source, target, weight=weight)
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NetworkXError as e:
            logger.warning(f"Error finding path from {source} to {target}: {e}")
            return None
    
    def shortest_path_length(
        self,
        source: int,
        target: int,
        weight: str = 'v'
    ) -> Optional[float]:
        """
        Get length of shortest path.
        
        Args:
            source: Source node ID
            target: Target node ID
            weight: Edge weight to minimize
            
        Returns:
            Path length, or None if no path exists
        """
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return None
        
        try:
            if weight is None:
                length = nx.shortest_path_length(self.graph, source, target)
            else:
                length = nx.dijkstra_path_length(self.graph, source, target, weight=weight)
            return float(length)
        except nx.NetworkXNoPath:
            return None
        except nx.NetworkXError:
            return None
    
    def has_path(self, source: int, target: int) -> bool:
        """
        Check if a path exists between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            True if path exists, False otherwise
        """
        return nx.has_path(self.graph, source, target)
    
    def get_statistics(self) -> dict:
        """
        Get network statistics.
        
        Returns:
            Dictionary with network statistics
        """
        stats = {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'is_weakly_connected': nx.is_weakly_connected(self.graph),
        }
        
        # Degree statistics
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        
        if in_degrees:
            stats['avg_in_degree'] = np.mean(in_degrees)
            stats['max_in_degree'] = np.max(in_degrees)
            stats['min_in_degree'] = np.min(in_degrees)
        
        if out_degrees:
            stats['avg_out_degree'] = np.mean(out_degrees)
            stats['max_out_degree'] = np.max(out_degrees)
            stats['min_out_degree'] = np.min(out_degrees)
        
        # Edge attribute statistics
        means = [self.graph[u][v]['m'] for u, v in self.graph.edges()]
        variances = [self.graph[u][v]['v'] for u, v in self.graph.edges()]
        
        stats['mean_range'] = (float(np.min(means)), float(np.max(means)))
        stats['variance_range'] = (float(np.min(variances)), float(np.max(variances)))
        stats['negative_means_count'] = int(np.sum(np.array(means) < 0))
        
        return stats
    
    def get_nodes(self) -> List[int]:
        """Get list of all node IDs."""
        return list(self.graph.nodes())
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get list of all edges as (source, target) tuples."""
        return list(self.graph.edges())

