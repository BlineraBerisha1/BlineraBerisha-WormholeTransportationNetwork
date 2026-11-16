"""
Path representation and manipulation utilities.

This module provides classes and functions for working with ship paths
through the wormhole network.
"""

from typing import List, Optional, Tuple
import numpy as np
from loguru import logger


# Type alias for path (list of node IDs)
Path = List[int]

# Type alias for list of paths (one per ship)
PathList = List[Path]


class PathUtils:
    """
    Utility functions for path manipulation and analysis.
    """
    
    @staticmethod
    def validate_path(
        path: Path,
        network,
        origin_set: Optional[set] = None,
        destination: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a path.
        
        Args:
            path: List of node IDs
            network: WormholeNetwork instance
            origin_set: Optional set of valid origin nodes
            destination: Optional destination node
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not path:
            errors.append("Path is empty")
            return False, errors
        
        # Check origin
        if origin_set is not None:
            if path[0] not in origin_set:
                errors.append(
                    f"Origin node {path[0]} not in valid origin set"
                )
        
        # Check path validity (all edges exist)
        if not network.is_valid_path(path):
            for i in range(len(path) - 1):
                if not network.has_edge(path[i], path[i + 1]):
                    errors.append(
                        f"Invalid edge: ({path[i]}, {path[i + 1]})"
                    )
                    break
        
        # Check destination
        if destination is not None:
            if path[-1] != destination:
                # Check if last node can reach destination
                if not network.has_edge(path[-1], destination):
                    errors.append(
                        f"Last node {path[-1]} cannot reach destination {destination}"
                    )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def compute_path_statistics(
        paths: PathList,
        network,
        delays: np.ndarray
    ) -> dict:
        """
        Compute statistics for a set of paths.
        
        Args:
            paths: List of paths (one per ship)
            network: WormholeNetwork instance
            delays: Initial delays for each ship
            
        Returns:
            Dictionary with path statistics
        """
        stats = {
            'path_lengths': [],
            'path_means': [],
            'path_variances': [],
            'arrival_times': [],
            'total_nodes_visited': set(),
        }
        
        for ship_idx, path in enumerate(paths):
            if not path:
                stats['path_lengths'].append(0)
                stats['path_means'].append(0.0)
                stats['path_variances'].append(0.0)
                stats['arrival_times'].append(delays[ship_idx])
                continue
            
            # Path length (excluding destination if present)
            path_length = len(path)
            stats['path_lengths'].append(path_length)
            
            # Path mean
            try:
                path_mean = network.compute_path_mean(path)
                stats['path_means'].append(path_mean)
            except Exception as e:
                logger.warning(f"Error computing path mean for ship {ship_idx}: {e}")
                stats['path_means'].append(0.0)
            
            # Path variance (base, without revisit penalties)
            try:
                path_variance = network.compute_path_variance(path)
                stats['path_variances'].append(path_variance)
            except Exception as e:
                logger.warning(f"Error computing path variance for ship {ship_idx}: {e}")
                stats['path_variances'].append(0.0)
            
            # Arrival time
            arrival_time = delays[ship_idx] + stats['path_means'][-1]
            stats['arrival_times'].append(arrival_time)
            
            # Nodes visited
            stats['total_nodes_visited'].update(path)
        
        stats['total_nodes_visited'] = len(stats['total_nodes_visited'])
        stats['arrival_window'] = (
            max(stats['arrival_times']) - min(stats['arrival_times'])
        )
        
        return stats
    
    @staticmethod
    def extend_path(
        path: Path,
        network,
        destination: int,
        max_length: int,
        strategy: str = 'random'
    ) -> Path:
        """
        Extend a path towards destination.
        
        Args:
            path: Current path
            network: WormholeNetwork instance
            destination: Destination node
            max_length: Maximum path length
            strategy: Extension strategy ('random', 'shortest', 'low_variance')
            
        Returns:
            Extended path
        """
        if not path:
            return path
        
        extended = path.copy()
        current = extended[-1]
        
        # If already at destination, return
        if current == destination:
            return extended
        
        # Extend until destination or max_length
        while len(extended) < max_length:
            neighbors = network.get_neighbors(current)
            
            if not neighbors:
                # Dead end
                break
            
            # Check if destination is reachable
            if destination in neighbors:
                extended.append(destination)
                break
            
            # Choose next node based on strategy
            if strategy == 'random':
                next_node = np.random.choice(neighbors)
            elif strategy == 'shortest':
                # Try to get closer to destination
                best_node = None
                best_distance = float('inf')
                for neighbor in neighbors:
                    distance = network.shortest_path_length(neighbor, destination)
                    if distance is not None and distance < best_distance:
                        best_distance = distance
                        best_node = neighbor
                if best_node is not None:
                    next_node = best_node
                else:
                    next_node = np.random.choice(neighbors)
            elif strategy == 'low_variance':
                # Choose neighbor with lowest variance edge
                best_node = None
                best_variance = float('inf')
                for neighbor in neighbors:
                    variance = network.get_edge_variance(current, neighbor)
                    if variance < best_variance:
                        best_variance = variance
                        best_node = neighbor
                next_node = best_node
            else:
                next_node = np.random.choice(neighbors)
            
            extended.append(next_node)
            current = next_node
        
        return extended
    
    @staticmethod
    def truncate_path(
        path: Path,
        network,
        destination: int,
        min_length: int = 1
    ) -> Path:
        """
        Truncate a path while maintaining validity.
        
        Args:
            path: Current path
            network: WormholeNetwork instance
            destination: Destination node
            min_length: Minimum path length
            
        Returns:
            Truncated path
        """
        if len(path) <= min_length:
            return path
        
        truncated = path.copy()
        
        # Remove nodes from end until we have a valid path to destination
        while len(truncated) > min_length:
            last_node = truncated[-1]
            
            # If last node can reach destination, keep it
            if network.has_edge(last_node, destination):
                break
            
            # Otherwise, remove it
            truncated.pop()
        
        return truncated
    
    @staticmethod
    def merge_paths(path1: Path, path2: Path) -> Path:
        """
        Merge two paths (for crossover operations).
        
        Args:
            path1: First path
            path2: Second path
            
        Returns:
            Merged path
        """
        # Simple concatenation (may create invalid path)
        merged = path1.copy()
        
        # Add path2, avoiding duplicates at junction
        if path2 and merged:
            if merged[-1] == path2[0]:
                merged.extend(path2[1:])
            else:
                merged.extend(path2)
        elif path2:
            merged = path2.copy()
        
        return merged
    
    @staticmethod
    def get_common_nodes(paths: PathList) -> set:
        """
        Get set of nodes visited by multiple ships.
        
        Args:
            paths: List of paths
            
        Returns:
            Set of nodes visited by more than one ship
        """
        node_counts = {}
        
        for path in paths:
            for node in path:
                node_counts[node] = node_counts.get(node, 0) + 1
        
        # Return nodes visited more than once
        return {node for node, count in node_counts.items() if count > 1}
    
    @staticmethod
    def compute_path_diversity(paths: PathList) -> float:
        """
        Compute diversity metric for a set of paths.
        
        Args:
            paths: List of paths
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not paths:
            return 0.0
        
        # Count unique nodes across all paths
        all_nodes = set()
        for path in paths:
            all_nodes.update(path)
        
        total_nodes = sum(len(path) for path in paths)
        
        if total_nodes == 0:
            return 0.0
        
        # Diversity = unique nodes / total nodes
        diversity = len(all_nodes) / total_nodes if total_nodes > 0 else 0.0
        
        return diversity

