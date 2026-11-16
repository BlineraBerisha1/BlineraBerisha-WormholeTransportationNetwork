"""
Chromosome encoding and decoding for Wormhole Transportation Network.

The chromosome is a fixed-length integer array of 6000 elements representing
paths for 12 ships, each with up to 500 jumps.
"""

from typing import List, Optional, Tuple
import numpy as np
from loguru import logger


class ChromosomeEncoder:
    """
    Encodes paths into chromosome representation.
    
    Chromosome format: [ship_0_path, ship_1_path, ..., ship_11_path]
    Each ship path: [n_0, n_1, ..., n_499] where n_k ∈ [0, 10000]
    Value 0 indicates path termination (padding).
    """
    
    def __init__(self, n_ships: int = 12, jump_limit: int = 500):
        """
        Initialize encoder.
        
        Args:
            n_ships: Number of ships (default: 12)
            jump_limit: Maximum jumps per ship (default: 500)
        """
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.chromosome_length = n_ships * jump_limit
    
    def encode(self, paths: List[List[int]]) -> np.ndarray:
        """
        Encode list of paths into chromosome.
        
        Args:
            paths: List of 12 paths, each path is a list of node IDs
            
        Returns:
            Chromosome as numpy array of shape (6000,)
        """
        if len(paths) != self.n_ships:
            raise ValueError(
                f"Expected {self.n_ships} paths, got {len(paths)}"
            )
        
        chromosome = np.zeros((self.chromosome_length,), dtype=np.int32)
        
        for ship_idx, path in enumerate(paths):
            if not path:
                logger.warning(f"Ship {ship_idx}: empty path")
                continue
            
            # Remove destination if present (it's implicit)
            path_copy = path.copy()
            if len(path_copy) > 1:
                # Keep all nodes except destination (will be added during evaluation)
                pass
            
            # Truncate if too long
            if len(path_copy) > self.jump_limit:
                logger.warning(
                    f"Ship {ship_idx}: path length {len(path_copy)} exceeds "
                    f"jump_limit {self.jump_limit}, truncating"
                )
                path_copy = path_copy[:self.jump_limit]
            
            # Encode path
            start_idx = ship_idx * self.jump_limit
            end_idx = start_idx + len(path_copy)
            chromosome[start_idx:end_idx] = path_copy
            
            # Remaining positions are already zero (padding)
        
        return chromosome
    
    def encode_with_destination(
        self,
        paths: List[List[int]],
        destination: int
    ) -> np.ndarray:
        """
        Encode paths, ensuring last node has edge to destination.
        
        Args:
            paths: List of 12 paths
            destination: Destination node ID
            
        Returns:
            Chromosome as numpy array
        """
        # Ensure all paths end with a node that can reach destination
        # (destination itself is not encoded, but last node must connect to it)
        return self.encode(paths)


class ChromosomeDecoder:
    """
    Decodes chromosome into list of paths.
    
    Interprets chromosome by:
    1. Splitting into 12 ship sections
    2. Truncating at first zero (path termination)
    3. Extracting node sequences
    """
    
    def __init__(self, n_ships: int = 12, jump_limit: int = 500):
        """
        Initialize decoder.
        
        Args:
            n_ships: Number of ships (default: 12)
            jump_limit: Maximum jumps per ship (default: 500)
        """
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.chromosome_length = n_ships * jump_limit
    
    def decode(
        self,
        chromosome: np.ndarray,
        destination: Optional[int] = None
    ) -> List[List[int]]:
        """
        Decode chromosome into list of paths.
        
        Args:
            chromosome: Chromosome array of shape (6000,)
            destination: Optional destination node to append to each path
            
        Returns:
            List of 12 paths, each path is a list of node IDs
        """
        if len(chromosome) != self.chromosome_length:
            raise ValueError(
                f"Chromosome length {len(chromosome)} does not match "
                f"expected length {self.chromosome_length}"
            )
        
        paths = []
        
        for ship_idx in range(self.n_ships):
            # Extract ship section
            start_idx = ship_idx * self.jump_limit
            end_idx = (ship_idx + 1) * self.jump_limit
            ship_section = chromosome[start_idx:end_idx]
            
            # Find first zero (path termination)
            zero_indices = np.where(ship_section == 0)[0]
            if len(zero_indices) > 0:
                # Path ends at first zero
                path_length = zero_indices[0]
                path = ship_section[:path_length].tolist()
            else:
                # No zeros, use full section
                path = ship_section.tolist()
            
            # Remove any trailing zeros (shouldn't happen, but safe)
            while path and path[-1] == 0:
                path.pop()
            
            # Append destination if provided
            if destination is not None and path:
                # Only append if not already destination
                if path[-1] != destination:
                    path.append(destination)
            
            paths.append(path)
        
        return paths
    
    def get_path_lengths(self, chromosome: np.ndarray) -> List[int]:
        """
        Get length of each path in chromosome.
        
        Args:
            chromosome: Chromosome array
            
        Returns:
            List of path lengths for each ship
        """
        paths = self.decode(chromosome, destination=None)
        return [len(path) for path in paths]
    
    def validate_chromosome(
        self,
        chromosome: np.ndarray,
        min_node: int = 0,
        max_node: int = 10000
    ) -> Tuple[bool, List[str]]:
        """
        Validate chromosome format.
        
        Args:
            chromosome: Chromosome array
            min_node: Minimum valid node ID
            max_node: Maximum valid node ID
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check length
        if len(chromosome) != self.chromosome_length:
            errors.append(
                f"Chromosome length {len(chromosome)} != {self.chromosome_length}"
            )
            return False, errors
        
        # Check dtype
        if chromosome.dtype != np.int32 and chromosome.dtype != np.int64:
            errors.append(f"Chromosome dtype {chromosome.dtype} is not integer")
        
        # Check node ID range
        invalid_nodes = chromosome[
            (chromosome < min_node) | (chromosome > max_node)
        ]
        if len(invalid_nodes) > 0:
            errors.append(
                f"Found {len(invalid_nodes)} node IDs outside range "
                f"[{min_node}, {max_node}]"
            )
        
        # Check for negative values (except 0 which is valid padding)
        negative_nodes = chromosome[chromosome < 0]
        if len(negative_nodes) > 0:
            errors.append(f"Found {len(negative_nodes)} negative node IDs")
        
        is_valid = len(errors) == 0
        return is_valid, errors


def encode_paths(
    paths: List[List[int]],
    n_ships: int = 12,
    jump_limit: int = 500
) -> np.ndarray:
    """
    Convenience function to encode paths.
    
    Args:
        paths: List of 12 paths
        n_ships: Number of ships
        jump_limit: Maximum jumps per ship
        
    Returns:
        Chromosome array
    """
    encoder = ChromosomeEncoder(n_ships, jump_limit)
    return encoder.encode(paths)


def decode_chromosome(
    chromosome: np.ndarray,
    destination: Optional[int] = None,
    n_ships: int = 12,
    jump_limit: int = 500
) -> List[List[int]]:
    """
    Convenience function to decode chromosome.
    
    Args:
        chromosome: Chromosome array
        destination: Optional destination node
        n_ships: Number of ships
        jump_limit: Maximum jumps per ship
        
    Returns:
        List of paths
    """
    decoder = ChromosomeDecoder(n_ships, jump_limit)
    return decoder.decode(chromosome, destination)

