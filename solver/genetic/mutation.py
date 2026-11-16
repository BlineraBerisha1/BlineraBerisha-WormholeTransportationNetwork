"""
Mutation operators for genetic algorithm.

This module implements various mutation strategies for modifying chromosomes.
"""

from typing import List
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder, ChromosomeEncoder
from ..model.network import WormholeNetwork


class MutationOperator:
    """
    Base class for mutation operators.
    """
    
    def __init__(
        self,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize mutation operator.
        
        Args:
            network: WormholeNetwork instance
            origins: List of origin sets for each ship
            destination: Destination node ID
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.network = network
        self.origins = origins
        self.destination = destination
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
        self.encoder = ChromosomeEncoder(n_ships, jump_limit)
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutate a chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        raise NotImplementedError


class NodeReplacementMutation(MutationOperator):
    """
    Mutation by replacing random nodes with valid neighbors.
    """
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutate by replacing nodes.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        paths = self.decoder.decode(chromosome, destination=None)
        
        # Mutate each ship's path with some probability
        for ship_idx, path in enumerate(paths):
            if not path or len(path) < 2:
                continue
            
            # Choose random node to replace (not origin)
            if len(path) > 1:
                replace_idx = np.random.randint(1, len(path))
                prev_node = path[replace_idx - 1]
                
                # Get valid neighbors
                neighbors = self.network.get_neighbors(prev_node)
                if neighbors:
                    # Replace with random neighbor
                    path[replace_idx] = np.random.choice(neighbors)
        
        # Re-encode
        mutated = self.encoder.encode(paths)
        return mutated


class PathExtensionMutation(MutationOperator):
    """
    Mutation by extending paths towards destination.
    """
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutate by extending paths.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        paths = self.decoder.decode(chromosome, destination=None)
        
        for ship_idx, path in enumerate(paths):
            if not path:
                continue
            
            # Only extend if path is shorter than jump_limit
            if len(path) < self.jump_limit:
                current = path[-1]
                
                # Try to extend towards destination
                neighbors = self.network.get_neighbors(current)
                if neighbors:
                    # Prefer destination if reachable
                    if self.destination in neighbors:
                        path.append(self.destination)
                    else:
                        # Add random neighbor
                        path.append(np.random.choice(neighbors))
        
        # Re-encode
        mutated = self.encoder.encode(paths)
        return mutated


class PathTruncationMutation(MutationOperator):
    """
    Mutation by truncating paths.
    """
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutate by truncating paths.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        paths = self.decoder.decode(chromosome, destination=None)
        
        for ship_idx, path in enumerate(paths):
            if len(path) > 2:
                # Truncate to random length (keep at least origin)
                new_length = np.random.randint(1, len(path))
                paths[ship_idx] = path[:new_length]
        
        # Re-encode
        mutated = self.encoder.encode(paths)
        return mutated


class CombinedMutation(MutationOperator):
    """
    Combined mutation using multiple strategies.
    """
    
    def __init__(
        self,
        network: WormholeNetwork,
        origins: List[set],
        destination: int,
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """Initialize combined mutation."""
        super().__init__(network, origins, destination, n_ships, jump_limit)
        
        self.node_mutation = NodeReplacementMutation(
            network, origins, destination, n_ships, jump_limit
        )
        self.extension_mutation = PathExtensionMutation(
            network, origins, destination, n_ships, jump_limit
        )
        self.truncation_mutation = PathTruncationMutation(
            network, origins, destination, n_ships, jump_limit
        )
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Mutate using random strategy.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutation_type = np.random.choice([
            'node_replacement',
            'extension',
            'truncation'
        ], p=[0.5, 0.3, 0.2])
        
        if mutation_type == 'node_replacement':
            return self.node_mutation.mutate(chromosome)
        elif mutation_type == 'extension':
            return self.extension_mutation.mutate(chromosome)
        else:
            return self.truncation_mutation.mutate(chromosome)

