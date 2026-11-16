"""
Origin node repair operator.

This module provides operators to fix origin node constraint violations.
"""

from typing import List
import numpy as np
from loguru import logger

from ..model.chromosome import ChromosomeDecoder, ChromosomeEncoder


class OriginRepair:
    """
    Repairs origin node constraint violations.
    """
    
    def __init__(
        self,
        origins: List[set],
        n_ships: int = 12,
        jump_limit: int = 500
    ):
        """
        Initialize origin repair operator.
        
        Args:
            origins: List of origin sets for each ship
            n_ships: Number of ships
            jump_limit: Maximum jumps per ship
        """
        self.origins = origins
        self.n_ships = n_ships
        self.jump_limit = jump_limit
        self.decoder = ChromosomeDecoder(n_ships, jump_limit)
        self.encoder = ChromosomeEncoder(n_ships, jump_limit)
    
    def repair(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Repair origin node violations.
        
        Args:
            chromosome: Chromosome to repair
            
        Returns:
            Repaired chromosome
        """
        paths = self.decoder.decode(chromosome, destination=None)
        
        for ship_idx, path in enumerate(paths):
            if not path:
                # Empty path - initialize with random origin
                if self.origins[ship_idx]:
                    path = [np.random.choice(list(self.origins[ship_idx]))]
                    paths[ship_idx] = path
            elif path[0] not in self.origins[ship_idx]:
                # Invalid origin - replace with valid one
                if self.origins[ship_idx]:
                    path[0] = np.random.choice(list(self.origins[ship_idx]))
                    paths[ship_idx] = path
        
        # Re-encode
        repaired = self.encoder.encode(paths)
        return repaired
    
    def repair_all(self, chromosomes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Repair multiple chromosomes.
        
        Args:
            chromosomes: List of chromosomes to repair
            
        Returns:
            List of repaired chromosomes
        """
        return [self.repair(chr) for chr in chromosomes]

