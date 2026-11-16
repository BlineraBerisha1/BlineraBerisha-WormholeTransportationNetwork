"""
Data validation module for Wormhole Transportation Network.

This module validates the integrity and consistency of loaded network data.
"""

from typing import List, Set, Tuple
import numpy as np
import networkx as nx
from loguru import logger


class DataValidator:
    """
    Validates wormhole network data for consistency and integrity.
    """
    
    def __init__(
        self,
        network: nx.DiGraph,
        origins: List[Set[int]],
        destination: int,
        delays: np.ndarray,
        jump_limit: int,
        window: float
    ):
        """
        Initialize validator.
        
        Args:
            network: NetworkX DiGraph
            origins: List of origin sets for each ship
            destination: Destination node ID
            delays: Initial delays for each ship
            jump_limit: Maximum jumps per ship
            window: Arrival time window
        """
        self.network = network
        self.origins = origins
        self.destination = destination
        self.delays = delays
        self.jump_limit = jump_limit
        self.window = window
    
    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Run all validation checks.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Run all checks
        errors.extend(self.validate_network_structure())
        errors.extend(self.validate_origins())
        errors.extend(self.validate_destination())
        errors.extend(self.validate_delays())
        errors.extend(self.validate_parameters())
        errors.extend(self.validate_reachability())
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("All validation checks passed")
        else:
            logger.warning(f"Validation found {len(errors)} issues")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors
    
    def validate_network_structure(self) -> List[str]:
        """Validate network graph structure."""
        errors = []
        
        if self.network is None:
            errors.append("Network is None")
            return errors
        
        if not isinstance(self.network, nx.DiGraph):
            errors.append("Network is not a directed graph")
        
        if self.network.number_of_nodes() == 0:
            errors.append("Network has no nodes")
        
        if self.network.number_of_edges() == 0:
            errors.append("Network has no edges")
        
        # Check all edges have mean and variance
        missing_attrs = 0
        for source, target in self.network.edges():
            if 'm' not in self.network[source][target]:
                missing_attrs += 1
            if 'v' not in self.network[source][target]:
                missing_attrs += 1
        
        if missing_attrs > 0:
            errors.append(f"{missing_attrs} edges missing mean/variance attributes")
        
        # Check for self-loops (might be valid, but worth noting)
        self_loops = list(nx.selfloop_edges(self.network))
        if len(self_loops) > 0:
            logger.debug(f"Network has {len(self_loops)} self-loops")
        
        return errors
    
    def validate_origins(self) -> List[str]:
        """Validate origin node sets."""
        errors = []
        
        if len(self.origins) != 12:
            errors.append(f"Expected 12 origin sets, got {len(self.origins)}")
        
        for ship_idx, origin_set in enumerate(self.origins):
            if not isinstance(origin_set, set):
                errors.append(f"Ship {ship_idx}: origin set is not a set")
                continue
            
            if len(origin_set) == 0:
                errors.append(f"Ship {ship_idx}: origin set is empty")
            
            # Check all origin nodes exist in network
            for origin_node in origin_set:
                if origin_node not in self.network.nodes:
                    errors.append(
                        f"Ship {ship_idx}: origin node {origin_node} not in network"
                    )
        
        return errors
    
    def validate_destination(self) -> List[str]:
        """Validate destination node."""
        errors = []
        
        if self.destination == 0:
            errors.append("Destination node is 0 (invalid)")
        
        if self.destination not in self.network.nodes:
            errors.append(f"Destination node {self.destination} not in network")
        
        return errors
    
    def validate_delays(self) -> List[str]:
        """Validate delay values."""
        errors = []
        
        if len(self.delays) != 12:
            errors.append(f"Expected 12 delays, got {len(self.delays)}")
        
        if np.any(np.isnan(self.delays)):
            errors.append("Delays contain NaN values")
        
        if np.any(np.isinf(self.delays)):
            errors.append("Delays contain infinite values")
        
        return errors
    
    def validate_parameters(self) -> List[str]:
        """Validate problem parameters."""
        errors = []
        
        if self.jump_limit <= 0:
            errors.append(f"Jump limit must be positive, got {self.jump_limit}")
        
        if self.jump_limit > 1000:
            errors.append(f"Jump limit seems unusually large: {self.jump_limit}")
        
        if self.window <= 0:
            errors.append(f"Window must be positive, got {self.window}")
        
        return errors
    
    def validate_reachability(self) -> List[str]:
        """Validate that destination is reachable from origins."""
        errors = []
        
        if self.destination not in self.network.nodes:
            return errors  # Already reported in validate_destination
        
        unreachable_ships = []
        for ship_idx, origin_set in enumerate(self.origins):
            reachable = False
            for origin in origin_set:
                if origin in self.network.nodes:
                    if nx.has_path(self.network, origin, self.destination):
                        reachable = True
                        break
            
            if not reachable:
                unreachable_ships.append(ship_idx)
        
        if len(unreachable_ships) > 0:
            errors.append(
                f"Ships {unreachable_ships} cannot reach destination from any origin"
            )
        
        return errors


def validate_network_data(
    network: nx.DiGraph,
    origins: List[Set[int]],
    destination: int,
    delays: np.ndarray,
    jump_limit: int,
    window: float
) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate network data.
    
    Args:
        network: NetworkX DiGraph
        origins: List of origin sets
        destination: Destination node
        delays: Ship delays
        jump_limit: Maximum jumps
        window: Arrival window
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = DataValidator(
        network, origins, destination, delays, jump_limit, window
    )
    return validator.validate_all()

