"""
Data loading module for Wormhole Transportation Network.

This module handles loading and parsing of the database.npz file containing
the network structure and problem parameters.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import networkx as nx
from loguru import logger


class DataLoader:
    """
    Loads and manages wormhole network data from database.npz file.
    
    Attributes:
        network: NetworkX DiGraph representing the wormhole network
        n_nodes: Number of nodes (black holes) in the network
        n_edges: Number of edges (wormholes) in the network
        origins: List of origin node sets for each ship (12 ships)
        destination: Destination node ID
        delays: Initial temporal offsets for each ship (12 ships)
        jump_limit: Maximum number of jumps allowed per ship
        window: Arrival time window constraint
        edges: Raw edge data (source, target pairs)
        meanvar: Mean and variance for each edge
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            database_path: Path to database.npz file. If None, uses default path.
        """
        self.database_path = database_path or "./data/database.npz"
        self._database_path = self.database_path  # Store for later use
        self.network: Optional[nx.DiGraph] = None
        self.n_nodes: int = 0
        self.n_edges: int = 0
        self.origins: List[Set[int]] = []
        self.destination: int = 0
        self.delays: np.ndarray = np.array([])
        self.jump_limit: int = 500
        self.window: float = 1.0
        self.edges: np.ndarray = np.array([])
        self.meanvar: np.ndarray = np.array([])
        self._loaded: bool = False
    
    def load(self, database_path: Optional[str] = None) -> None:
        """
        Load data from database.npz file.
        
        Args:
            database_path: Path to database.npz file. If None, uses instance path.
            
        Raises:
            FileNotFoundError: If database file does not exist
            ValueError: If database file is invalid or missing required keys
        """
        if database_path:
            self.database_path = database_path
        
        db_path = Path(self.database_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database file not found: {db_path.absolute()}"
            )
        
        logger.info(f"Loading database from: {db_path.absolute()}")
        
        try:
            # Load the compressed numpy archive
            loaded = np.load(db_path)
            
            # Extract required data
            self._extract_edges(loaded)
            self._extract_meanvar(loaded)
            self._extract_parameters(loaded)
            self._build_network()
            self._extract_origins(loaded)
            self._extract_destination(loaded)
            self._extract_delays(loaded)
            
            self._loaded = True
            logger.info(
                f"Successfully loaded network: {self.n_nodes} nodes, "
                f"{self.n_edges} edges, {len(self.origins)} ships"
            )
            
        except Exception as e:
            raise ValueError(
                f"Error loading database from {db_path.absolute()}: {e}"
            )
    
    def _extract_edges(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract edge data from loaded archive."""
        if 'edges' not in loaded:
            raise ValueError("Database missing 'edges' key")
        
        self.edges = loaded['edges'].astype(np.int32)
        
        if self.edges.shape[1] != 2:
            raise ValueError(
                f"Expected edges to have 2 columns, got {self.edges.shape[1]}"
            )
        
        logger.debug(f"Loaded {len(self.edges)} edges")
    
    def _extract_meanvar(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract mean and variance data from loaded archive."""
        if 'meanvar' not in loaded:
            raise ValueError("Database missing 'meanvar' key")
        
        self.meanvar = loaded['meanvar'].astype(np.float32)
        
        if self.meanvar.shape[1] != 2:
            raise ValueError(
                f"Expected meanvar to have 2 columns, got {self.meanvar.shape[1]}"
            )
        
        if len(self.meanvar) != len(self.edges):
            raise ValueError(
                f"Mismatch: {len(self.edges)} edges but {len(self.meanvar)} meanvar entries"
            )
        
        logger.debug(f"Loaded mean/variance for {len(self.meanvar)} edges")
    
    def _extract_parameters(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract problem parameters from loaded archive."""
        # Jump limit
        if 'jump_limit' in loaded:
            self.jump_limit = int(loaded['jump_limit'].item())
        else:
            logger.warning("jump_limit not found in database, using default: 500")
            self.jump_limit = 500
        
        # Window
        if 'window' in loaded:
            self.window = float(loaded['window'].item())
        else:
            logger.warning("window not found in database, using default: 1.0")
            self.window = 1.0
        
        logger.debug(f"Jump limit: {self.jump_limit}, Window: {self.window}")
    
    def _build_network(self) -> None:
        """Build NetworkX DiGraph from edge data."""
        self.network = nx.DiGraph()
        
        # Add edges with mean and variance attributes
        for idx, (source, target) in enumerate(self.edges):
            mean = self.meanvar[idx][0]
            variance = self.meanvar[idx][1]
            
            self.network.add_edge(
                int(source),
                int(target),
                m=float(mean),
                v=float(variance)
            )
        
        self.n_nodes = self.network.number_of_nodes()
        self.n_edges = self.network.number_of_edges()
        
        logger.debug(
            f"Built network: {self.n_nodes} nodes, {self.n_edges} edges"
        )
    
    def _extract_origins(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract origin node sets for each ship."""
        if 'origins' not in loaded:
            raise ValueError("Database missing 'origins' key")
        
        origins_data = loaded['origins']
        self.origins = []
        
        # Handle different possible shapes
        if origins_data.ndim == 1:
            # Single array, assume 12 ships with same structure
            # This case might need adjustment based on actual data format
            logger.warning("Origins data is 1D, may need special handling")
            for i in range(12):
                self.origins.append(set())
        elif origins_data.ndim == 2:
            # 2D array: (12, variable) - each row is origin set for a ship
            for ship_idx in range(origins_data.shape[0]):
                origin_set = set(origins_data[ship_idx].astype(int))
                # Remove zeros if present (they might be padding)
                origin_set.discard(0)
                self.origins.append(origin_set)
        else:
            raise ValueError(
                f"Unexpected origins shape: {origins_data.shape}"
            )
        
        if len(self.origins) != 12:
            raise ValueError(
                f"Expected 12 origin sets, got {len(self.origins)}"
            )
        
        logger.debug(f"Loaded origin sets for {len(self.origins)} ships")
        for ship_idx, origin_set in enumerate(self.origins):
            logger.debug(f"Ship {ship_idx}: {len(origin_set)} origin nodes")
    
    def _extract_destination(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract destination node."""
        if 'destination' not in loaded:
            raise ValueError("Database missing 'destination' key")
        
        self.destination = int(loaded['destination'].item())
        
        if self.destination not in self.network.nodes:
            logger.warning(
                f"Destination node {self.destination} not in network nodes"
            )
        
        # Mark destination in network
        self.network.nodes[self.destination]['d'] = True
        
        logger.debug(f"Destination node: {self.destination}")
    
    def _extract_delays(self, loaded: np.lib.npyio.NpzFile) -> None:
        """Extract initial delays for each ship."""
        if 'delays' not in loaded:
            raise ValueError("Database missing 'delays' key")
        
        self.delays = loaded['delays'].astype(np.float32)
        
        if len(self.delays) != 12:
            raise ValueError(
                f"Expected 12 delays, got {len(self.delays)}"
            )
        
        logger.debug(f"Loaded delays for {len(self.delays)} ships")
        logger.debug(f"Delay range: [{self.delays.min():.6f}, {self.delays.max():.6f}]")
    
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded
    
    def get_network(self) -> nx.DiGraph:
        """Get the NetworkX graph."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.network
    
    def get_origin_set(self, ship_idx: int) -> Set[int]:
        """Get origin node set for a specific ship."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        if ship_idx < 0 or ship_idx >= len(self.origins):
            raise ValueError(f"Invalid ship index: {ship_idx}")
        return self.origins[ship_idx]
    
    def get_delay(self, ship_idx: int) -> float:
        """Get initial delay for a specific ship."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        if ship_idx < 0 or ship_idx >= len(self.delays):
            raise ValueError(f"Invalid ship index: {ship_idx}")
        return float(self.delays[ship_idx])
    
    def get_edge_mean(self, source: int, target: int) -> float:
        """Get mean temporal offset for an edge."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        if not self.network.has_edge(source, target):
            raise ValueError(f"Edge ({source}, {target}) does not exist")
        return float(self.network[source][target]['m'])
    
    def get_edge_variance(self, source: int, target: int) -> float:
        """Get variance for an edge."""
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        if not self.network.has_edge(source, target):
            raise ValueError(f"Edge ({source}, {target}) does not exist")
        return float(self.network[source][target]['v'])


def load_wormhole_data(
    database_path: str = "./data/database.npz"
) -> DataLoader:
    """
    Convenience function to load wormhole data.
    
    Args:
        database_path: Path to database.npz file
        
    Returns:
        DataLoader instance with loaded data
    """
    loader = DataLoader(database_path)
    loader.load()
    return loader

