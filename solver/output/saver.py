"""
Solution saving utilities.

This module handles saving solutions, statistics, and logs.
"""

from typing import Dict, List, Optional
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger


class SolutionSaver:
    """
    Saves solutions and associated data.
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize solution saver.
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.solutions_dir = self.output_dir / "solutions"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        
        self.solutions_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
    
    def save_chromosome(
        self,
        chromosome: np.ndarray,
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save chromosome to .npy file.
        
        Args:
            chromosome: Chromosome to save
            filename: Optional filename. If None, generates timestamp-based name.
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"solution_{timestamp}.npy"
        
        filepath = self.solutions_dir / filename
        
        # Save chromosome
        np.save(filepath, chromosome)
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Solution saved: {filepath.absolute()}")
        return filepath
    
    def save_statistics(
        self,
        statistics: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save statistics to JSON file.
        
        Args:
            statistics: Statistics dictionary
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistics_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)
        
        logger.info(f"Statistics saved: {filepath.absolute()}")
        return filepath
    
    def save_history(
        self,
        history: List[Dict],
        filename: Optional[str] = None
    ) -> Path:
        """
        Save evolution history.
        
        Args:
            history: List of generation statistics
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"History saved: {filepath.absolute()}")
        return filepath
    
    def save_config(
        self,
        config: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save configuration.
        
        Args:
            config: Configuration dictionary
            filename: Optional filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"config_{timestamp}.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Configuration saved: {filepath.absolute()}")
        return filepath
    
    def save_summary(
        self,
        best_chromosome: np.ndarray,
        best_fitness: float,
        statistics: Dict,
        history: List[Dict],
        config: Dict,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save complete solution summary.
        
        Args:
            best_chromosome: Best solution chromosome
            best_fitness: Best fitness value
            statistics: Final statistics
            history: Evolution history
            config: Configuration used
            filename: Optional base filename (without extension)
            
        Returns:
            Path to summary directory
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"run_{timestamp}"
        
        summary_dir = self.output_dir / filename
        summary_dir.mkdir(exist_ok=True)
        
        # Save chromosome
        self.save_chromosome(
            best_chromosome,
            filename=f"{filename}_solution.npy",
            metadata={
                'best_fitness': best_fitness,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Save statistics
        stats_path = summary_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)
        
        # Save history
        history_path = summary_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Save config
        config_path = summary_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Create summary text file
        summary_path = summary_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Wormhole Transportation Network - Solution Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Best Fitness: {best_fitness:.6f}\n\n")
            f.write("Statistics:\n")
            for key, value in statistics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            f.write(f"Total Generations: {len(history)}\n")
            f.write(f"Total Evaluations: {statistics.get('evaluation_count', 'N/A')}\n")
        
        logger.info(f"Complete summary saved: {summary_dir.absolute()}")
        return summary_dir
    
    def load_chromosome(self, filepath: str) -> np.ndarray:
        """
        Load chromosome from .npy file.
        
        Args:
            filepath: Path to .npy file
            
        Returns:
            Loaded chromosome
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.absolute()}")
        
        chromosome = np.load(path)
        logger.info(f"Chromosome loaded: {path.absolute()}")
        return chromosome
    
    def load_statistics(self, filepath: str) -> Dict:
        """
        Load statistics from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Statistics dictionary
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.absolute()}")
        
        with open(path, 'r') as f:
            statistics = json.load(f)
        
        logger.info(f"Statistics loaded: {path.absolute()}")
        return statistics

