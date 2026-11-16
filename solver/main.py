"""
Main entry point for Wormhole Transportation Network Solver.

This script runs the genetic algorithm solver for the ESA SpOC2 challenge.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

from .config.settings import SolverConfig, load_config
from .data.loader import DataLoader, load_wormhole_data
from .model.network import WormholeNetwork
from .genetic.algorithm import GeneticAlgorithm
from .output.formatter import SolutionFormatter, create_submission_json
from .output.saver import SolutionSaver


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wormhole Transportation Network Solver (ESA SpOC2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="./data/database.npz",
        help="Path to database.npz file"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file (uses defaults if not provided)"
    )
    
    # Algorithm parameters (override config)
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Population size"
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Maximum number of generations"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=None,
        help="Crossover rate"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=None,
        help="Mutation rate"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--submission-file",
        type=str,
        default=None,
        help="Path to save submission JSON file"
    )
    parser.add_argument(
        "--save-solution",
        action="store_true",
        help="Save solution chromosome to .npy file"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    # Other options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data and configuration, don't run solver"
    )
    
    return parser.parse_args()


def update_config_from_args(config: SolverConfig, args: argparse.Namespace) -> SolverConfig:
    """Update configuration from command-line arguments."""
    if args.population_size is not None:
        config.population_size = args.population_size
    if args.max_generations is not None:
        config.max_generations = args.max_generations
    if args.crossover_rate is not None:
        config.crossover_rate = args.crossover_rate
    if args.mutation_rate is not None:
        config.mutation_rate = args.mutation_rate
    if args.output_dir:
        config.output_dir = args.output_dir
    
    return config


def main() -> int:
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger.info("=" * 70)
    logger.info("Wormhole Transportation Network Solver")
    logger.info("ESA SpOC2 Challenge")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Using default configuration")
            config = SolverConfig()
        
        # Update from command-line arguments
        config = update_config_from_args(config, args)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return 1
        
        logger.info(f"Configuration: population={config.population_size}, "
                   f"generations={config.max_generations}, "
                   f"crossover={config.crossover_rate}, "
                   f"mutation={config.mutation_rate}")
        
        # Load data
        logger.info(f"Loading data from: {args.data}")
        data_loader = load_wormhole_data(args.data)
        
        # Build network
        logger.info("Building network graph...")
        network = WormholeNetwork(data_loader.get_network())
        
        # Validate data
        from .data.validator import validate_network_data
        is_valid, validation_errors = validate_network_data(
            network=network.graph,
            origins=data_loader.origins,
            destination=data_loader.destination,
            delays=data_loader.delays,
            jump_limit=data_loader.jump_limit,
            window=data_loader.window
        )
        
        if not is_valid:
            logger.warning("Data validation found issues:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Data validation passed")
        
        # If validate-only, exit here
        if args.validate_only:
            logger.info("Validation complete. Exiting.")
            return 0
        
        # Initialize genetic algorithm
        logger.info("Initializing genetic algorithm...")
        ga = GeneticAlgorithm(
            config=config,
            data_loader=data_loader,
            network=network
        )
        
        # Run algorithm
        logger.info("Starting genetic algorithm...")
        best_solution, best_fitness = ga.run()
        
        # Get final statistics
        final_stats = ga.population.get_statistics()
        final_stats['best_fitness'] = best_fitness
        final_stats['best_generation'] = ga.best_generation
        final_stats['total_generations'] = ga.generation
        final_stats['total_evaluations'] = ga.evaluation_count
        
        logger.info("=" * 70)
        logger.info("Algorithm completed successfully!")
        logger.info(f"Best fitness: {best_fitness:.6f}")
        logger.info(f"Best generation: {ga.best_generation}")
        logger.info(f"Total generations: {ga.generation}")
        logger.info(f"Total evaluations: {ga.evaluation_count}")
        logger.info("=" * 70)
        
        # Save results
        saver = SolutionSaver(output_dir=config.output_dir)
        
        # Save complete summary
        summary_dir = saver.save_summary(
            best_chromosome=best_solution,
            best_fitness=best_fitness,
            statistics=final_stats,
            history=ga.history,
            config=config.to_dict()
        )
        logger.info(f"Results saved to: {summary_dir}")
        
        # Save submission file
        if args.submission_file:
            submission_path = args.submission_file
        else:
            submission_path = summary_dir / "submission.json"
        
        formatter = SolutionFormatter()
        formatter.create_submission_json(
            chromosome=best_solution,
            filepath=str(submission_path),
            submission_name=f"GA Solution (fitness={best_fitness:.6f})",
            submission_description=f"Genetic algorithm solution after {ga.generation} generations"
        )
        logger.info(f"Submission file saved: {submission_path}")
        
        # Save solution as .npy if requested
        if args.save_solution:
            solution_path = summary_dir / "solution.npy"
            saver.save_chromosome(
                best_solution,
                filename=solution_path.name,
                metadata={
                    'fitness': best_fitness,
                    'generation': ga.best_generation
                }
            )
            logger.info(f"Solution saved: {solution_path}")
        
        # Check feasibility
        is_feasible = ga.evaluator.is_feasible(best_solution)
        if is_feasible:
            logger.info("✓ Solution is FEASIBLE (all constraints satisfied)")
        else:
            violations = ga.constraint_checker.check_all(best_solution)
            logger.warning("✗ Solution is INFEASIBLE:")
            logger.warning(f"  Origin violations: {violations.origin_violations}")
            logger.warning(f"  Path violations: {violations.path_violations}")
            logger.warning(f"  Window violation: {violations.window_violation:.6f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

