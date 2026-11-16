"""
Main genetic algorithm implementation.

This module integrates all components into a complete GA solver.
"""

from typing import Optional, Dict, List, Tuple
import time
import numpy as np
from loguru import logger

from ..config.settings import SolverConfig
from ..data.loader import DataLoader
from ..model.network import WormholeNetwork
from ..evaluation.fitness import FitnessEvaluator
from ..evaluation.constraints import ConstraintChecker
from .population import Population
from .selection import SelectionOperator, TournamentSelection, RankBasedSelection
from .crossover import CrossoverOperator, SinglePointCrossover, UniformCrossover
from .mutation import MutationOperator, NodeReplacementMutation, CombinedMutation
from ..repair.origin_repair import OriginRepair
from ..repair.path_repair import PathRepair
from ..repair.window_repair import WindowRepair
from ..heuristics.initialization import (
    PopulationInitializer, RandomInitializer, HeuristicInitializer, MixedInitializer
)


class GeneticAlgorithm:
    """
    Main genetic algorithm solver for Wormhole Transportation Network.
    """
    
    def __init__(
        self,
        config: SolverConfig,
        data_loader: DataLoader,
        network: WormholeNetwork
    ):
        """
        Initialize genetic algorithm.
        
        Args:
            config: Solver configuration
            data_loader: DataLoader instance with loaded data
            network: WormholeNetwork instance
        """
        self.config = config
        self.data_loader = data_loader
        self.network = network
        
        # Initialize components
        self.evaluator = FitnessEvaluator(database_path=data_loader.database_path)
        self.constraint_checker = ConstraintChecker(
            network=network,
            origins=data_loader.origins,
            destination=data_loader.destination,
            delays=data_loader.delays,
            window=data_loader.window,
            n_ships=config.n_ships,
            jump_limit=config.jump_limit
        )
        
        # Repair operators
        self.origin_repair = OriginRepair(
            origins=data_loader.origins,
            n_ships=config.n_ships,
            jump_limit=config.jump_limit
        )
        self.path_repair = PathRepair(
            network=network,
            destination=data_loader.destination,
            n_ships=config.n_ships,
            jump_limit=config.jump_limit
        )
        self.window_repair = WindowRepair(
            network=network,
            destination=data_loader.destination,
            delays=data_loader.delays,
            window=data_loader.window,
            n_ships=config.n_ships,
            jump_limit=config.jump_limit
        )
        
        # Selection operator
        if config.selection_type == 'tournament':
            self.selection = TournamentSelection(config.tournament_size)
        elif config.selection_type == 'rank_based':
            self.selection = RankBasedSelection()
        else:
            self.selection = TournamentSelection(config.tournament_size)
        
        # Crossover operator
        if config.crossover_type == 'single_point':
            self.crossover = SinglePointCrossover(config.n_ships, config.jump_limit)
        elif config.crossover_type == 'uniform':
            self.crossover = UniformCrossover(n_ships=config.n_ships, jump_limit=config.jump_limit)
        else:
            self.crossover = SinglePointCrossover(config.n_ships, config.jump_limit)
        
        # Mutation operator
        if config.mutation_type == 'node_replacement':
            self.mutation = NodeReplacementMutation(
                network=network,
                origins=data_loader.origins,
                destination=data_loader.destination,
                n_ships=config.n_ships,
                jump_limit=config.jump_limit
            )
        else:
            self.mutation = CombinedMutation(
                network=network,
                origins=data_loader.origins,
                destination=data_loader.destination,
                n_ships=config.n_ships,
                jump_limit=config.jump_limit
            )
        
        # Population
        self.population: Optional[Population] = None
        
        # Statistics
        self.generation = 0
        self.evaluation_count = 0
        self.start_time = None
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: Optional[float] = None
        self.best_generation = 0
        self.history: List[Dict] = []
    
    def initialize_population(self) -> None:
        """Initialize population using configured strategy."""
        logger.info("Initializing population...")
        
        # Choose initializer
        if self.config.initialization_type == 'random':
            initializer = RandomInitializer()
        elif self.config.initialization_type == 'heuristic':
            initializer = HeuristicInitializer()
        else:
            initializer = MixedInitializer(
                heuristic_ratio=self.config.heuristic_ratio
            )
        
        # Generate chromosomes
        chromosomes = initializer.initialize(
            size=self.config.population_size,
            network=self.network,
            origins=self.data_loader.origins,
            destination=self.data_loader.destination,
            jump_limit=self.config.jump_limit
        )
        
        # Create population
        self.population = Population(
            chromosomes=chromosomes,
            evaluator=self.evaluator,
            n_ships=self.config.n_ships,
            jump_limit=self.config.jump_limit
        )
        
        # Evaluate initial population
        self.population.evaluate_all()
        self.evaluation_count += self.config.population_size
        
        # Track best
        best = self.population.get_best()
        if best is not None:
            self.best_solution = best[1]
            self.best_fitness = best[2]
        
        logger.info(
            f"Population initialized: {self.config.population_size} individuals, "
            f"best fitness: {self.best_fitness:.2f}"
        )
    
    def evolve(self) -> None:
        """Run one generation of evolution."""
        if self.population is None:
            raise RuntimeError("Population not initialized")
        
        offspring = []
        
        # Generate offspring
        while len(offspring) < self.config.population_size:
            # Selection
            parents = self.selection.select(self.population, n=2)
            parent1 = self.population.get(parents[0])
            parent2 = self.population.get(parents[1])
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child1 = self.mutation.mutate(child1)
            if np.random.random() < self.config.mutation_rate:
                child2 = self.mutation.mutate(child2)
            
            # Repair constraints
            child1 = self.origin_repair.repair(child1)
            child1 = self.path_repair.repair(child1)
            
            child2 = self.origin_repair.repair(child2)
            child2 = self.path_repair.repair(child2)
            
            offspring.append(child1)
            if len(offspring) < self.config.population_size:
                offspring.append(child2)
        
        # Evaluate offspring
        offspring_pop = Population(
            chromosomes=offspring,
            evaluator=self.evaluator,
            n_ships=self.config.n_ships,
            jump_limit=self.config.jump_limit
        )
        offspring_pop.evaluate_all()
        self.evaluation_count += len(offspring)
        
        # Replacement
        if self.config.replacement_strategy == 'generational':
            self.population = offspring_pop
        elif self.config.replacement_strategy == 'elitism':
            # Keep elite from current population
            elite_size = int(self.config.population_size * self.config.elitism_rate)
            elite = self.population.get_elite(elite_size)
            
            # Combine with best offspring
            offspring_elite = offspring_pop.get_elite(
                self.config.population_size - elite_size
            )
            
            new_chromosomes = elite + offspring_elite
            self.population = Population(
                chromosomes=new_chromosomes,
                evaluator=self.evaluator,
                n_ships=self.config.n_ships,
                jump_limit=self.config.jump_limit
            )
            self.population.evaluate_all()
        else:
            # Steady state: replace worst
            combined = self.population.chromosomes + offspring
            combined_pop = Population(
                chromosomes=combined,
                evaluator=self.evaluator,
                n_ships=self.config.n_ships,
                jump_limit=self.config.jump_limit
            )
            combined_pop.evaluate_all()
            
            # Keep best
            elite = combined_pop.get_elite(self.config.population_size)
            self.population = Population(
                chromosomes=elite,
                evaluator=self.evaluator,
                n_ships=self.config.n_ships,
                jump_limit=self.config.jump_limit
            )
            self.population.evaluate_all()
        
        # Update best solution
        best = self.population.get_best()
        if best is not None:
            if self.best_fitness is None or best[2] < self.best_fitness:
                self.best_solution = best[1]
                self.best_fitness = best[2]
                self.best_generation = self.generation
    
    def should_terminate(self) -> bool:
        """Check termination criteria."""
        # Max generations
        if self.generation >= self.config.max_generations:
            return True
        
        # Max evaluations
        if (self.config.max_evaluations is not None and
            self.evaluation_count >= self.config.max_evaluations):
            return True
        
        # Max time
        if self.config.max_time_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.config.max_time_seconds:
                return True
        
        # Target fitness
        if (self.config.target_fitness is not None and
            self.best_fitness is not None and
            self.best_fitness <= self.config.target_fitness):
            return True
        
        # Stagnation
        if len(self.history) >= self.config.stagnation_generations:
            recent_best = min(
                h['best_fitness']
                for h in self.history[-self.config.stagnation_generations:]
            )
            if self.best_fitness is not None and abs(self.best_fitness - recent_best) < 1e-6:
                return True
        
        return False
    
    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run the genetic algorithm.
        
        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        logger.info("Starting genetic algorithm...")
        self.start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        while not self.should_terminate():
            self.generation += 1
            
            # Evolve
            self.evolve()
            
            # Record history
            stats = self.population.get_statistics()
            stats['generation'] = self.generation
            stats['best_fitness'] = self.best_fitness
            stats['best_generation'] = self.best_generation
            stats['evaluation_count'] = self.evaluation_count
            stats['elapsed_time'] = time.time() - self.start_time
            self.history.append(stats)
            
            # Logging
            if self.generation % self.config.log_interval == 0:
                feasible_rate = self.constraint_checker.get_feasibility_rate(
                    self.population.chromosomes
                )
                logger.info(
                    f"Generation {self.generation}: "
                    f"best={self.best_fitness:.2f}, "
                    f"mean={stats.get('objective_mean', 0):.2f}, "
                    f"feasible={feasible_rate:.2%}"
                )
        
        elapsed = time.time() - self.start_time
        logger.info(
            f"Algorithm completed: {self.generation} generations, "
            f"{self.evaluation_count} evaluations, {elapsed:.2f}s"
        )
        logger.info(f"Best fitness: {self.best_fitness:.2f} (generation {self.best_generation})")
        
        if self.best_solution is None:
            raise RuntimeError("No solution found")
        
        return self.best_solution, self.best_fitness

