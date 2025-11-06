"""
Genetic algorithm strategy for BSEE.
"""

import random
import yaml
import os
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State
from bsee.operations.operations_registry import OperationsRegistry


class Individual:
    """Individual in the genetic algorithm population."""

    def __init__(self, genome: List[Tuple[str, Dict[str, Any]]], state: State):
        """Initialize an individual."""
        self.genome = genome  # List of (operation_name, params) tuples
        self.state = state
        self.fitness = state.score if hasattr(state, 'score') else 0.0
        self.age = 0

    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        return Individual(self.genome.copy(), self.state.copy())

    def __lt__(self, other: 'Individual') -> bool:
        """Less than comparison for sorting by fitness."""
        return self.fitness < other.fitness


class GeneticStrategy(BaseStrategy):
    """Genetic algorithm strategy with real implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize genetic strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = OperationsRegistry()
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self.config.get('config_file', 'config/strategies/strategy_genetic.yaml')

        # Try to load from file, otherwise use passed config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge file config with passed config
                self.config = {**file_config, **self.config}
            except Exception as e:
                print(f"Warning: Could not load Genetic config from {config_path}: {e}")

        # Extract parameters from config
        params = self.config.get('parameters', {})
        self.population_size = params.get('population_size', 50)
        self.crossover_rate = params.get('crossover_rate', 0.8)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.elite_size = params.get('elite_size', 5)
        self.max_genome_length = params.get('max_genome_length', 20)

        # Selection
        selection_config = self.config.get('selection', {})
        self.selection_method = selection_config.get('method', 'tournament')
        self.tournament_size = selection_config.get('tournament_size', 3)

        # Crossover
        crossover_config = self.config.get('crossover', {})
        self.crossover_method = crossover_config.get('method', 'single_point')
        self.max_crossover_points = crossover_config.get('max_points', 3)

        # Mutation
        mutation_config = self.config.get('mutation', {})
        self.mutation_method = mutation_config.get('method', 'point')
        self.mutation_strength = mutation_config.get('strength', 0.5)

        # Diversity
        diversity_config = self.config.get('diversity', {})
        self.maintain_diversity = diversity_config.get('enabled', True)
        self.diversity_threshold = diversity_config.get('threshold', 0.1)

    def initialize_population(self, initial_state: State) -> None:
        """Initialize the genetic algorithm population."""
        self.population = []
        self.generation = 0

        for _ in range(self.population_size):
            # Create random individual
            genome_length = random.randint(1, min(10, self.max_genome_length))
            genome = self._generate_random_genome(genome_length)
            individual_state = self._apply_genome_to_state(initial_state, genome)
            individual = Individual(genome, individual_state)
            self.population.append(individual)

        # Sort by fitness
        self.population.sort(reverse=True)
        self.best_individual = self.population[0] if self.population else None

    def _generate_random_genome(self, length: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate a random genome of specified length."""
        genome = []
        all_operations = self.operations_registry.list_operations()

        for _ in range(length):
            operation_name = random.choice(all_operations)
            params = self._generate_operation_params(operation_name)
            if params is not None:
                genome.append((operation_name, params))

        return genome

    def _generate_operation_params(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Generate random parameters for an operation."""
        try:
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = {}

            # Generate parameters based on metadata
            required_params = metadata.get('required_params', [])
            optional_params = metadata.get('optional_params', {})

            # Handle required parameters
            for param in required_params:
                if param == 'constant':
                    params[param] = random.randint(1, 255)
                elif param == 'shift':
                    params[param] = random.randint(1, 7)
                elif param == 'positions':
                    params[param] = random.sample(range(256), random.randint(1, min(10, 256)))
                elif param == 'value':
                    params[param] = random.randint(0, 255)
                elif param == 'seed':
                    params[param] = random.randint(0, 10000)
                else:
                    # Skip if we don't know how to generate this parameter
                    return None

            # Handle some optional parameters
            for param, param_type in optional_params.items():
                if random.random() < 0.3:  # 30% chance to include optional param
                    if param == 'count':
                        params[param] = random.randint(1, 10)
                    elif param == 'step':
                        params[param] = random.randint(1, 4)

            return params if params else {}

        except Exception:
            return None

    def _apply_genome_to_state(self, initial_state: State, genome: List[Tuple[str, Dict[str, Any]]]) -> State:
        """Apply a genome to a state to get the resulting state."""
        current_state = initial_state.copy()

        for operation_name, params in genome:
            try:
                operation_fn = self.operations_registry.get_operation(operation_name)
                new_data, inverse_fn, metadata = operation_fn(current_state.binary_data, **params)

                current_state = State(
                    binary_data=new_data,
                    parent_state_id=current_state.state_id,
                    operation_applied={
                        'operation': operation_name,
                        'params': params,
                        'cost': metadata.get('cost', 1.0)
                    },
                    operation_history=current_state.operation_history + [{
                        'operation': operation_name,
                        'params': params,
                        'cost': metadata.get('cost', 1.0)
                    }],
                    inverse_operations=current_state.inverse_operations + [inverse_fn],
                    generation=current_state.generation + 1
                )

            except Exception:
                # Operation failed, skip it
                continue

        return current_state

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using genetic algorithm."""
        # Initialize population if needed
        if not self.population:
            self.initialize_population(current_state)

        # Evolve population for one generation
        self.evolve_population(current_state)

        # Return the first operation from the best individual's genome
        if self.best_individual and self.best_individual.genome:
            return self.best_individual.genome[0]
        else:
            # Fallback to random operation
            return self._get_random_operation()

    def evolve_population(self, initial_state: State) -> None:
        """Evolve the population for one generation."""
        if not self.population:
            return

        # Selection
        selected_parents = self._select_parents()

        # Crossover
        offspring = self._crossover_parents(selected_parents)

        # Mutation
        offspring = self._mutate_individuals(offspring)

        # Create new population with elitism
        self.population = self._create_new_population(offspring)

        # Update generation and best individual
        self.generation += 1
        if self.population:
            self.best_individual = max(self.population, key=lambda ind: ind.fitness)

    def _select_parents(self) -> List[Individual]:
        """Select parents for reproduction."""
        if self.selection_method == 'tournament':
            return self._tournament_selection()
        elif self.selection_method == 'roulette_wheel':
            return self._roulette_wheel_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            return self._tournament_selection()

    def _tournament_selection(self) -> List[Individual]:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population,
                                     min(self.tournament_size, len(self.population)))
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner.copy())
        return selected

    def _roulette_wheel_selection(self) -> List[Individual]:
        """Roulette wheel selection."""
        selected = []
        total_fitness = sum(max(0, ind.fitness) for ind in self.population)

        if total_fitness == 0:
            # All fitnesses are 0 or negative, use uniform selection
            return [random.choice(self.population).copy() for _ in range(self.population_size)]

        for _ in range(self.population_size):
            r = random.uniform(0, total_fitness)
            cumulative = 0
            for ind in self.population:
                cumulative += max(0, ind.fitness)
                if cumulative >= r:
                    selected.append(ind.copy())
                    break

        return selected

    def _rank_selection(self) -> List[Individual]:
        """Rank-based selection."""
        # Sort by fitness
        sorted_pop = sorted(self.population, reverse=True)
        selected = []

        # Probability proportional to rank
        total_ranks = sum(range(1, len(sorted_pop) + 1))
        for _ in range(self.population_size):
            r = random.uniform(0, total_ranks)
            cumulative = 0
            for rank, ind in enumerate(sorted_pop, 1):
                cumulative += rank
                if cumulative >= r:
                    selected.append(ind.copy())
                    break

        return selected

    def _crossover_parents(self, parents: List[Individual]) -> List[Individual]:
        """Perform crossover on selected parents."""
        offspring = []

        # Pair up parents for crossover
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            if random.random() < self.crossover_rate:
                # Perform crossover
                child1, child2 = self._crossover_individuals(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                # No crossover, parents pass through unchanged
                offspring.extend([parent1.copy(), parent2.copy()])

        # Handle odd number of parents
        if len(parents) % 2 == 1:
            offspring.append(parents[-1].copy())

        return offspring

    def _crossover_individuals(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover two individuals to create offspring."""
        if self.crossover_method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        else:
            return self._single_point_crossover(parent1, parent2)

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        genome1 = parent1.genome
        genome2 = parent2.genome

        # Find crossover point
        max_point = min(len(genome1), len(genome2))
        if max_point == 0:
            return parent1.copy(), parent2.copy()

        crossover_point = random.randint(1, max_point)

        # Create offspring genomes
        child1_genome = genome1[:crossover_point] + genome2[crossover_point:]
        child2_genome = genome2[:crossover_point] + genome1[crossover_point:]

        # Create offspring individuals (state will be evaluated later)
        child1 = Individual(child1_genome, parent1.state.copy())
        child2 = Individual(child2_genome, parent2.state.copy())

        return child1, child2

    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        genome1 = parent1.genome
        genome2 = parent2.genome

        max_point = min(len(genome1), len(genome2))
        if max_point < 2:
            return self._single_point_crossover(parent1, parent2)

        point1 = random.randint(1, max_point - 1)
        point2 = random.randint(point1, max_point)

        # Create offspring genomes
        child1_genome = (genome1[:point1] + genome2[point1:point2] +
                        genome1[point2:])
        child2_genome = (genome2[:point1] + genome1[point1:point2] +
                        genome2[point2:])

        child1 = Individual(child1_genome, parent1.state.copy())
        child2 = Individual(child2_genome, parent2.state.copy())

        return child1, child2

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        genome1 = parent1.genome
        genome2 = parent2.genome

        child1_genome = []
        child2_genome = []

        max_len = max(len(genome1), len(genome2))
        for i in range(max_len):
            gene1 = genome1[i] if i < len(genome1) else None
            gene2 = genome2[i] if i < len(genome2) else None

            if gene1 is None:
                child1_genome.append(gene2)
                child2_genome.append(gene2)
            elif gene2 is None:
                child1_genome.append(gene1)
                child2_genome.append(gene1)
            else:
                if random.random() < 0.5:
                    child1_genome.append(gene1)
                    child2_genome.append(gene2)
                else:
                    child1_genome.append(gene2)
                    child2_genome.append(gene1)

        child1 = Individual(child1_genome, parent1.state.copy())
        child2 = Individual(child2_genome, parent2.state.copy())

        return child1, child2

    def _mutate_individuals(self, individuals: List[Individual]) -> List[Individual]:
        """Apply mutation to individuals."""
        for individual in individuals:
            if random.random() < self.mutation_rate:
                self._mutate_individual(individual)
        return individuals

    def _mutate_individual(self, individual: Individual) -> None:
        """Mutate a single individual."""
        if self.mutation_method == 'point':
            self._point_mutation(individual)
        elif self.mutation_method == 'insertion':
            self._insertion_mutation(individual)
        elif self.mutation_method == 'deletion':
            self._deletion_mutation(individual)
        elif self.mutation_method == 'swap':
            self._swap_mutation(individual)

    def _point_mutation(self, individual: Individual) -> None:
        """Point mutation: replace a random gene."""
        if not individual.genome:
            # Add a random gene
            individual.genome = self._generate_random_genome(1)
            return

        # Select random gene to mutate
        gene_index = random.randint(0, len(individual.genome) - 1)
        individual.genome[gene_index] = self._generate_random_gene()

    def _insertion_mutation(self, individual: Individual) -> None:
        """Insertion mutation: add a new gene."""
        if len(individual.genome) >= self.max_genome_length:
            return

        new_gene = self._generate_random_gene()
        insert_pos = random.randint(0, len(individual.genome))
        individual.genome.insert(insert_pos, new_gene)

    def _deletion_mutation(self, individual: Individual) -> None:
        """Deletion mutation: remove a random gene."""
        if len(individual.genome) > 1:
            delete_pos = random.randint(0, len(individual.genome) - 1)
            del individual.genome[delete_pos]

    def _swap_mutation(self, individual: Individual) -> None:
        """Swap mutation: swap two genes."""
        if len(individual.genome) >= 2:
            pos1, pos2 = random.sample(range(len(individual.genome)), 2)
            individual.genome[pos1], individual.genome[pos2] = individual.genome[pos2], individual.genome[pos1]

    def _generate_random_gene(self) -> Tuple[str, Dict[str, Any]]:
        """Generate a random gene (operation with parameters)."""
        all_operations = self.operations_registry.list_operations()
        operation_name = random.choice(all_operations)
        params = self._generate_operation_params(operation_name)
        return (operation_name, params or {})

    def _create_new_population(self, offspring: List[Individual]) -> List[Individual]:
        """Create new population with elitism."""
        # Evaluate offspring fitness
        for individual in offspring:
            individual.age += 1

        # Sort current population by fitness
        self.population.sort(reverse=True)

        # Keep elite individuals
        elite = self.population[:self.elite_size]

        # Sort offspring by fitness
        offspring.sort(reverse=True)

        # Fill remaining slots with best offspring
        remaining_slots = self.population_size - len(elite)
        new_population = elite + offspring[:remaining_slots]

        # Ensure we don't exceed population size
        return new_population[:self.population_size]

    def _get_random_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Get a random operation as fallback."""
        all_operations = self.operations_registry.list_operations()
        operation_name = random.choice(all_operations)
        params = self._generate_operation_params(operation_name)
        return (operation_name, params or {})

    def accept(self, new_state: State) -> bool:
        """Accept based on fitness improvement."""
        return new_state.score > self.best_score

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.population = []
        self.generation = 0
        self.best_individual = None

    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about the population."""
        if not self.population:
            return {'population_size': 0}

        fitnesses = [ind.fitness for ind in self.population]
        genome_lengths = [len(ind.genome) for ind in self.population]

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'worst_fitness': min(fitnesses) if fitnesses else 0,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            'avg_genome_length': sum(genome_lengths) / len(genome_lengths) if genome_lengths else 0,
            'diversity': self._calculate_diversity()
        }

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0

        # Simple diversity measure: average fitness difference
        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        diversity = sum(abs(f - avg_fitness) for f in fitnesses) / len(fitnesses)
        return diversity