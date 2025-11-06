"""
Simulated annealing strategy for BSEE.
"""

import random
import math
import yaml
import os
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State
from bsee.operations.operations_registry import OperationsRegistry


class AnnealingStrategy(BaseStrategy):
    """Simulated annealing strategy with proper implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize annealing strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = OperationsRegistry()
        self.current_temperature = self.initial_temperature
        self.last_state: Optional[State] = None
        self.moves_since_temperature_change = 0

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self.config.get('config_file', 'config/strategies/strategy_annealing.yaml')

        # Try to load from file, otherwise use passed config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge file config with passed config
                self.config = {**file_config, **self.config}
            except Exception as e:
                print(f"Warning: Could not load Annealing config from {config_path}: {e}")

        # Extract parameters from config
        params = self.config.get('parameters', {})
        self.initial_temperature = params.get('initial_temperature', 100.0)
        self.cooling_rate = params.get('cooling_rate', 0.95)
        self.min_temperature = params.get('min_temperature', 0.01)

        # Temperature schedule
        temp_schedule = self.config.get('temperature_schedule', {})
        self.schedule_type = temp_schedule.get('type', 'exponential')
        self.linear_cooling_step = temp_schedule.get('linear_step', 0.1)
        self.logarithmic_cooling_constant = temp_schedule.get('logarithmic_constant', 1.0)

        # Move generation
        move_config = self.config.get('move_generation', {})
        self.move_strategy = move_config.get('strategy', 'temperature_adaptive')
        self.neighborhood_size = move_config.get('neighborhood_size', 10)
        self.max_perturbation = move_config.get('max_perturbation', 50)

        # Acceptance criteria
        acceptance_config = self.config.get('acceptance', {})
        self.acceptance_criterion = acceptance_config.get('criterion', 'metropolis')
        self.use_adaptive_acceptance = acceptance_config.get('adaptive', False)

        # Reheating
        reheating_config = self.config.get('reheating', {})
        self.enable_reheating = reheating_config.get('enabled', False)
        self.reheating_threshold = reheating_config.get('threshold', 0.01)
        self.reheating_factor = reheating_config.get('factor', 2.0)

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using simulated annealing with temperature-dependent moves."""
        # Store current state for acceptance decision
        self.last_state = current_state

        # Generate move based on current temperature
        if self.move_strategy == 'temperature_adaptive':
            operation = self._generate_temperature_adaptive_move(current_state)
        elif self.move_strategy == 'neighborhood':
            operation = self._generate_neighborhood_move(current_state)
        elif self.move_strategy == 'random':
            operation = self._generate_random_move()
        else:
            operation = self._generate_temperature_adaptive_move(current_state)

        return operation

    def _generate_temperature_adaptive_move(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Generate move with perturbation based on temperature."""
        # Higher temperature = larger perturbations
        # Lower temperature = smaller, more local moves

        # Calculate perturbation scale based on temperature
        temperature_ratio = self.current_temperature / self.initial_temperature
        perturbation_scale = max(1, int(self.max_perturbation * temperature_ratio))

        # Select operation type based on temperature
        if temperature_ratio > 0.7:
            # High temperature: prefer disruptive operations
            operation_name = self._select_disruptive_operation()
        elif temperature_ratio > 0.3:
            # Medium temperature: balanced operations
            operation_name = self._select_balanced_operation()
        else:
            # Low temperature: prefer local operations
            operation_name = self._select_local_operation()

        # Generate parameters with appropriate perturbation
        params = self._generate_temperature_aware_params(operation_name, perturbation_scale)
        return (operation_name, params)

    def _generate_neighborhood_move(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Generate move from local neighborhood."""
        # Generate multiple candidate operations
        candidates = []
        all_operations = self.operations_registry.list_operations()

        # Sample neighborhood operations
        neighborhood_ops = random.sample(
            all_operations,
            min(self.neighborhood_size, len(all_operations))
        )

        for op_name in neighborhood_ops:
            params = self._generate_operation_params(op_name)
            if params is not None:
                # Score this operation based on how "local" it is
                local_score = self._calculate_locality_score(op_name, params)
                candidates.append((op_name, params, local_score))

        if candidates:
            # Select from neighborhood, weighted by locality
            candidates.sort(key=lambda x: x[2], reverse=True)
            # Use temperature to control selection from top candidates
            temp_ratio = self.current_temperature / self.initial_temperature
            selection_range = max(1, min(len(candidates), int(temp_ratio * len(candidates))))
            selected = random.choice(candidates[:selection_range])
            return (selected[0], selected[1])

        # Fallback to random move
        return self._generate_random_move()

    def _generate_random_move(self) -> Tuple[str, Dict[str, Any]]:
        """Generate completely random move."""
        all_operations = self.operations_registry.list_operations()
        operation_name = random.choice(all_operations)
        params = self._generate_operation_params(operation_name)
        return (operation_name, params or {})

    def _select_disruptive_operation(self) -> str:
        """Select operation that causes larger changes."""
        # Operations that tend to be more disruptive
        disruptive_ops = [
            'shuffle_bytes', 'reverse_bytes', 'rotate_bits',
            'swap_bytes', 'move_to_front', 'invert_bits'
        ]

        # Get available operations
        all_ops = self.operations_registry.list_operations()
        available_disruptive = [op for op in disruptive_ops if op in all_ops]

        if available_disruptive:
            return random.choice(available_disruptive)
        else:
            # Fallback to any operation
            return random.choice(all_ops)

    def _select_balanced_operation(self) -> str:
        """Select operation with balanced impact."""
        balanced_ops = [
            'xor_constant', 'add_constant', 'subtract_constant',
            'rotate_left', 'rotate_right', 'swap_pairs'
        ]

        all_ops = self.operations_registry.list_operations()
        available_balanced = [op for op in balanced_ops if op in all_ops]

        if available_balanced:
            return random.choice(available_balanced)
        else:
            return random.choice(all_ops)

    def _select_local_operation(self) -> str:
        """Select operation that causes small, local changes."""
        local_ops = [
            'flip_bit', 'increment_byte', 'decrement_byte',
            'swap_adjacent', 'nibble_swap'
        ]

        all_ops = self.operations_registry.list_operations()
        available_local = [op for op in local_ops if op in all_ops]

        if available_local:
            return random.choice(available_local)
        else:
            return random.choice(all_ops)

    def _generate_temperature_aware_params(self, operation_name: str, perturbation_scale: int) -> Dict[str, Any]:
        """Generate parameters with temperature-based perturbation."""
        try:
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = {}

            # Generate parameters based on metadata and perturbation scale
            required_params = metadata.get('required_params', [])
            optional_params = metadata.get('optional_params', {})

            # Handle required parameters with temperature awareness
            for param in required_params:
                if param == 'constant':
                    # Larger constants at high temperature, smaller at low temperature
                    max_const = min(255, 1 + perturbation_scale)
                    params[param] = random.randint(1, max_const)
                elif param == 'shift':
                    max_shift = min(7, 1 + perturbation_scale // 10)
                    params[param] = random.randint(1, max_shift)
                elif param == 'positions':
                    max_positions = min(256, perturbation_scale)
                    num_positions = min(10, max(1, perturbation_scale // 5))
                    params[param] = random.sample(range(max_positions), num_positions)
                elif param == 'value':
                    max_val = min(255, perturbation_scale)
                    params[param] = random.randint(0, max_val)
                elif param == 'seed':
                    params[param] = random.randint(0, 10000)
                else:
                    # Skip if we don't know how to generate this parameter
                    return None

            # Handle optional parameters
            for param, param_type in optional_params.items():
                if random.random() < 0.3:  # 30% chance to include optional param
                    if param == 'count':
                        params[param] = min(10, max(1, perturbation_scale // 2))
                    elif param == 'step':
                        params[param] = min(4, max(1, perturbation_scale // 10))

            return params if params else {}

        except Exception:
            return None

    def _calculate_locality_score(self, operation_name: str, params: Dict[str, Any]) -> float:
        """Calculate how "local" an operation is."""
        # Higher score for more local operations
        local_operations = {
            'flip_bit': 1.0, 'increment_byte': 0.9, 'decrement_byte': 0.9,
            'swap_adjacent': 0.8, 'nibble_swap': 0.7
        }

        base_score = local_operations.get(operation_name, 0.5)

        # Adjust score based on parameters
        if 'constant' in params:
            # Smaller constants are more local
            const_size = params['constant']
            const_score = max(0, 1.0 - const_size / 255.0)
            base_score = base_score * 0.7 + const_score * 0.3

        if 'shift' in params:
            # Smaller shifts are more local
            shift_size = params['shift']
            shift_score = max(0, 1.0 - shift_size / 8.0)
            base_score = base_score * 0.7 + shift_score * 0.3

        return base_score

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

    def accept(self, new_state: State) -> bool:
        """Accept based on simulated annealing criteria."""
        if not self.last_state:
            self.last_state = new_state
            return True

        delta_score = new_state.score - self.last_state.score

        # Always accept better solutions
        if delta_score > 0:
            self._update_temperature(accepted=True, improved=True)
            return True

        # Accept worse solutions based on temperature
        if self.current_temperature > self.min_temperature:
            if self.acceptance_criterion == 'metropolis':
                probability = math.exp(delta_score / self.current_temperature)
            elif self.acceptance_criterion == 'threshold':
                probability = 1.0 / (1.0 + math.exp(-delta_score / self.current_temperature))
            else:
                probability = math.exp(delta_score / self.current_temperature)

            # Adaptive acceptance
            if self.use_adaptive_acceptance:
                # Adjust probability based on recent acceptance history
                acceptance_rate = self._get_recent_acceptance_rate()
                if acceptance_rate < 0.2:  # Too few accepts, increase probability
                    probability = min(1.0, probability * 1.2)
                elif acceptance_rate > 0.8:  # Too many accepts, decrease probability
                    probability = max(0.0, probability * 0.8)

            if random.random() < probability:
                self._update_temperature(accepted=True, improved=False)
                return True

        # Reject and cool down
        self._update_temperature(accepted=False, improved=False)

        # Check for reheating
        if self.enable_reheating and self._should_reheat():
            self._reheat()

        return False

    def _update_temperature(self, accepted: bool, improved: bool) -> None:
        """Update temperature based on cooling schedule."""
        self.moves_since_temperature_change += 1

        # Update temperature every move (or could be every N moves)
        if self.schedule_type == 'exponential':
            if not accepted and not improved:
                self.current_temperature *= self.cooling_rate
        elif self.schedule_type == 'linear':
            self.current_temperature = max(
                self.min_temperature,
                self.current_temperature - self.linear_cooling_step
            )
        elif self.schedule_type == 'logarithmic':
            if self.iteration_count > 0:
                self.current_temperature = max(
                    self.min_temperature,
                    self.initial_temperature / (1 + self.logarithmic_cooling_constant * math.log(self.iteration_count + 1))
                )
        elif self.schedule_type == 'adaptive':
            # Adaptive cooling based on acceptance rate
            acceptance_rate = self._get_recent_acceptance_rate()
            if acceptance_rate > 0.6:  # Accepting too much, cool faster
                self.current_temperature *= self.cooling_rate * 0.9
            elif acceptance_rate < 0.2:  # Accepting too little, cool slower
                self.current_temperature *= self.cooling_rate * 1.1
            else:
                self.current_temperature *= self.cooling_rate

        # Ensure temperature doesn't go below minimum
        self.current_temperature = max(self.min_temperature, self.current_temperature)

    def _get_recent_acceptance_rate(self) -> float:
        """Calculate recent acceptance rate."""
        # This is a simplified version - in practice, you'd track recent moves
        return 0.5  # Placeholder

    def _should_reheat(self) -> bool:
        """Check if reheating should be triggered."""
        # Reheat if temperature is too low and we're not finding improvements
        return (self.current_temperature < self.min_temperature * 2 and
                self.no_improvement_count > 50)

    def _reheat(self) -> None:
        """Reheat the system."""
        self.current_temperature = min(
            self.initial_temperature,
            self.current_temperature * self.reheating_factor
        )

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.current_temperature = self.initial_temperature
        self.last_state = None
        self.moves_since_temperature_change = 0

    def get_annealing_stats(self) -> Dict[str, Any]:
        """Get statistics about the annealing process."""
        return {
            'current_temperature': self.current_temperature,
            'initial_temperature': self.initial_temperature,
            'temperature_ratio': self.current_temperature / self.initial_temperature,
            'moves_since_change': self.moves_since_temperature_change,
            'is_frozen': self.current_temperature <= self.min_temperature
        }