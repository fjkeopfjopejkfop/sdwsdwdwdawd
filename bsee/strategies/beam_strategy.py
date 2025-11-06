"""
Beam search strategy for BSEE.
"""

import random
import yaml
import os
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State
from bsee.operations.operations_registry import OperationsRegistry


class BeamCandidate:
    """Candidate in the beam search."""

    def __init__(self, state: State, operation: Optional[Tuple[str, Dict[str, Any]]] = None,
                 parent: Optional['BeamCandidate'] = None):
        """Initialize a beam candidate."""
        self.state = state
        self.operation = operation  # Operation that led to this state
        self.parent = parent
        self.score = state.score if hasattr(state, 'score') else 0.0
        self.depth = state.generation if hasattr(state, 'generation') else 0

    def __lt__(self, other: 'BeamCandidate') -> bool:
        """Less than comparison for sorting by score."""
        return self.score < other.score


class BeamStrategy(BaseStrategy):
    """Beam search strategy with real implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize beam strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = OperationsRegistry()
        self.beam: List[BeamCandidate] = []
        self.current_candidate: Optional[BeamCandidate] = None
        self.beam_initialized = False

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self.config.get('config_file', 'config/strategies/strategy_beam.yaml')

        # Try to load from file, otherwise use passed config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge file config with passed config
                self.config = {**file_config, **self.config}
            except Exception as e:
                print(f"Warning: Could not load Beam config from {config_path}: {e}")

        # Extract parameters from config
        params = self.config.get('parameters', {})
        self.beam_width = params.get('beam_width', 10)
        self.expansion_factor = params.get('expansion_factor', 5)
        self.max_depth = params.get('max_depth', 20)

        # Beam management
        beam_mgmt = self.config.get('beam_management', {})
        self.diversify_beam = beam_mgmt.get('diversify', True)
        self.diversity_threshold = beam_mgmt.get('diversity_threshold', 0.1)
        self.prune_duplicates = beam_mgmt.get('prune_duplicates', True)

        # Expansion
        expansion_config = self.config.get('expansion', {})
        self.expansion_strategy = expansion_config.get('strategy', 'best_first')
        self.random_expansion_chance = expansion_config.get('random_chance', 0.1)

        # Selection
        selection_config = self.config.get('selection', {})
        self.final_selection = selection_config.get('method', 'best_score')
        self.consider_depth = selection_config.get('consider_depth', True)
        self.depth_penalty = selection_config.get('depth_penalty', 0.01)

    def initialize_beam(self, initial_state: State) -> None:
        """Initialize the beam with the initial state."""
        self.beam = [BeamCandidate(initial_state)]
        self.current_candidate = self.beam[0]
        self.beam_initialized = True

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using beam search."""
        # Initialize beam if needed
        if not self.beam_initialized:
            self.initialize_beam(current_state)

        # Expand beam with new candidates
        expanded_candidates = self._expand_beam()

        # Prune beam to maintain beam width
        self._prune_beam(expanded_candidates)

        # Select best operation from current best candidate
        best_operation = self._select_best_operation()

        # Update current candidate
        if best_operation and self.current_candidate:
            # Find the child that corresponds to this operation
            for candidate in self.beam:
                if (candidate.operation == best_operation and
                    candidate.parent == self.current_candidate):
                    self.current_candidate = candidate
                    break

        return best_operation if best_operation else self._get_random_operation()

    def _expand_beam(self) -> List[BeamCandidate]:
        """Expand each beam candidate with operations."""
        new_candidates = []

        for candidate in self.beam:
            if candidate.depth >= self.max_depth:
                continue  # Don't expand if max depth reached

            # Generate expansion candidates
            expansion_candidates = self._generate_expansion_candidates(candidate)
            new_candidates.extend(expansion_candidates)

        return new_candidates

    def _generate_expansion_candidates(self, parent: BeamCandidate) -> List[BeamCandidate]:
        """Generate expansion candidates from a parent candidate."""
        candidates = []
        all_operations = self.operations_registry.list_operations()

        # Determine how many operations to try
        num_operations = min(len(all_operations), self.expansion_factor)

        if self.expansion_strategy == 'best_first':
            # Try best operations based on heuristics
            selected_operations = self._select_best_operations(all_operations, num_operations)
        elif self.expansion_strategy == 'random':
            # Random selection
            selected_operations = random.sample(all_operations, num_operations)
        elif self.expansion_strategy == 'diverse':
            # Select diverse operations
            selected_operations = self._select_diverse_operations(all_operations, num_operations)
        else:
            # Default to random
            selected_operations = random.sample(all_operations, num_operations)

        # Apply each selected operation
        for operation_name in selected_operations:
            # Sometimes add random exploration
            if random.random() < self.random_expansion_chance:
                operation_name = random.choice(all_operations)

            params = self._generate_operation_params(operation_name)
            if params is None:
                continue

            try:
                # Apply operation
                operation_fn = self.operations_registry.get_operation(operation_name)
                new_data, inverse_fn, metadata = operation_fn(parent.state.binary_data, **params)

                # Create new state
                new_state = State(
                    binary_data=new_data,
                    parent_state_id=parent.state.state_id,
                    operation_applied={
                        'operation': operation_name,
                        'params': params,
                        'cost': metadata.get('cost', 1.0)
                    },
                    operation_history=parent.state.operation_history + [{
                        'operation': operation_name,
                        'params': params,
                        'cost': metadata.get('cost', 1.0)
                    }],
                    inverse_operations=parent.state.inverse_operations + [inverse_fn],
                    generation=parent.state.generation + 1
                )

                # Create candidate
                candidate = BeamCandidate(
                    state=new_state,
                    operation=(operation_name, params),
                    parent=parent
                )
                candidates.append(candidate)

            except Exception:
                # Operation failed, skip it
                continue

        return candidates

    def _select_best_operations(self, all_operations: List[str], count: int) -> List[str]:
        """Select best operations based on heuristics."""
        # Simple heuristic: prefer operations that have worked well in the past
        # For now, just return a mix of different operation categories
        categories = self.operations_registry.get_operation_categories()
        selected = []

        for category in categories:
            category_ops = self.operations_registry.get_operations_by_category(category)
            if category_ops and len(selected) < count:
                # Add one operation from each category
                selected.append(random.choice(list(category_ops.keys())))

        # Fill remaining slots randomly
        while len(selected) < count:
            selected.append(random.choice(all_operations))

        return selected[:count]

    def _select_diverse_operations(self, all_operations: List[str], count: int) -> List[str]:
        """Select diverse operations to maintain beam diversity."""
        # Simple diversity: ensure we pick operations from different categories
        categories = self.operations_registry.get_operation_categories()
        selected = set()

        # First, try to get operations from different categories
        for category in categories:
            if len(selected) >= count:
                break
            category_ops = self.operations_registry.get_operations_by_category(category)
            if category_ops:
                op_name = random.choice(list(category_ops.keys()))
                selected.add(op_name)

        # Fill remaining slots randomly
        while len(selected) < count:
            selected.add(random.choice(all_operations))

        return list(selected)

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

    def _prune_beam(self, new_candidates: List[BeamCandidate]) -> None:
        """Prune beam to maintain beam width."""
        # Combine current beam and new candidates
        all_candidates = self.beam + new_candidates

        # Remove duplicates if enabled
        if self.prune_duplicates:
            all_candidates = self._remove_duplicates(all_candidates)

        # Add depth penalty if considering depth
        if self.consider_depth:
            for candidate in all_candidates:
                depth_penalty = candidate.depth * self.depth_penalty
                candidate.score -= depth_penalty

        # Sort by score (descending)
        all_candidates.sort(reverse=True)

        # Select top candidates
        self.beam = all_candidates[:self.beam_width]

        # Apply diversity maintenance if enabled
        if self.diversify_beam and len(self.beam) > 1:
            self.beam = self._maintain_diversity(self.beam)

        # Update current candidate to the best one
        if self.beam:
            self.current_candidate = self.beam[0]

    def _remove_duplicates(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
        """Remove duplicate candidates based on state."""
        seen_states = set()
        unique_candidates = []

        for candidate in candidates:
            state_hash = candidate.state.state_id
            if state_hash not in seen_states:
                seen_states.add(state_hash)
                unique_candidates.append(candidate)

        return unique_candidates

    def _maintain_diversity(self, candidates: List[BeamCandidate]) -> List[BeamCandidate]:
        """Maintain diversity in the beam."""
        if len(candidates) <= 2:
            return candidates

        # Simple diversity: ensure we have candidates with different operations
        diverse_candidates = [candidates[0]]  # Always keep the best one
        used_operations = {candidates[0].operation[0] if candidates[0].operation else None}

        for candidate in candidates[1:]:
            op_name = candidate.operation[0] if candidate.operation else None
            if op_name not in used_operations:
                diverse_candidates.append(candidate)
                used_operations.add(op_name)

            if len(diverse_candidates) >= self.beam_width:
                break

        # Fill remaining slots if needed
        if len(diverse_candidates) < self.beam_width:
            for candidate in candidates:
                if candidate not in diverse_candidates:
                    diverse_candidates.append(candidate)
                    if len(diverse_candidates) >= self.beam_width:
                        break

        return diverse_candidates[:self.beam_width]

    def _select_best_operation(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Select the best operation from the current beam."""
        if not self.beam:
            return None

        if self.final_selection == 'best_score':
            best_candidate = max(self.beam, key=lambda c: c.score)
        elif self.final_selection == 'shallow':
            # Prefer shallower candidates
            best_candidate = min(self.beam, key=lambda c: c.depth)
        elif self.final_selection == 'balanced':
            # Balance between score and depth
            best_candidate = max(self.beam,
                               key=lambda c: c.score - c.depth * self.depth_penalty)
        else:
            # Default to best score
            best_candidate = max(self.beam, key=lambda c: c.score)

        return best_candidate.operation if best_candidate else None

    def _get_random_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Get a random operation as fallback."""
        all_operations = self.operations_registry.list_operations()
        operation_name = random.choice(all_operations)
        params = self._generate_operation_params(operation_name)
        return (operation_name, params or {})

    def accept(self, new_state: State) -> bool:
        """Accept based on beam evaluation."""
        # In beam search, we accept if the new state improves the score
        # or if it adds diversity to the beam
        if new_state.score > self.best_score:
            return True

        # Small probability of accepting to maintain diversity
        if self.diversify_beam and random.random() < 0.05:
            return True

        return False

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.beam = []
        self.current_candidate = None
        self.beam_initialized = False

    def get_beam_stats(self) -> Dict[str, Any]:
        """Get statistics about the beam."""
        if not self.beam:
            return {'beam_size': 0}

        scores = [c.score for c in self.beam]
        depths = [c.depth for c in self.beam]

        return {
            'beam_size': len(self.beam),
            'best_score': max(scores) if scores else 0,
            'worst_score': min(scores) if scores else 0,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'max_depth': max(depths) if depths else 0,
            'min_depth': min(depths) if depths else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'diversity': self._calculate_beam_diversity()
        }

    def _calculate_beam_diversity(self) -> float:
        """Calculate diversity in the current beam."""
        if len(self.beam) < 2:
            return 0.0

        # Simple diversity measure: score variance
        scores = [c.score for c in self.beam]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        return variance