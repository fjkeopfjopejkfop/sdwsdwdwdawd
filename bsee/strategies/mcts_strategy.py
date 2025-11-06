"""
Monte Carlo Tree Search strategy for BSEE.
"""

import math
import random
import yaml
import os
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State
from bsee.operations.operations_registry import OperationsRegistry


class TreeNode:
    """Node in the MCTS search tree."""

    def __init__(self, state: State, parent: Optional['TreeNode'] = None,
                 operation: Optional[Tuple[str, Dict[str, Any]]] = None):
        """Initialize a tree node."""
        self.state = state
        self.parent = parent
        self.operation = operation  # (operation_name, params) that led to this node
        self.children: List['TreeNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_operations: List[Tuple[str, Dict[str, Any]]] = []
        self.is_fully_expanded = False

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def has_untried_operations(self) -> bool:
        """Check if there are untried operations from this node."""
        return len(self.untried_operations) > 0

    def get_best_child(self, exploration_constant: float = 1.4) -> 'TreeNode':
        """Get the best child using UCT (Upper Confidence Bound for Trees)."""
        best_child = None
        best_uct = float('-inf')

        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def add_child(self, child_state: State, operation: Tuple[str, Dict[str, Any]]) -> 'TreeNode':
        """Add a child node."""
        child = TreeNode(child_state, parent=self, operation=operation)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.value += reward


class MCTSStrategy(BaseStrategy):
    """Monte Carlo Tree Search strategy with real implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MCTS strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = OperationsRegistry()
        self.root: Optional[TreeNode] = None
        self.current_node: Optional[TreeNode] = None

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = self.config.get('config_file', 'config/strategies/strategy_mcts.yaml')

        # Try to load from file, otherwise use passed config
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                # Merge file config with passed config
                self.config = {**file_config, **self.config}
            except Exception as e:
                print(f"Warning: Could not load MCTS config from {config_path}: {e}")

        # Extract parameters from config
        params = self.config.get('parameters', {})
        self.exploration_constant = params.get('exploration_constant', 1.4)
        self.simulation_count = params.get('simulation_count', 100)
        self.rollout_depth = params.get('rollout_depth', 10)
        self.max_tree_depth = params.get('max_tree_depth', 50)

        # Tree management
        tree_mgmt = self.config.get('tree_management', {})
        self.max_children = tree_mgmt.get('max_children', 20)
        self.prune_tree = tree_mgmt.get('prune_tree', True)
        self.prune_threshold = tree_mgmt.get('prune_threshold', 0.01)

        # Simulation
        sim_config = self.config.get('simulation', {})
        self.simulation_policy = sim_config.get('policy', 'random')
        self.early_termination = sim_config.get('early_termination', True)
        self.early_termination_threshold = sim_config.get('early_termination_threshold', 0.8)

        # Selection
        selection_config = self.config.get('selection', {})
        self.final_selection = selection_config.get('final_selection', 'most_visited')

        # Memory management
        memory_config = self.config.get('memory', {})
        self.max_tree_nodes = memory_config.get('max_tree_nodes', 10000)

    def initialize_tree(self, initial_state: State) -> None:
        """Initialize the MCTS tree with the initial state."""
        self.root = TreeNode(initial_state)
        self.current_node = self.root
        self._initialize_untried_operations(self.root)

    def _initialize_untried_operations(self, node: TreeNode) -> None:
        """Initialize the list of untried operations for a node."""
        # Get a sample of operations to try
        all_operations = self.operations_registry.list_operations()

        # Limit operations to keep tree manageable
        max_ops = min(len(all_operations), self.max_children)
        selected_ops = random.sample(all_operations, max_ops)

        for op_name in selected_ops:
            # Generate random parameters for this operation
            params = self._generate_operation_params(op_name)
            if params is not None:  # Valid parameters generated
                node.untried_operations.append((op_name, params))

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

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using MCTS."""
        # Initialize tree if needed
        if self.root is None or self.current_node is None:
            self.initialize_tree(current_state)

        # Run MCTS iterations
        for _ in range(self.simulation_count):
            # Selection phase
            selected_node = self._select(self.root)

            # Expansion phase
            expanded_node = self._expand(selected_node)

            # Simulation phase
            reward = self._simulate(expanded_node)

            # Backpropagation phase
            self._backpropagate(expanded_node, reward)

        # Select best operation based on configured policy
        best_operation = self._select_best_operation()

        # Update current node to the selected child
        if best_operation and self.current_node:
            # Find the child that corresponds to this operation
            for child in self.current_node.children:
                if child.operation == best_operation:
                    self.current_node = child
                    break

        return best_operation if best_operation else ('xor_constant', {'constant': 1})

    def _select(self, node: TreeNode) -> TreeNode:
        """Selection phase: traverse tree using UCT."""
        current = node

        while not current.is_leaf() and not current.has_untried_operations():
            current = current.get_best_child(self.exploration_constant)

        return current

    def _expand(self, node: TreeNode) -> TreeNode:
        """Expansion phase: add a new child node."""
        if not node.has_untried_operations():
            return node

        # Select an untried operation
        operation = random.choice(node.untried_operations)
        node.untried_operations.remove(operation)

        # Apply operation to create new state
        operation_name, params = operation
        try:
            operation_fn = self.operations_registry.get_operation(operation_name)
            new_data, inverse_fn, metadata = operation_fn(node.state.binary_data, **params)

            # Create new state
            new_state = State(
                binary_data=new_data,
                parent_state_id=node.state.state_id,
                operation_applied={
                    'operation': operation_name,
                    'params': params,
                    'cost': metadata.get('cost', 1.0),
                    'timestamp': node.state.timestamp.isoformat()
                },
                operation_history=node.state.operation_history + [{
                    'operation': operation_name,
                    'params': params,
                    'cost': metadata.get('cost', 1.0),
                    'timestamp': node.state.timestamp.isoformat()
                }],
                inverse_operations=node.state.inverse_operations + [inverse_fn],
                generation=node.state.generation + 1
            )

            # Create child node
            child = node.add_child(new_state, operation)

            # Initialize untried operations for the child
            if new_state.generation < self.max_tree_depth:
                self._initialize_untried_operations(child)

            return child

        except Exception:
            # Operation failed, return current node
            return node

    def _simulate(self, node: TreeNode) -> float:
        """Simulation phase: random playout from node."""
        current_state = node.state

        for depth in range(self.rollout_depth):
            if self.early_termination and random.random() < self.early_termination_threshold:
                break

            # Select random operation
            operation_name = random.choice(self.operations_registry.list_operations())
            params = self._generate_operation_params(operation_name)

            if params is None:
                continue

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
                continue

        # Return the score of the final state
        return current_state.score if hasattr(current_state, 'score') else 0.0

    def _backpropagate(self, node: TreeNode, reward: float) -> None:
        """Backpropagation phase: update statistics up the tree."""
        current = node

        while current is not None:
            current.update(reward)
            current = current.parent

    def _select_best_operation(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Select the best operation from the root node."""
        if not self.root or not self.root.children:
            return None

        if self.final_selection == 'most_visited':
            best_child = max(self.root.children, key=lambda c: c.visits)
        elif self.final_selection == 'highest_value':
            best_child = max(self.root.children, key=lambda c: c.value / max(c.visits, 1))
        elif self.final_selection == 'robust':
            # Select child with high visits and good value
            robust_children = [c for c in self.root.children if c.visits > 10]
            if robust_children:
                best_child = max(robust_children, key=lambda c: c.value / max(c.visits, 1))
            else:
                best_child = max(self.root.children, key=lambda c: c.visits)
        else:
            # Default to most visited
            best_child = max(self.root.children, key=lambda c: c.visits)

        return best_child.operation if best_child else None

    def accept(self, new_state: State) -> bool:
        """Accept based on MCTS evaluation."""
        # In MCTS, we accept if the new state improves the score
        # or with some probability to encourage exploration
        if new_state.score > self.best_score:
            return True

        # Small probability of accepting worse moves to avoid local optima
        exploration_prob = 0.1 * math.exp(-self.no_improvement_count / 10)
        return random.random() < exploration_prob

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.root = None
        self.current_node = None

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the MCTS tree."""
        if not self.root:
            return {'tree_size': 0}

        def count_nodes(node: TreeNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)

        def max_depth(node: TreeNode) -> int:
            if not node.children:
                return 0
            return 1 + max(max_depth(child) for child in node.children)

        return {
            'tree_size': count_nodes(self.root),
            'max_depth': max_depth(self.root),
            'root_visits': self.root.visits,
            'root_children': len(self.root.children),
            'current_node_depth': self.current_node.generation if self.current_node else 0
        }