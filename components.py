"""
Component registry for swappable NHL DFS pipeline components.

Instead of hard-coding imports in main.py, components can be selected
at runtime via CLI flags or configuration. This enables switching between:

- optimizer.py (greedy heuristic) vs optimizer_ilp.py (ILP solver)
- simulator.py (deterministic/independent MC) vs simulation_engine.py (correlated MC)

Usage:
    from components import get_optimizer, get_simulator

    OptimizerClass = get_optimizer("ilp")   # or "greedy" (default)
    SimulatorClass = get_simulator("correlated")  # or "basic" (default)
"""

from typing import Type


# ============================================================================
# Optimizer Registry
# ============================================================================

_OPTIMIZER_REGISTRY = {}


def register_optimizer(name: str):
    """Decorator to register an optimizer implementation."""
    def decorator(cls):
        _OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator


def get_optimizer(name: str = "greedy") -> Type:
    """
    Get an optimizer class by name.

    Available optimizers:
        "greedy" - Heuristic greedy optimizer (optimizer.py) — default
        "ilp"    - Integer Linear Programming optimizer (optimizer_ilp.py)

    Returns:
        The NHLLineupOptimizer class (not an instance).
    """
    if name in _OPTIMIZER_REGISTRY:
        return _OPTIMIZER_REGISTRY[name]

    # Lazy-load to avoid import cycles and optional dependency issues
    if name == "greedy":
        from optimizer import NHLLineupOptimizer
        _OPTIMIZER_REGISTRY["greedy"] = NHLLineupOptimizer
        return NHLLineupOptimizer
    elif name == "ilp":
        try:
            from optimizer_ilp import NHLLineupOptimizer as ILPOptimizer
            _OPTIMIZER_REGISTRY["ilp"] = ILPOptimizer
            return ILPOptimizer
        except ImportError as e:
            print(f"Warning: ILP optimizer not available ({e}). Falling back to greedy.")
            return get_optimizer("greedy")
    else:
        raise ValueError(f"Unknown optimizer: {name!r}. Available: greedy, ilp")


# ============================================================================
# Simulator Registry
# ============================================================================

_SIMULATOR_REGISTRY = {}


def register_simulator(name: str):
    """Decorator to register a simulator implementation."""
    def decorator(cls):
        _SIMULATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_simulator(name: str = "basic") -> Type:
    """
    Get a simulator class by name.

    Available simulators:
        "basic"      - Team-pair frequency analysis (simulator.py) — default
        "correlated" - Correlated Monte Carlo with zero-inflated lognormal (simulation_engine.py)

    Returns:
        The simulator class (not an instance).
    """
    if name in _SIMULATOR_REGISTRY:
        return _SIMULATOR_REGISTRY[name]

    if name == "basic":
        from simulator import OptimalLineupSimulator
        _SIMULATOR_REGISTRY["basic"] = OptimalLineupSimulator
        return OptimalLineupSimulator
    elif name == "correlated":
        try:
            from simulation_engine import CorrelatedSimulator
            _SIMULATOR_REGISTRY["correlated"] = CorrelatedSimulator
            return CorrelatedSimulator
        except ImportError as e:
            print(f"Warning: Correlated simulator not available ({e}). Falling back to basic.")
            return get_simulator("basic")
    else:
        raise ValueError(f"Unknown simulator: {name!r}. Available: basic, correlated")


# ============================================================================
# Ownership Model Registry
# ============================================================================

_OWNERSHIP_REGISTRY = {}


def get_ownership_model(name: str = "gpp") -> Type:
    """
    Get an ownership model class by name.

    Available models:
        "gpp" - GPP ownership model (ownership.py) — default
        "se"  - Single-entry ownership model (se_ownership.py)

    Returns:
        The ownership model class (not an instance).
    """
    if name in _OWNERSHIP_REGISTRY:
        return _OWNERSHIP_REGISTRY[name]

    if name == "gpp":
        from ownership import OwnershipModel
        _OWNERSHIP_REGISTRY["gpp"] = OwnershipModel
        return OwnershipModel
    elif name == "se":
        try:
            from se_ownership import SEOwnershipModel
            _OWNERSHIP_REGISTRY["se"] = SEOwnershipModel
            return SEOwnershipModel
        except ImportError as e:
            print(f"Warning: SE ownership model not available ({e}). Falling back to GPP.")
            return get_ownership_model("gpp")
    else:
        raise ValueError(f"Unknown ownership model: {name!r}. Available: gpp, se")
