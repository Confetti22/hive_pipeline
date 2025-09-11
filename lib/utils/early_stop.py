# early_stopping.py
from __future__ import annotations
from copy import deepcopy

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        mode: 'min' for metrics that should decrease (e.g., val_loss),
              'max' for metrics that should increase (e.g., val_acc).
        patience: #epochs to wait after last improvement.
        min_delta: minimum absolute change to qualify as an improvement.
        verbose: print messages on improvements/stop.
        restore_best_state: if True, keeps a copy of the best model state_dict to restore later.
    """
    def __init__(
        self,
        mode: str = 'min',
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True,
        restore_best_state: bool = True,
    ):
        assert mode in ('min', 'max')
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_state = restore_best_state

        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self.best_state = None  # state_dict to restore

        if mode == 'min':
            self._is_better = lambda current, best: (best - current) > self.min_delta
            self.best_score = float('inf')
        else:
            self._is_better = lambda current, best: (current - best) > self.min_delta
            self.best_score = -float('inf')

    def step(self, current_score: float, model=None) -> bool:
        """
        Update with the latest validation score, optionally capture best weights.

        Returns:
            True if training should stop (patience exceeded), else False.
        """
        if self._is_better(current_score, self.best_score):
            if self.verbose:
                print(f"[EarlyStopping] Improvement: {self.best_score:.6f} -> {current_score:.6f}")
            self.best_score = current_score
            self.num_bad_epochs = 0
            if self.restore_best_state and model is not None:
                # deepcopy() to avoid reference to the same tensor storage
                self.best_state = deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.num_bad_epochs}/{self.patience})")

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop

    def restore_best(self, model):
        """Restore the best recorded state (if any)."""
        if self.restore_best_state and self.best_state is not None:
            model.load_state_dict(self.best_state)