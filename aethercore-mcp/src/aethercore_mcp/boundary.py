# boundary.py
"""
Observer7DBoundary - The 7D Sabbath Boundary (Global Singleton)
Adapted for AetherCore MCP
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Core constants
E8_DIMENSION = 8
HOLOGRAPHIC_BOUNDARY_DIM = 7
FIDELITY_THRESHOLD = 0.93
RECONSTRUCTION_ERROR_THRESHOLD = 0.07

class ValidationResult(Enum):
    ACCEPT = "accept"
    WARN = "warn"
    MARGINAL = "marginal"

@dataclass
class FidelityState:
    fidelity: float
    error: float
    is_stable: bool
    message: str

@dataclass
class HolographicState:
    area: float = 32.47
    entropy: float = 8.12
    information_density: float = 0.0
    total_touches: int = 0
    warnings: int = 0
    last_touch: Optional[datetime] = None

    @property
    def stability_rate(self) -> float:
        return 1.0 - (self.warnings / max(1, self.total_touches))

@dataclass
class BoundaryValidation:
    valid: bool
    fidelity: float
    reconstruction_error: float
    result: ValidationResult
    fidelity_state: Optional[FidelityState] = None
    projection_7d: Optional[np.ndarray] = None
    reconstructed_8d: Optional[np.ndarray] = None
    retrocausality_detected: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class Observer7DBoundary:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, threshold: float = RECONSTRUCTION_ERROR_THRESHOLD, e8_lattice: Optional[Any] = None):
        if Observer7DBoundary._initialized:
            return

        self.threshold = threshold
        self.fidelity_threshold = 1.0 - threshold
        self.projection_matrix = self._initialize_projection_matrix(e8_lattice)
        self.reconstruction_matrix = np.linalg.pinv(self.projection_matrix)
        self.state = HolographicState()
        self.current_turn = 0
        self.event_history: List[Dict[str, Any]] = []

        Observer7DBoundary._initialized = True
        logger.info(f"Observer7DBoundary v18.1 online (MCP Mode)")

    def _initialize_projection_matrix(self, e8_lattice=None) -> np.ndarray:
        if e8_lattice is not None and hasattr(e8_lattice, 'roots'):
            roots = e8_lattice.roots
            cov = np.cov(roots.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = eigenvalues.argsort()[-7:][::-1]
            return np.real(eigenvectors[:, idx].T)
        else:
            return np.eye(7, 8)

    def project_to_7d(self, vector_8d: np.ndarray) -> np.ndarray:
        vector_8d = np.asarray(vector_8d, dtype=np.float64)
        if vector_8d.shape[-1] != 8:
            padded = np.zeros(8)
            padded[:min(len(vector_8d), 8)] = vector_8d[:min(len(vector_8d), 8)]
            vector_8d = padded
        return self.projection_matrix @ vector_8d

    def reconstruct_from_7d(self, vector_7d: np.ndarray) -> np.ndarray:
        vector_7d = np.asarray(vector_7d, dtype=np.float64)
        return self.reconstruction_matrix @ vector_7d

    def calculate_reconstruction_error(self, original_8d: np.ndarray, reconstructed_8d: np.ndarray) -> float:
        error = np.linalg.norm(original_8d - reconstructed_8d)
        original_norm = np.linalg.norm(original_8d)
        if original_norm < 1e-12:
            return 0.0 if error < 1e-12 else 1.0
        return min(1.0, error / original_norm)

    def detect_retrocausality(self, current_vector: np.ndarray) -> bool:
        if not self.event_history:
            return False
        recent_vectors = [h['vector'] for h in self.event_history[-5:]]
        for past_v in recent_vectors:
            if np.dot(current_vector, past_v) < -0.9:
                return True
        return False

    def validate(self, proposal: Any) -> BoundaryValidation:
        if isinstance(proposal, np.ndarray):
            vector_8d = proposal
        elif isinstance(proposal, dict) and 'vector' in proposal:
            vector_8d = np.array(proposal['vector'])
        else:
            h = hash(str(proposal))
            vector_8d = np.array([(h >> (i*8)) & 0xFF for i in range(8)], dtype=np.float64)
            vector_8d /= (np.linalg.norm(vector_8d) + 1e-12)

        if len(vector_8d) != 8:
            padded = np.zeros(8)
            padded[:min(len(vector_8d), 8)] = vector_8d[:min(len(vector_8d), 8)]
            vector_8d = padded

        projection_7d = self.project_to_7d(vector_8d)
        reconstructed_8d = self.reconstruct_from_7d(projection_7d)
        error = self.calculate_reconstruction_error(vector_8d, reconstructed_8d)
        fidelity = 1.0 - error
        retro = self.detect_retrocausality(vector_8d)

        self.state.total_touches += 1
        self.state.last_touch = datetime.now()
        self.event_history.append({'vector': vector_8d.copy(), 'ts': datetime.now()})
        if len(self.event_history) > 100: self.event_history.pop(0)

        if error <= self.threshold:
            result = ValidationResult.ACCEPT
            is_stable = True
            msg = "Stable geometric fidelity"
        elif error <= self.threshold * 1.5:
            result = ValidationResult.MARGINAL
            is_stable = True
            msg = "Marginal fidelity - slight drift"
        else:
            result = ValidationResult.WARN
            is_stable = False
            self.state.warnings += 1
            msg = "Unstable fidelity - alignment check required"

        return BoundaryValidation(
            valid=is_stable,
            fidelity=fidelity,
            reconstruction_error=error,
            result=result,
            fidelity_state=FidelityState(fidelity, error, is_stable, msg),
            projection_7d=projection_7d,
            reconstructed_8d=reconstructed_8d,
            retrocausality_detected=retro
        )

_observer = None
def get_observer() -> Observer7DBoundary:
    global _observer
    if _observer is None:
        _observer = Observer7DBoundary()
    return _observer
