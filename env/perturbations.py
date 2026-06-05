"""
Orbital perturbation models for satellite formation flying.

This module implements simplified but physically meaningful external
perturbations that can occur during orbital flight, such as:
- Solar radiation pressure (SRP)
- Residual atmospheric drag (LEO)
- Actuation errors from micro-thrusters

All perturbations are designed to be:
- Low magnitude
- Smooth or discrete but non-impulsive
- Fully decoupled from the controller
"""

import numpy as np


class OrbitalPerturbations:
    """
    Class that encapsulates external orbital perturbations applied
    to each satellite in the formation.
    """

    def __init__(
        self,
        dt: float,
        srp_magnitude: float = 1e-4,
        drag_coefficient: float = 1e-3,
        thruster_error_period: int = 50,
        thruster_error_max: float = 5e-4,
        wind_magnitude: float = 0.02,  # orbital wind magnitude
        wind_direction: np.ndarray | None = None,
        wind_slow_variation: float = 0.001,  # slow time variation
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        dt : float
            Simulation timestep.
        srp_magnitude : float
            Magnitude of the solar radiation pressure acceleration.
        drag_coefficient : float
            Coefficient for residual atmospheric drag.
        thruster_error_period : int
            Number of steps between discrete thruster errors.
        thruster_error_max : float
            Maximum magnitude of thruster actuation error.
        wind_magnitude : float
            Magnitude of the continuous orbital wind.
        wind_direction : np.ndarray
            Direction vector of the wind.
        wind_slow_variation : float
            # slow variation factor for direction or magnitude
        seed : int or None
            Random seed for reproducibility.
        """

        self.dt = dt
        self.srp_magnitude = srp_magnitude
        self.drag_coefficient = drag_coefficient
        self.thruster_error_period = thruster_error_period
        self.thruster_error_max = thruster_error_max

        # Orbital wind
        self.wind_magnitude = wind_magnitude
        if wind_direction is None:
            self.wind_direction = np.array([1.0, 0.0])
        else:
            self.wind_direction = wind_direction
        self.wind_direction /= np.linalg.norm(self.wind_direction)
        self.wind_slow_variation = wind_slow_variation

        # Fixed or slowly varying Sun direction (normalized)
        self.sun_direction = np.array([1.0, 0.0])
        self.sun_direction /= np.linalg.norm(self.sun_direction)

        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Individual perturbation models
    # ------------------------------------------------------------------

    def solar_radiation_pressure(self) -> np.ndarray:
        """
        Solar radiation pressure (SRP).

        Modeled as a constant low-magnitude acceleration in the
        direction of the Sun.
        """
        return self.srp_magnitude * self.sun_direction

    def atmospheric_drag(self, velocity: np.ndarray) -> np.ndarray:
        """
        Residual atmospheric drag (LEO).

        Modeled as a force opposite to the velocity vector, proportional
        to the speed magnitude.
        """
        speed = np.linalg.norm(velocity)

        if speed < 1e-8:
            return np.zeros_like(velocity)

        drag_direction = -velocity / speed
        drag_magnitude = self.drag_coefficient * speed

        return drag_magnitude * drag_direction

    def thruster_error(self, step: int) -> np.ndarray:
        """
        Discrete actuation error due to imperfect micro-thrusters.

        This perturbation is applied only every N steps and represents
        small mismatches between commanded and executed thrust.
        """
        if step % self.thruster_error_period != 0:
            return np.zeros(2)

        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)

        magnitude = np.random.uniform(0.0, self.thruster_error_max)

        return magnitude * direction

    def orbital_wind(self, step: int) -> np.ndarray:
        """
        Perturbación de viento orbital constante o lentamente variable.
        """
        variation_factor = np.sin(self.wind_slow_variation * step)
        return self.wind_magnitude * variation_factor * self.wind_direction

    # ------------------------------------------------------------------
    # Combined perturbation interface
    # ------------------------------------------------------------------

    def total_perturbation(
        self,
        velocity: np.ndarray,
        step: int,
        use_srp: bool = True,
        use_drag: bool = True,
        use_thruster_error: bool = True,
        use_wind: bool = True,  # enable wind
    ) -> np.ndarray:
        """
        Compute the total external perturbation acting on the satellite.

        Returns the sum of all enabled perturbations.
        """

        perturbation = np.zeros(2)

        if use_srp:
            perturbation += self.solar_radiation_pressure()

        if use_drag:
            perturbation += self.atmospheric_drag(velocity)

        if use_thruster_error:
            perturbation += self.thruster_error(step)
            
        if use_wind:
            perturbation += self.orbital_wind(step)
        return perturbation
