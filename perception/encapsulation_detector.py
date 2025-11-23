"""
Encapsulation Detection Module for Microfluidics

Detects and quantifies particle encapsulation in droplets.
Uses two-stage detection: large circles (droplets) and small circles (particles).

Author: Auto-Fluidics Project
"""

import numpy as np
from typing import List, Tuple, Dict


class EncapsulationDetector:
    """
    Detects particle encapsulation in droplets using two-stage circle detection.

    Workflow:
    1. Detect large circles (droplets)
    2. Detect small circles (particles)
    3. Determine spatial containment (which particles are inside which droplets)
    4. Calculate encapsulation statistics
    """

    def __init__(self,
                 size_ratio_threshold: float = 2.0,
                 containment_margin: float = 0.9):
        """
        Initialize encapsulation detector.

        Args:
            size_ratio_threshold: Minimum ratio between droplet and particle radius
                to consider them as droplet/particle pair (default 2.0)
            containment_margin: Margin factor for containment check (default 0.9)
                particle is inside if: distance + particle_radius < droplet_radius * margin
        """
        self.size_ratio_threshold = size_ratio_threshold
        self.containment_margin = containment_margin

    def detect_encapsulation(self,
                            centres: List[Tuple[int, int]],
                            radii: List[float]) -> Dict:
        """
        Detect encapsulation from detected circles.

        Args:
            centres: List of (x, y) circle centres
            radii: List of circle radii

        Returns:
            Dictionary containing:
                - droplets: List of (x, y, r) for droplets
                - particles: List of (x, y, r) for particles
                - encapsulated_particles: List of particle indices that are encapsulated
                - encapsulation_map: Dict mapping droplet_idx -> [particle_indices]
                - encapsulation_rate: Fraction of particles that are encapsulated
                - mean_particles_per_droplet: Average particles per droplet
                - poisson_lambda: Estimated λ for Poisson loading
                - encapsulation_distribution: Histogram [0, 1, 2, 3+] particles per droplet
        """
        if len(centres) == 0 or len(radii) == 0:
            return self._empty_result()

        # Stage 1: Classify circles into droplets and particles based on size
        droplets, particles = self._classify_circles(centres, radii)

        if len(droplets) == 0:
            # No droplets detected - all are probably particles
            return {
                'droplets': droplets,
                'particles': [(c[0], c[1], r) for c, r in zip(centres, radii)],
                'encapsulated_particles': [],
                'encapsulation_map': {},
                'encapsulation_rate': 0.0,
                'mean_particles_per_droplet': 0.0,
                'poisson_lambda': 0.0,
                'encapsulation_distribution': [0, 0, 0, 0]
            }

        # Stage 2: Determine which particles are inside which droplets
        encapsulation_map, encapsulated_set = self._map_encapsulation(droplets, particles)

        # Stage 3: Calculate statistics
        stats = self._calculate_statistics(droplets, particles, encapsulation_map, encapsulated_set)

        return {
            'droplets': droplets,
            'particles': particles,
            'encapsulated_particles': list(encapsulated_set),
            'encapsulation_map': encapsulation_map,
            **stats
        }

    def _classify_circles(self,
                         centres: List[Tuple[int, int]],
                         radii: List[float]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Classify detected circles into droplets (large) and particles (small).

        Uses median radius as reference point:
        - Above median * threshold = droplet
        - Below = particle

        Returns:
            (droplets, particles) as lists of (x, y, r) tuples
        """
        if len(radii) == 0:
            return [], []

        median_radius = np.median(radii)

        # If size variation is small, might all be one type
        std_radius = np.std(radii)
        if std_radius / median_radius < 0.3:
            # Low variation - assume all are same type (droplets)
            return [(c[0], c[1], r) for c, r in zip(centres, radii)], []

        # Classify based on size threshold
        droplets = []
        particles = []

        threshold_radius = median_radius * 0.7  # Below 70% of median = particle

        for centre, radius in zip(centres, radii):
            circle = (centre[0], centre[1], radius)
            if radius > threshold_radius:
                droplets.append(circle)
            else:
                particles.append(circle)

        return droplets, particles

    def _map_encapsulation(self,
                          droplets: List[Tuple],
                          particles: List[Tuple]) -> Tuple[Dict, set]:
        """
        Map which particles are inside which droplets.

        Args:
            droplets: List of (x, y, r) for droplets
            particles: List of (x, y, r) for particles

        Returns:
            (encapsulation_map, encapsulated_set)
                encapsulation_map: {droplet_idx: [particle_indices]}
                encapsulated_set: Set of particle indices that are encapsulated
        """
        encapsulation_map = {i: [] for i in range(len(droplets))}
        encapsulated_set = set()

        for p_idx, (px, py, pr) in enumerate(particles):
            for d_idx, (dx, dy, dr) in enumerate(droplets):
                # Check if particle is inside droplet
                distance = np.sqrt((px - dx)**2 + (py - dy)**2)

                # Containment check: particle center + radius must be inside droplet
                if distance + pr < dr * self.containment_margin:
                    encapsulation_map[d_idx].append(p_idx)
                    encapsulated_set.add(p_idx)
                    break  # Particle can only be in one droplet

        return encapsulation_map, encapsulated_set

    def _calculate_statistics(self,
                             droplets: List[Tuple],
                             particles: List[Tuple],
                             encapsulation_map: Dict,
                             encapsulated_set: set) -> Dict:
        """
        Calculate encapsulation statistics.

        Returns:
            Dictionary of statistics
        """
        n_droplets = len(droplets)
        n_particles = len(particles)
        n_encapsulated = len(encapsulated_set)

        # Encapsulation rate
        encapsulation_rate = n_encapsulated / n_particles if n_particles > 0 else 0.0

        # Particles per droplet
        particles_per_droplet = [len(encapsulation_map[i]) for i in range(n_droplets)]
        mean_particles_per_droplet = np.mean(particles_per_droplet) if n_droplets > 0 else 0.0

        # Poisson loading parameter λ
        # From Poisson distribution: P(0) = e^(-λ), so λ = -ln(P(0))
        # Estimate P(0) from fraction of empty droplets
        empty_droplets = sum(1 for count in particles_per_droplet if count == 0)
        p_empty = empty_droplets / n_droplets if n_droplets > 0 else 1.0

        if p_empty > 0 and p_empty < 1:
            poisson_lambda = -np.log(p_empty)
        elif p_empty == 0:
            poisson_lambda = mean_particles_per_droplet  # All droplets have particles
        else:
            poisson_lambda = 0.0  # All droplets empty

        # Distribution: [0, 1, 2, 3+] particles per droplet
        distribution = [0, 0, 0, 0]
        for count in particles_per_droplet:
            if count == 0:
                distribution[0] += 1
            elif count == 1:
                distribution[1] += 1
            elif count == 2:
                distribution[2] += 1
            else:
                distribution[3] += 1

        return {
            'encapsulation_rate': encapsulation_rate,
            'mean_particles_per_droplet': mean_particles_per_droplet,
            'poisson_lambda': poisson_lambda,
            'encapsulation_distribution': distribution,
            'num_droplets': n_droplets,
            'num_particles': n_particles,
            'num_encapsulated': n_encapsulated,
            'num_empty_droplets': distribution[0]
        }

    def _empty_result(self) -> Dict:
        """Return empty result when no circles detected."""
        return {
            'droplets': [],
            'particles': [],
            'encapsulated_particles': [],
            'encapsulation_map': {},
            'encapsulation_rate': 0.0,
            'mean_particles_per_droplet': 0.0,
            'poisson_lambda': 0.0,
            'encapsulation_distribution': [0, 0, 0, 0],
            'num_droplets': 0,
            'num_particles': 0,
            'num_encapsulated': 0,
            'num_empty_droplets': 0
        }


def visualize_encapsulation(image, encapsulation_result, color_droplet=(0, 255, 0),
                            color_particle=(255, 0, 0), color_encapsulated=(0, 255, 255)):
    """
    Helper function to visualize encapsulation on an image.

    Args:
        image: Input image (numpy array)
        encapsulation_result: Result dictionary from detect_encapsulation()
        color_droplet: BGR color for droplets (default green)
        color_particle: BGR color for free particles (default blue)
        color_encapsulated: BGR color for encapsulated particles (default cyan)

    Returns:
        Annotated image
    """
    import cv2
    annotated = image.copy()

    # Draw droplets (thick circles)
    for (x, y, r) in encapsulation_result['droplets']:
        cv2.circle(annotated, (int(x), int(y)), int(r), color_droplet, 3)

    # Draw particles
    encapsulated_set = set(encapsulation_result['encapsulated_particles'])
    for idx, (x, y, r) in enumerate(encapsulation_result['particles']):
        color = color_encapsulated if idx in encapsulated_set else color_particle
        cv2.circle(annotated, (int(x), int(y)), int(r), color, 2)

    return annotated
