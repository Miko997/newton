# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU arithmetic reference for the contact-roundoff example, not a solver backend."""

from fractions import Fraction

import numpy as np


def round_float32(value: Fraction) -> int:
    """Round a rational to FP32 bits, ties to even, with +0 for exact zero."""
    if value == 0:
        return 0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    largest = Fraction.from_float(float(np.finfo(np.float32).max))
    if value >= (largest + 2**128) / 2:
        return sign | 0x7F800000
    # Correct possible double rounding through float64 using exact distances.
    with np.errstate(over="ignore", under="ignore"):
        candidate = np.float32(float(value))
        if not np.isfinite(candidate):
            candidate = np.finfo(np.float32).max
        neighbors = (
            candidate,
            np.nextafter(candidate, np.float32(-np.inf)),
            np.nextafter(candidate, np.float32(np.inf)),
        )
    distances = []
    for neighbor in neighbors:
        if np.isfinite(neighbor) and neighbor >= 0:
            bits = int(np.asarray(neighbor).view(np.uint32))
            distances.append((abs(value - Fraction.from_float(float(neighbor))), bits & 1, bits))
    return sign | min(distances)[2]


def sum_contact_outputs(initial: np.ndarray, bodies: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum finite FP32 records exactly, returning one uint32 word per component.

    Args:
        initial: Incoming force/torque, float32[B, 6].
        bodies: Destination body IDs, int32[N]; -1 marks an inactive record.
        values: Isolated force/torque outputs, float32[N, 6].
    """
    if initial.dtype != np.float32 or initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("initial must be float32[B, 6]")
    if bodies.dtype != np.int32 or bodies.ndim != 1:
        raise ValueError("bodies must be int32[N]")
    if values.dtype != np.float32 or values.shape != (len(bodies), 6):
        raise ValueError("values must be float32[N, 6]")
    if np.any(bodies < -1) or np.any(bodies >= len(initial)):
        raise ValueError("invalid destination body")
    if not np.isfinite(initial).all() or not np.isfinite(values[bodies >= 0]).all():
        raise ValueError("nonfinite active input")
    totals = [[Fraction.from_float(float(x)) for x in row] for row in initial]
    for body, row in zip(bodies, values, strict=True):
        if body >= 0:
            for component, value in enumerate(row):
                totals[body][component] += Fraction.from_float(float(value))
    return np.array([[round_float32(value) for value in row] for row in totals], dtype=np.uint32).reshape(-1, 6)
