# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Arithmetic and public-API observation checks for the contact diagnostic."""

import contextlib
import io
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.basic._contact_reference import round_float32, sum_contact_outputs
from newton.examples.basic.example_basic_contact_roundoff import Example, ForceObserver
from newton.tests.unittest_utils import add_function_test, get_test_devices
from newton.viewer import ViewerNull


class TestContactReference(unittest.TestCase):
    def test_cancellation_and_permutation(self):
        """Recover a small term lost by ordinary FP32 addition in either order."""
        values = np.zeros((3, 6), dtype=np.float32)
        values[:, 0] = [2**24, 1, -(2**24)]
        initial = np.zeros((1, 6), dtype=np.float32)
        bodies = np.zeros(3, dtype=np.int32)
        expected = np.zeros((1, 6), dtype=np.uint32)
        expected[0, 0] = 0x3F800000
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            np.testing.assert_array_equal(sum_contact_outputs(initial, bodies, values[order]), expected)

    def test_rounding_ties(self):
        """Round both even and odd halfway candidates correctly, including negatives."""
        self.assertEqual(round_float32(Fraction(1) + Fraction(1, 2**24)), 0x3F800000)
        self.assertEqual(round_float32(Fraction(1) + Fraction(3, 2**24)), 0x3F800002)
        self.assertEqual(round_float32(-Fraction(1) - Fraction(3, 2**24)), 0xBF800002)
        # The float64 approximation lands on a tie; the exact value is above it.
        self.assertEqual(round_float32(Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**100)), 0x3F800001)

    def test_subnormal_zero_overflow(self):
        """Preserve subnormals, canonicalize exact zero and round overflow explicitly."""
        self.assertEqual(round_float32(Fraction(1, 2**149)), 1)
        self.assertEqual(round_float32(Fraction(-1, 2**149)), 0x80000001)
        self.assertEqual(round_float32(Fraction(0)), 0)
        largest = Fraction.from_float(float(np.finfo(np.float32).max))
        boundary = (largest + 2**128) / 2
        self.assertEqual(round_float32(boundary - 1), 0x7F7FFFFF)
        self.assertEqual(round_float32(boundary), 0x7F800000)
        self.assertEqual(round_float32(-boundary), 0xFF800000)

    def test_initial_inactive_and_empty(self):
        """Include initial forces once and ignore inactive payloads without losing duplicates."""
        initial = np.ones((2, 6), dtype=np.float32)
        bodies = np.array([0, 0, -1], dtype=np.int32)
        values = np.ones((3, 6), dtype=np.float32)
        values[2] = np.nan
        expected = initial.copy()
        expected[0] = 3
        np.testing.assert_array_equal(sum_contact_outputs(initial, bodies, values), expected.view(np.uint32))
        np.testing.assert_array_equal(sum_contact_outputs(initial, bodies[:0], values[:0]), initial.view(np.uint32))

    def test_invalid_inputs(self):
        """Reject invalid destinations, shapes, dtypes and nonfinite active values."""
        initial = np.zeros((1, 6), dtype=np.float32)
        bodies = np.zeros(1, dtype=np.int32)
        values = np.zeros((1, 6), dtype=np.float32)
        for bad in (np.array([-2], np.int32), np.array([1], np.int32), bodies.astype(np.int64)):
            with self.assertRaises(ValueError):
                sum_contact_outputs(initial, bad, values)
        for bad in (np.full((1, 6), np.inf, np.float32), values.astype(np.float64), values[:, :5]):
            with self.assertRaises(ValueError):
                sum_contact_outputs(initial, bodies, bad)
        with self.assertRaises(ValueError):
            sum_contact_outputs(initial.astype(np.float64), bodies, values)


def make_example(device):
    """Keep device tests small and independent of the recorded research fixture."""
    with wp.ScopedDevice(device):
        return Example(ViewerNull(num_frames=1), SimpleNamespace(world_count=1, layers=1, repeats=2, capture=None))


def test_isolated_contact_example(test, device):
    """Run the diagnostic without assuming native variation or a fixed mismatch count."""
    example = make_example(device)
    example.step()
    before = example.state_0.body_q.numpy().view(np.uint32).copy()
    velocity_before = example.state_0.body_qd.numpy().view(np.uint32).copy()
    with contextlib.redirect_stdout(io.StringIO()):
        report = example.diagnose()
    test.assertGreater(report["contact_count"], 0)
    test.assertLessEqual(report["contact_peak"], report["contact_capacity"])
    np.testing.assert_array_equal(example.state_0.body_q.numpy().view(np.uint32), before)
    np.testing.assert_array_equal(example.state_0.body_qd.numpy().view(np.uint32), velocity_before)
    if device.is_cuda:
        for mode in ("run_to_run", "gpu_to_gpu"):
            test.assertEqual(report["modes"][mode]["distinct_arrays"], 1)


def test_observer_trajectory(test, device):
    """Preserve force-buffer ownership and the original solver's deterministic trajectory."""
    example = make_example(device)
    mode = wp.DeterministicMode.GPU_TO_GPU if device.is_cuda else wp.DeterministicMode.NOT_GUARANTEED
    original = newton.solvers.SolverSemiImplicit(example.model, deterministic=mode)
    observer = ForceObserver(example.model, deterministic=mode)
    states = [[example.model.state(), example.model.state()] for _ in range(2)]
    contacts = [example.pipeline.contacts(), example.pipeline.contacts()]
    for _ in range(8):
        for index, solver in enumerate((original, observer)):
            current, output = states[index]
            current.clear_forces()
            # Exercise the actual incoming buffer as well as the joint temporary.
            current.body_f.fill_(wp.spatial_vector(0.125, -0.25, 0.5, 0.25, 0.0, -0.125))
            buffer = current.body_f
            example.pipeline.collide(current, contacts[index])
            solver.step(current, output, example.control, contacts[index], example.sim_dt)
            test.assertIs(current.body_f, buffer)
            states[index] = [output, current]
        for name in ("body_q", "body_qd"):
            np.testing.assert_array_equal(
                getattr(states[0][0], name).numpy().view(np.uint32),
                getattr(states[1][0], name).numpy().view(np.uint32),
            )


def test_observer_graph(test, device):
    """Replay the observation copy with changed incoming forces while retaining its buffers."""
    if not device.is_cuda:
        test.skipTest("CUDA graph capture requires CUDA")
    example = make_example(device)
    example.collide()
    solver = ForceObserver(example.model, deterministic=wp.DeterministicMode.GPU_TO_GPU)
    state, output = example.model.state(), example.model.state()
    solver.step(state, output, example.control, example.contacts, example.sim_dt)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, output, example.control, example.contacts, example.sim_dt)
    for value in (0.0, 0.125, -0.5, 0.0):
        state.body_f.fill_(wp.spatial_vector(value, value, value, value, value, value))
        wp.capture_launch(capture.graph)
        observed = solver.force.numpy().view(np.uint32).copy()
        solver.step(state, output, example.control, example.contacts, example.sim_dt)
        np.testing.assert_array_equal(solver.force.numpy().view(np.uint32), observed)


def test_cpu_output_snapshot(test, device):
    """Keep each evaluation when NumPy aliases the reusable CPU observation buffer."""
    if device.is_cuda:
        test.skipTest("This regression targets CPU NumPy buffer aliasing")

    class ChangingObserver(ForceObserver):
        def __init__(self, model, *, deterministic):
            super().__init__(model, deterministic=deterministic)
            self.native = deterministic == wp.DeterministicMode.NOT_GUARANTEED
            self.evaluation = 0

        def step(self, *args, **kwargs):
            super().step(*args, **kwargs)
            if self.native:
                self.evaluation += 1
                self.force.fill_(wp.spatial_vector(float(self.evaluation), 0.0, 0.0, 0.0, 0.0, 0.0))

    example = make_example(device)
    with tempfile.TemporaryDirectory() as directory:
        example.args.capture = str(Path(directory) / "capture.npz")
        with (
            patch("newton.examples.basic.example_basic_contact_roundoff.ForceObserver", ChangingObserver),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = example.diagnose()
        test.assertEqual(report["modes"]["native"]["distinct_arrays"], 2)
        with np.load(example.args.capture, allow_pickle=False) as saved:
            test.assertEqual(int(saved["native"][0, 0, 0]), 0x3F800000)
            test.assertEqual(int(saved["native"][1, 0, 0]), 0x40000000)


class TestContactRoundoff(unittest.TestCase):
    pass


for device in get_test_devices():
    for function in (
        test_isolated_contact_example,
        test_observer_trajectory,
        test_observer_graph,
        test_cpu_output_snapshot,
    ):
        add_function_test(TestContactRoundoff, function.__name__, function, devices=[device])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
