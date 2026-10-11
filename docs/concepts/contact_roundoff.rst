.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _contact-roundoff:

Diagnosing contact accumulation roundoff
========================================

Repeating a floating-point reduction in a fixed order and correctly rounding
its exact mathematical sum are different guarantees. The
``basic_contact_roundoff`` example compares them on a small
:class:`~newton.solvers.SolverSemiImplicit` box stack, using the normal native,
``RUN_TO_RUN`` and ``GPU_TO_GPU`` modes. Different answers do not by themselves
indicate a bug or a physically significant error.

Run the diagnostic
------------------

From a checkout with the examples dependencies installed:

.. code-block:: console

   python -m newton.examples basic_contact_roundoff --viewer null --capture contact_roundoff.npz

The default is four worlds containing 320 boxes, 128 settling frames, and
100 evaluations of one frozen state. Each settling frame has four 0.0002 s
substeps. The example checks contact capacity at every substep. For a quick
check, use:

.. code-block:: console

   python -m newton.examples basic_contact_roundoff --viewer null --num-frames 2 --world-count 1 --layers 1 --repeats 3

The JSON output reports how many distinct body-force arrays each mode
produced and how many components differ from the reference. Native variation
is not guaranteed to appear in every scene or run. A deterministic result
can also differ from the reference. CPU execution reports the native mode
only; comparisons between GPU determinism modes require CUDA.

What is being compared?
-----------------------

After settling, the example freezes the body state and contact geometry.
It replays each contact separately through the public solver, with zero
incoming forces and controls. A small subclass copies the actual force buffer
in :meth:`~newton.solvers.SolverBase.integrate_bodies`, including the temporary
buffer used for free-joint forces. It then delegates integration unchanged.
No force law is copied or replaced.

For each body, the example sums the isolated FP32 force/torque outputs with
Python rational arithmetic and rounds once to FP32, ties to even. The six
components are independent. The arithmetic helper includes incoming forces
once, canonicalizes exact zero to +0 and handles finite-input overflow; the
example uses zero incoming forces and rejects nonfinite observed forces.
The reference is independent of input order, unlike fixing one addition order.

This reference is the sum of **isolated solver outputs**, not a general export
of the original pre-atomic operands. A replay can already include rounding or
subnormal flushing inside the solver. The example cannot recover information
lost there. It is deliberately limited to its free-body scene with zero
controls and no particles, verifies that a zero-contact call produces zero
force, and rejects contacts whose endpoints map to the same body. Do not
extend the procedure to arbitrary joints, other solvers, or nonzero background
forces by subtracting two rounded force arrays.

The diagnostic uses synchronous readbacks and one solver invocation per
contact. It is a debugging example, not a simulation backend or performance
benchmark. It adds no library API, dependency, default setting, or solver
instrumentation. A general and efficient export of original endpoint words
would need a separately agreed extension to the solver diagnostics.

Inspect a capture
-----------------

``--capture`` creates a new NPZ file and refuses to overwrite an existing
file. It stores the initial force, destination IDs, isolated FP32 values,
reference uint32 words, frozen body state, and each mode's uint32 outputs.
Zero-force contacts remain represented. Inactive endpoints use body ID -1.

For example, inspect the first differing component of one saved evaluation:

.. code-block:: python

   import numpy as np

   with np.load("contact_roundoff.npz", allow_pickle=False) as capture:
       actual = capture["run_to_run"][0]
       reference = capture["reference"]
       differences = np.argwhere(actual != reference)
       if len(differences):
           body, component = differences[0]
           print(body, component, hex(int(actual[body, component])), hex(int(reference[body, component])))

Comparing isolated values and destinations from two captures can help check
whether generated inputs changed before attributing a discrepancy to
accumulation. Contact order can change, so compare records as a multiset,
retaining duplicates. The example does not establish whole-simulation or
cross-device determinism, or improved physical accuracy. Use the existing
deterministic mode if repeatability alone answers the debugging question.
