# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare repeatability and exact sums of isolated SemiImplicit contact outputs.

Command: python -m newton.examples basic_contact_roundoff --viewer null
This intentionally slow diagnostic uses public solver calls, not a copied force law.
"""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._contact_reference import sum_contact_outputs


class ForceObserver(newton.solvers.SolverSemiImplicit):
    """Observe the actual force buffer, including the solver's joint temporary."""

    def __init__(self, model, *, deterministic):
        super().__init__(model, deterministic=deterministic)
        self.force = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)

    def integrate_bodies(self, model, state_in, state_out, dt, angular_damping=0.0):
        wp.copy(self.force, state_in.body_f)
        super().integrate_bodies(model, state_in, state_out, dt, angular_damping)


class Example:
    """A bounded free-body scene; this is not a general contact-export interface."""

    def __init__(self, viewer, args):
        if not 1 <= args.world_count <= 4 or not 1 <= args.layers <= 5 or not 1 <= args.repeats <= 100:
            raise ValueError("Use 1..4 worlds, 1..5 layers and 1..100 repetitions")
        self.viewer = viewer
        self.args = args
        if args.capture and Path(args.capture).exists():
            raise FileExistsError(args.capture)
        self.sim_time = 0.0
        self.sim_dt = 0.0002
        self.report = None
        config = newton.ModelBuilder.ShapeConfig(density=500, ke=15000, kd=80, kf=1000, mu=0.5, margin=0, gap=0.015)
        world = newton.ModelBuilder()
        world.add_ground_plane(cfg=config)
        for z in range(args.layers):
            for y in range(4):
                for x in range(4):
                    body = world.add_body(
                        xform=wp.transform(
                            wp.vec3((x - 1.5) * 0.20, (y - 1.5) * 0.20, 0.098 + z * 0.196), wp.quat_identity()
                        )
                    )
                    world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=config)
        builder = newton.ModelBuilder()
        for i in range(args.world_count):
            builder.add_world(world, xform=wp.transform(wp.vec3(2 * i, 0, 0), wp.quat_identity()))
        self.model = builder.finalize()
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=4096 * args.world_count, deterministic=True, broad_phase="sap"
        )
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, deterministic=wp.DeterministicMode.GPU_TO_GPU)
        self.contact_peak = 0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.5, -3, 2), pitch=-25, yaw=130)

    def collide(self):
        """Check capacity on every evolving step and before recording a fixture."""
        self.pipeline.collide(self.state_0, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if not 0 <= count <= self.contacts.rigid_contact_max:
            raise RuntimeError("contact capacity exceeded")
        self.contact_peak = max(self.contact_peak, count)
        return count

    def step(self):
        for _ in range(4):
            self.state_0.clear_forces()
            self.collide()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += 4 * self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check the scene and run the diagnostic without requiring native variation."""
        if not np.isfinite(self.state_0.body_q.numpy()).all() or not np.isfinite(self.state_0.body_qd.numpy()).all():
            raise RuntimeError("nonfinite simulation state")
        self.diagnose()

    def diagnose(self):
        """Replay a frozen fixture; never advance the simulation's saved state."""
        if self.report is not None:
            return self.report
        if self.model.particle_count or np.any(self.model.joint_type.numpy() != int(newton.JointType.FREE)):
            raise ValueError("This diagnostic supports only its free-body scene without particles")
        if np.any(self.control.joint_f.numpy()):
            raise ValueError("This diagnostic requires zero joint controls")
        count = self.collide()
        state, output = self.model.state(), self.model.state()
        state.assign(self.state_0)
        state.clear_forces()
        initial = state.body_f.numpy()
        one = self.pipeline.contacts()
        one.rigid_contact_count.assign(np.array([0], dtype=np.int32))
        isolated = ForceObserver(self.model, deterministic=wp.DeterministicMode.RUN_TO_RUN)
        isolated.step(state, output, self.control, one, self.sim_dt)
        if np.any(isolated.force.numpy()):
            raise ValueError("Non-contact forces would invalidate isolated replay")
        shape_body = self.model.shape_body.numpy()
        shapes = (self.contacts.rigid_contact_shape0.numpy(), self.contacts.rigid_contact_shape1.numpy())
        bodies = np.full((count, 2), -1, dtype=np.int32)
        values = np.zeros((count, 2, 6), dtype=np.float32)
        fields = (
            "point0",
            "point1",
            "normal",
            "shape0",
            "shape1",
            "margin0",
            "margin1",
            "surface_velocity",
            "stiffness",
            "damping",
            "friction",
        )
        one.rigid_contact_count.assign(np.array([1], dtype=np.int32))
        for contact in range(count):
            for endpoint in range(2):
                shape = int(shapes[endpoint][contact])
                if not -1 <= shape < self.model.shape_count:
                    raise ValueError("invalid contact shape")
                bodies[contact, endpoint] = shape_body[shape] if shape >= 0 else -1
            a, b = bodies[contact]
            if a >= 0 and a == b:
                raise ValueError("Cannot separate both endpoints on the same body")
            for name in fields:
                source = getattr(self.contacts, "rigid_contact_" + name)
                target = getattr(one, "rigid_contact_" + name)
                if source is not None and source.size:
                    wp.copy(target, source, src_offset=contact, count=1)
            state.clear_forces()
            isolated.step(state, output, self.control, one, self.sim_dt)
            force = isolated.force.numpy()
            for endpoint, body in enumerate(bodies[contact]):
                if body >= 0:
                    values[contact, endpoint] = force[body]
        bodies, values = bodies.reshape(-1), values.reshape(-1, 6)
        expected = sum_contact_outputs(initial, bodies, values)
        captures = {
            "initial": initial,
            "bodies": bodies,
            "values": values,
            "reference": expected,
            "body_q": state.body_q.numpy(),
            "body_qd": state.body_qd.numpy(),
        }
        modes = {"native": wp.DeterministicMode.NOT_GUARANTEED}
        if self.model.device.is_cuda:
            modes.update(run_to_run=wp.DeterministicMode.RUN_TO_RUN, gpu_to_gpu=wp.DeterministicMode.GPU_TO_GPU)
        report = {
            "body_count": self.model.body_count,
            "contact_count": count,
            "contact_peak": self.contact_peak,
            "contact_capacity": self.contacts.rigid_contact_max,
            "repeats": self.args.repeats,
            "reference": "RN-even FP32 exact sum of isolated contact outputs; not raw pre-atomic operands",
            "device": str(self.model.device),
            "modes": {},
        }
        for name, mode in modes.items():
            solver = ForceObserver(self.model, deterministic=mode)
            outputs = []
            for _ in range(self.args.repeats):
                state.assign(self.state_0)
                state.clear_forces()
                solver.step(state, output, self.control, self.contacts, self.sim_dt)
                force = solver.force.numpy()
                if not np.isfinite(force).all():
                    raise RuntimeError("nonfinite accumulated force")
                # CPU numpy() may alias the observation buffer reused next step.
                outputs.append(force.view(np.uint32).copy())
            words = np.stack(outputs)
            captures[name] = words
            report["modes"][name] = {
                "distinct_arrays": len({row.tobytes() for row in words}),
                "mismatched_components_min": int(np.count_nonzero(words != expected, axis=(1, 2)).min()),
                "mismatched_components_max": int(np.count_nonzero(words != expected, axis=(1, 2)).max()),
            }
        if self.args.capture:
            # Refuse overwrites, including incomplete results from an earlier run.
            with Path(self.args.capture).open("xb") as stream:
                np.savez_compressed(stream, **captures)
        self.report = report
        print(json.dumps(report, sort_keys=True))
        return report

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=128)
        parser.add_argument("--world-count", type=int, default=4)
        parser.add_argument("--layers", type=int, default=5)
        parser.add_argument("--repeats", type=int, default=100)
        parser.add_argument("--capture", type=str, help="Write a new NPZ with isolated outputs and comparisons")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
    example.diagnose()
