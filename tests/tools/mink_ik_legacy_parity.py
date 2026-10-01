"""Generate or audit Mink golden streams against an explicit legacy source tree.

No hardware, camera, or JAX imports are required. Example, from the Axol root:

    PYTHONPATH=. .venv/bin/python tests/tools/mink_ik_legacy_parity.py \
      --legacy-root /path/to/shiraz_axol/xr1 \
      --write-fixture tests/data/mink_ik_legacy

Without --write-fixture, compare the two real solvers and print a report only.
The legacy files are imported under an isolated module namespace, so the
reference solver uses its own source and its own pinned assets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

LEGACY_COMMIT = "b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b"
COMMITTED_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "mink_ik_legacy"
INPUT_NAMES = (
    "rest14",
    "left_pos",
    "left_rot",
    "right_pos",
    "right_rot",
    "reset_before",
    "fk_joints",
    "pose6",
)
REST14 = np.array(
    [
        -0.8708,
        0.0,
        0.0,
        1.395,
        0.0,
        0.0,
        0.3442,
        0.8784,
        0.0,
        0.0,
        -1.403,
        0.0,
        0.0,
        -0.3408,
    ],
    dtype=np.float32,
)


def load_reference(source: Path):
    # Check every imported numerical source and asset before executing any of
    # the reference tree. A current-port checkout is not an independent oracle.
    expected = json.loads((COMMITTED_FIXTURES / "provenance.json").read_text())
    if expected["legacy_commit"] != LEGACY_COMMIT:
        raise ValueError("Legacy fixture commit does not match the pinned baseline")
    if source_hashes(source) != expected["source_sha256"]:
        raise ValueError(
            "Legacy reference source/assets differ from the pinned baseline"
        )
    validate_shared_name_tables()
    namespace = "_xr1_mink_reference"
    for name in list(sys.modules):
        if name == namespace or name.startswith(namespace + "."):
            del sys.modules[name]
    package = types.ModuleType(namespace)
    package.__path__ = [str(source)]
    sys.modules[namespace] = package
    return (
        importlib.import_module(f"{namespace}.ik"),
        importlib.import_module(f"{namespace}.fk"),
    )


def validate_shared_name_tables():
    """The legacy solver's only SDK imports are these unchanged name tables."""
    from almond_axol.constants import Joint, urdf_arm_joint_names, urdf_body_name

    for side, is_left in (("left", True), ("right", False)):
        expected = [
            f"{side}_{suffix}"
            for suffix in ("s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0")
        ]
        if urdf_arm_joint_names(is_left=is_left) != expected:
            raise ValueError("Shared SDK joint name table differs from pinned legacy")
        for joint, suffix in (
            (Joint.GRIPPER, "gripper"),
            (Joint.ELBOW, "e2"),
            (Joint.SHOULDER_1, "s2"),
        ):
            if urdf_body_name(joint, is_left=is_left) != f"{side}_{suffix}":
                raise ValueError(
                    "Shared SDK body name table differs from pinned legacy"
                )


def runtime_versions() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "mink", "daqp", "qpsolvers", "numpy")
    }
    expected = json.loads((COMMITTED_FIXTURES / "provenance.json").read_text())
    if versions != expected["runtime_versions"]:
        raise RuntimeError(
            "Legacy numerical parity requires the pinned numerical runtime: "
            f"expected {expected['runtime_versions']}, found {versions}"
        )
    return versions


def runtime_platform() -> dict[str, str]:
    return {"system": platform.system(), "machine": platform.machine()}


def load_fixture(path: Path, *, require_native: bool = False):
    """Validate stored expectations before tests use them as an oracle.

    Native overrides must use the exact frozen inputs, sources, configuration,
    dependency versions, and current architecture. Tests never fetch a baseline
    or regenerate expectations implicitly.
    """
    provenance = json.loads((path / "provenance.json").read_text())
    if provenance["legacy_commit"] != LEGACY_COMMIT:
        raise ValueError("Fixture legacy commit differs from the pinned baseline")
    fixture = path / "stream.npz"
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != provenance["fixture_sha256"]:
        raise ValueError("Fixture archive hash does not match provenance")
    with np.load(fixture, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if require_native:
        frozen, original = load_fixture(COMMITTED_FIXTURES)
        for name in ("source_sha256", "config", "stream", "runtime_versions"):
            if provenance[name] != original[name]:
                raise ValueError(
                    f"Native fixture {name} differs from the pinned baseline"
                )
        if provenance.get("input_fixture_sha256") != original["fixture_sha256"]:
            raise ValueError(
                "Native fixture was not replayed from the committed inputs"
            )
        if provenance.get("runtime_platform") != runtime_platform():
            raise ValueError("Native fixture was generated on a different platform")
        if provenance["runtime_versions"] != runtime_versions():
            raise ValueError(
                "Native fixture numerical runtime differs from this process"
            )
        for name in INPUT_NAMES:
            if arrays[name].dtype != frozen[name].dtype or not np.array_equal(
                arrays[name], frozen[name]
            ):
                raise ValueError(f"Native fixture changed frozen input {name}")
    return arrays, provenance


def source_hashes(source: Path) -> dict[str, str]:
    paths = [
        source / name
        for name in (
            "ik.py",
            "ik_config.py",
            "ik_mink_backend.py",
            "ik_mujoco_model.py",
            "vendor_io.py",
            "fk.py",
        )
    ]
    paths.extend(path for path in (source / "assets").rglob("*") if path.is_file())
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def generate_reference(source: Path):
    versions = runtime_versions()
    reference, checkpoint_fk = load_reference(source)
    solver = reference.MinkIK()
    solver.set_rest_posture(REST14)
    solver.reset_tracking_state()
    base = solver.fk(REST14)

    def target(t, side):
        pos, rotation = base[side]
        offset = np.array(
            [
                0.04 * math.sin(2 * math.pi * 0.4 * t),
                0.03 * math.sin(2 * math.pi * 0.3 * t + 1),
                0.05 * math.sin(2 * math.pi * 0.5 * t),
            ]
        )
        angle = 0.25 * math.sin(2 * math.pi * 0.35 * t)
        c, s = math.cos(angle), math.sin(angle)
        rotate_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return (pos + offset).astype(np.float32), (rotation @ rotate_z).astype(
            np.float32
        )

    n = 360
    lp, rp = np.empty((n, 3), np.float32), np.empty((n, 3), np.float32)
    lr, rr = np.empty((n, 3, 3), np.float32), np.empty((n, 3, 3), np.float32)
    q_out = np.empty((n, 14), np.float32)
    reset_before = np.zeros(n, dtype=bool)
    failures = np.empty(n, dtype=np.int64)
    q = REST14.copy()
    for tick in range(n):
        if tick == 330:
            q = REST14.copy()
            solver.reset_tracking_state()
            reset_before[tick] = True
        if 300 <= tick < 330:
            left = (np.array([1.5, 0.3, 0.2], np.float32), np.eye(3, dtype=np.float32))
            right = target(0.0, 1)
        else:
            t = (tick if tick < 300 else tick - 330) / 30.0
            left, right = target(t, 0), target(t, 1)
        lp[tick], lr[tick] = left
        rp[tick], rr[tick] = right
        q = solver.solve(q, left, right)
        q_out[tick] = q
        failures[tick] = solver.fail_count

    joint_ids = [
        mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in solver.joint_names
    ]
    ranges = solver.model.jnt_range[joint_ids]
    rng = np.random.default_rng(20260902)
    fk_q = rng.uniform(ranges[:, 0], ranges[:, 1], size=(24, 14)).astype(np.float32)
    fk_pos = np.empty((24, 2, 3), np.float64)
    fk_rot = np.empty((24, 2, 3, 3), np.float64)
    for sample, joints in enumerate(fk_q):
        for side, name in enumerate(("left", "right")):
            rotation, position = checkpoint_fk.gripper_pose_from_joints(
                joints[side * 7 : (side + 1) * 7], name
            )
            fk_pos[sample, side], fk_rot[sample, side] = position, rotation

    poses = rng.normal(size=(64, 6)).astype(np.float32)
    poses[0] = 0
    poses[1, 3:] = [1e-12, -1e-12, 1e-12]
    poses[2, 3:] = [np.pi, 0, 0]
    poses[3, 3:] = [0, np.pi - 1e-6, 0]
    converted = [reference.pose6_to_pos_rot_np(pose) for pose in poses]
    arrays = {
        "rest14": REST14,
        "left_pos": lp,
        "left_rot": lr,
        "right_pos": rp,
        "right_rot": rr,
        "expected_joints": q_out,
        "reset_before": reset_before,
        "expected_fail_count": failures,
        "fk_joints": fk_q,
        "fk_positions": fk_pos,
        "fk_rotations": fk_rot,
        "pose6": poses,
        "pose_positions": np.stack([p for p, _ in converted]),
        "pose_rotations": np.stack([r for _, r in converted]),
    }
    provenance = {
        "legacy_commit": LEGACY_COMMIT,
        "reference_module": "shiraz_axol/xr1/ik.py",
        "runtime_versions": versions,
        "runtime_platform": runtime_platform(),
        "config": asdict(solver.config),
        "source_sha256": source_hashes(source),
        "stream": {
            "fps": 30,
            "regular_steps": 300,
            "unreachable_steps": 30,
            "reset_then_regular_steps": 30,
            "total_steps": n,
        },
        "fk_reference": checkpoint_fk.FK_VERSION,
        "fk_sample_count": len(fk_q),
        "fk_seed": 20260902,
        "pose_conversion_count": len(poses),
        "hardware_access": False,
    }
    return arrays, provenance


def replay_reference(source: Path):
    """Compute native legacy expectations using the committed stream inputs.

    Cross-architecture floating point equality is not promised by MuJoCo/DAQP
    or NumPy. Equality between the independent old and new solver on the same
    platform remains exact, including recurrent previous-solution seeds.
    """
    versions = runtime_versions()
    arrays, original = load_fixture(COMMITTED_FIXTURES)
    reference, checkpoint_fk = load_reference(source)
    arrays = {name: arrays[name] for name in INPUT_NAMES}
    solver = reference.MinkIK()
    if asdict(solver.config) != original["config"]:
        raise ValueError("Legacy reference config differs from the committed baseline")
    rest = arrays["rest14"]
    solver.set_rest_posture(rest)
    solver.reset_tracking_state()
    q = rest.copy()
    joints, failures = [], []
    for tick in range(len(arrays["reset_before"])):
        if arrays["reset_before"][tick]:
            q = rest.copy()
            solver.reset_tracking_state()
        q = solver.solve(
            q,
            (arrays["left_pos"][tick], arrays["left_rot"][tick]),
            (arrays["right_pos"][tick], arrays["right_rot"][tick]),
        )
        joints.append(q.copy())
        failures.append(solver.fail_count)
    arrays["expected_joints"] = np.stack(joints)
    arrays["expected_fail_count"] = np.asarray(failures, dtype=np.int64)
    poses = [reference.pose6_to_pos_rot_np(pose) for pose in arrays["pose6"]]
    arrays["pose_positions"] = np.stack([p for p, _ in poses])
    arrays["pose_rotations"] = np.stack([r for _, r in poses])
    fk = [
        [
            checkpoint_fk.gripper_pose_from_joints(
                joints[side * 7 : (side + 1) * 7], name
            )
            for side, name in enumerate(("left", "right"))
        ]
        for joints in arrays["fk_joints"]
    ]
    arrays["fk_positions"] = np.asarray([[p for _, p in sample] for sample in fk])
    arrays["fk_rotations"] = np.asarray([[r for r, _ in sample] for sample in fk])
    provenance = {
        key: original[key]
        for key in (
            "legacy_commit",
            "reference_module",
            "source_sha256",
            "config",
            "stream",
            "fk_reference",
            "fk_sample_count",
            "fk_seed",
            "pose_conversion_count",
            "hardware_access",
        )
    }
    provenance.update(
        runtime_versions=versions,
        runtime_platform=runtime_platform(),
        input_fixture_sha256=original["fixture_sha256"],
    )
    return arrays, provenance


def compare_port(arrays):
    from almond_axol.policy.mink_ik import MinkIK, pose6_to_pos_rot_np

    solver = MinkIK()
    rest = arrays["rest14"]
    solver.set_rest_posture(rest)
    solver.reset_tracking_state()
    q = rest.copy()
    worst = 0.0
    for tick in range(len(arrays["expected_joints"])):
        if arrays["reset_before"][tick]:
            q = rest.copy()
            solver.reset_tracking_state()
        q = solver.solve(
            q,
            (arrays["left_pos"][tick], arrays["left_rot"][tick]),
            (arrays["right_pos"][tick], arrays["right_rot"][tick]),
        )
        np.testing.assert_array_equal(
            q, arrays["expected_joints"][tick], err_msg=f"tick {tick}"
        )
        assert solver.fail_count == arrays["expected_fail_count"][tick]
        worst = max(worst, float(np.max(np.abs(q - arrays["expected_joints"][tick]))))
    worst_pos, worst_rot = 0.0, 0.0
    for sample, joints in enumerate(arrays["fk_joints"]):
        for side, (pos, rotation) in enumerate(solver.fk(joints)):
            worst_pos = max(
                worst_pos,
                float(np.max(np.abs(pos - arrays["fk_positions"][sample, side]))),
            )
            worst_rot = max(
                worst_rot,
                float(np.max(np.abs(rotation - arrays["fk_rotations"][sample, side]))),
            )
    assert worst_pos <= 1e-5 and worst_rot <= 1e-4
    for sample, pose in enumerate(arrays["pose6"]):
        position, rotation = pose6_to_pos_rot_np(pose)
        np.testing.assert_array_equal(position, arrays["pose_positions"][sample])
        np.testing.assert_array_equal(rotation, arrays["pose_rotations"][sample])
    return {
        "solver_steps": len(arrays["expected_joints"]),
        "max_joint_difference_rad": worst,
        "fk_position_max_difference_m": worst_pos,
        "fk_rotation_max_difference": worst_rot,
        "pose_conversion_exact": True,
        "fail_count": solver.fail_count,
    }


def add_current_wire_reference(arrays, provenance, source: Path, robot_path: Path):
    """Use the independent legacy solver on the actual new server's wire data.

    The inverse rotation below is deliberately independent of the port's
    frames.py. Its geometry is separately checked against both real URDFs.
    """
    spec = importlib.util.spec_from_file_location("_xr1_wire_reference", robot_path)
    robot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = robot
    spec.loader.exec_module(robot)
    reference, _ = load_reference(source)
    solver = reference.MinkIK()
    rest = arrays["rest14"]
    solver.set_rest_posture(rest)
    solver.reset_tracking_state()
    q = rest.copy()
    n = len(arrays["expected_joints"])
    wire = np.empty((n, 2, 6), dtype=np.float32)
    positions = np.empty((n, 2, 3), dtype=np.float32)
    rotations = np.empty((n, 2, 3, 3), dtype=np.float32)
    expected = np.empty((n, 14), dtype=np.float32)
    inverse_yaw = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([-2.77556e-17, -6.93889e-18, 0.86])
    for tick in range(n):
        if arrays["reset_before"][tick]:
            q = rest.copy()
            solver.reset_tracking_state()
        row = {}
        for side in ("left", "right"):
            rotation, position = robot.pose_to_current_axol_root(
                arrays[f"{side}_rot"][tick], arrays[f"{side}_pos"][tick]
            )
            row.update(
                {
                    f"{side}_ee_pos": position,
                    f"{side}_ee_rotm": rotation,
                    f"{side}_gripper_pos": np.array([0.5]),
                }
            )
        action, _ = robot.targets_to_axol_action(row)
        targets = []
        for side, keys in enumerate((robot.LEFT_EE_KEYS, robot.RIGHT_EE_KEYS)):
            wire[tick, side] = [action[key] for key in keys]
            position, rotation = reference.pose6_to_pos_rot_np(wire[tick, side])
            positions[tick, side] = origin + inverse_yaw @ (position - origin)
            rotations[tick, side] = inverse_yaw @ rotation
            targets.append((positions[tick, side], rotations[tick, side]))
        q = solver.solve(q, *targets)
        expected[tick] = q
    arrays.update(
        {
            "current_wire_pose6": wire,
            "current_wire_positions": positions,
            "current_wire_rotations": rotations,
            "current_wire_joints": expected,
        }
    )
    provenance["current_wire_reference"] = {
        "producer": "applications/inference/src/inference/xr1/robot.py",
        "producer_sha256": hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        "description": "Actual current-frame XR-1 wire decoded by independent legacy NumPy "
        "function, inverse root rotation, then independent legacy solver.",
        "fail_count": solver.fail_count,
    }


def audit_wire_sensitivity(arrays, source: Path):
    """Compare actual old adapter packing with current-frame wire packing.

    Only two pure functions are AST-extracted from the old hardware adapter;
    its module-level runtime, cameras and CAN ownership are never imported.
    """
    reference, _ = load_reference(source)
    old_obs = importlib.import_module("_xr1_mink_reference.obs")
    adapter_path = source / "adapter.py"
    tree = ast.parse(adapter_path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"rotm_to_rotvec", "targets_to_axol_action"}
    ]
    axes = ("x", "y", "z", "rx", "ry", "rz")
    keys = [[f"{side}_ee.{axis}" for axis in axes] for side in ("left", "right")]
    namespace = {
        "np": np,
        "math": math,
        "obs": old_obs,
        "LEFT_EE_KEYS": keys[0],
        "RIGHT_EE_KEYS": keys[1],
        "LEFT_GRIPPER_KEY": "left_gripper.pos",
        "RIGHT_GRIPPER_KEY": "right_gripper.pos",
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(adapter_path), "exec"),
        namespace,
    )
    old_targets = []
    new_targets = []
    position_error = rotation_error = 0.0
    for tick in range(len(arrays["expected_joints"])):
        row = {}
        for side in ("left", "right"):
            row.update(
                {
                    f"{side}_ee_pos": arrays[f"{side}_pos"][tick],
                    f"{side}_ee_rotm": arrays[f"{side}_rot"][tick],
                    f"{side}_gripper_pos": np.array([0.5]),
                }
            )
        action, _ = namespace["targets_to_axol_action"](row)
        old = tuple(
            reference.pose6_to_pos_rot_np([action[key] for key in arm_keys])
            for arm_keys in keys
        )
        new = tuple(
            (
                arrays["current_wire_positions"][tick, side],
                arrays["current_wire_rotations"][tick, side],
            )
            for side in range(2)
        )
        old_targets.append(old)
        new_targets.append(new)
        for (op, ore), (np_, nr) in zip(old, new, strict=True):
            position_error = max(position_error, float(np.max(np.abs(op - np_))))
            rotation_error = max(rotation_error, float(np.max(np.abs(ore - nr))))

    def new_solver(config=None):
        solver = reference.MinkIK() if config is None else reference.MinkIK(config)
        solver.set_rest_posture(arrays["rest14"])
        solver.reset_tracking_state()
        return solver

    results = {}
    for recurrent in (False, True):
        old_solver, new_solver_ = new_solver(), new_solver()
        old_q = arrays["rest14"].copy()
        new_q = old_q.copy()
        differences = []
        for tick, (old, new) in enumerate(zip(old_targets, new_targets, strict=True)):
            if arrays["reset_before"][tick]:
                old_q = arrays["rest14"].copy()
                new_q = old_q.copy()
                old_solver.reset_tracking_state()
                new_solver_.reset_tracking_state()
            # Shared reference seeds isolate each call's output sensitivity.
            new_q = new_solver_.solve(new_q if recurrent else old_q, *new)
            old_q = old_solver.solve(old_q, *old)
            differences.append(float(np.max(np.abs(new_q - old_q))))
        results[
            "independent_recurrent_seeds" if recurrent else "shared_old_reference_seed"
        ] = {
            "max_joint_difference_rad": max(differences),
            "worst_tick": int(np.argmax(differences)),
            "regular_300_max_rad": max(differences[:300]),
            "unreachable_30_max_rad": max(differences[300:330]),
            "after_reset_30_max_rad": max(differences[330:]),
            "ticks_over_1e_4_rad": sum(value > 1e-4 for value in differences),
            "old_fail_count": old_solver.fail_count,
            "new_fail_count": new_solver_.fail_count,
        }
    first_tick = []
    for iterations in (1, 2, 3, 4):
        for collision in (False, True):
            config = replace(
                reference.MinkIKConfig(),
                mink_iterations=iterations,
                mink_collision=collision,
            )
            old_solver, new_solver_ = new_solver(config), new_solver(config)
            old_q = old_solver.solve(arrays["rest14"], *old_targets[0])
            new_q = new_solver_.solve(arrays["rest14"], *new_targets[0])
            first_tick.append(
                {
                    "iterations": iterations,
                    "collision": collision,
                    "max_joint_difference_rad": float(np.max(np.abs(old_q - new_q))),
                }
            )
    return {
        "synthetic_not_field_measurement": True,
        "old_adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "old_packing": "AST-extracted original rotm_to_rotvec/targets_to_axol_action; "
        "original obs module",
        "wire_position_max_difference_m": position_error,
        "wire_rotation_max_difference": rotation_error,
        "streams": results,
        "first_tick_collision_diagnostic": first_tick,
        "diagnostic_only": "Production solver collision settings and iteration count "
        "remain unchanged.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--write-fixture", type=Path)
    parser.add_argument(
        "--replay-fixture",
        action="store_true",
        help="Replay exact committed inputs to create architecture-native expectations",
    )
    parser.add_argument(
        "--xr1-robot",
        type=Path,
        help="Optionally record actual current-frame XR-1 wire parity",
    )
    parser.add_argument(
        "--audit-wire",
        action="store_true",
        help="Compare old/new wire quantization (requires --xr1-robot)",
    )
    args = parser.parse_args()
    generate = replay_reference if args.replay_fixture else generate_reference
    arrays, provenance = generate(args.legacy_root.resolve())
    report = compare_port(arrays)
    provenance["port_comparison"] = report
    if args.xr1_robot is not None:
        add_current_wire_reference(
            arrays, provenance, args.legacy_root.resolve(), args.xr1_robot
        )
    if args.audit_wire:
        if args.xr1_robot is None:
            parser.error("--audit-wire requires --xr1-robot")
        report["wire_sensitivity"] = audit_wire_sensitivity(
            arrays, args.legacy_root.resolve()
        )
    if args.write_fixture is not None:
        args.write_fixture.mkdir(parents=True, exist_ok=True)
        fixture = args.write_fixture / "stream.npz"
        np.savez_compressed(fixture, **arrays)
        provenance["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        (args.write_fixture / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
