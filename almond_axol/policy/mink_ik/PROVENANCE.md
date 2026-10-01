# Legacy XR-1 Mink solver

The four `ik*.py` files and the entire `assets/` directory are copied without
numerical changes from `shiraz_axol/xr1/` at commit
`b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b` of `shirazai/axol`.
`vendor_io.py` contains only that revision's exact `aa2rotm` function; its
unrelated image and model-normalization helpers are not needed here.

This package has no JAX, jaxlie, pyroki, camera, or CAN dependency. The
qpsolvers dependency probes optional solvers and can import an installed
JAX during discovery, but Mink/DAQP does not execute it; the real warmup
test runs with all those optional imports unavailable. The package exposes
`MinkIK`, `MinkIKConfig`, and `pose6_to_pos_rot_np`.
`MinkIK.model` and `.tracker` expose the native MuJoCo model and tracker;
the solver's mutable data is private and belongs to one control thread.

The numerical runtime validated by the legacy deployment uses
`mujoco==3.11.0`, `mink==1.2.0`, and DAQP through qpsolvers. The checked-in
parity fixture also records the exact DAQP, qpsolvers, and NumPy versions
used to generate it. Earlier MuJoCo versions produce different collision
constraint rows and are not equivalent merely because the imports succeed.

The pinned URDF and meshes originate at `80e7a8c`, as documented in
`assets/README.md`. Keep the yam mesh license with the assets. The solver
operates in the checkpoint's yaw-zero root frame. Do not replace this model
with the current Axol URDF: its world frame is rotated by 90 degrees.
`pose6_current_to_legacy` performs the inverse frame transform for current
Cartesian wire commands, after the original NumPy pose decoder. This adapter
is outside the verbatim solver files.

`tests/data/mink_ik_legacy/provenance.json` records source and asset hashes.
The fixture includes a 30 Hz two-arm stream, unreachable-target steps,
tracking resets, and independent checkpoint-FK samples. Regenerate or
compare against an explicit legacy source checkout using
`tests/tools/mink_ik_legacy_parity.py`.

## Architecture-native exact parity

The committed fixture was generated on Linux x86_64. MuJoCo/DAQP and NumPy
do not promise identical floating point output across CPU architectures.
The first ARM CI run differed from that fixture by up to 4.96e-5 rad at the
first solver tick and 1.1920929e-7 in pose rotation elements. This does not
establish a difference between the old and new solver on ARM.

CI extracts the committed `tests/data/mink_ik_legacy/reference-b32002c.tar.gz`
snapshot of the immutable legacy commit above outside the working tree. This
minimal independent reference contains the six numerical source modules and
their complete assets; it avoids requiring access to the private source fork
from upstream pull-request CI. The generator verifies every legacy source
and asset against the committed hashes before importing it, and checks the
shared SDK joint/body name tables against their pinned values. With
`--replay-fixture`, it feeds the exact committed target poses, starting joints,
reset markers, FK samples, and rotation samples to that independent solver
on the runner. The new solver must then match those native expectations
exactly, including all 360 recurrent joint solutions and pose conversions.
The existing FK tolerances remain unchanged.

`AXOL_MINK_LEGACY_FIXTURE` selects the resulting directory for pytest. Its
archive hash, input-fixture hash, individual input arrays/dtypes, legacy
source hashes, configuration, numerical dependency versions, and platform
are validated before use. The committed x86 archive hash and copied source
and asset hashes are still tested on every platform. Neither tests nor the
generator download a reference or fall back to the current implementation.
CI publishes the native archive and provenance as an artifact.

To run exact parity on another architecture, extract the pinned reference and
run from the Axol root (use destinations outside this repository):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export AXOL_MINK_LEGACY_FIXTURE=/tmp/axol-mink-native
mkdir -p /tmp/axol-mink-legacy
tar -xzf tests/data/mink_ik_legacy/reference-b32002c.tar.gz -C /tmp/axol-mink-legacy
PYTHONPATH=. .venv/bin/python tests/tools/mink_ik_legacy_parity.py \
  --legacy-root /tmp/axol-mink-legacy/shiraz_axol/xr1 \
  --replay-fixture --write-fixture "$AXOL_MINK_LEGACY_FIXTURE"
.venv/bin/python -m pytest tests/test_mink_ik_parity.py
```

Linux x86_64 tests can use the committed fixture directly without a legacy
checkout. Other platforms require an explicit native fixture; missing or
incompatible provenance fails rather than weakening or skipping parity.

## Wire quantization and collision sensitivity

The direct 360-step solver comparison on the same platform is bit exact.
The independent legacy solver also produces bit-exact output for the same decoded current-frame
wire targets. The current URDF's gripper FK is the pinned FK rotated by
90 degrees (24 random samples: maximum position difference 4.44e-16 m,
rotation-matrix difference 1.05e-15). The actual XR-1 packing/inverse path
matches the pinned FK to 1e-6 after float32 wire conversion.

This does **not** mean the previous and new wire encodings produce identical
joint trajectories. Encoding axis-angle in different world frames introduces
float32 rotation differences even after undoing the frame transform. The
recorded synthetic 360-step audit compares the original adapter's actual
packing functions and the independent legacy solver against the new wire
encoding; it measures no position difference and a maximum rotation-matrix
difference of 1.3411e-7. The collision constraints can amplify this rounding.

| Comparison | Largest joint difference |
| --- | --- |
| Same reference seed for each tick, first 300 smooth targets | 0.0154244 rad |
| Same reference seed, all 360 including unreachable targets | 0.0181587 rad |
| Independent recurrent previous-solution seeds, all 360 | 0.0239702 rad |
| First tick, one IK iteration with collision constraints | 1.7487e-9 rad |
| First tick, two IK iterations with collision constraints | 0.00458157 rad |
| First tick, four IK iterations with collision constraints | 0.00227177 rad |
| First tick, four IK iterations with collision disabled diagnostically | 2.9802e-8 rad |

All default-configuration streams reported zero solver failures. The
diagnostic iteration/collision variants localize the amplification to the
collision-constrained multi-iteration solve; they do not identify a specific
MuJoCo contact row or prove behavior on hardware. Production collision
settings, iteration counts, and safety limits are unchanged. These are
synthetic numerical tests, without a robot plant or Rust actuator filter.

`provenance.json` includes the complete audit. Reproduce from the Axol root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  tests/tools/mink_ik_legacy_parity.py \
  --legacy-root /path/to/shiraz_axol/xr1 \
  --xr1-robot ../../applications/inference/src/inference/xr1/robot.py \
  --audit-wire
```

Add `--write-fixture tests/data/mink_ik_legacy` only to regenerate the checked-in
reference fixture and provenance. The old adapter's two pure packing functions
are extracted by AST; its hardware-owning module is not imported.
