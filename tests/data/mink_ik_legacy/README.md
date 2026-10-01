# Independent legacy Mink reference

`reference-b32002c.tar.gz` contains the numerical reference used to reproduce
the committed fixture on another CPU architecture. It is a byte-for-byte
snapshot of these files from `shirazai/axol` commit
`b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b`:

- `shiraz_axol/xr1/ik.py`, `ik_config.py`, `ik_mink_backend.py`,
  `ik_mujoco_model.py`, `vendor_io.py`, and `fk.py`;
- the complete `shiraz_axol/xr1/assets/` directory;
- the repository's root `LICENSE`.

The archive additionally carries `LICENSE-Xiaomi-Robotics-1.txt`, copied
from Xiaomi-Robotics-1's Apache-2.0 license. The original reference
`vendor_io.py` preserves its Xiaomi copyright and source attribution.

Archive SHA-256:
`6026a1dd09158fed38f2786b19ef6c209b559a7df4e97bbbb52000eb54bd2b9d`.

`provenance.json` pins every reference source and asset SHA-256. The replay
tool checks those hashes before loading the reference under an isolated
namespace. It also checks the numerical dependency versions and unchanged
joint/body naming tables. The production solver is not used as its own
reference. The archive allows public CI to run these checks without access
to the private source repository.

To generate native expectations from exactly the committed inputs:

```bash
mkdir -p /tmp/mink-legacy-source
tar -xzf tests/data/mink_ik_legacy/reference-b32002c.tar.gz \
  -C /tmp/mink-legacy-source
PYTHONPATH=. python tests/tools/mink_ik_legacy_parity.py \
  --legacy-root /tmp/mink-legacy-source/shiraz_axol/xr1 \
  --replay-fixture --write-fixture /tmp/mink-native
AXOL_MINK_LEGACY_FIXTURE=/tmp/mink-native python -m pytest \
  tests/test_mink_ik_parity.py
```

The reference solver and URDF retain the repository's MIT license
(Copyright 2026 Almond AI, Inc.). The XR-1 numerical helpers are derived
from Xiaomi-Robotics-1, commit `7c20088b5328`, under Apache-2.0
(Copyright 2026 Xiaomi Corporation). The yam gripper meshes retain
`assets/meshes/LICENSE-yam_linear4310.txt` and their documented attribution.

Archive member names are sorted; file metadata and the gzip timestamp are
zeroed. Recreating it from the pinned files and bundled license produces the
same archive hash. Do not update it when changing the production controller:
a reference update needs an explicit new source revision and parity review.
