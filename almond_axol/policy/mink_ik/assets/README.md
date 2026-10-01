# Pinned Axol URDF (fork main 80e7a8c) — shiraz #550, RUSTCORE_DESIGN.md 5.2

`axol_fork_80e7a8c.urdf` and `meshes/` are byte-identical copies of
`almond_axol/kinematics/urdf/axol.urdf` and `almond_axol/kinematics/urdf/meshes/*`
at fork-main commit `80e7a8c` (`git show 80e7a8c:almond_axol/kinematics/urdf/axol.urdf`).
`shiraz_axol/tests/test_xr1_rt_ik_verbatim.py` re-checks that identity.

Why pin instead of loading the vendor tree's URDF:

- The XR-1 checkpoints were trained on EE poses from `shiraz_axol/xr1/fk.py`
  (`fk.axol_urdf_bimanual.numpy.v1`: yaw-0 root, EE = the `*_gripper` link
  origin). The IK must invert THAT FK.
- The chemical-speak URDF's only kinematic change is a +90 deg yaw on
  `fixed_node_to_root_joint_0` (arm chain bit-identical, FK to 4.4e-16 after
  `Rz(+90 deg)` about the 0.86 m root origin). An un-rotated fk.py v1 pose
  solved against it would be up to 1.119 m off — the hazard this pin removes.
- The chemical-speak tree also deletes the three `yam_linear4310_*.stl`
  meshes (+ their LICENSE) and the finger/TCP frames the fork tests target,
  so the asset directory must be self-contained.

Mesh references inside the URDF are `package://assembly/meshes/<name>.stl`;
`ik_mujoco_model.load_mj_model` injects `<compiler strippath="true"
meshdir=".../assets/meshes">` at load time, exactly as the fork loader does, so
the URDF text itself stays unmodified.

`meshes/LICENSE-yam_linear4310.txt` covers the three yam meshes (i2rt YAM).
