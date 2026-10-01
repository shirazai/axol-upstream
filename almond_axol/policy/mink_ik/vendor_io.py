"""Only the exact legacy Rodrigues conversion required by the Mink solver.

Copied from shiraz_axol/xr1/vendor_io.py at
b32002c0507db5ab03a421c9fb1f2ebf4b7fd49b, originally XR-1's
mibot/utils/io.py:125-132. Preserve float32 coercion and operation order.
"""

import numpy as np


def aa2rotm(axis_angle) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = float(np.linalg.norm(axis_angle))
    axis = axis_angle / (angle + 1e-10)
    x, y, z = axis.tolist()
    axis_hat = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float32)
    eye = np.identity(3, dtype=np.float32)
    return eye + np.sin(angle) * axis_hat + (1.0 - np.cos(angle)) * axis_hat @ axis_hat
