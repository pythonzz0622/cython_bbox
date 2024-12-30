from cython_bbox import bbox_overlaps, bbox_ious
import numpy as np

gt = np.random.random((5, 4))
dt = np.random.random((10, 4))

# intersection over A
overlaps = bbox_overlaps(
        np.ascontiguousarray(dt, dtype=np.float32),
        np.ascontiguousarray(gt, dtype=np.float32)
    )

# intersection over union
ious = bbox_ious(
        np.ascontiguousarray(dt, dtype=np.float32),
        np.ascontiguousarray(gt, dtype=np.float32)
    )

print(overlaps)
print(ious)
