from cython_bbox import bbox_overlaps
import numpy as np

gt = np.random.random((5, 4))
dt = np.random.random((10, 4))

overlaps = bbox_overlaps(
        np.ascontiguousarray(dt, dtype=np.float32),
        np.ascontiguousarray(gt, dtype=np.float32)
    )

print(overlaps)