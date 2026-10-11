"""Camera rotation from images (ORB features + RANSAC similarity on the 1/4-scale frames)."""
import cv2
import numpy as np


def _similarity_angle(ka, da, kb, db, bf, min_inliers):
    if da is None or db is None:
        return None
    m = bf.match(da, db)
    if len(m) < min_inliers:
        return None
    a = np.float32([ka[x.queryIdx].pt for x in m])
    b = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inl.sum() < min_inliers:
        return None
    return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def frame_rotation(small):
    """Rotation phi_t (deg) between consecutive frames, content p_t ~ R(phi) p_(t-1) + T; 0 where unreliable."""
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    phi = np.zeros(len(small))
    prev = orb.detectAndCompute(np.ascontiguousarray(small[0]), None)
    for t in range(1, len(small)):
        cur = orb.detectAndCompute(np.ascontiguousarray(small[t]), None)
        a = _similarity_angle(*prev, *cur, bf, 12)
        phi[t] = a if a is not None else 0.0
        prev = cur
    return phi


class PairRotation:
    """Rotation between two (non-consecutive) views of the same ground, e.g. an earlier pass and the current frame."""

    def __init__(self, small, min_deg):
        self.small, self.min_deg = small, min_deg
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.cache = {}

    def _features(self, f):
        if f not in self.cache:
            self.cache[f] = self.orb.detectAndCompute(np.ascontiguousarray(self.small[f]), None)
        return self.cache[f]

    def __call__(self, s, t):
        a = _similarity_angle(*self._features(s), *self._features(t), self.bf, 15)
        return a if a is not None and abs(a) >= self.min_deg else 0.0
