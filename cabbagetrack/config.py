"""Settings of the final CabbageTrack model. All values were fixed on the development flights; nothing is tuned per field."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # detections
    plant_class: int = 1               # detector class index of the plant (Cabbage / Cauliflower)
    min_track_frames: int = 3          # a tracker track must be this old before it is given an identity
    # camera odometry (frame-to-frame phase correlation, computed by the extraction stage)
    max_frame_disp: float = 25.0       # px per frame; larger steps are treated as failed and set to 0
    small_scale: int = 4               # stored frames are 1/4 resolution (480 x 270)
    # "been here before?" lookup
    recent_gap: int = 45               # frames: only memory older than this counts as an earlier visit
    min_overlap: float = 0.25          # an earlier frame is a candidate when it shares >= 25 % of the image (map)
    visit_gap: int = 15                # frames: candidate frames separated by more than this are different visits
    max_visits: int = 4                # earlier visits checked per frame (closest first)
    n_link_candidates: int = 3         # frames verified per visit (closest on the map, >= 10 frames apart)
    # link verification (phase correlation of the overlapping parts, placed by the predicted shift)
    min_response: float = 0.08         # minimum phase-correlation peak
    shift_tol: float = 400.0           # px: measured shift must agree with the (drift-corrected) prediction
    # rotation and drift
    rot_min_deg: float = 1.0           # rotation between the two views (ORB similarity) applied when >= 1 deg
    drift_recent: int = 60             # frames: residuals of links made this recently correct the prediction ...
    drift_window: int = 300            # frames: ... when their earlier frame lies within this many frames
    drift_min_peak: float = 0.2        # ... and only links with a clear alignment peak update the drift correction
    turn_rate: float = 0.3             # deg/frame: frames rotating faster than this are inside a fast turn ...
    turn_margin: int = 15              # frames: ... together with this many frames on either side
    # matching remembered plants to new tracks
    window: int = 60                   # frames around the linked earlier frame whose plants are remembered
    gate: float = 60.0                 # px: maximum distance between a registered plant and a new track
    patience: int = 60                 # frames a track over known ground waits for a match before a new identity (settings check A+B+C)
    live_gap: int = 5                  # frames: identities seen this recently (still in view) are not candidates
    cross_pass_gap: int = 300          # frames (10 s): a link spanning more than this is a cross-pass revisit
