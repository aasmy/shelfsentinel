from dataclasses import dataclass, field

# COCO keypoint indices we need
KEYPOINTS = {
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
}


@dataclass
class Person:
    track_id: int
    t: float  # seconds from start of video
    bbox: list  # [x1, y1, x2, y2]
    keypoints: dict = field(default_factory=dict)  # name -> [x, y], only the confident ones

    def to_dict(self):
        return {"track_id": self.track_id, "t": self.t, "bbox": self.bbox, "keypoints": self.keypoints}
