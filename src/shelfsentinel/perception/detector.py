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


class PoseDetector:
    def __init__(self, model="yolo11n-pose.pt", tracker="bytetrack.yaml", conf=0.4, keypoint_conf=0.5):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.tracker = tracker
        self.conf = conf
        self.keypoint_conf = keypoint_conf

    def track(self, video_path):
        # yields the raw yolo result for every frame, ids stay the same across frames
        return self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            classes=[0],  # people only
            verbose=False,
        )

    def get_people(self, result, t):
        # turns one yolo result into a list of Person
        boxes = result.boxes
        if boxes is None or boxes.id is None:  # nobody tracked in this frame
            return []

        ids = boxes.id.int().tolist()
        bboxes = boxes.xyxy.tolist()
        kp_xy = result.keypoints.xy.tolist()
        kp_conf = result.keypoints.conf.tolist()

        people = []
        for i, track_id in enumerate(ids):
            keypoints = {}
            for name, idx in KEYPOINTS.items():
                if kp_conf[i][idx] >= self.keypoint_conf:  # skip points yolo isnt sure about
                    keypoints[name] = [round(v, 1) for v in kp_xy[i][idx]]
            bbox = [round(v, 1) for v in bboxes[i]]
            people.append(Person(track_id, round(t, 3), bbox, keypoints))
        return people
