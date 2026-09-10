from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def read(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def fit(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 242, np.uint8)
    if image is None: return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1]*scale)), max(1, int(image.shape[0]*scale))))
    y, x = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas


def generate_video(output_root: Path, cfg: dict) -> Path | None:
    vc = cfg["visualization"]
    if not vc.get("enabled", True): return None
    events = [x for x in read(output_root / "global/association_events.csv") if x.get("accepted") == "1"]
    events.sort(key=lambda x: float(x.get("association_score") or 999))
    persons = {x["local_uid"]: x for x in read(output_root / "local/persons.csv")}
    mapping = {x["local_uid"]: x["global_id"] for x in read(output_root / "global/id_mapping.csv")}
    bev = cv2.imread(cfg["bev_image"])
    if bev is None or not events: return None
    width, height, fps = int(vc["width"]), int(vc["height"]), int(vc["fps"])
    target = output_root / "visualization/best_matches.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    max_cases = int(vc["max_cases"])
    selected = []
    # Include both overlap de-duplication and topology handoff examples.
    for match_type in ("simultaneous_overlap", "handoff"):
        quota = max(1, max_cases // 2)
        selected.extend([event for event in events if event["match_type"] == match_type][:quota])
    for event in events:
        if len(selected) >= max_cases:
            break
        if event not in selected:
            selected.append(event)
    for event in selected[:max_cases]:
        left, right = persons[event["left_uid"]], persons[event["right_uid"]]
        left_img = cv2.imread(left.get("crop_path", "")); right_img = cv2.imread(right.get("crop_path", ""))
        frames = fps * int(vc["seconds_per_case"])
        for frame_no in range(frames):
            canvas = np.full((height,width,3), (18,29,42), np.uint8)
            map_panel = fit(bev, width//2, height-120)
            canvas[90:height-30, :width//2] = map_panel[:height-120]
            map_h, map_w = height-120, width//2
            p1 = (int(float(left["end_bev_x"])*map_w), 90+int(float(left["end_bev_y"])*map_h))
            p2 = (int(float(right["start_bev_x"])*map_w), 90+int(float(right["start_bev_y"])*map_h))
            progress = min(1.0, frame_no/max(1,frames*.65))
            current = (int(p1[0]+(p2[0]-p1[0])*progress), int(p1[1]+(p2[1]-p1[1])*progress))
            cv2.line(canvas,p1,p2,(65,190,225),3); cv2.circle(canvas,p1,10,(62,210,130),-1); cv2.circle(canvas,p2,10,(245,160,55),-1); cv2.circle(canvas,current,13,(255,255,255),3)
            crop_w=(width//2-60)//2; crop_h=300
            canvas[130:130+crop_h,width//2+20:width//2+20+crop_w]=fit(left_img,crop_w,crop_h)
            canvas[130:130+crop_h,width//2+40+crop_w:width-20]=fit(right_img,crop_w,crop_h)
            cv2.putText(canvas,"MCMOT MATCH EXPLAINER",(30,48),cv2.FONT_HERSHEY_SIMPLEX,1.05,(235,245,250),2)
            gid=mapping.get(event["left_uid"],"-")
            cv2.putText(canvas,f"Global ID  {gid}",(width//2+25,70),cv2.FONT_HERSHEY_SIMPLEX,.75,(80,220,170),2)
            cv2.putText(canvas,f"Camera {left['camera_id']}",(width//2+20,115),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,230,235),2)
            cv2.putText(canvas,f"Camera {right['camera_id']}",(width//2+40+crop_w,115),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,230,235),2)
            lines=[f"match type       {event['match_type']}",f"embedding L2     {float(event['reid_distance']):.4f}",f"accept threshold {float(event['reid_threshold']):.4f}",f"BEV distance     {float(event['bev_distance']):.4f}",f"time gap         {float(event['temporal_gap_seconds']):.2f} sec",f"combined score   {float(event['association_score']):.4f}","DECISION          SAME PERSON"]
            for i,line in enumerate(lines): cv2.putText(canvas,line,(width//2+25,475+i*30),cv2.FONT_HERSHEY_SIMPLEX,.58,(75,220,160) if i==6 else (215,225,232),2 if i==6 else 1)
            writer.write(canvas)
    writer.release()
    return target
