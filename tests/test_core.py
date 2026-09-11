import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from mcmot.export_html import export_standalone_html
from mcmot.config import select_output_root
from mcmot.geometry import bbox_xyxy, distance_to_polygon, iou, is_watch, point_in_polygon
from mcmot.global_match import union_duration
from mcmot.global_match import build_global
from mcmot.tracker import BBoxKalman, Track, assignment, paired_frames


class ExportHtmlTest(unittest.TestCase):
    def test_embeds_local_assets_and_keeps_external_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pixel.png").write_bytes(b"\x89PNG\r\n")
            source = root / "index.html"
            target = root / "index_export.html"
            source.write_text('<img src="pixel.png"><a href="https://example.com">external</a>', encoding="utf-8")
            export_standalone_html(source, target)
            exported = target.read_text(encoding="utf-8")
            self.assertIn('src="data:image/png;base64,', exported)
            self.assertIn('href="https://example.com"', exported)


class OutputSelectionTest(unittest.TestCase):
    def test_fresh_output_uses_next_number_and_latest_reuses_highest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "output_site"
            base.mkdir()
            (root / "output_site2").mkdir()
            (root / "output_site4").mkdir()
            cfg = {"output_root": str(base)}
            self.assertEqual(select_output_root(cfg, fresh=True)["output_root"], str(root / "output_site5"))
            self.assertEqual(select_output_root(cfg, fresh=False)["output_root"], str(root / "output_site4"))

    def test_unused_output_keeps_base_name(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "output_site"
            self.assertEqual(select_output_root({"output_root": str(base)}, fresh=True)["output_root"], str(base))


class GeometryTest(unittest.TestCase):
    def test_polygon_and_distance(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertTrue(point_in_polygon((0.5, 0.5), square))
        self.assertFalse(point_in_polygon((2, 2), square))
        self.assertAlmostEqual(distance_to_polygon((0.5, 0.5), square), 0.5)

    def test_iou(self):
        self.assertAlmostEqual(iou(bbox_xyxy(.5, .5, .2, .2), bbox_xyxy(.5, .5, .2, .2)), 1.0)

    def test_watch_condition(self):
        row = {"w": .2, "nose_conf": 1, "left_eye_conf": 1, "right_eye_conf": 1,
               "left_ear_conf": 1, "right_ear_conf": 1, "nose_x": .5, "nose_y": .45,
               "left_eye_x": .45, "left_eye_y": .4, "right_eye_x": .55, "right_eye_y": .4,
               "left_ear_x": .4, "left_ear_y": .45, "right_ear_x": .6, "right_ear_y": .45}
        cfg = {"keypoint_confidence": .05, "normal_angle_deg": 75, "small_bbox_width": .08,
               "small_angle_deg": 55}
        self.assertTrue(is_watch(row, cfg))


class TimeTest(unittest.TestCase):
    def test_cpp_style_threshold_not_backfilled(self):
        start = datetime(2026, 9, 4, 10)
        track = Track(1, BBoxKalman(np.array([.5, .5, .1, .2])), start, start, start)
        cfg = {"watch_threshold_seconds": 1, "attention_threshold_seconds": 3}
        for index in range(41):
            track.apply_watch(start + timedelta(seconds=index / 10), True, cfg)
        self.assertAlmostEqual(track.watch_total, 3.1)
        self.assertAlmostEqual(track.attention_total, 1.1)

    def test_union_duration(self):
        start = datetime(2026, 9, 4, 10)
        intervals = [(start, start + timedelta(seconds=5)),
                     (start + timedelta(seconds=3), start + timedelta(seconds=7))]
        self.assertEqual(union_duration(intervals), 7)


class TrackingRecoveryTest(unittest.TestCase):
    def test_bev_recovers_confirmed_track_after_bbox_jump(self):
        start = datetime(2026, 9, 4, 10)
        track = Track(1, BBoxKalman(np.array([.2, .2, .1, .2])), start, start, start,
                      hits=5, confirmed=True)
        track.kf.P = np.eye(8) * .001
        track.bev_history.append((start, .4, .4))
        detection = {"cx": .8, "cy": .8, "w": .1, "h": .2, "bev_x": .41, "bev_y": .4}
        matches, _, _ = assignment([track], [detection], .05, 13.28,
                                   start + timedelta(seconds=1), .025, .012)
        self.assertEqual(matches, [(0, 0)])

    def test_bev_recovery_does_not_attach_distant_detection(self):
        start = datetime(2026, 9, 4, 10)
        track = Track(1, BBoxKalman(np.array([.2, .2, .1, .2])), start, start, start,
                      hits=5, confirmed=True)
        track.kf.P = np.eye(8) * .001
        track.bev_history.append((start, .4, .4))
        detection = {"cx": .8, "cy": .8, "w": .1, "h": .2, "bev_x": .7, "bev_y": .7}
        matches, _, _ = assignment([track], [detection], .05, 13.28,
                                   start + timedelta(seconds=1), .025, .012)
        self.assertEqual(matches, [])


class IOTest(unittest.TestCase):
    def test_paired_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, bev = Path(tmp) / "raw.csv", Path(tmp) / "bev.csv"
            raw.write_text("timestamp,frame_index,confidence,cx,cy,w,h\n2026-09-04 10:00:00.000,0,.9,.5,.5,.1,.2\n", encoding="utf-8")
            bev.write_text("timestamp,frame_index,x,y\n2026-09-04 10:00:00.000,0,.4,.3\n", encoding="utf-8")
            groups = list(paired_frames(raw, bev))
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0][2][0]["bev_x"], .4)

    def test_global_association(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local, attrs, output = root / "local", root / "attrs.csv", root / "global"
            local.mkdir()
            fields = ["local_uid","date","recording_id","camera_id","local_id","first_seen","last_seen",
                      "last_observed","track_span_seconds","observed_seconds","predicted_gap_seconds",
                      "watch_seconds","attention_seconds","detection_count","start_boundary_distance",
                      "end_boundary_distance","start_bev_x","start_bev_y","end_bev_x","end_bev_y",
                      "crop_watch_condition","crop_area","crop_path","quality_flags"]
            base = {"date":"2026-09-04","local_id":"1","first_seen":"2026-09-04 10:00:00",
                    "last_seen":"2026-09-04 10:00:05","last_observed":"2026-09-04 10:00:05",
                    "track_span_seconds":"5","observed_seconds":"5","predicted_gap_seconds":"0",
                    "watch_seconds":"0","attention_seconds":"0","detection_count":"10",
                    "start_boundary_distance":".01","end_boundary_distance":".01","start_bev_x":".5",
                    "start_bev_y":".5","end_bev_x":".5","end_bev_y":".5","crop_watch_condition":"1",
                    "crop_area":"100","crop_path":"","quality_flags":""}
            people = [{**base,"local_uid":"2026-09-04:1:100000:L1","recording_id":"d_a_100000","camera_id":"1"},
                      {**base,"local_uid":"2026-09-04:2:100000:L1","recording_id":"d_b_100000","camera_id":"2",
                       "start_bev_x":".51","end_bev_x":".51"}]
            with (local/"persons.csv").open("w",newline="") as s:
                w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(people)
            (local/"watch_segments.csv").write_text("local_uid,segment_index,start,end,raw_duration_seconds,credited_watch_seconds,credited_attention_seconds\n")
            vectors=[]
            for index in range(2):
                path=root/f"e{index}.npy";np.save(path,np.zeros(4,dtype=np.float32));vectors.append(path)
            with attrs.open("w",newline="") as s:
                w=csv.DictWriter(s,fieldnames=["local_uid","reid_embedding_path"]);w.writeheader()
                for person,path in zip(people,vectors):w.writerow({"local_uid":person["local_uid"],"reid_embedding_path":path})
            cfg={"reid":{"distance_threshold":.9},"watch":{"watch_threshold_seconds":1,"attention_threshold_seconds":3},
                 "global_matching":{"sync_tolerance_seconds":.2,"simultaneous_bev_distance":.05,
                 "max_handoff_seconds":8,"max_bev_speed_per_second":.2,"boundary_quantile":.15,
                 "min_topology_observations":2,"weights":{"reid":.55,"bev":.3,"time":.15}}}
            build_global(local/"persons.csv",local/"tracks.csv",attrs,output,cfg)
            with (output/"persons.csv").open() as stream:
                result=list(csv.DictReader(stream))
            self.assertEqual(len(result),1)
            self.assertEqual(float(result[0]["exposure_seconds"]),5)


if __name__ == "__main__":
    unittest.main()
