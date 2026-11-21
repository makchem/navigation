import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from logger import RunLogger


class RunLoggerTest(unittest.TestCase):
    def test_full_schema_success_false_flush_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLogger(tmp) as log:
                self.assertTrue(log.path.exists())
                gt = SimpleNamespace(lat=51.0688, lon=13.7198,
                                     yaw_rad=1.2, alt_m=60.0)
                good = SimpleNamespace(lat=51.06881, lon=13.71981,
                    heading_rad=1.21, sigma_m=0.7, ok=True,
                    n_inliers=19, n_total=25, inlier_ratio=.76)
                nofix = SimpleNamespace(lat=0.0, lon=0.0, heading_rad=0.0,
                    sigma_m=float('inf'), ok=False,
                    n_inliers=3, n_total=15, inlier_ratio=.2)
                log.log_tick(t_sim=12.3, frame_id=0, gt=gt, fix=good,
                             sent_to_ekf=True, gps_state='off')
                log.log_tick(t_sim=12.5, frame_id=1, gt=gt, fix=nofix,
                             sent_to_ekf=False, gps_state='ok')
                for i in range(2, 10):
                    log.log_tick(t_sim=12.5+.2*(i-1), frame_id=i,
                                 gt=gt, fix=None, gps_state='ok')
                with log.path.open(newline='') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 10)
            with log.path.open(newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(tuple(reader.fieldnames), RunLogger.FIELDS)
                rows = list(reader)
            self.assertEqual(rows[0]['fix_lat'], '51.06881')
            self.assertEqual(rows[0]['ok'], 'True')
            self.assertEqual(rows[0]['sent_to_ekf'], 'True')
            self.assertEqual(rows[1]['fix_lat'], '')
            self.assertEqual(rows[1]['fix_lon'], '')
            self.assertEqual(rows[1]['fix_heading'], '')
            self.assertEqual(rows[1]['sigma_m'], 'inf')
            self.assertEqual(rows[1]['n_total'], '15')
            self.assertEqual(rows[1]['n_inliers'], '3')
            self.assertEqual(rows[1]['ok'], 'False')
            self.assertEqual(rows[2]['gt_alt'], '60.0')

    def test_one_file_per_run_and_optional_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLogger(tmp) as one:
                p1 = one.path
                one.log_branch(t_sim=5.0, frame_id=0, branch='texture',
                    n_keypoints=50, n_matches=14, n_inliers=10,
                    rms_px=1.2, time_ms=24.7)
                self.assertTrue(one.branch_path.exists())
            with RunLogger(tmp) as two:
                self.assertNotEqual(p1, two.path)
                self.assertFalse(two.path.with_name(two.path.stem + '_matcher.csv').exists())

    def test_invalid_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLogger(tmp) as log:
                log.log_tick(t_sim=0., frame_id=0, gt=None, fix=None)
                with self.assertRaises(ValueError):
                    log.log_tick(t_sim=0.2, frame_id=0, gt=None, fix=None)
                with self.assertRaises(ValueError):
                    log.log_tick(t_sim=0.3, frame_id=1, gt=None, fix=None,
                                 branch_won='made_up')

if __name__ == '__main__':
    unittest.main()
