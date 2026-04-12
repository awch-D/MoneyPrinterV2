import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from video_motion import plan_transition_durations


class VideoMotionTests(unittest.TestCase):
    def test_transition_preserves_total_duration(self) -> None:
        durs = [1.5, 2.0, 2.5, 1.8]
        for mode in ("page_flip", "random_page_flip"):
            adj, trans = plan_transition_durations(
                durs, mode, 1.0, 0.3, random_seed=42, min_flip_seconds=0.1
            )
            self.assertEqual(len(adj), len(durs))
            self.assertEqual(len(trans), len(durs) - 1)
            self.assertAlmostEqual(
                sum(adj) + sum(trans), sum(durs), places=5, msg=mode
            )

    def test_none_leaves_unchanged(self) -> None:
        durs = [1.0, 2.0]
        adj, trans = plan_transition_durations(durs, "none", 0.5, 0.4, 0)
        self.assertEqual(adj, durs)
        self.assertEqual(trans, [0.0])


if __name__ == "__main__":
    unittest.main()
