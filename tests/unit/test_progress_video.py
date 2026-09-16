from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from tools.train_progressive import DataSaver


class ProgressVideoTests(unittest.TestCase):
    def _saver(self, root, mode="camera", stride=1):
        saver = DataSaver()
        saver.SaveDirRootPath = str(Path(root) / ".work" / "video_frames")
        saver.MetricsRootPath = str(Path(root) / "metrics")
        saver.VisualizationRootPath = str(Path(root) / "visualizations")
        saver.ProgressVideo = mode
        saver.ProgressVideoStride = stride
        saver.ProgressVideoFPS = 4.0
        saver.MakeDirs()
        return saver

    def test_disabled_video_creates_no_placeholder_directories(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory, mode="none")

            self.assertFalse(Path(saver.SaveDirRootPath).exists())
            self.assertFalse((Path(directory) / "visualizations").exists())

    def test_camera_video_has_no_black_prefix_and_pads_odd_dimensions(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory)
            first = np.full((9, 15, 3), 20, dtype=np.uint8)
            second = np.full((9, 15, 3), 200, dtype=np.uint8)
            saver.SaveImage(first, "camera-a", 13)
            saver.SaveImage(second, "camera-b", 14)

            saver.GetDemo()

            video = Path(directory) / "visualizations" / "progressive_camera_views.mp4"
            capture = cv2.VideoCapture(str(video))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) % 2, 0)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) % 2, 0)
            ok, decoded = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(float(decoded.mean()), 1.0)
            self.assertFalse((Path(saver.SaveDirRootPath) / "camera").exists())

    def test_stride_keeps_last_frame(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory, stride=3)
            for index in range(5):
                saver.SaveImage(
                    np.full((8, 8, 3), index * 20, dtype=np.uint8),
                    "camera-{}".format(index),
                    index,
                )

            saver.GetDemo()

            video = Path(directory) / "visualizations" / "progressive_camera_views.mp4"
            capture = cv2.VideoCapture(str(video))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()

    def test_tdom_frames_with_growing_extents_use_one_canvas(self):
        with TemporaryDirectory() as directory:
            saver = self._saver(directory, mode="tdom")
            saver.SaveTDOM(
                np.full((8, 10, 3), 30, dtype=np.uint8),
                {"geometry_stage": 13},
            )
            saver.SaveTDOM(
                np.full((13, 17, 3), 60, dtype=np.uint8),
                {"geometry_stage": 14},
            )

            saver.GetDemo()

            video = Path(directory) / "visualizations" / "progressive_tdom.mp4"
            capture = cv2.VideoCapture(str(video))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 18)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 14)
            capture.release()


if __name__ == "__main__":
    unittest.main()
