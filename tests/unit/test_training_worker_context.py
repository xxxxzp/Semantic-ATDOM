from argparse import Namespace
import unittest

from tools.train_progressive import copy_dataset_context


class TrainingWorkerContextTests(unittest.TestCase):
    def test_preflight_context_is_copied_to_independent_worker_arguments(self):
        source = Namespace(
            dataset_cache_root="/output/cache/dataset",
            dataset_preflight_path="/output/cache/dataset/preflight.json",
            dataset_preflight_sha256="a" * 64,
            semantic_authoritative_exact_complete=False,
        )
        worker = Namespace()

        copy_dataset_context(source, worker)

        self.assertEqual(worker.dataset_cache_root, source.dataset_cache_root)
        self.assertEqual(
            worker.dataset_preflight_path, source.dataset_preflight_path
        )
        self.assertEqual(
            worker.dataset_preflight_sha256,
            source.dataset_preflight_sha256,
        )
        self.assertIs(
            worker.semantic_authoritative_exact_complete,
            source.semantic_authoritative_exact_complete,
        )


if __name__ == "__main__":
    unittest.main()
