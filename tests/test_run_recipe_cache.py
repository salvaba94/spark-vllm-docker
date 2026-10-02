#!/usr/bin/env python3
"""A partial Hub snapshot must not be mistaken for a downloaded model."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

module_path = Path(__file__).resolve().parents[1] / 'run-recipe.py'
spec = importlib.util.spec_from_file_location('run_recipe', module_path)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class CacheCompletenessTests(unittest.TestCase):
    def test_incomplete_shards_and_transfers_are_not_cache_hits(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'HF_HUB_CACHE': temporary}):
            model = Path(temporary) / 'models--example--sharded'
            snapshot = model / 'snapshots' / 'revision'
            snapshot.mkdir(parents=True)
            (model / 'refs').mkdir()
            (model / 'refs' / 'main').write_text('revision')
            (snapshot / 'config.json').write_text('{}')
            first = snapshot / 'model-00001-of-00002.safetensors'
            second = snapshot / 'model-00002-of-00002.safetensors'
            first.write_bytes(b'weight')
            self.assertFalse(recipe.check_model_exists('example/sharded'))
            second.write_bytes(b'weight')
            self.assertTrue(recipe.check_model_exists('example/sharded'))
            (model / 'blobs').mkdir()
            (model / 'blobs' / 'transfer.incomplete').write_bytes(b'partial')
            self.assertFalse(recipe.check_model_exists('example/sharded'))


if __name__ == '__main__':
    unittest.main()
