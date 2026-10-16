"""Small deterministic checks for scientific selection/counting contracts."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ensemble_occurrence import (counts_for, main, metrics, minimum_counts,
                                 normalize_labels, select_indices)


class OccurrenceTests(unittest.TestCase):
    def test_strict_and_inclusive_thresholds(self):
        self.assertEqual(minimum_counts(100), [41,61,81,100])
        self.assertEqual(minimum_counts(200), [81,121,161,200])
        self.assertEqual(minimum_counts(100,False), [40,60,80,100])
        self.assertEqual(minimum_counts(3), [2,2,3,3])
        count = np.array([40,41,60,61,80,81,99,100])
        self.assertEqual((count >= minimum_counts(100)[0]).tolist(),
                         [False,True,True,True,True,True,True,True])

    def test_hard_labels_and_counts(self):
        labels = np.array([[[[9,0]]],[[[9,9]]],[[[0,9]]]])
        count = counts_for(labels,np.array([0,1]),9)
        np.testing.assert_array_equal(count,np.array([[[2,1]]]))
        with self.assertRaises(ValueError):
            normalize_labels(labels.astype(float))

    def test_physics_ranking_has_no_truth_argument(self):
        scores = np.array([0.4,0.1,0.1,0.9])
        selected = select_indices(["a","b","c","d"],"physics",2,scores)
        np.testing.assert_array_equal(selected,[1,2])
        np.testing.assert_array_equal(select_indices(["a","b","c","d"],"oracle-iou",2,scores,higher=True),[3,0])

    def test_empty_prediction_is_not_perfect(self):
        empty = np.zeros((2,2,2),bool)
        truth = empty.copy()
        truth[0,0,0] = True
        self.assertEqual(metrics(empty,truth)["iou"],0)
        self.assertIsNone(metrics(empty,truth)["precision"])
        self.assertIsNone(metrics(empty,empty)["iou"])

    def test_truth_cannot_change_physics_selected_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            labels = np.zeros((4,3,3,3),np.int8)
            for i in range(4):
                labels[i,i%3,0,0] = 9
            np.save(root/"labels.npy",labels)
            (root/"ids.json").write_text(json.dumps(["a","b","c","d"]))
            (root/"provenance.json").write_text(json.dumps({"case_id":"toy","method_id":"test_only"}))
            (root/"score.csv").write_text("sample_id,hard_physics_rmse\na,4\nb,1\nc,2\nd,3\n")
            common = ["--labels",str(root/"labels.npy"),"--ids",str(root/"ids.json"),
                      "--provenance",str(root/"provenance.json"),"--target-label","9",
                      "--mode","physics","--score-csv",str(root/"score.csv"),
                      "--keep","2","--expected-count","4","--skip-plots"]
            main(common+["--out",str(root/"without_truth")])
            truth = np.zeros((3,3,3),np.int8)
            truth[2,0,0] = 9
            np.save(root/"truth.npy",truth)
            main(common+["--truth",str(root/"truth.npy"),"--out",str(root/"with_truth")])
            self.assertEqual((root/"without_truth/selected/selected_ids.json").read_text(),
                             (root/"with_truth/selected/selected_ids.json").read_text())

    def test_duplicate_ids_are_rejected(self):
        from ensemble_occurrence import read_ids
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p = Path(tmp)/"ids.json"
            p.write_text('["a","a"]')
            with self.assertRaises(ValueError):
                read_ids(p,2)


if __name__ == "__main__":
    unittest.main()
