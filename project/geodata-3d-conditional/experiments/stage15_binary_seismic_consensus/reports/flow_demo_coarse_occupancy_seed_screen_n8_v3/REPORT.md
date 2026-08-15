# Stage15 exploratory Flow demonstration

This is a post-hoc, truth-informed visualization selection from all eight strictly paired seeds. It demonstrates a possible mechanism and is not evidence of generalization.

- Positive label9 IoU pairs: 8/8
- Positive label9 recall pairs: 8/8
- Median delta IoU / precision / recall: +0.019956 / +0.061548 / +0.025033
- Median global mIoU delta: -0.013113
- Selected seed: 142
- Label9 IoU: 0.018964 -> 0.053887 (+0.034923)
- Label9 precision: 0.073012 -> 0.139791
- Label9 recall: 0.024978 -> 0.080620 (+0.055642)
- Centroid distance: 22.133287 -> 12.452924
- Label9 connected components: 45 -> 890
- Largest-component fraction: 0.729791 -> 0.188322
- Hard-condition violations: 0 / 0

The coarse occupancy formulation removes the previous repeated-block target, but it does not equal Phase1: localization improves while the hard decoded label9 becomes excessively fragmented. The Phase1-style figure uses the same fixed-camera surface rendering so that this difference remains visible rather than being hidden by plotting choices.
