# Stage19 learned evidence adapter

This directory is new and immutable with respect to Stage1--18.  Run the
pipeline in order: cohort, observations, evidence, validation audit, CUDA
smoke, one formal training, five-arm test inference, retrospective evaluation.
Every writer refuses a non-empty output directory.

The continuous evidence is the frozen Stage17A full-trace binary impedance
score.  It is a normalized binary-property evidence channel, not calibrated
`P(label9)`.  The observation model is noiseless and inverse-crime.
