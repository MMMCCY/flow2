# Stage20 continuous impedance adapter

Stage20 keeps the pretrained 3-D categorical Flow model frozen, reuses the
Phase4c multiclass seismic forward model and Phase5a log-impedance inversion
kernel, builds one deterministic borehole-derived low-frequency acoustic prior
per case, and trains only the validated one-channel residual velocity adapter.

The frozen order is reuse audit, cohort, observations, continuous evidence,
VAL evidence audit, engineering smoke, one formal training, one formal
inference, and retrospective evaluation.  A failed gate stops the protocol;
no Stage20 v1 parameter tuning or TEST-based selection is permitted.
