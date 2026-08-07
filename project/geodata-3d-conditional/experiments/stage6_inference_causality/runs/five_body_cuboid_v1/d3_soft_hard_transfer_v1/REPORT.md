# Phase 6Q D3 soft/hard transfer report

U-Net was not loaded. Truth geology was used only for post-update mechanism diagnostics.

| Level | Control | Max/final soft attainment | Max/final hard attainment | Final soft/hard closer | Best hard step |
|---|---|---:|---:|---|---:|
| probability | correct | 1.0000/1.0000 | 1.0000/1.0000 | truth/truth | 27 |
| expected_property | correct | 1.0000/1.0000 | 1.0000/1.0000 | truth/truth | 26 |
| blurred_property | correct | 1.0000/1.0000 | 1.0000/1.0000 | truth/truth | 166 |
| reflectivity_spikes | correct | 0.1080/0.0000 | 0.0000/0.0000 | baseline/baseline | 0 |
| reflectivity_spikes | zero | 0.2909/nan | nan/nan | baseline/baseline | 0 |
| reflectivity_spikes | shuffled_xy | 0.6496/nan | nan/nan | baseline/baseline | 0 |
| seismic | correct | 0.1909/0.1909 | 0.0244/-0.0080 | baseline/baseline | 126 |
| seismic | zero | 0.3140/nan | nan/nan | baseline/baseline | 0 |
| seismic | shuffled_xy | 0.8765/nan | nan/nan | baseline/baseline | 163 |
| gravity | correct | 0.0696/0.0000 | 0.0000/0.0000 | baseline/baseline | 0 |
| gravity | zero | 25.9918/25.5201 | 25.5371/25.5371 | truth/truth | 200 |
| gravity | shuffled_xy | 5.0514/5.0477 | 5.0475/5.0475 | truth/truth | 200 |
