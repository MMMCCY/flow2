# Phase 6Q D5 StructuralGeo-native audit

Native history: five IntrusionSpec(kind=hemisphere) events; audit label 9 is their recorded union, not a dike assumption.
Prior support: 8/8 samples contain target voxels beyond exact conditions; size-compatible component fraction 0.875.

| Control | Mode | Best/final hard attainment | Final soft attainment | Final target IoU |
|---|---|---:|---:|---:|
| correct | BASE | 0.0000/0.0000 | 0.0000 | 0.0556 |
| correct | MATCHED_NORM_PHYSICS_ONLY | 0.0956/-14.6162 | 0.6709 | 0.0177 |
| correct | BASE_PLUS_PHYSICS | 0.3148/0.1444 | 0.1256 | 0.0557 |
| correct | LATE_PHYSICS | 0.1247/0.0898 | 0.0948 | 0.0555 |
| zero | BASE | 0.0000/0.0000 | 0.0000 | 0.0556 |
| zero | MATCHED_NORM_PHYSICS_ONLY | 0.0976/-18.7008 | 0.9438 | 0.0103 |
| zero | BASE_PLUS_PHYSICS | 0.2958/0.2731 | 0.2557 | 0.0558 |
| zero | LATE_PHYSICS | 0.1381/0.1056 | 0.1156 | 0.0554 |
| shuffled_xy | BASE | 0.0000/0.0000 | 0.0000 | 0.0556 |
| shuffled_xy | MATCHED_NORM_PHYSICS_ONLY | 0.0949/-18.5765 | 0.8847 | 0.0107 |
| shuffled_xy | BASE_PLUS_PHYSICS | 0.3156/0.1883 | 0.1693 | 0.0527 |
| shuffled_xy | LATE_PHYSICS | 0.1394/0.1065 | 0.1139 | 0.0560 |

Correct-control hard-specificity margin over the strongest control: -0.128703.
