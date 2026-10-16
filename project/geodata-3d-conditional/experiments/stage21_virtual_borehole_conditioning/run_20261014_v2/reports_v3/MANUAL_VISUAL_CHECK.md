# Visual inspection

The primary common-domain curves were opened and checked against the tables.
The earlier v2 case05 well map revealed a legend over the top-left well14;
v3 places the legend below the axes and the opened v3 map shows all25 markers.
This is a presentation correction only. Metric summary hashes agree across
reports_v1/v2/v3; no predictions were regenerated.

Opened images included:

- `figures/common_domain_dose_curves.png` (identical v2/v3 export)
- `figures/fullgeo_case05_seed142_targets_view1.png` (identical v2/v3 export)
- `figures/fullgeo_case05_well_target_hits.png` (v3 legend outside data)
- `figures/fullgeo_case02_seed142_targets_view2.png`
- `figures/fullgeo_case03_seed142_targets_view1.png`
- `figures/fullgeo_case01_seed142_targets_view1.png`

The curves retain case01's extreme volume error rather than dropping the case;
this compresses other cases on the linear volume-error axis, so exact values
are available in CSV. Case05's nonmonotonic target IoU remains visible.
Full-domain3D plots reveal widespread false positives, very poor case01/03
recovery, and case02's partial overlap rather than only cropping correct areas.
Truth/N9/N16/N25 use equal0..63 axes and fixed cameras at elevation22,
azimuth-60/145, for every case and fixed source142. Boundary-voxel scatter can
occlude interiors and is not a solid geological rendering; the second camera
and complete saved volumes remain necessary for structural inspection.

No result-dependent case or seed selection was introduced. These plots do not
establish reliable original hidden-major recovery or physical consistency.
