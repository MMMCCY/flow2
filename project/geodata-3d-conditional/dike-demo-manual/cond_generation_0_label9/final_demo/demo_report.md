# Dike Guidance Demo Report



This report summarizes post-processing artifacts for baseline vs guided dike-like target reconstruction.



Terminology: all geophysical fields are lightweight gravity-proxy fields.



- target_label: `9`

- target_label_source: `--target-label`

- density_config: `project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/density_config.json`

- observed_gravity: `project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/observed_gravity/observed_gravity.pt`

- recommended_for_demo: `true`

- observability_reason: recommended: target has measurable lightweight gravity-proxy response

- paired_by_seed: `true`

- paired_by_seed_reason: matching guided-sampler config with zero-guidance baseline

- baseline best geo_misfit: 0.8615928888320923

- guided best geo_misfit: 0.6609110832214355

- baseline best target_iou/recall/volume_error: 0.02631578966975212 / 0.028880463913083076 / -7835.0

- guided best target_iou/recall/volume_error: 0.026254434138536453 / 0.028880463913083076 / -7812.0



## Warnings



- none



## Artifacts



- `truth_label_qa/`: manual truth_model label QA figures and label summary

- `observed_gravity/`: observed lightweight gravity-proxy generated from truth_model + density_config when requested

- `case_selection/manifest.json`: legacy automatic target-label evidence when `--allow-auto-target-selection` is used

- `observability/summary.json`: target-label lightweight gravity-proxy observability

- `baseline_target/target_metrics.csv`: baseline target-label metrics

- `guided_target/target_metrics.csv`: guided target-label metrics

- `figures/`: ensemble probability and target realization figures

- `residuals/`: gravity-proxy residual comparison

- `sample_selection/selected_samples.csv`: samples selected for display

- `sweep/combined_guidance_response.png`: guidance sweep response when `guidance_sweep_summary.csv` exists



## Commands



```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/visualize_truth_model_labels.py --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/truth_label_qa --device cuda --boreholes project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/boreholes.pt
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_geophysics.py --samples-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/baseline_alpha0 --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/baseline_global_evaluation --kernel-size 9 --device cuda --boreholes project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/boreholes.pt --observed-gravity project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/observed_gravity/observed_gravity.pt --density-config project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/density_config.json
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_geophysics.py --samples-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/guided_alpha005 --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/guided_global_evaluation --kernel-size 9 --device cuda --boreholes project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/boreholes.pt --observed-gravity project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/observed_gravity/observed_gravity.pt --density-config project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/density_config.json
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/analyze_dike_observability.py --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --target-label 9 --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/observability --kernel-size 9 --device cuda --boreholes project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/boreholes.pt --density-config project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/density_config.json
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_target_feature.py --samples-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/baseline_alpha0 --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --target-label 9 --metrics-csv project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/baseline_global_evaluation/metrics.csv --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/baseline_target --device cuda
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_target_feature.py --samples-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/guided_alpha005 --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --target-label 9 --metrics-csv project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/guided_global_evaluation/metrics.csv --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/guided_target --device cuda
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/select_dike_demo_samples.py --baseline-metrics project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/baseline_global_evaluation/metrics.csv --guided-metrics project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/guided_global_evaluation/metrics.csv --baseline-target-metrics project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/baseline_target/target_metrics.csv --guided-target-metrics project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/guided_target/target_metrics.csv --baseline-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/baseline_alpha0 --guided-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/guided_alpha005 --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/sample_selection
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/visualize_dike_ensemble.py --baseline-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/baseline_alpha0 --guided-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/guided_alpha005 --truth-model project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/true_model.pt --target-label 9 --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/figures --device cuda --boreholes project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0/boreholes.pt --sample-id 12 --sample-id 13 --sample-id 7 --sample-id 15
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/compare_gravity_residuals.py --baseline-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/baseline_alpha0 --guided-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/guided_alpha005 --sample-id 12 --kernel-size 9 --output-dir project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/final_demo/residuals --device cuda --observed-gravity project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/observed_gravity/observed_gravity.pt --density-config project/geodata-3d-conditional/dike-demo-manual/cond_generation_0_label9/density_config.json
```
