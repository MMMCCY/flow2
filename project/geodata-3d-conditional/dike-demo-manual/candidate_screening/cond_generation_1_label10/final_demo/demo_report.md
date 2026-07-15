# Dike Guidance Demo Report



This report summarizes post-processing artifacts for baseline vs guided dike-like target reconstruction.



Terminology: all geophysical fields are lightweight gravity-proxy fields.



- target_label: `10`

- target_label_source: `--target-label`

- density_config: `/home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/density_config.json`

- observed_gravity: `/home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/observed_gravity/observed_gravity.pt`

- recommended_for_demo: `true`

- observability_reason: recommended: target has measurable lightweight gravity-proxy response

- paired_by_seed: `true`

- paired_by_seed_reason: matching guided-sampler config with zero-guidance baseline

- baseline best geo_misfit: 0.19465568661689758

- guided best geo_misfit: 0.14726769924163818

- baseline best target_iou/recall/volume_error: 0.520257294178009 / 0.7833075523376465 / 2804.0

- guided best target_iou/recall/volume_error: 0.5237177014350891 / 0.783822774887085 / 2722.0



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
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/visualize_truth_model_labels.py --truth-model /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/true_model.pt --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/truth_label_qa --device cuda --boreholes /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/boreholes.pt
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/analyze_dike_observability.py --truth-model /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/true_model.pt --target-label 10 --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/observability --kernel-size 9 --device cuda --boreholes /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/boreholes.pt --density-config /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/density_config.json
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_target_feature.py --samples-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/baseline_alpha0 --truth-model /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/true_model.pt --target-label 10 --metrics-csv /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/screening/baseline_global_evaluation/metrics.csv --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/baseline_target --device cuda
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/evaluate_target_feature.py --samples-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/guided_alpha0_05 --truth-model /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/true_model.pt --target-label 10 --metrics-csv /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/screening/guided_global_evaluation/metrics.csv --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/guided_target --device cuda
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/select_dike_demo_samples.py --baseline-metrics /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/screening/baseline_global_evaluation/metrics.csv --guided-metrics /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/screening/guided_global_evaluation/metrics.csv --baseline-target-metrics /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/baseline_target/target_metrics.csv --guided-target-metrics /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/guided_target/target_metrics.csv --baseline-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/baseline_alpha0 --guided-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/guided_alpha0_05 --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/sample_selection
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/visualize_dike_ensemble.py --baseline-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/baseline_alpha0 --guided-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/guided_alpha0_05 --truth-model /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/true_model.pt --target-label 10 --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/figures --device cuda --boreholes /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_1/boreholes.pt --sample-id 14 --sample-id 4 --sample-id 5 --sample-id 12 --sample-id 13
```

```bash
/home/mcy/miniconda3/envs/geoflow/bin/python /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/compare_gravity_residuals.py --baseline-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/baseline_alpha0 --guided-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/guided_alpha0_05 --sample-id 4 --kernel-size 9 --output-dir /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/final_demo/residuals --device cuda --observed-gravity /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/observed_gravity/observed_gravity.pt --density-config /home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional/dike-demo-manual/candidate_screening/cond_generation_1_label10/density_config.json
```
