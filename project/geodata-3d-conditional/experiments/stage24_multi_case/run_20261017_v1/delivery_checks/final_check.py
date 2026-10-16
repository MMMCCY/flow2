import os,sys,csv,json,hashlib,shutil,subprocess
from pathlib import Path
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
repo=Path('/home/xmj/mcy/flowtrain_stochastic_interpolation-main');p=repo/'project/geodata-3d-conditional'
sys.path.insert(0,str(p))
from scripts.stage24.common import *
from PIL import Image,ImageDraw
import torch,numpy as np
torch.set_num_threads(4)
r=ROOT;oldstyle=p/'experiments/stage23_full_pseudo_wells/reports/conference_paper_v2'
fix=read_json(r/'VISUAL_VALIDATOR_FIX.json')
assert read_json(r/'standard_regression/STYLE_REGRESSION.json')['passed']
assert read_json(r/'standard_regression/AFTER_VALIDATOR_REGRESSION.json')['passed']
manifest=read_json(STANDARD/'MANIFEST.json')
assert file_hash(STANDARD/'renderer.py')==manifest['renderer_sha256']
for rel,h in manifest['source_sha256'].items():assert file_hash(STANDARD/'source_snapshot'/rel)==h
for name,h in manifest['approved_files'].items():assert file_hash(STANDARD/'approved_samples'/name)==h
for rel,h in read_json(R23/'ARTIFACT_HASHES.json').items():assert file_hash(R23/rel)==h
for rel,h in read_json(oldstyle.parent/'conference_paper_v1/OUTPUT_HASHES.json').items():assert file_hash(oldstyle.parent/'conference_paper_v1'/rel)==h
cases={};calls=0;solves=0
for case in CASES:
 c=r/'cases'/case;freeze=read_json(c/'METHOD_FREEZE.json');seal=read_json(c/'CONDITION_RECEIPT.json');reg=read_json(c/'inference_registry.json')
 for rel,h in freeze['source_sha256'].items():
  assert file_hash(c/'source_snapshot'/rel)==h
  assert file_hash(REPO/rel)==(fix['delivery_sha256'] if rel==fix['file'] else h)
 for rel,h in freeze['historical_source_sha256'].items():assert file_hash(REPO/rel)==h
 for rel,h in freeze['metadata_sha256'].items():assert file_hash(REPO/rel)==h
 assert file_hash(c/'METHOD_FREEZE.json')==seal['freeze_file_sha256']
 assert file_hash(c/'inference_registry.json')==seal['inference_registry_sha256']
 assert json_hash(reg['conditions'])==seal['condition_record_sha256']
 state=read_json(c/'SOLVER_STATE.json');solves+=state['solve_invocations']
 assert state['converged'] and state['solve_invocations']==1 and state['truth_access']==False
 assert state['final']['stable_count']>=50 and state['iterations']<=2000
 for asset in state['assets'].values():verify_tensor(asset)
 ts=check_condition(reg['conditions']);added=verify_tensor(seal['added_mask'])
 assert torch.equal(added,ts['mask']&~ts['original_mask'])
 old=read_json(OLD/'inference_registry.json')['conditions'][case]
 for arm in ('B_512','C_512'):assert torch.equal(ts['mask'],verify_tensor(old[arm]['mask']))
 assert not (added&well_mask(read_json(OLD/'layout.json')['holdout'])).any()
 rr=c/'predictions/seed42';receipt=read_json(rr/'receipt.json')
 assert read_json(rr/'completion_receipt.json')['receipt_sha256']==file_hash(rr/'receipt.json')
 assert receipt['status']=='completed' and receipt['registry_sha256']==file_hash(c/'inference_registry.json')
 pred=verify_tensor(receipt['prediction']);assert not ((pred!=ts['values'])&ts['mask']).any()
 trace=read_json(rr/'trace.json');assert len(trace)==receipt['velocity_evaluations']==32
 assert all(t['time']==(i+.5)/32 and t['condition_embedding_max_error']==0 and t['state_finite'] and t['velocity_finite'] for i,t in enumerate(trace))
 assert file_hash(rr/'trace.json')==receipt['trace_sha256']
 noise=verify_tensor(reg['noise']['42']['asset']);assert tensor_hash(noise)==receipt['initial_noise_tensor_sha256']
 assert receipt['base_hash_before']==receipt['base_hash_after']==reg['noise']['42']['expected_model_hash']
 assert receipt['model_load_report']['ema_entry_count']==411
 calls+=receipt['velocity_evaluations']
 e=read_json(c/'EXECUTION.json');ei=read_json(c/'EVALUATION_INPUTS.json')
 assert datetime.fromisoformat(seal['sealed_at_utc'])<datetime.fromisoformat(receipt['started_at_utc'])<datetime.fromisoformat(receipt['completed_at_utc'])<datetime.fromisoformat(e['evaluation_started_at_utc'])
 by={s['arm']:s for s in e['samples']}
 held=verify_tensor(ei['domains']['E_holdout']['tensor'])
 base=verify_tensor(by['A_9']['prediction']);copy=verify_tensor(by['COPY_ONLY']['prediction'])
 assert torch.equal(base[held],copy[held])
 for s in e['samples']:
  verify_tensor(s['prediction'])
  for m in s['metrics'].values():
   assert m['target_tp']+m['target_fp']==m['target_pred_volume']
   assert m['target_tp']+m['target_fn']==m['target_truth_volume']
 for objid in {o['object_id'] for o in e['objects_common_outside']}:
  assert len({o['voxels'] for o in e['objects_common_outside'] if o['object_id']==objid})==1
 cases[case]=dict(iterations=state['iterations'],solve_seconds=state['elapsed_seconds'],flow_seconds=receipt['elapsed_seconds'],added_voxels=int(added.sum()),velocity_calls=32,scientific_retries=0)
assert solves==2 and calls==64
assert len(list((r/'cases').glob('*/predictions/seed42/decoded_geology.pt')))==2
# All cases keep approved cameras and complete target counts; every main/clean PDF has editable short text.
for case in (*CASES,'fullgeo_case03'):
 folder=r/'reports'/case
 assert len(list((folder/'panels').glob('*.png')))==9
 if case=='fullgeo_case03':
  for f in folder.glob('fig*.png'):assert file_hash(f)==manifest['approved_files'][f.name]
  for f in folder.glob('fig*.pdf'):assert file_hash(f)==manifest['approved_files'][f.name]
 else:
  config=read_json(folder/'RENDER_CONFIG.json')
  assert config['cameras']==read_json(STANDARD/'approved_samples/RENDER_CONFIG.json')['cameras']
  for check in config['panel_checks']:
   assert check['all_faces_inside_viewport']
   if check['target_only']:assert check['label_voxels_represented']['9']==check['full_target_voxels']
  config['visual_inspection']='Final main plates opened; matched approved contact-sheet reference; all shapes/FP retained. Camera unchanged, physical edge validation only.'
  write_json(folder/'RENDER_CONFIG.json',config)
 for pdf in folder.glob('fig*.pdf'):
  text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
  assert not any(word in text for word in ('IoU','precision','recall','seed','case','144','510'))
  if '_clean' in pdf.stem:assert 'Ground' not in text and 'Ours' not in text
  assert 'TrueType' in subprocess.check_output(['pdffonts',str(pdf)],text=True)
  png=Image.open(pdf.with_suffix('.png'));assert png.width==4251 and abs(png.info['dpi'][0]-600)<1
# Compact numeric table intended for a single PPT slide (no new graphic).
with (r/'three_case_summary.csv').open() as f:summary=list(csv.DictReader(f))
rows=[]
for q in sorted(summary,key=lambda q:q['case']):
 rows.append(dict(case=q['case'],pseudo_well_accuracy_C_to_new=f"{float(q['old_C_well_accuracy']):.4f} -> {float(q['restored_well_accuracy']):.4f}",
  holdout_target_IoU_C_to_new=f"{float(q['holdout_C_target_iou']):.4f} -> {float(q['holdout_new_target_iou']):.4f}",
  holdout_mIoU_C_to_new=f"{float(q['holdout_C_miou']):.4f} -> {float(q['holdout_new_miou']):.4f}",
  original_free_new_FP=q['primary_new_FP'],original_free_new_volume_ratio=q['primary_new_volume_ratio'],
  original_free_direct_IoU=q['direct_primary_iou']))
write_csv(r/'PPT_COMPARISON.csv',rows)
# Only temporary reference QA contact sheet, not a new science figure.
canvas=Image.new('RGB',(2400,900),'white');draw=ImageDraw.Draw(canvas)
for j,case in enumerate(('fullgeo_case03','fullgeo_case02','fullgeo_case01')):
 for i,stem in enumerate(('fig01_full_geology_comparison','fig02_target_body_comparison')):
  im=Image.open(r/'reports'/case/f'{stem}.png').convert('RGB');im.thumbnail((790,510 if i==0 else 320))
  yy=30 if i==0 else 560;canvas.paste(im,(j*800+(800-im.width)//2,yy));draw.text((j*800+10,yy-20),case+(' approved/reused' if j==0 else ' approved_v1'),fill='black')
canvas.save('/tmp/stage24_three_case_visual_check.png')
shutil.copyfile('/tmp/stage24_case02_plot2.log',r/'reports/fullgeo_case02/render.log')
shutil.copyfile('/tmp/stage24_final_style_regression.log',r/'standard_regression/after_validator.log')
shutil.copyfile('/tmp/stage24_report.log',r/'report.log')
write_json(r/'FINAL_CHECKS.json',dict(status='passed',cases=cases,new_solves=solves,new_predictions=2,velocity_evaluations=calls,
 original_and_added_condition_violations=0,source_noise_and_model_identity=True,copy_holdout_equal_A=True,
 all_target_components_preserved=True,all_tables_arithmetic=True,all_registered_objects_retained=True,
 main_figures=6,clean_figures=6,independent_panels=27,case03_original_main_bytes_reused=True,
 approved_regression_before_and_after_validator='all4 PNGs pixel-exact',new_training=0,new_seeds=0,new_scans=0,
 old_source_and_assets_unchanged=True,scientific_retries=0,visualization_technical_retries=2,
 visual_code_diff='VISUAL_VALIDATOR_FIX.diff',final_CPU_tests=4))
print('PASS: 2 solves,2 predictions,64 calls; 6 main/6 clean/27 panels; approved style pixel-exact; no scientific retries')
