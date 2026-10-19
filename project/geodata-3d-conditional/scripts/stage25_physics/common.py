"""Optional single-member physics bridge; main science remains frozen."""
from scripts.stage25.common import *
from guidance.seismic import acoustic_tables_from_config,seismic_operator_from_config,hard_labels_to_acoustic
PHYS=ROOT/'physics'
ACOUSTIC=PROJECT/'experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json'
OBS_CONFIG=PROJECT/'experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json'
INV_CONFIG=PROJECT/'experiments/stage5_acoustic_inversion/configs/model_based_log_impedance_v1.json'
def physics_sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[PROJECT/'guidance/seismic.py',PROJECT/'guidance/seismic_inversion.py',ACOUSTIC,OBS_CONFIG,INV_CONFIG]
    return {str(p.relative_to(REPO)):file_hash(p) for p in paths}
def setup():
    tables,_=acoustic_tables_from_config(read_json(PHYS/'acoustic_codebook.json'),15)
    operator,_=seismic_operator_from_config(read_json(PHYS/'observation_config.json'))
    return tables.property_table.double(),operator

def inversion_guard():
    guard('restore')
    def deny(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=(isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or (isinstance(args[2],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        forbidden_history=any((ROOT/'arms'/a)==p or (ROOT/'arms'/a) in p.parents for a in ('B','C')) or p.name=='gaussian_sigma1.pt'
        forbidden_field_read='preprocessing' in p.parts and not (writing and PHYS in p.parents)
        if forbidden_history or forbidden_field_read:raise PermissionError(f'Inversion only A prior, original conditions, seismic observation; own output writes allowed: {p}')
    sys.addaudithook(deny)
