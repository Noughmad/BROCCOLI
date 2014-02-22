from nipype.interfaces.broccoli import secondlevel
import os

BROCCOLI_DIR = '/home/miha/Programiranje/BROCCOLI'
OpenfMRI_DIR = '/data/miha/OpenfMRI/RhymeJudgment/ds003'
SUBJECT_DIR = os.path.join(OpenfMRI_DIR, subject)

secondlevel.SecondLevelAnalysis.help()
print(SUBJECT_DIR)

interface = secondlevel.SecondLevelAnalysis(
    MNI_brain_mask_file = BROCCOLI_DIR + '/brain_templates/MNI152_T1_2mm_brain_mask.nii',
    first_level_results_basepath = 
    GLM_path = os.path.join(SUBJECT_DIR, 'model/model001/onsets/task001_run001'),
    
    
    use_temporal_derivatives = True,
    regress_motion = True,
    regress_confounds = False,

    EPI_smoothing = 6.0,
    AR_smoothing = 7.0,
    
    opencl_device = 0,
    show_results = True,
)


results = interface.run()
print(repr(results))

