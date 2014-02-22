from nipype.interfaces.base import TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File, Directory, isdefined, traits
from nipype.utils.filemanip import split_filename

import scipy.io
import scipy.signal
import os
import os.path as op
import nibabel as nb
import numpy as np

import broccoli
from base import BroccoliInputSpec, BroccoliInterface

class SecondLevelAnalysisInputSpec(BroccoliInputSpec):
    first_level_results_basepath = Directory(exists=True, mandatory=True)
    MNI_brain_mask_file = File(exists=True)
    GLM_path = Directory(exists=True, mandatory=True)

class SecondLevelAnalysisOutputSpec(TraitedSpec):
    statistical_map = File()
  
class SecondLevelAnalysis(BroccoliInterface):
    input_spec = SecondLevelAnalysisInputSpec
    output_spec = SecondLevelAnalysisOutputSpec
    
    def load_regressor(self, filename, st, samples):
        d = np.loadtxt(filename)
        hr = np.zeros(samples * st)
        tr = 2
        
        for row in d:
            start = int(round(row[0] * samples / tr))
            duration = int(round(row[1] * samples / tr))
            for i in range(duration):
                hr[start + i] = row[2]
        
        print(hr.shape)
        print(np.count_nonzero(hr))
        print(hr)
        lr = scipy.signal.decimate(hr, samples)
        return lr
    
    def load_regressors(self, st):
        files = [f for f in os.listdir(self.inputs.GLM_path) if op.isfile(op.join(self.inputs.GLM_path, f))]
        data = [self.load_regressor(op.join(self.inputs.GLM_path, f), st, 10) for f in files]
        return np.array(data).transpose()
    
    def load_first_level_result(self, subject):
        print("Loading first level result for subject %s" % subject)
        filename = os.path.join(self.inputs.GLM_PATH, subject, 'model/model001/task001.gfeat/cope1.feat/stats/cope1.nii.gz')
        data, _ = broccoli.load_nni(filename)
        return data
        
    def _run_interface(self, runtime):
        MNI_brain_mask, _ = broccoli.load_nni(self.inputs.MNI_brain_mask_file)
        

        first_level_results = np.array([self.load_first_level_result(subject) for subject in os.listdir(self.inputs.GLM_path)])
        number_of_subjects = first_level_results.shape[3]

        X_GLM = self.load_regressors(number_of_subjects)
        xtx = np.linalg.inv(np.dot(X_GLM.T, X_GLM))
        xtxxt_GLM = xtx.dot(X_GLM.T)

        contrasts = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        ctxtxc_GLM = [contrasts[i:i+1].dot(xtx).dot(contrasts[i:i+1].T) for i in range(len(contrasts))]
        
        broccoli.performSecondLevelAnalysis(
            first_level_results, MNI_brain_mask,
            X_GLM, xtxxt_GLM, contrasts, ctxtxc_GLM,
            statistical_test, permutation_matrix, number_of_permutations, inference_mode, cluster_defining_threshold,
            self.inputs.opencl_platform, self.inputs.opencl_device, self.inputs.show_results,
        )
        
        return runtime
            
    def _list_outputs(self):
        outputs = self.output_spec().get()
        for k in outputs.keys():
            outputs[k] = self._get_output_filename(k + '.nii')
        return outputs
