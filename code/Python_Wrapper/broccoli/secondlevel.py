import broccoli_common as broccoli
import numpy

import matplotlib.pyplot as plot
import matplotlib.cm as cm
    
def plotVolume(data, sliceYrel, sliceZrel):
  sliceY = int(round(sliceYrel * data.shape[0]))

  # Data is first ordered [y][x][z]
  plot.imshow(numpy.flipud(data[sliceY].transpose()), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

  sliceZ = int(round(sliceZrel * data.shape[2])) - 1

  plot.imshow(numpy.fliplr(data[:,:,sliceZ]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()

def performSecondLevelAnalysis(
  first_level_results, MNI_brain_mask_data,
  X_GLM, xtxxt_GLM, contrasts, ctxtxc_GLM,
  statistical_test, permutation_matrix, number_of_permutations, inference_mode, cluster_defining_threshold,
  opencl_platform, opencl_device, show_results = False,
  ):
    
  BROCCOLI = broccoli.BROCCOLI_LIB()
  print("Initializing OpenCL...")

  BROCCOLI.OpenCLInitiate(opencl_platform, opencl_device)
  ok = BROCCOLI.GetOpenCLInitiated()

  if ok == 0:
    BROCCOLI.printSetupErrors()
    print("OpenCL initialization failed, aborting")
    return

  print("OpenCL initialization successful, proceeding...")
  
  number_of_subjects = first_level_results.shape[3]
  number_of_GLM_regressors = X_GLM.shape[1]
  number_of_contrasts = contrasts.shape[0]
  
  BROCCOLI.SetInputFirstLevelResults(BROCCOLI.packVolume(first_level_results))
  
  BROCCOLI.SetMNIHeight(MNI_brain_mask_data.shape[0])
  BROCCOLI.SetMNIWidth(MNI_brain_mask_data.shape[1])
  BROCCOLI.SetMNIDepth(MNI_brain_mask_data.shape[2])
  BROCCOLI.SetInputMNIBrainMask(BROCCOLI.packVolume(MNI_brain_mask_data))
  
  BROCCOLI.SetStatisticalTest(statistical_test)
  BROCCOLI.SetInferenceMode(inference_mode)
  BROCCOLI.SetClusterDefiningThreshold(cluster_defining_threshold)

  BROCCOLI.SetNumberOfSubjects(number_of_subjects)
  BROCCOLI.SetNumberOfPermutations(number_of_permutations)
  BROCCOLI.SetNumberOfGLMRegressors(number_of_GLM_regressors)
  BROCCOLI.SetNumberOfContrasts(number_of_contrasts)
  BROCCOLI.SetDesignMatrix(BROCCOLI.packVolume(X_GLM), BROCCOLI.packVolume(xtxxt_GLM))
  BROCCOLI.SetContrasts(BROCCOLI.packVolume(contrasts))
  BROCCOLI.SetGLMScalars(BROCCOLI.packVolume(ctxtxc_GLM))
  BROCCOLI.SetPermutationMatrix(BROCCOLI.packVolume(permutation_matrix))
  
  design_matrix_1 = BROCCOLI.createOutputArray((number_of_subjects, number_of_GLM_regressors))
  design_matrix_2 = BROCCOLI.createOutputArray((number_of_subjects, number_of_GLM_regressors))
  BROCCOLI.SetOutputDesignMatrix(design_matrix_1, design_matrix_2)
  
  beta_volumes = BROCCOLI.createOutputArray(MNI_brain_mask_data.shape[0:3] + (number_of_GLM_regressors,))
  BROCCOLI.SetOutputBetaVolumes(beta_volumes)
  
  residuals = BROCCOLI.createOutputArray(MNI_brain_mask_data.shape[0:3] + (number_of_subjects,))
  BROCCOLI.SetOutputResiduals(residuals)
  
  residual_variances = BROCCOLI.createOutputArray(MNI_brain_mask_data.shape[0:3])
  BROCCOLI.SetOutputResidualVariances(residual_variances)
  
  statistical_maps = BROCCOLI.createOutputArray(MNI_brain_mask_data.shape[0:3] + (number_of_contrasts,))
  BROCCOLI.SetOutputStatisticalMaps(statistical_maps)
  
  cluster_indices = BROCCOLI.createOutputArray(MNI_brain_mask_data.shape[0:3], dtype=numpy.int32)
  BROCCOLI.SetOutputClusterIndices(cluster_indices)
  
  permutation_distribution = BROCCOLI.createOutputArray((number_of_permutations, 1))
  BROCCOLI.SetOutputPermutationDistribution(permutation_distribution)
  
  permuted_first_level_results = BROCCOLI.createOutputArray(first_level_results.shape)
  BROCCOLI.SetOutputPermutedFirstLevelResults(permuted_first_level_results)    

  BROCCOLI.PerformSecondLevelAnalysisWrapper()
  
