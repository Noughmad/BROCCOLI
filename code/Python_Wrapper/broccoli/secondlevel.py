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
  first_level_results, MNI_brain_mask_data
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
  BROCCOLI.SetPermutationMatrix(BROCCOLI.packVolume(permutation_matrix)
  
  BROCCOLI.SetOutputDesignMatrix(h_Design_Matrix, h_Design_Matrix2);        
  BROCCOLI.SetOutputBetaVolumes(h_Beta_Volumes);        
  BROCCOLI.SetOutputResiduals(h_Residuals);        
  BROCCOLI.SetOutputResidualVariances(h_Residual_Variances);        
  BROCCOLI.SetOutputStatisticalMaps(h_Statistical_Maps);        
  BROCCOLI.SetOutputClusterIndices(h_Cluster_Indices);
  BROCCOLI.SetOutputPermutationDistribution(h_Permutation_Distribution);
  BROCCOLI.SetOutputPermutedFirstLevelResults(h_Permuted_First_Level_Results);       

  BROCCOLI.PerformSecondLevelAnalysisWrapper()
  
  if show_results:
    for volume in [aligned_T1_Volume_nonparametric, aligned_EPI_volume, MNI_brain_data]:
      plotVolume(volume, 0.45, 0.47)
      
  plot.plot(motion_parameters[0],'g')
  plot.plot(motion_parameters[1],'r')
  plot.plot(motion_parameters[2],'b')
  plot.title('Translation (mm)')
  plot.legend('X','Y','Z')
  plot.draw()
  plot.figure()
  
  plot.plot(motion_parameters[3],'g')
  plot.plot(motion_parameters[4],'r')
  plot.plot(motion_parameters[5],'b')
  plot.title('Rotation (degrees)')
  plot.legend('X','Y','Z')
  plot.draw()
  plot.figure()
  
  
  if beta_space == broccoli.MNI:
      slice = int(MNI_brain_data.shape[2] / 2)
  else:
      slice = int(fMRI_data.shape[2] / 2)
  
  plot.imshow(numpy.flipud(MNI_brain_data[:,:,slice]), cmap = cm.Greys_r, interpolation="nearest")
  plot.draw()
  plot.figure()
  
  print(statistical_maps.shape)
  print(statistical_maps[..., 0].shape)
  plot.imshow(numpy.flipud(statistical_maps[:,:,slice,0]), interpolation="nearest")
  plot.draw()
  plot.figure()

  plot.close()
  plot.show()
  
  # TODO: Return more parameters in proper order
  return statistical_maps
