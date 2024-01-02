#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import warnings
import matplotlib.pyplot as plt
import csv
import time
import random
from scipy.fftpack import dct
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def prod_gaus(mu1, mu2, sigma1, sigma2):
    mu1 = np.array(mu1)
    mu2 = np.array(mu2)
    sigma1 = np.array(sigma1)
    sigma2 = np.array(sigma2)
    
    a = 1 / (2 * sigma1**2)
    b = 1 / (2 * sigma2**2)
    
    # New mean
    muN = (sigma2**2 * mu1 + sigma1**2 * mu2) / (sigma1**2 + sigma2**2)

    # New sigma
    sigmaN = np.sqrt(1 / (2 * (a + b)))

    # Integral
    pre = np.exp(-(a * b / (a + b) * (mu1 - mu2)**2)) / (sigma1 * sigma2 * 2 * np.pi)
    c = pre * np.sqrt(np.pi / (a + b))

    return c, muN, sigmaN, pre


def prod3gauss(mu1, mu2, mu3, sigma1, sigma2, sigma3):
    c, muN, sigmaN, _ = prod_gaus(mu1, mu2, sigma1, sigma2)
    c *= prod_gaus(muN, mu3, sigmaN, sigma3)[0]

    return c, muN, sigmaN


def simulateVV_GUI(paras, n, data, biOnly = 1,  strategy = 'ave' , fitType = 'dif', es_para = [1,1,1,1,1,0,0], fixvalue = [0.5,0.4,0.8,4000,2,70000.5,70000.5]):
    # Import Data into Python
    responses = data[:, [2, 3]]
    stimuli = data[:, [0, 1]]
    N = np.max(stimuli) + 1
    modelprop = []
    dataprop = []
    plt.clf()
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    # Default Parameters
    sU = fixvalue[5] 
    sD = fixvalue[6]
    sUm = 0
    sDm = 0
    PZ_center = fixvalue[4]
    sigmaZ = fixvalue[3]
    p_cutoff = 0.5
    pa_index = 0

    if es_para[0]==1:
        pcommon = min(abs(paras[pa_index]), 1)
        pa_index = pa_index + 1
    else:
        pcommon = fixvalue[0]

    if es_para[1]==1:
        sigmaU = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaU = fixvalue[1]

    if es_para[2]==1:
        sigmaD = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaD = fixvalue[2]

    if es_para[3]==1:
        sigmaZ = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sigmaZ = fixvalue[3]

    if es_para[4]==1:
        PZ_center = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        PZ_center = fixvalue[4]
    
    if es_para[5]==1:
        sU = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sU = fixvalue[5]
    
    if es_para[6]==1:
        sD = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sD = fixvalue[6]


        
    conditions = np.unique(stimuli, axis=0)
    
    # Real Stimuli, Repeat n times
    real = np.tile(conditions, (n, 1))
    
    # Create Mean of Distribution 
    # sU and sD is the 1/rate of increase away from center
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)

    # Make sure no missing stimuli
    sigma_like[real[:, 0] == 0, 0] = 1000
    sigma_like[real[:, 1] == 0, 1] = 1000
    
    #########
    # Calculate p(C|D,U)
    #########
    
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))
    #print(np.shape(PCDU))
    #########
    # Calculate Sc_hat
    #########
    
    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )
    
    # Mean Responses (Sim)
    
    responsesSim = np.zeros((Sc.shape[0], 2))
    if strategy == 'ave' :
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
    
    elif strategy == 'sel': 
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
    
    elif strategy == 'mat': 
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
      
    
            

    
    # Define Trial Type 
    trialType = 1 + np.dot(stimuli, [N, 1])
    trialTypeSim = 1 + np.dot(real, [N, 1])
    
    #########################################################
    # Create model probabilities/proportions from simulated responses
    modelprop = np.zeros((2, int(np.max(trialType).item()), int(N.item())))
    dataprop = np.zeros((2, int(np.max(trialType).item()), int(N.item())))
    # Numerical list of bimodal conditions
    vector1 = np.ceil(np.arange(1, (N - 1) ** 2 + 1) / (N - 1))
    vector2 = np.tile(np.arange(1, N), int(N.item()) - 1)
    matrix2 = np.vstack((vector1, vector2))
    matrix1 = np.array([N, 1])
    bimodalList = matrix1 @ matrix2
    
    bimodalList = bimodalList.astype(int)
    #print (np.max(trialType))
    '''
    for i in range(2):
        for j in range(1, int(np.max(trialType).item())+1):
            k1 = np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeSim==j),i]).astype(int)),N-1)#need to check
            k1 = k1.ravel().astype(int)
           
            for k in range(len(k1)):
                modelprop[i,j-1,k1[k]] += 1
            print(modelprop)
            k2 = np.minimum(np.maximum(0, np.round(responses[np.where(trialType==j),i]).astype(int)),N-1)
            k2 = k2.ravel().astype(int)
            for k in range(len(k2)):
                 dataprop[i,j-1,k2[k]] += 1
            
            dataprop[i,bimodalList,0] = 0
            dataprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(dataprop[i,j-1,:])))
            modelprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(modelprop[i,j-1,:])))
    print(modelprop)
    '''
    for i in range(2):
        for j in range(1, int(np.max(trialType).item())+1):
            k1 = np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeSim==j),i]).astype(int)), N-1)
            k1 = k1.ravel().astype(int)
            counts = np.bincount(k1, minlength=int(N)) 
            modelprop[i, j-1, :] = counts
            
            #print(modelprop)

            k2 = np.minimum(np.maximum(0, np.round(responses[np.where(trialType==j),i]).astype(int)), N-1)
            k2 = k2.ravel().astype(int)
            counts2 = np.bincount(k2, minlength=int(N)) 
            dataprop[i,j-1,:] = np.pad(counts2, (0, 0), mode='constant')

            dataprop[i, bimodalList, 0] = 0
            dataprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(dataprop[i,j-1,:])))
            modelprop[i,j-1,:] /= (1e-10+np.sum(np.squeeze(modelprop[i,j-1,:])))
     
    '''
    for i in range(2):
            
        trial_types = np.unique(trialType)
        max_trial_type = int(np.max(trialType).item())

        trialTypeMaskSim = trialTypeSim[:, np.newaxis] == trial_types[np.newaxis, :]
        trialTypeMask = trialType[:, np.newaxis] == trial_types[np.newaxis, :]

        k1_counts = np.bincount(np.minimum(np.maximum(1, np.round(responsesSim[np.where(trialTypeMaskSim[:, :, i])]).astype(int)), N-1).ravel(), minlength=4)
        modelprop[i, trial_types-1, :] = k1_counts

        k2_counts = np.bincount(np.minimum(np.maximum(0, np.round(responses[np.where(trialTypeMask[:, :, i])]).astype(int)), N-1).ravel(), minlength=4)
        dataprop[i, trial_types-1, :] = k2_counts


    dataprop[:, bimodalList, 0] = 0
    dataprop /= (1e-10 + np.sum(dataprop, axis=2, keepdims=True))
    modelprop /= (1e-10 + np.sum(modelprop, axis=2, keepdims=True))
    '''
    
    dataprop = dataprop[:, 1:, :]
    modelprop = modelprop[:, 1:, :]

    bimodalList = np.sum(conditions > 0, axis=1) > 1
    unimodal = np.sum(conditions > 0, axis=1) < 2
    
    modelprop[0, conditions[:, 0] == 0, 0] = 1 #recognize no stimulus
    modelprop[0, conditions[:, 0] == 0, 1:] = 0 #never respond 1, 2, 3...N

    modelprop[1, conditions[:, 1] == 0, 0] = 1
    modelprop[1, conditions[:, 1] == 0, 1:] = 0
    
    # Log likelihood
    modelTH = (1-0.001)*modelprop + 0.001*(1/N)
    
    # revert to numbers of responses
    npc = len(stimuli) // (N**2 - 1)  # n responses per condition
    loglike = npc * dataprop * np.log(modelTH)
    
    if biOnly == 1:  # bimodal only
        biORuni = bimodalList
    else:
        biORuni = np.logical_or(unimodal, bimodalList)
        
    model_dat = modelprop[:, biORuni, :]  
    data_dat = dataprop[:, biORuni, :] 
    
    # Multiple ways to define error, to be minimized:
    # Negative sum of log-likelihood
    minus_sll = -np.sum(np.sum(np.sum(loglike[:,biORuni,:])))

    # Negative R2
    x = np.corrcoef(model_dat.ravel(), data_dat.ravel())
    mr2 = -(x[1, 0]**2)

    # Sum Squared Error
    sse = np.sum(np.sum(np.sum((model_dat - data_dat)**2)))

    print("R2 on all")
    print(x[1, 0] ** 2)

    # Calculate sum log likelihood, and the 'best' model if it perfectly matched data
    print("Sum Loglike    Optimal")
    sum_loglike = np.sum(
        npc * dataprop[:, biORuni, :] * np.log((1 - 0.001) * modelprop[:, biORuni, :] + 0.001 / N)
    )
    optimal = np.sum(
        npc * dataprop[:, biORuni, :] * np.log((1 - 0.001) * dataprop[:, biORuni, :] + 0.001 / N)
    )
    print(sum_loglike, optimal)

    # Different way of calculating R2
    A = dataprop[:, biORuni, :]
    B = modelprop[:, biORuni, :]
    avg_data = np.mean(A)
    var_of_data = np.sum((A - avg_data) ** 2)
    avg_diff = np.mean(A - B)
    var_of_diff = np.sum(((A - B - avg_diff) ** 2))
    print("Explainable variance, r2:")
    r_square = 1 - var_of_diff / var_of_data
    print(r_square)
   
    '''
    #Plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(dataprop[0, :, :])
    axs[0, 0].set_title('Experimental Data, cond V resp')

    axs[0, 1].imshow(dataprop[1, :, :])
    axs[0, 1].set_title('Experimental Data, cond A resp')

    axs[1, 0].imshow(modelprop[0, :, :])
    axs[1, 0].set_title('Model, cond V resp')

    axs[1, 1].imshow(modelprop[1, :, :])
    axs[1, 1].set_title('Model, cond A resp')
    
    plt.savefig("output2.png") 
    plt.figure()
    
    plt.figure()
            
    plt.show()
    '''
    
    #print (loglike)
    
    if fitType == 'mll':
        error = minus_sll
    elif fitType == 'mr2':
        error = mr2
    else:
        error = sse
    
    error += 10000000 * int((sigmaU + sigmaD) < 0)
    
    np.random.set_state(s1)
    
    #np.savetxt('array10.txt', dataprop[1, :, :], fmt='%.6f')
    #np.savetxt('array13.txt', trialType, fmt='%.6f')
    
   
    return error, modelprop, dataprop, responsesSim#, real, sigma_like, noisy, PCDU, Sc




def simulateLC_GUI (paras, n, data, strategy = 'ave', fittype = 'fre', es_para = [1,1,1,1,1,0,0], fixvalue = [0.5,2,8,4000,0,70000.5,70000.5]):
    # Import Data into Python
    responses = data[:, [2, 3]]
    stimuli = data[:, [0, 1]]
    modelprop = []
    dataprop = []
    plt.clf()
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    # Default Parameters
    sU = fixvalue[5] 
    sD = fixvalue[6]
    sUm = 0
    sDm = 0
    PZ_center = fixvalue[4]
    sigmaZ = fixvalue[3]
    p_cutoff = 0.5
    pa_index = 0

    if es_para[0]==1:
        pcommon = min(abs(paras[pa_index]), 1)
        pa_index = pa_index + 1
    else:
        pcommon = fixvalue[0]

    if es_para[1]==1:
        sigmaU = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaU = fixvalue[1]

    if es_para[2]==1:
        sigmaD = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaD = fixvalue[2]

    if es_para[3]==1:
        sigmaZ = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sigmaZ = fixvalue[3]

    if es_para[4]==1:
        PZ_center = paras[pa_index]
        pa_index = pa_index + 1
    else:
        PZ_center = fixvalue[4]
    
    if es_para[5]==1:
        sU = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sU = fixvalue[5]
    
    if es_para[6]==1:
        sD = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sD = fixvalue[6]
    
    stimuli_cp = np.copy(stimuli)
    nan_marker = -999
    stimuli_cp[np.isnan(stimuli_cp)] = nan_marker
    conditions = np.unique(stimuli_cp, axis=0)
    unique_elements = np.unique(conditions[:, 0], axis=0)
    N=len(unique_elements)
    conditions[conditions == -999] = np.nan
    
    real = np.tile(conditions, (n, 1))
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)

    # Make sure no missing stimuli
    sigma_like[np.isnan(sigma_like)] = 1000
    noisy[np.isnan(noisy)] = 0
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))

    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    responsesSim = np.zeros((Sc.shape[0], 2))
    if strategy == 'ave':
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])

        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
        
    elif strategy == 'mat':
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
        
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])
        
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
    
    else:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
        
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])
        
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
    
    def assign_indices(matrix):
        _, indices = np.unique(matrix, axis=0, return_inverse=True)
        return indices.reshape(-1, 1)

    nan_marker = -999
    stimuli_cp = np.copy(stimuli)
    stimuli_cp[np.isnan(stimuli_cp)] = nan_marker
    real_cp = np.copy(real)
    real_cp[np.isnan(real_cp)] = nan_marker

    trialType = assign_indices(stimuli_cp)
    trialTypeSim = assign_indices(real_cp)

    modelmat = np.zeros((2, int(np.max(trialType).item())+1, int(n)))
    datamat = np.zeros((2, int(np.max(trialType).item())+1, int(len(stimuli)/len(conditions))))


    for i in range(2):
        for j in range(int(np.max(trialType).item())+1):
            k1 = responsesSim[np.where(trialTypeSim==j)[0],i]
            modelmat[i, j, :] = k1
            
            #print(modelprop)
            k2 = responses[np.where(trialType==j)[0],i]
            datamat[i, j, :] = k2

    loglike = 0
    emdall = 0
    rangei= np.arange(-50, 51, 5)
    freqmat_data=[]
    freqmat_model=[]
    r2values = []
    klmat = []
    for m in range(int(np.max(trialType).item())+1):
        '''
        llV=np.log(np.mean(norm.pdf(np.repeat(datamat[0,m,:], n, axis=0), np.repeat(modelmat[0,m,:], len(datamat[0,m,:]), axis=0))))
        llA=np.log(np.mean(norm.pdf(np.repeat(datamat[1,m,:], n, axis=0), np.repeat(modelmat[1,m,:], len(datamat[0,m,:]), axis=0))))
        llV=np.nan_to_num(llV)
        llA=np.nan_to_num(llA)
        loglike = loglike + llV + llA
        
        
        
        data_mean, data_std = np.mean(datamat[0,m,:]), np.std(datamat[0,m,:])
        llDV = np.sum(norm.logpdf(datamat[0,m,:], loc=data_mean, scale=data_std))
        model_mean, model_std = np.mean(modelmat[0,m,:]), np.std(modelmat[0,m,:])
        llMV = np.sum(norm.logpdf(modelmat[0,m,:], loc=model_mean, scale=model_std))
        llDV = np.nan_to_num(llDV)
        llMV = np.nan_to_num(llMV)
        llV = abs(llDV-llMV)
        
        data_mean, data_std = np.mean(datamat[1,m,:]), np.std(datamat[1,m,:])
        llDA = np.sum(norm.logpdf(datamat[1,m,:], loc=data_mean, scale=data_std))
        model_mean, model_std = np.mean(modelmat[1,m,:]), np.std(modelmat[1,m,:])
        llMA = np.sum(norm.logpdf(modelmat[1,m,:], loc=model_mean, scale=model_std))
        llDA = np.nan_to_num(llDA)
        llMA = np.nan_to_num(llMA)
        llA = abs(llDA-llMA)
        
        loglike = loglike + llV + llA
        
        '''
        if fittype == 'fre':
            hist_VD,kk= np.histogram(datamat[0,m,:], bins=rangei)
            freq_VD = np.maximum(0.001,hist_VD/np.sum(hist_VD))
            hist_VM,kk= np.histogram(modelmat[0,m,:], bins=rangei)
            freq_VM = np.maximum(0.001,hist_VM/np.sum(hist_VM))
            llV = np.sum(freq_VD * np.log(freq_VM))
            llV = np.nan_to_num(llV)
        
            hist_AD,kk= np.histogram(datamat[1,m,:], bins=rangei)
            freq_AD = np.maximum(0.001,hist_AD/np.sum(hist_AD))
            hist_AM,kk= np.histogram(modelmat[1,m,:], bins=rangei)
            freq_AM = np.maximum(0.001,hist_AM/np.sum(hist_AM))
            llA = np.sum(freq_AD * np.log(freq_AM))
            llA = np.nan_to_num(llA)
        
            loglike =loglike + llV + llA
            
        if fittype == 'gk':
            llV=np.sum(norm.logpdf(np.repeat(datamat[0,m,:], n, axis=0), np.repeat(modelmat[0,m,:], len(datamat[0,m,:]),         axis=0),5))
            llA=np.sum(norm.logpdf(np.repeat(datamat[1,m,:], n, axis=0), np.repeat(modelmat[1,m,:], len(datamat[0,m,:]), axis=0),5))
            llV=np.nan_to_num(llV, nan=0)
            llA=np.nan_to_num(llA, nan=0)

            loglike = loglike + llV + llA
        
        if fittype =='mr2':
            hist_VD,kk= np.histogram(datamat[0,m,:], bins=rangei)
            freq_VD = np.maximum(0.001,hist_VD/np.sum(hist_VD))
            hist_VM,kk= np.histogram(modelmat[0,m,:], bins=rangei)
            freq_VM = np.maximum(0.001,hist_VM/np.sum(hist_VM))
            
            hist_AD,kk= np.histogram(datamat[1,m,:], bins=rangei)
            freq_AD = np.maximum(0.001,hist_AD/np.sum(hist_AD))
            hist_AM,kk= np.histogram(modelmat[1,m,:], bins=rangei)
            freq_AM = np.maximum(0.001,hist_AM/np.sum(hist_AM))
            if datamat[0,m,0] < 1000:
                freqmat_data.append(freq_VD)
                freqmat_model.append(freq_VM)
            if datamat[1,m,0] < 1000:
                freqmat_data.append(freq_AD)
                freqmat_model.append(freq_AM)

        if fittype=='emd':
            if datamat[0,m,1]> -1000:
                kde_1 = KernelDensity(kernel='gaussian', bandwidth=1).fit(datamat[0,m,:].reshape(-1, 1))
                kde_2 = KernelDensity(kernel='gaussian', bandwidth=1).fit(modelmat[0,m,:].reshape(-1, 1))
                x = np.linspace(-50, 50, 50).reshape(-1, 1)
                log_density_1 = kde_1.score_samples(x)
                log_density_2 = kde_2.score_samples(x)
        
                emd_distance = wasserstein_distance(np.exp(log_density_1), np.exp(log_density_2))
                emdall = emdall + emd_distance
            
            if datamat[1,m,1]>-1000:    
                kde_1 = KernelDensity(kernel='gaussian', bandwidth=1).fit(datamat[1,m,:].reshape(-1, 1))
                kde_2 = KernelDensity(kernel='gaussian', bandwidth=1).fit(modelmat[1,m,:].reshape(-1, 1))
                x = np.linspace(-50, 50, 50).reshape(-1, 1)
                log_density_1 = kde_1.score_samples(x)
                log_density_2 = kde_2.score_samples(x)
        
                emd_distance = wasserstein_distance(np.exp(log_density_1), np.exp(log_density_2))
                emdall = emdall+emd_distance
        
        

    
    if fittype == 'fre':
        error = -loglike
        print("Sum Loglike:    Paras:")
        print(loglike,paras)


    elif fittype == 'emd':
        print("Distance:    Paras:")
        print(emdall,paras)
        error = emdall
        
    elif fittype == 'mr2':
        
        corr_matrix = np.corrcoef(np.concatenate(freqmat_data),np.concatenate(freqmat_model))
        r2 = corr_matrix[0, 1]**2
        
        print("r2:    Paras:")
        print(r2, paras)
        error = -r2
        
    elif fittype =='gk':
        print("Sum Loglike:    Paras:")
        print(loglike, paras)
        error = -loglike


   
    return error, modelmat, datamat, conditions

def simulateLCD_GUI (paras, n, data, strategy = 'ave', fittype = 'mll', es_para = [1,1,1,1,1,0,0], fixvalue = [0.5,0.4,0.8,4000,2,70000.5,70000.5]):
    # Import Data into Python
    responses = data[:, [2, 3]]
    stimuli = data[:, [0, 1]]
    modelprop = []
    dataprop = []
    plt.clf()
    
    # Set Random Seed
    s1 = np.random.get_state()
    s2 = np.random.get_state()
    np.random.seed(13)
    
    # Default Parameters
    sU = fixvalue[5] 
    sD = fixvalue[6]
    sUm = 0
    sDm = 0
    PZ_center = fixvalue[4]
    sigmaZ = fixvalue[3]
    p_cutoff = 0.5
    pa_index = 0

    if es_para[0]==1:
        pcommon = min(abs(paras[pa_index]), 1)
        pa_index = pa_index + 1
    else:
        pcommon = fixvalue[0]

    if es_para[1]==1:
        sigmaU = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaU = fixvalue[1]

    if es_para[2]==1:
        sigmaD = max(0.1, abs(paras[pa_index]))
        pa_index = pa_index + 1
    else:
        sigmaD = fixvalue[2]

    if es_para[3]==1:
        sigmaZ = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sigmaZ = fixvalue[3]

    if es_para[4]==1:
        PZ_center = paras[pa_index]
        pa_index = pa_index + 1
    else:
        PZ_center = fixvalue[4]
    
    if es_para[5]==1:
        sU = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sU = fixvalue[5]
    
    if es_para[6]==1:
        sD = abs(paras[pa_index])
        pa_index = pa_index + 1
    else:
        sD = fixvalue[6]
    
    stimuli_cp = np.copy(stimuli)
    nan_marker = -999
    stimuli_cp[np.isnan(stimuli_cp)] = nan_marker
    conditions = np.unique(stimuli_cp, axis=0)
    unique_elements = np.unique(conditions[:, 0], axis=0)
    N=len(unique_elements)
    conditions[conditions == -999] = np.nan
    
    response_cp = np.copy(responses)
    response_cp[np.isnan(response_cp)] = nan_marker
    conditions_r = np.unique(response_cp, axis=0)
    unique_elements_r1 = np.unique(conditions_r[:, 0], axis=0)
    unique_elements_r2 = np.unique(conditions_r[:, 1], axis=0)
    unique_elements_r = np.unique(conditions_r)
    N_l1 = len(unique_elements_r)
    N_l2 = len(unique_elements_r2)
    N_l = np.max([N_l1, N_l2])
    
    real = np.tile(conditions, (n, 1))
    sigma_like = np.zeros((real.shape[0], 2))
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - real[:, 0]), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - real[:, 1]), 1) / sD)
    # Add noise
    noisy = np.zeros_like(real)
    noisy[:, 0] = real[:, 0] + np.random.randn(real.shape[0]) * sigma_like[:, 0]
    noisy[:, 1] = real[:, 1] + np.random.randn(real.shape[0]) * sigma_like[:, 1]
    
    # New sigma_like based on the added noise
    sigma_like[:, 0] = sigmaU * (1 + np.power(np.abs(sUm - np.maximum(0, noisy[:, 0])), 1) / sU)
    sigma_like[:, 1] = sigmaD * (1 + np.power(np.abs(sDm - np.maximum(0, noisy[:, 1])), 1) / sD)

    # Make sure no missing stimuli
    sigma_like[np.isnan(sigma_like)] = 1000
    noisy[np.isnan(noisy)] = 0
    # CalculetP(U,D|C)
    # Integral of P(U|Z)*P(D|Z)*P(Z)
    PDUC = prod3gauss(noisy[:, 0], noisy[:, 1], PZ_center, sigma_like[:, 0], sigma_like[:, 1], sigmaZ)[0]
    # Integral of P(U|Z)*P(Z) times integral of P(D|Z)*P(Z)
    PDUnC = prod_gaus(noisy[:, 0], PZ_center, sigma_like[:, 0], sigmaZ)[0] * prod_gaus(noisy[:, 1], PZ_center, sigma_like[:, 1],
                                                                                       sigmaZ)[0]

    # Posterior of Common Cause Given Signals
    PCDU = np.multiply(PDUC, pcommon) / (np.multiply(PDUC, pcommon) + np.multiply(PDUnC, 1 - pcommon))

    Sc = (
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2 * noisy[:, 0]) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2 * noisy[:, 1]) +
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2 * PZ_center)
    ) / (
    (sigma_like[:, 0] ** 2 * sigma_like[:, 1] ** 2) +
    (sigma_like[:, 1] ** 2 * sigmaZ ** 2) +
    (sigma_like[:, 0] ** 2 * sigmaZ ** 2)
    )
    
    Snc1 = (
    (sigmaZ ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * noisy[:, 0] +
    (sigma_like[:, 0] ** 2 / (sigma_like[:, 0] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    Snc2 = (
    (sigmaZ ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * noisy[:, 1] +
    (sigma_like[:, 1] ** 2 / (sigma_like[:, 1] ** 2 + sigmaZ ** 2)) * PZ_center
    )

    responsesSim = np.zeros((Sc.shape[0], 2))
    if strategy == 'ave':
        # Averaging 
        responsesSim[:, 0] = PCDU * Sc + (1 - PCDU) * Snc1
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])

        responsesSim[:, 1] = PCDU * Sc + (1 - PCDU) * Snc2
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
        
    elif strategy == 'mat':
        # Matching
        p_cutoff = np.random.rand(Sc.shape[0])
        responsesSim[:, 0] = np.where(PCDU > p_cutoff, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > p_cutoff, Sc, Snc2)
        
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])
        
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
    
    else:
        # Selecting 
        responsesSim[:, 0] = np.where(PCDU > 0.5, Sc, Snc1)
        responsesSim[:, 1] = np.where(PCDU > 0.5, Sc, Snc2)
        
        responsesSim[:, 0] = np.where(sigma_like[:, 0] > 500, np.nan, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 0] > 500, Snc2, responsesSim[:,1])
        
        responsesSim[:, 0] = np.where(sigma_like[:, 1] > 500, Snc1, responsesSim[:,0])
        responsesSim[:, 1] = np.where(sigma_like[:, 1] > 500, np.nan, responsesSim[:,1])
    
    def assign_indices(matrix):
        _, indices = np.unique(matrix, axis=0, return_inverse=True)
        return indices.reshape(-1, 1)
    
    L = len(responsesSim[:,0])
    c_L = len(conditions)

    abs_diff = np.abs(np.tile(responsesSim[:,0].reshape(L, 1), (1, N_l)) - np.tile(unique_elements_r.reshape(1, N_l), (n*c_L, 1)))
    tV = np.argmin(abs_diff, axis=1)
    responsesSim[:, 0] = unique_elements_r[tV]

    abs_diff2 = np.abs(np.tile(responsesSim[:,1].reshape(L, 1), (1, N_l)) - np.tile(unique_elements_r.reshape(1, N_l), (n*c_L, 1)))
    tA = np.argmin(abs_diff2, axis=1)
    responsesSim[:, 1] = unique_elements_r[tA]

    nan_marker = -999
    stimuli_cp = np.copy(stimuli)
    stimuli_cp[np.isnan(stimuli_cp)] = nan_marker
    real_cp = np.copy(real)
    real_cp[np.isnan(real_cp)] = nan_marker

    trialType = assign_indices(stimuli_cp)
    trialTypeSim = assign_indices(real_cp)

    modelmat = np.zeros((2, int(np.max(trialType).item())+1, N_l))
    datamat = np.zeros((2, int(np.max(trialType).item())+1, N_l))

    binM = unique_elements_r - 0.01
    binM = np.append(binM, binM[-1]+1)
    for i in range(2):
        for j in range(int(np.max(trialType).item())+1):
            k1 = responsesSim[np.where(trialTypeSim==j)[0],i]


            hist_values = np.histogram(k1, bins=binM)[0]
            #freq_predV = np.maximum(0.00001, hist_values / n)
            freq_predV = hist_values / n
            modelmat[i, j, :] = freq_predV
            k2 = responses[np.where(trialType==j)[0],i]
            hist_values = np.histogram(k2, bins=binM)[0]
            #freq_predA = np.maximum(0.00001, hist_values / (len(stimuli)/c_L))
            freq_predA = hist_values / (len(stimuli)/c_L)
            datamat[i, j, :] = freq_predA

    
       
    
    if fittype == 'mll':
        modelTH = (1-0.001)*modelmat + 0.001*(1/N)
    
        # revert to numbers of responses
        npc = len(stimuli) // (N**2 - 1)  # n responses per condition
        loglike = npc * datamat * np.log(modelTH)

     
        # Negative sum of log-likelihood
        minus_sll = -np.sum(np.sum(np.sum(loglike)))
        error = minus_sll
        print("Sum Loglike:    Paras:")
        print(minus_sll , paras)


    elif fittype == 'mr2':
        x = np.corrcoef(modelmat.ravel(), datamat.ravel())
        mr2 = -(x[1, 0]**2)
        
        A = datamat
        B = modelmat
        avg_data = np.mean(A)
        var_of_data = np.sum((A - avg_data) ** 2)
        avg_diff = np.mean(A - B)
        var_of_diff = np.sum(((A - B - avg_diff) ** 2))
        print("Explainable variance, r2:    Paras:  ")
        r_square = 1 - var_of_diff / var_of_data
        print(r_square, paras)
        error = mr2
        
    elif fittype == 'sse':
        
        sse = np.sum(np.sum(np.sum((modelmat - datamat)**2)))
        print("Sum of Squared Errors:    Paras:")
        print(sse, paras)
        error = sse
     
    error += 10000000 * int((sigmaU + sigmaD) < 0)

   
    return error, modelmat, datamat, conditions




def plotKonrads(data, model=None, save_path=None):
    a = data.shape
    condi = int((a[1] + 1) ** 0.5)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  

    for i in range(a[1]):
        if condi < 11:
            plt.subplot(condi, condi, i + 2)
        else:
            plt.figure(int(i/condi) + 1)
            plt.subplot(condi, 1, (i % condi) + 1)
        
        plt.plot(data[0, i, :], 'b')
     
        plt.plot(data[1, i, :], 'r')
        plt.axis([0, a[2]-1, 0, 1])

    if model is not None:
        for i in range(a[1]):
            if condi < 11:
                plt.subplot(condi, condi, i+2)
            else:
                plt.figure(int(i/condi) + 1)
                plt.subplot(condi, 1, (i % condi) + 1)
            
           
            plt.plot(model[0, i, :], 'b-.')
            plt.plot(model[1, i, :], 'r-.')
            plt.axis([0, a[2]-1, 0, 1])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    


