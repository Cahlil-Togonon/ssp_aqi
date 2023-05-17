# Check modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import qr
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import cv2

# to edit: make this work for interpolated data

#%% Load Yale Database
Ni = []                     # Index of people in Yale Database                    
Nx = [14]                   # 14th person corrupted
for i in range(1,39+1):
    if i in Nx:
        continue
    else:
        Ni+=[i]
        
m = 192
n = 168

dirn = 'CroppedYale/'
cnt_tr = 0
cnt_te = 0
count = 0

avgFace = np.zeros((m*n,1))

Xtrain = []
Xtest = []

P1 = 38
P2 = 39
test = [P1,P2]
P_tr = []
P_te = []

fnames = []
for j in Ni:
    print('person: ',j)
    per = 'yaleB%02d'%j
    with open(dirn+per+'/'+per+'_P00.info') as fid: # read file with image name list
        fn = fid.read().splitlines()
    fn.pop(0)   # remove ambient filename

    if j in test:
        P_te += [cnt_te]
    else:
        P_tr += [cnt_tr]
    
    # scan all images for each person
    for i in fn:
        # check if file exists. Some images with file extension .bad are corrupted
        file = dirn+per+'/'+i    
        if os.path.exists(file):
            fnames += fn
            image = cv2.imread(file,0)
            R = np.reshape(image,(m*n,1))        # flatten
            if j in test:
                if cnt_te==0:
                    Xtest = R
                else:
                    Xtest = np.hstack((Xtest,R))
                cnt_te += 1
            else:
                if cnt_tr==0:
                    Xtrain = R
                else:
                    Xtrain = np.hstack((Xtrain,R))
                cnt_tr += 1
        else:
            continue        

#%% Get Average Face
avgFace = np.sum(Xtrain,axis=1)/cnt_tr
avgFaceIm = np.reshape(avgFace,(m,n))
plt.figure()
plt.imshow(avgFaceIm,cmap='gray',vmin=0,vmax=255)
plt.title('Yale Database Average Face')

#%% Subtract the mean
Xtrain_s = np.zeros(Xtrain.shape)
for j in range(0,cnt_tr):
    Xtrain_s[:,j] = Xtrain[:,j] - avgFace
    if j%10==0:
        print(j)
#%% Compute the SVD
U,S,Vh = np.linalg.svd(Xtrain_s,full_matrices=False)

#%% Item 1 Spare Sensor Placement
# Item 1 a) Reconstucting in sample
# Code 6.1 
#--------------------------------------------------------------------------------------------------------
file = 'face.pgm'   # image filename here
Im = cv2.imread(file,0)
I = cv2.resize(Im,(n,m))
plt.figure()
plt.imshow(I,cmap='gray')
Ir = np.reshape(I,(m*n,1))      # image reshaped flattened
avgFace2 = np.atleast_2d(avgFace).T
    
#%% Code 6.2
rp = [50,100,500]      # input values for no. of modes and sensors

fig,ax = plt.subplots(len(rp),4,figsize=(19,9.2),sharex='all',sharey='all')
ax = np.atleast_2d(ax)

for k in range(0,len(rp)):
    print('P = '+str(rp[k]))
    r = rp[k]     # No. of modes
    p = rp[k]     # No. of sensors
    Psi = U[:,:r]
    
    Q,R,P = qr(Psi.T,pivoting=True)
    Pr = np.random.choice(np.arange(1,m*n,1),p)
    
    pC = np.zeros((p,2))    # qr sensor coordinates
    pCr = np.zeros((p,2))    # random sensor coordinates
    # Construct measurement matrix
    C = np.zeros((p,m*n)) 
    Cr = np.zeros((p,m*n))
    
    for j in range(p):
        C[j,P[j]] = 1
        Cr[j,Pr[j]] = 1
        
        xp = P[j]%n
        yp = np.ceil(P[j]/n)
        
        xpr = Pr[j]%n
        ypr = np.ceil(Pr[j]/n)
        
        pC[j,:] = np.array([xp,yp])
        pCr[j,:] = np.array([xpr,ypr])
    
    Ic = Ir
    #-------------------------------------------------------------------------
    # Item 1 c) insert code here
    
    Irs = Ir - avgFace2           # use this expression
    Ic = avgFace2 + np.dot(Psi,np.dot(Psi.T,Irs))
    
    #-------------------------------------------------------------------------
    # QR Sensors    
    Theta = np.dot(C,Psi)
    y = np.take(Ic,P[:p])
    # get sparse vector s and reconstruct
    s,res,rank,sig = np.linalg.lstsq(Theta,y,rcond=None)
    recon = np.dot(Psi,s)
    ax[k,0].imshow(I,cmap='gray')
    ax[k,0].plot(pC[:,0],pC[:,1],'ro',markersize=2.5)
    ax[k,0].set_xlim([0,n])
    ax[k,0].set_ylim([m,0])
    ax[k,1].imshow(recon.reshape((m,n)),cmap='gray')
    
    # Random Sensors
    Theta = np.dot(Cr,Psi)
    y = np.take(Ic,Pr[:p])
    # get sparse vector s and reconstruct
    s,res,rank,sig = np.linalg.lstsq(Theta,y,rcond=None)
    recon = np.dot(Psi,s)
    ax[k,2].imshow(I,cmap='gray')
    ax[k,2].plot(pCr[:,0],pCr[:,1],'ro',markersize=2.5)
    ax[k,2].set_xlim([0,n])
    ax[k,2].set_ylim([m,0])
    ax[k,3].imshow(recon.reshape((m,n)),cmap='gray')
    
    ax[k,0].set_ylabel('P = '+str(rp[k]))
    if k == 0:
        ax[k,0].set_title('QR sensors')
        ax[k,1].set_title('QR reconstruction')
        ax[k,2].set_title('Random sensors')
        ax[k,3].set_title('Random reconstruction')

