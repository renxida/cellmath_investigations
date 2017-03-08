from skimage.io import imread, imsave, imshow
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sobel_h, sobel_v
from math import sqrt
import numpy as np
#%%

LINE_TYPE = 'bright'
img = imread('./blur.jpg')
# sigma: 0.1113
rxx, rxy, ryy = hessian_matrix(img, sigma = 0.1113, mode = 'constant', cval = 0)
rx = sobel_v(img)
ry = sobel_h(img)
hmev=hessian_matrix_eigvals(rxx, rxy, ryy)
#find greater eigenvalue that reflects maximum curvature direction.
# currently configured to find only ^ curves not v curves. Change key to 
# lambda x: x to find v curves
from itertools import product

# maxev done - but not used
nx, ny = np.zeros_like(rxx), np.zeros_like(ryy)

is_linepoint = np.full_like(rxy, False, dtype = bool)
# find nx, ny for 

if LINE_TYPE == 'bright':
    evindex = 0 # more negative eigenvalue. corresponds to a cap
elif LINE_TYPE=='dark':
    evindex = 1 #more positive eigenvalue. corresponds to a cup
else:
    raise Exception('invalid line type (must be \'bright\' or \'dark\')')
for r, c in product(range(512), repeat = 2):
    M = np.matrix([[rxx[r][c], rxy[r][c]], [rxy[r][c], ryy[r][c]]])
    vals, vects = np.linalg.eigh(M)
    # np.linalg.eigh sort eigenvalues in increasing order of face, non-absolute
    # value, so the most negative eigenvalue would be the one to use
    is_linepoint[r][c]=(abs(vals[evindex]) > abs(vals[1-evindex])) # if the our eigenvalue is the major eigenvalue
    if not is_linepoint[r][c]:
        continue
    nx[r][c], ny[r][c] = vects[0, evindex], vects[1, evindex] # normalized eigenvector in direction perpendicular to line direction which is also the direction of maximum curvature
    


# READY: rxx,rxy,ryy,rx,ry,hmev,maxev,nx,ny
a = (rxx*nx**2+2*rxy*nx*ny+ryy*ny**2)
b = -(rx*nx+ry*ny)
t = b/a
px = nx*t
py = ny*t
is_linepoint = is_linepoint * (a!=0) * (np.abs(px)<0.5) * (np.abs(py)<0.5)
print(np.sum(is_linepoint)/is_linepoint.size)
imshow(is_linepoint)
# PROBLEM WITH STEGER: sigma has to be tuned to detect curves correctly. Better
# to use some scale-invariant method