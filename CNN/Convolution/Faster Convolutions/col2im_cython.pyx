from __future__ import print_function
import numpy as np
cimport numpy as np

cimport cython

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

"""
Function takes in 2d input, gives 2d output
Input        : w_shp, x_shp, stride,   X_val 
Input shape  :  1x3 ,  1x3 ,  1x1  , (x0*x1*x2,x3)

Output       :       X_res
Output shape : (r0*r2*r3,w1*w2*w3)

""" 
cpdef col2im(np.ndarray[np.int_t, ndim=1] w_shp,np.ndarray[np.int_t, ndim=1] x_shp,int stride,np.ndarray[np.float64_t, ndim=2] x_grad):
    
    cdef int w0 = w_shp[0]
    cdef int w1 = w_shp[1]
    cdef int w2 = w_shp[2]
    cdef int w3 = w_shp[3]
    cdef int x0 = x_shp[0]
    cdef int x1 = x_shp[1]
    cdef int x2 = x_shp[2]
    cdef int x3 = x_shp[3]
    cdef int st = stride
    
    cdef int r0 = x0
    cdef int r1 = w1
    cdef int r2 = (x2-w2)/st+1
    cdef int r3 = (x3-w3)/st+1
    cdef np.ndarray[np.float64_t, ndim = 2] X_grad = np.zeros((x0*x1*x2,x3))

    col2im_c(x0,x2,w1,w2, w3, r2, r3, X_grad, x_grad, st)
    return X_grad

@cython.boundscheck(False) 
@cython.wraparound(False)    
cdef int col2im_c(int x0,int x2,int w1,int w2, int w3, int r2, int r3,np.ndarray[np.float64_t, ndim=2] X_grad,np.ndarray[np.float64_t, ndim=2] x_grad,int st) except ? -1:
    cdef int iter1 = -1
    cdef int iter2 = -1

    cdef int i,j,l,k,m,n
    cdef int cach1,cach2,cach3,cach4
    
    for k in range(x0):
        iter1=-1
        cach2 = k*r2*r3
        cach3 = k*w1*x2
        for l in range(w1):
            cach4 = l*x2
            for i in range(w2):
                for j in range(w3):
                    
                    iter1+=1
                    iter2 = -1
                    for m in range(r2):
                        cach1 = m*st+i
                        for n in range(r3):
                            iter2+=1
                            X_grad[cach3+cach4+cach1,n*st+j] += x_grad[cach2+iter2,iter1]  
                            
