from __future__ import print_function
#import numpy as np
cimport numpy as np

cimport cython

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
cpdef im2col(np.ndarray[np.int_t, ndim=1] w_shp,np.ndarray[np.int_t, ndim=1] x_shp,np.ndarray[np.int_t, ndim=1] res_shp,np.ndarray[np.float64_t, ndim=4] X_val,np.ndarray[np.float64_t, ndim=3] X_res,int stride):
    
    cdef int w0 = w_shp[0]
    cdef int w1 = w_shp[1]
    cdef int w2 = w_shp[2]
    cdef int w3 = w_shp[3]
    cdef int x0 = x_shp[0]
    cdef int x2 = x_shp[2]
    cdef int x3 = x_shp[3]
    cdef int r0 = res_shp[0]
    cdef int r2 = res_shp[2]
    cdef int r3 = res_shp[3]
    cdef int st = stride
    im2col_c(x0,w1,w2, w3, r2, r3, X_val, X_res, st)
    return X_res

@cython.boundscheck(False)    
cdef int im2col_c(int x0,int w1,int w2, int w3, int r2, int r3,np.ndarray[np.float64_t, ndim=4] X_val,np.ndarray[np.float64_t, ndim=3] X_res,int st) except ? -1:
    cdef int iter1 = -1
    cdef int iter2 = -1

    cdef int i,j,l,k,m,n
    cdef int cach1
    
    for k in range(x0):
        iter1=-1
        for l in range(w1):   
            for i in range(w2):
                for j in range(w3):
                    
                    iter1+=1
                    iter2 = -1
                    for m in range(r2):
                        cach1 = m*st+i

                        for n in range(r3):
                            iter2+=1
                            X_res[k,iter2,iter1]  =X_val[k,l,cach1,n*st+j]
                            
