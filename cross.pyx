import numpy as np
cimport numpy as np
cimport cython

ctypedef np.double_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
def cross(np.ndarray[object,ndim=1] df1_symbols,np.ndarray[long, ndim=1] df1_times,np.ndarray[long, ndim=1] df2_times,
            np.ndarray[object,ndim=1] df2_columns,np.ndarray[DTYPE_t, ndim=2] df2):
    """ Frame 1 Symbols, Frame 1 times (RAW-EPOCH-LONG-FORM), Frame 2 times,
    Frame 2 Column Names to Join, Frame 2 (ENTIRE)"""
    cdef:
        long sym_len = df1_symbols.shape[0],df1_times_len=df1_times.shape[0]
        long df2_times_len=df2_times.shape[0],df2_len=df2.shape[0]
        long i,j=0
        np.ndarray[DTYPE_t, ndim=1] res = np.zeros(sym_len, dtype=np.double) * np.NaN
        dict sym_lookup = dict(zip(df2_columns,range(len(df2_columns))))
    assert(sym_len==df1_times_len)
    assert(df2_times_len==df2_len)
    for i from 0 <= i < sym_len:
        while (j+1) < df2_len and df1_times[i] >= df2_times[j+1]:
            j+=1
        if df1_times[i] >= df2_times[j] and sym_lookup.has_key(df1_symbols[i]):
            res[i] = df2[j,sym_lookup[df1_symbols[i]]]
        else:
            res[i] = np.NaN
    return res