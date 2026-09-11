/*
 Copyright (c) 2026, InterDigital AI Lab

 Python C extension entry point for NeoRadium's nrext package.
 * Each functional area has its own .c/.h pair (e.g. crc.c / crc.h).
 * To add a new area: implement it in area.c / area.h, include the header
 * here, add a py_<function> wrapper, register it in NrextMethods, and add
 * area.c to the sources list in setup.py.
 */

/* Note:
   To see if this extension is actually being used in python, do the following:
 
     from neoradium.nrext import HAS_C_EXT
     print(HAS_C_EXT)   # should print True
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>

#include "crc.h"
#include "LdpcLBP.h"

/* ****************************************************************************************************************** */
/* CRC                                                                                                                */
/* ****************************************************************************************************************** */

/* Map a polynomial name string to its shift-register integer and degree. */
static int get_poly(const char *name, uint32_t *poly_val, int *crc_len) {
    if (strcmp(name, "6")   == 0) { *poly_val = 0x21u;     *crc_len = 6;  return 0; }
    if (strcmp(name, "11")  == 0) { *poly_val = 0x621u;    *crc_len = 11; return 0; }
    if (strcmp(name, "16")  == 0) { *poly_val = 0x1021u;   *crc_len = 16; return 0; }
    if (strcmp(name, "24A") == 0) { *poly_val = 0x864CFBu; *crc_len = 24; return 0; }
    if (strcmp(name, "24B") == 0) { *poly_val = 0x800063u; *crc_len = 24; return 0; }
    if (strcmp(name, "24C") == 0) { *poly_val = 0xB2B117u; *crc_len = 24; return 0; }
    return -1;
}

/* ****************************************************************************************************************** */
static PyObject *py_getCrc(PyObject *self, PyObject *args) {
    PyObject   *bits_obj;
    const char *poly_str;
    if (!PyArg_ParseTuple(args, "Os", &bits_obj, &poly_str))    return NULL;

    uint32_t poly_val;
    int      crc_len;
    if (get_poly(poly_str, &poly_val, &crc_len) < 0) {
        PyErr_SetString(PyExc_ValueError, "'poly' must be one of: '6', '11', '16', '24A', '24B', '24C'");
        return NULL;
    }

    /* Accept any integer dtype; convert to uint8 contiguous array. */
    PyArrayObject *bits_arr = (PyArrayObject *)PyArray_FROM_OTF(bits_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (bits_arr == NULL)   return NULL;

    int ndim = PyArray_NDIM(bits_arr);
    int m, n;
    if (ndim == 1) {
        m = 1;
        n = (int)PyArray_DIM(bits_arr, 0);
    }
    else if (ndim == 2) {
        m = (int)PyArray_DIM(bits_arr, 0);
        n = (int)PyArray_DIM(bits_arr, 1);
    }
    else {
        Py_DECREF(bits_arr);
        PyErr_SetString(PyExc_ValueError, "'bits' must be a 1D or 2D array");
        return NULL;
    }

    /* Allocate output: same shape as input but with last dimension = crc_len */
    npy_intp    out_dims[2];
    PyArrayObject *out_arr;
    if (ndim == 1) {
        out_dims[0] = crc_len;
        out_arr = (PyArrayObject *)PyArray_SimpleNew(1, out_dims, NPY_UINT8);
    }
    else {
        out_dims[0] = m;
        out_dims[1] = crc_len;
        out_arr = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_UINT8);
    }
    if (out_arr == NULL) { Py_DECREF(bits_arr); return NULL; }

    nr_getCrc((const uint8_t *)PyArray_DATA(bits_arr), m, n,
              poly_val, crc_len,
              (uint8_t *)PyArray_DATA(out_arr));

    Py_DECREF(bits_arr);
    return (PyObject *)out_arr;
}

/* ****************************************************************************************************************** */
/* LDPC Layered Belief Propagation                                                                                    */
/* ****************************************************************************************************************** */

/*
 * decodeLBP(llrs, bg, z, numIter, damping) -> ndarray
 *
 * llrs    : float32 ndarray, shape (c, (n_cols-2)*z)
 * bg      : int16  ndarray, shape (n_rows, n_cols)   base-graph cyclic shifts
 * z       : int    lifting size
 * numIter : int    number of BP iterations
 * damping : float  min-sum damping factor (0.75)
 *
 * Returns float32 ndarray of shape (c, n_cols*z) containing the output beliefs.
 */
static PyObject *py_decodeLBP(PyObject *self, PyObject *args) {
    PyObject *llrs_obj, *bg_obj;
    int       z, numIter;
    float     damping;

    if (!PyArg_ParseTuple(args, "OOiif", &llrs_obj, &bg_obj, &z, &numIter, &damping))
        return NULL;

    PyArrayObject *llrs_arr = (PyArrayObject *)PyArray_FROM_OTF(llrs_obj, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *bg_arr   = (PyArrayObject *)PyArray_FROM_OTF(bg_obj,   NPY_INT16, NPY_ARRAY_IN_ARRAY);
    
    if (!llrs_arr || !bg_arr) {
        Py_XDECREF(llrs_arr); Py_XDECREF(bg_arr);
        return NULL;
    }

    if (PyArray_NDIM(llrs_arr) != 2 || PyArray_NDIM(bg_arr) != 2) {
        Py_DECREF(llrs_arr); Py_DECREF(bg_arr);
        PyErr_SetString(PyExc_ValueError, "'llrs' must be 2D and 'bg' must be 2D");
        return NULL;
    }

    int c      = (int)PyArray_DIM(llrs_arr, 0);
    int n_rows = (int)PyArray_DIM(bg_arr,   0);
    int n_cols = (int)PyArray_DIM(bg_arr,   1);

    /* Allocate output: (c, n_cols * z) float32 */
    npy_intp out_dims[2] = { c, n_cols * z };
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_SimpleNew(2, out_dims, NPY_FLOAT);
    if (!out_arr) { Py_DECREF(llrs_arr); Py_DECREF(bg_arr); return NULL; }

    int ok = ldpc_lbp_decode(
        (const float   *)PyArray_DATA(llrs_arr),
        (float         *)PyArray_DATA(out_arr),
        c, n_cols, n_rows, z,
        (const int16_t *)PyArray_DATA(bg_arr),
        numIter, damping);

    Py_DECREF(llrs_arr); Py_DECREF(bg_arr);

    if (!ok) {
        Py_DECREF(out_arr);
        PyErr_NoMemory();
        return NULL;
    }
    return (PyObject *)out_arr;
}

/* ── Method table ─────────────────────────────────────────────────────── */

static PyMethodDef NrextMethods[] = {
    {"getCrc", py_getCrc, METH_VARARGS,
     "getCrc(bits, poly) -> ndarray\n\n"
     "Compute CRC bits for one or more bitstreams.\n"
     "bits: uint8 ndarray, 1D (n,) or 2D (m, n).\n"
     "poly: one of '6', '11', '16', '24A', '24B', '24C'.\n"
     "Returns uint8 ndarray of shape (crc_len,) or (m, crc_len)."},
    
    {"decodeLBP", py_decodeLBP, METH_VARARGS,
     "decodeLBP(llrs, bg, z, numIter, damping) -> ndarray\n\n"
     "LDPC layered belief-propagation decoder (min-sum with damping).\n"
     "llrs:    float32 ndarray, shape (c, (n_cols-2)*z).\n"
     "bg:      int16  ndarray, shape (n_rows, n_cols) — base-graph cyclic shifts.\n"
     "z:       int    lifting size (Zc).\n"
     "numIter: int    number of BP iterations.\n"
     "damping: float  min-sum damping factor (0.75).\n"
     "Returns float32 ndarray of shape (c, n_cols*z) containing output beliefs."},
    
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef NrextModule = {
    PyModuleDef_HEAD_INIT, "_ext", NULL, -1, NrextMethods
};

PyMODINIT_FUNC PyInit__ext(void) {
    import_array();   /* initialise numpy C API; returns NULL on failure */
    return PyModule_Create(&NrextModule);
}
