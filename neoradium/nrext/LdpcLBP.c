/* Copyright (c) 2026, InterDigital AI Lab */
/*
 * LDPC Layered Belief Propagation decoder — min-sum with damping.
 *
 * Direct port of LdpcCodecCW.decodeCodeBlocks() from ldpccodec.py.
 * The algorithm processes each row of the base graph as a "layer", updating
 * extrinsic messages (ll) and beliefs (r) in place.
 */

#include "LdpcLBP.h"

#include <stdlib.h>
#include <string.h>

/* ──────────────────────────────────────────────────────────────────────────── */

int ldpc_lbp_decode(
    const float   *rx,
    float         *out,
    int            c,
    int            n_cols,
    int            n_rows,
    int            z,
    const int16_t *bg,
    int            numIter,
    float          damping) {
    /*
     * Memory layout notes:
     *   r       [c, n_cols, z]          — beliefs; initialised from rx with 2z zeros prepended
     *   ll      [total_nnz * c * z]     — extrinsic messages, one slice per row; indexed via row_offsets
     *   tmp/nl  [c * max_q * z]         — workspace for the current layer's messages
     *   nn_cols / nn_shifts — per-row lists of non-zero column indices and their cyclic-shift values
     */

    int   *row_q       = NULL;
    int   *row_offsets = NULL;
    int   *nn_cols_all = NULL;   /* [n_rows * n_cols] — stores q non-neg col indices for each row */
    int   *nn_shft_all = NULL;   /* [n_rows * n_cols] — matching cyclic-shift values */
    float *ll          = NULL;
    float *r           = NULL;
    float *tmp         = NULL;
    float *nl          = NULL;

    /* ── Pre-compute per-row metadata ─────────────────────────────────────── */

    row_q       = (int *)malloc(n_rows * sizeof(int));
    row_offsets = (int *)malloc((n_rows + 1) * sizeof(int));
    nn_cols_all = (int *)malloc(n_rows * n_cols * sizeof(int));
    nn_shft_all = (int *)malloc(n_rows * n_cols * sizeof(int));
    if (!row_q || !row_offsets || !nn_cols_all || !nn_shft_all) goto fail;

    int max_q = 0;
    row_offsets[0] = 0;
    for (int row = 0; row < n_rows; row++) {
        int q = 0;
        for (int col = 0; col < n_cols; col++) {
            int16_t s = bg[row * n_cols + col];
            if (s >= 0) {
                nn_cols_all[row * n_cols + q] = col;
                nn_shft_all[row * n_cols + q] = (int)s;
                q++;
            }
        }
        row_q[row] = q;
        if (q > max_q) max_q = q;
        row_offsets[row + 1] = row_offsets[row] + c * q * z;
    }

    /* ── Allocate working arrays ──────────────────────────────────────────── */

    ll  = (float *)calloc((size_t)row_offsets[n_rows], sizeof(float));
    r   = (float *)calloc((size_t)(c * n_cols * z),    sizeof(float));
    tmp = (float *)malloc((size_t)(c * max_q * z)      * sizeof(float));
    nl  = (float *)malloc((size_t)(c * max_q * z)      * sizeof(float));
    if (!ll || !r || !tmp || !nl) goto fail;

    /* Initialise r: the first two columns are zero-filled (punctured bits);
       the remaining (n_cols-2)*z values come from rx. */
    for (int ci = 0; ci < c; ci++) {
        const float *src = rx + (size_t)ci * (n_cols - 2) * z;
        float       *dst = r  + (size_t)ci *  n_cols      * z + 2 * z;
        memcpy(dst, src, (size_t)(n_cols - 2) * z * sizeof(float));
    }

    /* ── Main BP loop ─────────────────────────────────────────────────────── */

    for (int iter = 0; iter < numIter; iter++) {
        for (int row = 0; row < n_rows; row++) {
            const int   q        = row_q[row];
            const int  *nn_cols  = nn_cols_all + row * n_cols;
            const int  *nn_shfts = nn_shft_all + row * n_cols;
            float      *ll_row   = ll + row_offsets[row];

            /* Step 1 — subtract old extrinsic messages from beliefs */
            for (int ci = 0; ci < c; ci++) {
                for (int j = 0; j < q; j++) {
                    float *r_col   = r      + (size_t)ci * n_cols * z + nn_cols[j] * z;
                    float *ll_cur  = ll_row + (size_t)ci * q      * z + j          * z;
                    for (int b = 0; b < z; b++) r_col[b] -= ll_cur[b];
                }
            }

            /* Step 2 — forward cyclic shift: tmp[ci,j,b] = r[ci, col_j, (k_j+b)%z] */
            for (int ci = 0; ci < c; ci++) {
                for (int j = 0; j < q; j++) {
                    const float *r_col = r   + (size_t)ci * n_cols * z + nn_cols[j] * z;
                    float       *t     = tmp + (size_t)ci * q      * z + j          * z;
                    int k = nn_shfts[j];
                    for (int b = 0; b < z; b++) t[b] = r_col[(k + b) % z];
                }
            }

            /* Step 3 — min-sum: for each (ci, b) find min1/min2/parity,
               output nl[ci,j,b] = magnitude * parity_without_j */
            for (int ci = 0; ci < c; ci++) {
                for (int b = 0; b < z; b++) {
                    float parity  = 1.0f;
                    float min1    = 1.0e38f;
                    float min2    = 1.0e38f;
                    int   min_idx = 0;

                    for (int j = 0; j < q; j++) {
                        float v  = tmp[(size_t)ci * q * z + j * z + b];
                        float av = (v < 0.0f) ? -v : v;
                        if (v < 0.0f) parity = -parity;
                        if (av < min1) { min2 = min1; min1 = av; min_idx = j; }
                        else if (av < min2) { min2 = av; }
                    }

                    for (int j = 0; j < q; j++) {
                        float v              = tmp[(size_t)ci * q * z + j * z + b];
                        float sign_j         = (v < 0.0f) ? -1.0f : 1.0f;
                        float parity_wo_j    = parity * sign_j;  /* sign_j^2 == 1 → this is prod_{k≠j} sign_k */
                        float mag            = (j == min_idx) ? min2 : min1;
                        nl[(size_t)ci * q * z + j * z + b] = mag * parity_wo_j;
                    }
                }
            }

            /* Step 4 — inverse cyclic shift + damping:
               ll[row][ci,j,b] = nl[ci,j,(b-k_j+z)%z] * damping */
            for (int ci = 0; ci < c; ci++) {
                for (int j = 0; j < q; j++) {
                    int   k      = nn_shfts[j];
                    float *ll_cur = ll_row + (size_t)ci * q * z + j * z;
                    float *n      = nl     + (size_t)ci * q * z + j * z;
                    for (int b = 0; b < z; b++) ll_cur[b] = n[(b - k + z) % z] * damping;
                }
            }

            /* Step 5 — add new extrinsic messages back to beliefs */
            for (int ci = 0; ci < c; ci++) {
                for (int j = 0; j < q; j++) {
                    float *r_col  = r      + (size_t)ci * n_cols * z + nn_cols[j] * z;
                    float *ll_cur = ll_row + (size_t)ci * q      * z + j          * z;
                    for (int b = 0; b < z; b++) r_col[b] += ll_cur[b];
                }
            }
        }
    }

    /* Copy beliefs to caller's output buffer */
    memcpy(out, r, (size_t)c * n_cols * z * sizeof(float));

    free(row_q); free(row_offsets); free(nn_cols_all); free(nn_shft_all);
    free(ll); free(r); free(tmp); free(nl);
    return 1;

fail:
    free(row_q); free(row_offsets); free(nn_cols_all); free(nn_shft_all);
    free(ll); free(r); free(tmp); free(nl);
    return 0;
}
