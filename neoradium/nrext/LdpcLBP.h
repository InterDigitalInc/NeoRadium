/* Copyright (c) 2026, InterDigital AI Lab */
#ifndef NREXT_LDPC_LBP_H
#define NREXT_LDPC_LBP_H

#include <stdint.h>

/*
 * LDPC Layered Belief Propagation decoder — min-sum with damping.
 *
 * Implements the same algorithm as LdpcCodecCW.decodeCodeBlocks() in ldpccodec.py.
 *
 * rx       - float32 input LLRs, shape (c, (n_cols-2)*z), row-major.
 *            The two punctured columns are prepended as zeros internally.
 * out      - float32 output beliefs, shape (c, n_cols*z), caller-allocated.
 * c        - number of code blocks
 * n_cols   - base-graph columns (68 for BG1, 52 for BG2)
 * n_rows   - base-graph rows   (46 for BG1, 42 for BG2)
 * z        - lifting size (Zc)
 * bg       - base-graph shift values, (n_rows, n_cols) int16, row-major.
 *            Negative entries indicate no edge.
 * numIter  - number of belief-propagation iterations
 * damping  - min-sum damping factor (0.75 per 3GPP convention)
 *
 * Returns 1 on success, 0 on allocation failure.
 */
int ldpc_lbp_decode(
    const float   *rx,
    float         *out,
    int            c,
    int            n_cols,
    int            n_rows,
    int            z,
    const int16_t *bg,
    int            numIter,
    float          damping
);

#endif /* NREXT_LDPC_LBP_H */
