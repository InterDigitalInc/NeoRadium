/* Copyright (c) 2026, InterDigital AI Lab */
#ifndef NREXT_CRC_H
#define NREXT_CRC_H

#include <stddef.h>
#include <stdint.h>

/* Compute CRC for m bitstreams of length n each.
 *
 * bits     - flat uint8 array, row-major: bits[i*n + j] is bit j of bitstream i
 * m        - number of bitstreams
 * n        - number of bits per bitstream
 * poly_val - generator polynomial with leading 1 excluded (e.g. 0x864CFBu for CRC-24A)
 * crc_len  - CRC degree / output length in bits (6, 11, 16, or 24)
 * out      - caller-allocated uint8 output, row-major, m * crc_len bytes; bits written MSB-first
 *
 * Polynomial values for 3GPP TS 38.212, Section 5.1:
 *   CRC-6   poly_val = 0x21      crc_len = 6
 *   CRC-11  poly_val = 0x621     crc_len = 11
 *   CRC-16  poly_val = 0x1021    crc_len = 16
 *   CRC-24A poly_val = 0x864CFB  crc_len = 24
 *   CRC-24B poly_val = 0x800063  crc_len = 24
 *   CRC-24C poly_val = 0xB2B117  crc_len = 24
 */
void nr_getCrc(const uint8_t *bits, int m, int n,
               uint32_t poly_val, int crc_len,
               uint8_t *out);

#endif /* NREXT_CRC_H */
