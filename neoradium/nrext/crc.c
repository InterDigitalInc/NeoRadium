/* Copyright (c) 2026, InterDigital AI Lab */
#include "crc.h"

/* Shift-register CRC matching 3GPP TS 38.212, Section 5.1 polynomial long division.
 *
 * The algorithm feeds all n data bits into the shift register, then feeds crc_len
 * appended zero bits to flush the register — equivalent to computing
 * (input_polynomial * x^crc_len) mod g(x) over GF(2).
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
               uint8_t *out) {
    uint32_t mask = (1u << crc_len) - 1u;

    for (int i = 0; i < m; i++) {
        const uint8_t *row     = bits + (size_t)i * (size_t)n;
        uint8_t       *row_out = out  + (size_t)i * (size_t)crc_len;
        uint32_t reg = 0u;

        /* Feed data bits MSB-first */
        for (int j = 0; j < n; j++) {
            uint32_t msb = (reg >> (crc_len - 1)) & 1u;
            reg = ((reg << 1) | (row[j] & 1u)) & mask;
            if (msb) reg ^= poly_val;
        }

        /* Flush: feed crc_len appended zero bits */
        for (int j = 0; j < crc_len; j++) {
            uint32_t msb = (reg >> (crc_len - 1)) & 1u;
            reg = (reg << 1) & mask;
            if (msb) reg ^= poly_val;
        }

        /* Write CRC bits MSB-first */
        for (int k = 0; k < crc_len; k++) {
            row_out[k] = (uint8_t)((reg >> (crc_len - 1 - k)) & 1u);
        }
    }
}
