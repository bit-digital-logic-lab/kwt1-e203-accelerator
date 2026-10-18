#ifndef KWT_NICE_DRIVER_H
#define KWT_NICE_DRIVER_H

#include <stdint.h>

enum { KWT_FFN1_ROWS = 99, KWT_FFN1_K = 64, KWT_FFN1_COLS = 256,
       KWT_FFN1_ROW_TILES = 13, KWT_FFN1_COL_TILES = 16,
       KWT_FFN1_TILES = 208 };

/* Two 128-bit descriptor beats in the current NICE-like simulation model. */
typedef struct __attribute__((aligned(16))) {
    uint32_t a_addr, w_addr, bias_addr, out_addr;
    uint32_t control, masks, reserved0, reserved1;
} kwt_nice_descriptor;

typedef struct {
    uint32_t descriptor_base, a_base, w_base, bias_base, out_base;
    uint32_t shared_bytes;
} kwt_nice_layout;

/* Prepare 208 descriptors for already packed INT16 A/W and INT32 bias/Y. */
int kwt_nice_prepare_ffn1(volatile kwt_nice_descriptor *dst,
                          const kwt_nice_layout *layout);

/* Submit every descriptor with E203 custom-0/funct3=110, stop on first error. */
int kwt_nice_run_ffn1(uint32_t descriptor_base, uint32_t *failed_tile);

#endif
