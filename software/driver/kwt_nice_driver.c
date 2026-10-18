#include "kwt_nice_driver.h"

#include <stddef.h>

_Static_assert(sizeof(kwt_nice_descriptor) == 32, "descriptor must be 32 bytes");
_Static_assert(_Alignof(kwt_nice_descriptor) == 16, "descriptor must align to 16 bytes");

static int within(uint32_t base, uint32_t length, uint32_t limit)
{
    return (base & 15u) == 0 && (uint64_t)base + length <= limit;
}

static int disjoint(uint32_t a, uint32_t a_length,
                    uint32_t b, uint32_t b_length)
{
    return (uint64_t)a + a_length <= b || (uint64_t)b + b_length <= a;
}

int kwt_nice_prepare_ffn1(volatile kwt_nice_descriptor *dst,
                          const kwt_nice_layout *layout)
{
    if (dst == NULL || layout == NULL) return -1;
    const uint32_t bases[5] = {layout->descriptor_base, layout->a_base,
        layout->w_base, layout->bias_base, layout->out_base};
    const uint32_t sizes[5] = {KWT_FFN1_TILES * 32u, 13u * 64u * 8u * 2u,
        16u * 64u * 16u * 2u, 256u * 4u, KWT_FFN1_TILES * 512u};
    for (unsigned i = 0; i < 5; ++i) {
        if (!within(bases[i], sizes[i], layout->shared_bytes)) return -2;
        for (unsigned j = 0; j < i; ++j)
            if (!disjoint(bases[i], sizes[i], bases[j], sizes[j])) return -3;
    }
    for (uint32_t rt = 0; rt < KWT_FFN1_ROW_TILES; ++rt) {
        for (uint32_t ct = 0; ct < KWT_FFN1_COL_TILES; ++ct) {
            const uint32_t tile = rt * KWT_FFN1_COL_TILES + ct;
            dst[tile].a_addr = layout->a_base + rt * 1024u;
            dst[tile].w_addr = layout->w_base + ct * 2048u;
            dst[tile].bias_addr = layout->bias_base + ct * 64u;
            dst[tile].out_addr = layout->out_base + tile * 512u;
            dst[tile].control = 64u; /* K=64, ReLU disabled. GELU is optional. */
            dst[tile].masks = (rt == 12u ? 7u : 255u) | (0xffffu << 8);
            dst[tile].reserved0 = 0;
            dst[tile].reserved1 = 0;
        }
    }
    return 0;
}

int kwt_nice_run_ffn1(uint32_t descriptor_base, uint32_t *failed_tile)
{
    if (failed_tile != NULL) *failed_tile = UINT32_MAX;
    if (descriptor_base & 15u) return -1;
#if defined(__riscv) && __riscv_xlen == 32
    for (uint32_t tile = 0; tile < KWT_FFN1_TILES; ++tile) {
        uint32_t status;
        const uint32_t address = descriptor_base + tile * 32u;
        __asm__ volatile(".insn r 0x0b, 6, 0, %0, %1, zero"
                         : "=r"(status) : "r"(address) : "memory");
        if (status != 0) {
            if (failed_tile != NULL) *failed_tile = tile;
            return (int)status;
        }
    }
    return 0;
#else
    (void)descriptor_base;
    return -2; /* Host builds can validate descriptors but cannot issue NICE. */
#endif
}
