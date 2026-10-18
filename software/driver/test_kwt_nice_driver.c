#include "kwt_nice_driver.h"

#include <assert.h>
#include <stdio.h>

static kwt_nice_descriptor descriptors[KWT_FFN1_TILES];

int main(void)
{
    const kwt_nice_layout map = {0x1000, 0x3000, 0x7000, 0xf000,
                                 0x10000, 512u * 1024u};
    assert(kwt_nice_prepare_ffn1(descriptors, &map) == 0);
    assert(descriptors[0].a_addr == 0x3000);
    assert(descriptors[0].w_addr == 0x7000);
    assert(descriptors[0].bias_addr == 0xf000);
    assert(descriptors[0].out_addr == 0x10000);
    assert(descriptors[0].control == 64);
    assert(descriptors[0].masks == 0x00ffffffu);
    assert(descriptors[207].a_addr == 0x3000 + 12 * 1024);
    assert(descriptors[207].w_addr == 0x7000 + 15 * 2048);
    assert(descriptors[207].bias_addr == 0xf000 + 15 * 64);
    assert(descriptors[207].out_addr == 0x10000 + 207 * 512);
    assert(descriptors[207].masks == 0x00ffff07u);
    for (unsigned rt = 0; rt < KWT_FFN1_ROW_TILES; ++rt)
        for (unsigned ct = 0; ct < KWT_FFN1_COL_TILES; ++ct) {
            const unsigned tile = rt * KWT_FFN1_COL_TILES + ct;
            assert(descriptors[tile].a_addr == 0x3000u + rt * 1024u);
            assert(descriptors[tile].w_addr == 0x7000u + ct * 2048u);
            assert(descriptors[tile].bias_addr == 0xf000u + ct * 64u);
            assert(descriptors[tile].out_addr == 0x10000u + tile * 512u);
            assert(descriptors[tile].control == 64u);
            assert(descriptors[tile].masks ==
                   ((rt == 12u ? 7u : 255u) | (0xffffu << 8)));
            assert(descriptors[tile].reserved0 == 0u);
            assert(descriptors[tile].reserved1 == 0u);
        }
    kwt_nice_layout overlap = map;
    overlap.w_base = overlap.a_base;
    assert(kwt_nice_prepare_ffn1(descriptors, &overlap) == -3);
    kwt_nice_layout out_of_range = map;
    out_of_range.shared_bytes = 0x20000;
    assert(kwt_nice_prepare_ffn1(descriptors, &out_of_range) == -2);
    puts("PASS provisional NICE FFN1 descriptor layout and bounds");
    return 0;
}
