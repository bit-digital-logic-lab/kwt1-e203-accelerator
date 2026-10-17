    .section .text
    .globl _start
    .option norvc
#ifndef KWT_TILE_START
#define KWT_TILE_START 0
#endif
#ifndef KWT_TILE_COUNT
#define KWT_TILE_COUNT 208
#endif
_start:
    li a0, 0x1000 + 32*KWT_TILE_START  # first descriptor
    li a2, KWT_TILE_COUNT              # number of tiles to submit
next_tile:
    .insn r 0x0b, 6, 0, a1, a0, zero
    bnez a1, failed
    addi a0, a0, 32
    addi a2, a2, -1
    bnez a2, next_tile
    li a3, 1            # all tiles completed
passed:
    j passed
failed:
    li a3, 2
    j failed
