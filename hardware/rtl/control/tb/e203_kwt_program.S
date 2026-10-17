    .section .text
    .globl _start
    .option norvc
_start:
    li a0, 0x500
    li a1, 0x7f
    .insn r 0x0b, 6, 0, a1, a0, zero
done:
    j done
