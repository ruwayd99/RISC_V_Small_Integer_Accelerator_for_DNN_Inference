# riscvbench_scalar.s
# Full 4x4 tile scalar benchmark using base ISA mul/add (IKJ order).
#
# Data memory layout (from benchdata.txt):
#   A32  at 0x00, BT32 at 0x40
#   OUT  at 0xC0 (16 words)
#
# Computes C = A32 x BT32 and stores all 16 outputs.
# Expected matrix: C[0][0]=25, all others 0.

main:
        addi x14, x0, 0
        addi x15, x0, 64
        addi x17, x0, 192
        addi x2, x0, 4

loop_i:
        addi x20, x0, 0
        addi x21, x0, 0
        addi x22, x0, 0
        addi x23, x0, 0
        addi x6, x14, 0
        addi x4, x15, 0
        addi x5, x0, 4

loop_k:
        lw   x11, 0(x6)
        addi x7, x4, 0
        lw   x12, 0(x7)
        mul  x13, x11, x12
        add  x20, x20, x13
        lw   x12, 16(x7)
        mul  x13, x11, x12
        add  x21, x21, x13
        lw   x12, 32(x7)
        mul  x13, x11, x12
        add  x22, x22, x13
        lw   x12, 48(x7)
        mul  x13, x11, x12
        addi x6, x6, 4
        add  x23, x23, x13
        addi x4, x4, 4
        addi x5, x5, -1
        beq  x5, x0, end_k
        beq  x0, x0, loop_k

end_k:
        sw   x20, 0(x17)
        sw   x21, 4(x17)
        sw   x22, 8(x17)
        sw   x23, 12(x17)
        addi x17, x17, 16
        addi x14, x14, 16
        addi x2, x2, -1
        beq  x2, x0, finish
        beq  x0, x0, loop_i

finish:
done:   beq  x0, x0, done
