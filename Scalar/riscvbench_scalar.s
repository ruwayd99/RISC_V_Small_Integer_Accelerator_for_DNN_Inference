# riscvbench_scalar.s
# Full 4x4 tile scalar benchmark using base ISA mul/add.
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
        addi x2, x0, 4
        addi x17, x0, 192

loop_i:
        addi x4, x15, 0
        addi x5, x0, 4

loop_j:
        addi x6, x14, 0
        addi x7, x4, 0
        addi x8, x0, 4
        addi x9, x0, 0

loop_k:
        lw   x11, 0(x6)
        lw   x12, 0(x7)
        mul  x13, x11, x12
        add  x9, x9, x13
        addi x6, x6, 4
        addi x7, x7, 4
        addi x8, x8, -1
        beq  x8, x0, end_k
        beq  x0, x0, loop_k

end_k:
        sw   x9, 0(x17)
        addi x17, x17, 4
        addi x4, x4, 16
        addi x5, x5, -1
        beq  x5, x0, end_j
        beq  x0, x0, loop_j

end_j:
        addi x14, x14, 16
        addi x2, x2, -1
        beq  x2, x0, finish
        beq  x0, x0, loop_i

finish:
done:   beq  x0, x0, done
