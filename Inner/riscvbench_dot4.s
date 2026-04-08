# riscvbench_dot4.s
# Full 4x4 tile benchmark using packed-byte dot4.
#
# Data memory layout (from benchdata.txt):
#   Apack  at 0x80, BTpack at 0x90
#   OUT    at 0xC0 (16 words)
#
# Computes C = Apack x BTpack and stores all 16 outputs.
# Expected matrix: C[0][0]=25, all others 0.

main:
        addi x14, x0, 128
        addi x15, x0, 144
        addi x2, x0, 4
        addi x17, x0, 192

loop_i:
        addi x4, x15, 0
        addi x5, x0, 4

loop_j:
        lw   x11, 0(x14)
        lw   x12, 0(x4)
        addi x9, x0, 0
        .word 0x00C5948B      # dot4 x9, x11, x12
        sw   x9, 0(x17)
        addi x17, x17, 4
        addi x4, x4, 4
        addi x5, x5, -1
        beq  x5, x0, end_j
        beq  x0, x0, loop_j

end_j:
        addi x14, x14, 4
        addi x2, x2, -1
        beq  x2, x0, finish
        beq  x0, x0, loop_i

finish:
done:   beq  x0, x0, done
