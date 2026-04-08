# riscvbench_outer.s
# Full 4x4 tile benchmark using mload/mmatmul/mstore.
#
# Data memory layout (from benchdata.txt):
#   Acol at 0xA0, Brow at 0xB0
#   OUT  at 0xC0 (16 words)
#
# Computes C = A x B and stores all 16 outputs.
# Expected matrix: C[0][0]=25, all others 0.

main:
        lw   x2, 160(x0)
        lw   x3, 176(x0)
        .word 0x0031200B      # mmatmul x2, x3
        lw   x2, 164(x0)
        lw   x3, 180(x0)
        .word 0x0031200B      # mmatmul x2, x3
        lw   x2, 168(x0)
        lw   x3, 184(x0)
        .word 0x0031200B      # mmatmul x2, x3
        lw   x2, 172(x0)
        lw   x3, 188(x0)
        .word 0x0031200B      # mmatmul x2, x3

        .word 0x0C00202B      # mstore C[0],  192(x0)
        .word 0x0C10222B      # mstore C[1],  196(x0)
        .word 0x0C20242B      # mstore C[2],  200(x0)
        .word 0x0C30262B      # mstore C[3],  204(x0)
        .word 0x0C40282B      # mstore C[4],  208(x0)
        .word 0x0C502A2B      # mstore C[5],  212(x0)
        .word 0x0C602C2B      # mstore C[6],  216(x0)
        .word 0x0C702E2B      # mstore C[7],  220(x0)
        .word 0x0E80202B      # mstore C[8],  224(x0)
        .word 0x0E90222B      # mstore C[9],  228(x0)
        .word 0x0EA0242B      # mstore C[10], 232(x0)
        .word 0x0EB0262B      # mstore C[11], 236(x0)
        .word 0x0EC0282B      # mstore C[12], 240(x0)
        .word 0x0ED02A2B      # mstore C[13], 244(x0)
        .word 0x0EE02C2B      # mstore C[14], 248(x0)
        .word 0x0EF02E2B      # mstore C[15], 252(x0)

done:   beq  x0, x0, done
