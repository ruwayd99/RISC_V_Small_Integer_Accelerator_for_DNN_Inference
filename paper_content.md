# Paper Content — Section by Section

> This file contains word-for-word content for each section of paper.tex.
> Transfer each section into LaTeX. Placeholders for figures/tables/additional data are marked with **[TODO: ...]**.

---

## Keywords

RISC-V, custom ISA extensions, matrix multiply, dot product, outer product, DNN inference, INT8, FPGA, accelerator

---

## I. Introduction

Deep neural network (DNN) inference on resource-constrained embedded devices demands high arithmetic throughput at minimal area and power cost. The dominant computation in DNN inference is matrix multiplication, specifically the multiply-accumulate (MAC) operation applied across large matrices of quantized integer weights and activations. General-purpose processors execute matrix multiplication element by element, resulting in poor utilization of available datapath hardware and excessive instruction fetch and decode overhead.

RISC-V International has recognized this bottleneck and is developing several instruction set extensions to accelerate dot product and matrix multiply computations. The proposed RISC-V Matrix (Zmatrix) extension targets maximum performance with support for a wide range of floating-point data types, but it requires significant area overhead that is impractical for small embedded cores. At the other extreme, a simple dot product instruction is easy to implement but captures only modest parallelism.

In this paper, we explore the design space between these two extremes by implementing and comparing three progressively more parallel approaches to 4x4 integer matrix multiplication on a minimal RV32I processor:

1. **Scalar baseline**: Using the standard `mul` and `add` instructions with software loops, representing the performance of an unmodified RISC-V core with the M extension.
2. **Inner dot product**: A custom `dot4` instruction that computes four packed 8-bit multiply-accumulate operations in a single cycle, exploiting data-level parallelism within each output element.
3. **Outer product**: A custom `mmatmul` instruction that performs 16 parallel 8-bit MAC operations in a single cycle, updating an entire 4x4 tile of accumulator registers in one instruction.

All three approaches are implemented on the same 2-stage pipelined RV32I base processor, synthesized to an Intel Cyclone V FPGA, and evaluated on an identical 4x4 matrix multiplication workload. Our results show that the outer product approach achieves a 29.2x reduction in cycle count over the scalar baseline while requiring only a modest increase in hardware resources (411 additional ALMs and 16 additional DSP blocks), demonstrating that significant acceleration is achievable with simple custom extensions on a minimal RISC-V core.

The remainder of this paper is organized as follows. Section II provides background on RISC-V custom extensions and related work in matrix acceleration. Section III describes the base processor architecture and the three matrix multiply implementations in detail. Section IV presents our experimental evaluation. Section V discusses the results and trade-offs, and Section VI concludes the paper.

---

## II. Background and Related Work

### A. RISC-V Custom Extension Space

The RISC-V ISA is designed with extensibility as a first-class feature. The base integer instruction set (RV32I) reserves four custom opcode ranges — custom-0 through custom-3 — specifically for user-defined extensions. This allows designers to add application-specific instructions without conflicting with the standard ISA or future ratified extensions. In this work, we use the custom-0 opcode (0001011) for compute instructions and the custom-1 opcode (0101011) for coprocessor data movement instructions.

### B. Matrix Multiply Acceleration Approaches

Matrix multiplication C = C + A x B can be decomposed in three fundamental ways, each exposing different parallelism:

**Scalar (element-wise):** The innermost operation computes a single MAC: C[i][j] += A[i][k] * B[k][j]. This requires three nested loops and achieves no data-level parallelism beyond what the base multiplier provides.

**Inner product (dot product):** A row of A is multiplied element-wise with a column of B and the products are summed to produce one element of C. By packing multiple small integers into a single register, a dot product instruction can compute several MACs in parallel. This approach requires two nested loops (over i and j), with the inner k-loop replaced by the hardware dot product.

**Outer product:** A column of A is multiplied with a row of B to produce a rank-1 update to the entire C matrix. For 4-element vectors, one outer product operation performs 16 MACs simultaneously, updating all 16 elements of a 4x4 tile. This eliminates all software loops over the tile and requires only iteration over the shared dimension k.

### C. Related Work

Google's Tensor Processing Unit (TPU) employs a systolic array that computes outer products to achieve high throughput for DNN inference. NVIDIA's Tensor Cores similarly operate on small matrix tiles (e.g., 4x4) using mixed-precision arithmetic. At the embedded scale, ARM's Helium (MVE) extension for Cortex-M processors provides SIMD dot product instructions for INT8 data, analogous to our inner product approach. The RISC-V Packed SIMD (P extension) proposal includes dot product operations for packed integers but has not been ratified. Our work differs from these efforts by comparing inner and outer product approaches on the same minimal RISC-V core to quantify the performance and area trade-offs at the smallest practical scale.

**[TODO: Add 2-3 more citations to recent RISC-V matrix extension proposals, e.g., the Zve/Zvl vector extensions, any published RISC-V matrix accelerator papers, and the official RISC-V Matrix extension draft specification.]**

---

## III. Architecture and Implementation

### A. Base Processor

Our base processor is derived from the single-cycle RV32I implementation published in the textbook *Digital Design and Computer Architecture: RISC-V Edition* by Harris and Harris. The original design supports approximately 12 instructions including `add`, `sub`, `and`, `or`, `slt`, `addi`, `lw`, `sw`, `beq`, and `jal`.

We convert this to a 2-stage pipeline with Fetch and Execute stages separated by a pipeline register that captures the instruction word, program counter, and PC+4. The Fetch stage reads from instruction memory and computes the next sequential address. The Execute stage performs instruction decoding, register file access, ALU computation, data memory access, and register writeback — all within a single clock cycle.

Branch handling follows a flush-on-taken policy: when a branch or jump is resolved in the Execute stage, the incorrectly fetched instruction in the pipeline register is replaced with a NOP (encoded as `addi x0, x0, 0`, opcode 0x00000013). This incurs a one-cycle penalty per taken branch. Data hazards between consecutive dependent instructions are resolved through register file forwarding, where the write occurs at the rising clock edge and the read sees the updated value in the same cycle.

We extend the base ISA with the full RISC-V M extension multiply instructions by adding a dedicated 32-bit multiplier alongside the ALU. The multiplier supports all four standard variants: `mul` (lower 32 bits of signed product), `mulh` (upper 32 bits of signed-by-signed product), `mulhsu` (upper 32 bits of signed-by-unsigned product), and `mulhu` (upper 32 bits of unsigned-by-unsigned product). These variants are distinguished by the funct3 field (000, 001, 010, 011 respectively) under the R-type opcode 0110011 with funct7 = 0000001. The multiplier result is selected through an existing result multiplexer. This extended processor serves as the common platform for all three matrix multiply approaches.

**[TODO: Insert Figure 1 — Block diagram of the 2-stage pipelined base processor showing Fetch stage, pipeline register, and Execute stage with ALU, multiplier, register file, and data memory.]**

### B. Scalar Matrix Multiply

The scalar approach uses only the base RV32I instructions plus `mul` from the M extension. Matrix multiplication is performed with three nested software loops iterating over the row index i, column index j, and shared dimension k. The innermost loop body loads one 32-bit element from matrix A and one from the transposed matrix B^T, multiplies them, and accumulates the product into a running sum:

```
loop_k:
    lw   x11, 0(x6)        // Load A[i][k]
    lw   x12, 0(x7)        // Load BT[j][k]
    mul  x13, x11, x12     // temp = A[i][k] * BT[j][k]
    add  x9, x9, x13       // C[i][j] += temp
    addi x6, x6, 4         // Advance A pointer
    addi x7, x7, 4         // Advance BT pointer
    addi x8, x8, -1        // Decrement k counter
    beq  x8, x0, end_k     // Exit if k == 0
    beq  x0, x0, loop_k    // Unconditional branch back
```

Each element of the output matrix requires a full traversal of the k-loop (4 iterations for a 4x4 tile), and the complete 4x4 matrix multiply requires 4 x 4 x 4 = 64 scalar MAC operations. Including loop overhead (counter updates, pointer arithmetic, and branches), the total program length is 54 instructions with an execution cost of 818 cycles.

Matrix data is stored in 32-bit integer format. Matrix A occupies addresses 0x00–0x3F (16 words), and the pre-transposed matrix B^T occupies 0x40–0x7F. The output matrix C is written to addresses 0xC0–0xFF.

### C. Inner Dot Product (dot4)

The inner dot product approach introduces two custom instructions that share opcode custom-0 (0001011) and are distinguished by the funct3 field:

**MAC instruction (funct3 = 000):** Computes `rd = rd + rs1 * rs2` as a 32-bit multiply-accumulate operation. This requires a third read port on the register file to access the current value of rd.

**DOT4 instruction (funct3 = 001):** Computes the packed 8-bit dot product with accumulation:

```
rd = rd + rs1[7:0]*rs2[7:0] + rs1[15:8]*rs2[15:8]
       + rs1[23:16]*rs2[23:16] + rs1[31:24]*rs2[31:24]
```

Each source register holds four packed 8-bit integers. The hardware extracts the four byte lanes from each source operand, performs four parallel 8x8-bit unsigned multiplications producing 16-bit products, sums the four products through an adder tree, and accumulates the result into the 32-bit destination register.

The `mac` and `dot4` instructions are implemented in a shared `mac_dot4` hardware module that contains the four 8x8-bit multipliers and an adder tree. A mode signal selects between `mac` (accumulate with the full 32-bit multiplier result) and `dot4` (accumulate with the packed 8-bit dot product sum). The module receives the 32-bit multiplier result as an input, allowing the `mac` instruction to reuse the existing multiplier that also serves the `mul`/`mulh`/`mulhsu`/`mulhu` instructions. The module output is routed to the result multiplexer as a fourth input (ResultSrc = 11).

With `dot4`, the k-loop is entirely eliminated. Each `dot4` instruction processes all four k-elements in parallel, so computing one output element requires only one `lw` (load packed A row), one `lw` (load packed B^T row), one register clear, and one `dot4`. The program reduces to two nested loops over i and j:

```
loop_j:
    lw   x11, 0(x14)       // Load packed A[i][0..3]
    lw   x12, 0(x4)        // Load packed BT[j][0..3]
    addi x9, x0, 0         // Clear accumulator
    dot4 x9, x11, x12      // C[i][j] = dot(A_row, BT_row)
    sw   x9, 0(x17)        // Store result
    ...                     // Pointer and counter updates
```

The packed 8-bit data is stored in a separate region of data memory: packed A rows at 0x80–0x8F (4 words) and packed B^T rows at 0x90–0x9F (4 words). The total program length is 41 instructions with an execution cost of 194 cycles.

**[TODO: Insert Figure 2 — Block diagram of the mac_dot4 module showing the four 8x8 multipliers, adder tree, 32-bit multiplier, and mode selection mux.]**

### D. Outer Product (mmatmul)

The outer product approach introduces a coprocessor with 16 dedicated 32-bit accumulator registers (C[0] through C[15]) that sit outside the main register file. Three new custom instructions control this coprocessor:

**MMATMUL (opcode 0001011, funct3 = 010):** Performs a rank-1 outer product update on all 16 accumulator registers simultaneously. Source registers rs1 and rs2 each hold four packed 8-bit integers representing a column of A and a row of B, respectively. The hardware computes:

```
for r in 0..3:
    for c in 0..3:
        C[r*4 + c] += A[r] * B[c]
```

This is 16 parallel signed 8-bit MAC operations executed in a single cycle.

**MLOAD (opcode 0101011, funct3 = 000):** Loads a 32-bit value from data memory into one of the 16 accumulator registers. The accumulator index is encoded in the rd field (bits [11:7], lower 4 bits used). The memory address is computed as rs1 + immediate using standard I-type encoding.

**MSTORE (opcode 0101011, funct3 = 010):** Stores one accumulator register value to data memory. The accumulator index is encoded in the rs2 field (bits [24:20], lower 4 bits used). The memory address is computed as rs1 + immediate using standard S-type encoding.

The outer product coprocessor is implemented as a separate `outer_product` module instantiated in the datapath alongside the main ALU and multiplier. It receives `do_mmatmul` and `do_mload` control signals from the expanded 14-bit main decoder. The `mstore` instruction routes the selected accumulator value through a multiplexer onto the data memory write port: `WriteData = do_mstore ? c_read_data : RD2`.

With the outer product approach, the entire 4x4 matrix multiply is expressed as straight-line code with no software loops over the tile:

```
lw   x2, 160(x0)       // Load A column 0
lw   x3, 176(x0)       // Load B row 0
mmatmul x2, x3         // Rank-1 update: C += A_col0 * B_row0
lw   x2, 164(x0)       // Load A column 1
lw   x3, 180(x0)       // Load B row 1
mmatmul x2, x3         // Rank-1 update: C += A_col1 * B_row1
lw   x2, 168(x0)       // Load A column 2
lw   x3, 184(x0)       // Load B row 2
mmatmul x2, x3         // Rank-1 update: C += A_col2 * B_row2
lw   x2, 172(x0)       // Load A column 3
lw   x3, 188(x0)       // Load B row 3
mmatmul x2, x3         // Rank-1 update: C += A_col3 * B_row3
mstore C[0], 192(x0)   // Store 16 results
mstore C[1], 196(x0)
...
mstore C[15], 252(x0)
```

Column-major data for A is stored at 0xA0–0xAF (4 words), and row-major data for B at 0xB0–0xBF (4 words). The total program is 29 instructions (including the terminal branch) with a measured execution cost of 28 cycles. Of these, 8 are load instructions (4 A columns + 4 B rows), 4 are mmatmul, 16 are mstore, and 1 is the terminal branch.

**[TODO: Insert Figure 3 — Block diagram of the outer product coprocessor showing the 16 accumulator registers, 16 parallel 8x8 multipliers, and connections to the main datapath (rs1, rs2, memory interface).]**

**[TODO: Insert Figure 4 — Diagram illustrating the three matrix multiply decompositions (scalar element-wise, inner dot product, outer product rank-1 update) side by side.]**

### E. Integrated Design

All three approaches are implemented within a single unified processor design, sharing the same 2-stage pipeline, register file, instruction memory, and data memory. The main decoder is expanded from 11 bits to 14 bits to accommodate the three outer product control signals (`do_mmatmul`, `do_mload`, `do_mstore`). The custom-0 opcode (0001011) is shared between `mac` (funct3=000), `dot4` (funct3=001), and `mmatmul` (funct3=010), with the funct3 field used for disambiguation. The custom-1 opcode (0101011) handles `mload` (funct3=000) and `mstore` (funct3=010).

For standalone benchmarking, we also produce three separate processor variants — one per approach — each containing only the hardware modules relevant to that approach. This ensures that FPGA resource measurements reflect the true cost of each extension in isolation.

Table I summarizes the instruction encodings for all supported instructions, including the base RV32I subset, the M extension multiply variants, and our custom extensions.

**Table I: Instruction Encoding Summary**

| Instruction | Type | Opcode (6:0) | funct3 (14:12) | funct7 (31:25) | Operation |
|---|---|---|---|---|---|
| **Base RV32I** | | | | | |
| add | R | 0110011 | 000 | 0000000 | rd = rs1 + rs2 |
| sub | R | 0110011 | 000 | 0100000 | rd = rs1 - rs2 |
| and | R | 0110011 | 111 | 0000000 | rd = rs1 & rs2 |
| or | R | 0110011 | 110 | 0000000 | rd = rs1 \| rs2 |
| slt | R | 0110011 | 010 | 0000000 | rd = (rs1 < rs2) ? 1 : 0 |
| addi | I | 0010011 | 000 | — | rd = rs1 + imm |
| andi | I | 0010011 | 111 | — | rd = rs1 & imm |
| ori | I | 0010011 | 110 | — | rd = rs1 \| imm |
| slti | I | 0010011 | 010 | — | rd = (rs1 < imm) ? 1 : 0 |
| lw | I | 0000011 | 010 | — | rd = mem[rs1 + imm] |
| sw | S | 0100011 | 010 | — | mem[rs1 + imm] = rs2 |
| beq | B | 1100011 | 000 | — | if (rs1 == rs2) PC += imm |
| jal | J | 1101111 | — | — | rd = PC+4; PC += imm |
| **M Extension** | | | | | |
| mul | R | 0110011 | 000 | 0000001 | rd = (rs1 * rs2)[31:0] |
| mulh | R | 0110011 | 001 | 0000001 | rd = (rs1 * rs2)[63:32] (signed x signed) |
| mulhsu | R | 0110011 | 010 | 0000001 | rd = (rs1 * rs2)[63:32] (signed x unsigned) |
| mulhu | R | 0110011 | 011 | 0000001 | rd = (rs1 * rs2)[63:32] (unsigned x unsigned) |
| **Custom Extensions** | | | | | |
| mac | R | 0001011 | 000 | 0000000 | rd = rd + rs1 * rs2 |
| dot4 | R | 0001011 | 001 | 0000000 | rd = rd + sum(rs1[i]*rs2[i]) for 4 packed bytes |
| mmatmul | R | 0001011 | 010 | 0000000 | C[r*4+c] += A[r] * B[c] for all r,c in 0..3 |
| mload | I | 0101011 | 000 | — | C[rd[3:0]] = mem[rs1 + imm] |
| mstore | S | 0101011 | 010 | — | mem[rs1 + imm] = C[rs2[3:0]] |

The custom-0 opcode (0001011) is shared among `mac`, `dot4`, and `mmatmul`, with the funct3 field providing disambiguation. The custom-1 opcode (0101011) is used for the coprocessor data movement instructions `mload` and `mstore`. The M extension multiply variants share the standard R-type opcode (0110011) with funct7 = 0000001 and are distinguished by funct3.

---

## IV. Evaluation

### A. Experimental Setup

All designs are synthesized using Intel Quartus Prime targeting the Cyclone V 5CSEMA5F31C6 FPGA on the DE1-SoC development board. Functional verification is performed using Verilator on EDA Playground. Each benchmark program computes the same 4x4 matrix multiplication C = A x B with identical input matrices and verifies that the output matches the expected result matrix. Cycle counts are measured in the simulation testbench using a free-running cycle counter that increments on each rising clock edge.

The maximum operating frequency (Fmax) achieved after synthesis is 52.56 MHz across all three variants, as the critical path is dominated by the base processor logic rather than the custom extensions.

### B. Instruction Count Analysis

Table II summarizes the static instruction count and dynamic instruction mix for each benchmark program.

**Table II: Instruction Count Breakdown**

| Metric | Scalar | Dot4 | Outer Product |
|---|---|---|---|
| Total static instructions | 54 | 41 | 29 |
| Load instructions (lw) | 2 per inner iter | 2 per j iter | 8 total |
| Store instructions (sw) | 1 per j iter | 1 per j iter | 16 total (mstore) |
| Multiply instructions | 1 per inner iter (mul) | 1 per j iter (dot4) | 4 total (mmatmul) |
| Loop overhead instructions | ~5 per inner iter | ~4 per j iter | 0 |
| Branch instructions | 2 per inner iter | 2 per j iter | 1 (terminal) |
| Software loops | 3 nested (i, j, k) | 2 nested (i, j) | 0 |

The scalar approach requires three nested loops because each output element is computed by iterating over the shared dimension k one element at a time. The dot4 approach eliminates the k-loop by processing all four k-elements in parallel, reducing the loop nest to two levels. The outer product approach eliminates all loops over the 4x4 tile entirely, as each mmatmul instruction updates all 16 output elements simultaneously.

### C. Cycle Count and Execution Time

Table III presents the measured cycle counts and wall-clock execution times at the synthesized Fmax of 52.56 MHz.

**Table III: Cycle Count and Execution Time**

| Approach | Cycles | Execution Time | Speedup vs. Scalar |
|---|---|---|---|
| Scalar (mul + add) | 818 | 15.563 us | 1.0x |
| Inner dot product (dot4) | 194 | 3.691 us | 4.2x |
| Outer product (mmatmul) | 28 | 532.7 ns | 29.2x |

The scalar baseline requires 818 cycles due to the deep loop nesting and high ratio of loop overhead to useful computation. Each scalar MAC requires approximately 9 instructions (2 loads, 1 multiply, 1 add, 3 pointer/counter updates, 2 branches), of which only the multiply and add are productive computation.

The dot4 approach achieves a 4.2x speedup by replacing the inner k-loop with a single instruction that performs four MACs in parallel. The remaining cycles are spent on the i and j loops, loads, stores, and loop control.

The outer product approach achieves a 29.2x speedup by eliminating all software loops. The entire computation consists of 8 loads (to feed A columns and B rows to the coprocessor), 4 mmatmul instructions, and 16 mstore instructions to write back the results. Every instruction performs useful work — there is zero loop overhead.

**[TODO: Insert Figure 5 — Bar chart comparing cycle counts for the three approaches. Use log scale on y-axis. Label each bar with the exact cycle count.]**

**[TODO: Insert Figure 6 — Bar chart comparing execution times (in microseconds) for the three approaches.]**

### D. FPGA Resource Utilization

Table IV presents the FPGA synthesis results for each standalone processor variant, synthesized independently to isolate the incremental cost of each extension.

**Table IV: FPGA Resource Utilization by Variant (Cyclone V 5CSEMA5F31C6)**

| Resource | Scalar | Dot Product | Outer Product | Available |
|---|---|---|---|---|
| Logic ALMs | 1,401 (4.4%) | 1,484 (4.6%) | 1,812 (5.6%) | 32,070 |
| Total Registers | 2,418 | 2,416 | 3,003 | — |
| Block Memory Bits | 3,072 | 3,072 | 3,072 | 4,065,280 |
| DSP Blocks | 8 (9.2%) | 12 (13.8%) | 24 (27.6%) | 87 |

The scalar baseline uses 8 DSP blocks for the 32-bit multiplier (supporting `mul`, `mulh`, `mulhsu`, and `mulhu`). The dot product variant adds 4 DSP blocks (total 12) for the four 8x8-bit multipliers in the `mac_dot4` module. The outer product variant uses 24 DSP blocks — the largest increase — due to the 16 parallel 8x8-bit multipliers in the coprocessor, in addition to the base multiplier.

The dot product extension adds only 83 ALMs (a 5.9% increase) over the scalar baseline, while the outer product extension adds 411 ALMs (a 29.3% increase), primarily due to the 16 accumulator registers (adding 585 registers) and the wider multiplier array. Block memory usage is identical across all variants (3,072 bits for instruction and data memory) since the memory architecture is unchanged.

Despite being the most resource-intensive variant, the outer product processor occupies only 5.6% of the available ALMs, confirming that even the most aggressive extension remains a small design well within the capacity of modest FPGAs.

**[TODO: Insert Figure 7 — Grouped bar chart comparing ALM, register, and DSP usage across the three variants.]**

### E. Efficiency Metrics

To compare the approaches on a normalized basis, we compute the number of useful MAC operations per cycle.

**Table V: Compute Efficiency**

| Approach | MACs per tile | Cycles | MACs/cycle | Efficiency vs. Scalar |
|---|---|---|---|---|
| Scalar | 64 | 818 | 0.078 | 1.0x |
| Dot4 | 64 | 194 | 0.330 | 4.2x |
| Outer product | 64 | 28 | 2.286 | 29.2x |

A 4x4 matrix multiply requires 64 MAC operations (4^3) regardless of the approach. The scalar baseline achieves 0.078 MACs per cycle because most cycles are spent on non-compute overhead. The outer product approach achieves 2.3 MACs per cycle, meaning each cycle contributes more than two useful MACs on average. The four mmatmul instructions alone contribute 16 MACs each (64 total in 4 cycles = 16 MACs/cycle for the compute-only portion), with the remaining cycles spent on data movement.

### F. Power Analysis

Table VI presents the power dissipation estimates from Quartus Power Analyzer for each standalone variant at the synthesized Fmax of 52.56 MHz.

**Table VI: Power Dissipation by Variant**

| Power Component | Scalar | Dot Product | Outer Product |
|---|---|---|---|
| Core Dynamic | 20.08 mW | 21.86 mW | 33.19 mW |
| Core Static | 411.38 mW | 411.39 mW | 411.46 mW |
| I/O | 9.10 mW | 8.87 mW | 8.87 mW |
| **Total** | **440.56 mW** | **442.12 mW** | **453.52 mW** |

The total power is dominated by core static dissipation (approximately 411 mW across all variants), which is inherent to the FPGA fabric and independent of the design. The meaningful comparison is in the core dynamic power, which reflects the switching activity of the active logic. The dot product variant increases dynamic power by only 1.78 mW (8.9%) over the scalar baseline, while the outer product variant increases dynamic power by 13.11 mW (65.3%) due to the 16 parallel multipliers and additional accumulator registers.

However, when normalized by useful work (MACs per unit energy), the outer product approach is significantly more energy-efficient. For a single 4x4 tile:

**Table VII: Energy Efficiency**

| Approach | Execution Time | Energy (Dynamic) | MACs/uJ |
|---|---|---|---|
| Scalar | 15.563 us | 312.5 nJ | 204.8 |
| Dot4 | 3.691 us | 80.7 nJ | 793.1 |
| Outer product | 532.7 ns | 17.7 nJ | 3,616.1 |

The outer product approach completes the computation 29.2x faster, and despite consuming 65% more dynamic power, its total dynamic energy per tile is 17.7x lower than the scalar baseline. This results in a 17.7x improvement in energy efficiency (MACs per microjoule), making it the most suitable approach for energy-constrained embedded DNN inference.

---

## V. Discussion

### A. Parallelism and Loop Elimination

The progression from scalar to dot4 to outer product represents increasing exploitation of data-level parallelism. The scalar approach extracts no parallelism: one MAC per multiply instruction. The dot4 instruction exploits SIMD parallelism along the k-dimension, packing four 8-bit multiplies into a single instruction and eliminating the innermost loop. The outer product instruction exploits parallelism in both the i and j dimensions simultaneously, performing 16 MACs per instruction and eliminating all loops over the tile.

This analysis reveals that the primary source of cycle reduction is not faster computation per se, but rather the elimination of instruction fetch, decode, and loop control overhead. In the scalar approach, only 2 of approximately 9 inner-loop instructions (22%) perform useful computation. The outer product approach makes every instruction productive, achieving near-unity utilization of the instruction stream.

### B. Data Layout and Memory Access

Each approach requires a different data layout in memory to enable efficient single-instruction loads:

- **Scalar**: Standard 32-bit row-major layout for A and column-major (transposed) layout for B, allowing sequential word-addressable access within each row or column.
- **Dot4**: Packed 8-bit row-major layout for A and packed 8-bit transposed layout for B, where four consecutive elements are packed into a single 32-bit word.
- **Outer product**: Column-major layout for A (each word holds one column as four packed 8-bit elements) and row-major layout for B (each word holds one row as four packed 8-bit elements).

In a real system, the data reformatting cost must be amortized over multiple tile operations. For a 64x64 matrix decomposed into 4x4 tiles, the packing and transposition are performed once and reused across 16x16 = 256 tile computations, making the overhead negligible.

### C. Hardware Cost

The outer product coprocessor adds 16 accumulator flip-flop registers (585 additional registers, bringing the total from 2,418 to 3,003) and 16 additional DSP blocks for the parallel 8x8-bit multipliers. In total, the outer product variant uses 411 more ALMs (29.3% increase) and 16 more DSP blocks (200% increase) compared to the scalar baseline. Despite being the most hardware-intensive extension, the outer product processor occupies only 5.6% of the Cyclone V's ALMs and 27.6% of DSP blocks, while delivering a 29.2x speedup. The performance-per-ALM ratio (speedup per additional ALM) is 0.071x per ALM, and the dynamic power increase is only 13.11 mW — a modest cost for nearly 30x acceleration.

The dot4 extension provides a middle ground: it adds only 83 ALMs (5.9% increase), 4 DSP blocks, and 1.78 mW of dynamic power over the scalar baseline, achieving a 4.2x speedup. For applications where DSP block budget is constrained, the dot4 approach offers an attractive performance-per-resource ratio of 0.051x speedup per additional ALM, comparable to the outer product approach but with far lower absolute resource consumption.

### D. Scaling to Larger Matrices

The benchmarks in this paper evaluate a single 4x4 tile operation. For larger matrices (e.g., 64x64), the computation is decomposed into 4x4 tiles, and the tile operation is invoked repeatedly within additional software loops. The relative advantage of the outer product approach grows with matrix size because the tile-level speedup is preserved while the per-tile loop overhead remains constant. For a 64x64 matrix (256 tile operations), the outer product approach would require approximately 256 x 28 = 7,168 cycles (plus inter-tile loop overhead), compared to approximately 256 x 818 = 209,408 cycles for the scalar approach.

**[TODO: If time permits, run the 64x64 benchmarks and replace the estimates above with measured cycle counts.]**

### E. Limitations

Our evaluation has several limitations. First, the 2-stage pipeline and single-cycle instruction execution mean that the outer product coprocessor's 16 parallel multipliers must all complete within one clock period, which may limit achievable Fmax on faster FPGA families or ASIC implementations. Second, the current design uses unsigned 8-bit arithmetic for dot4 and signed 8-bit arithmetic for mmatmul; a production implementation would need to support both. Third, our cycle counts do not include the cost of data packing and transposition, which must be performed in software or with additional custom instructions in a complete system.

**[TODO: Add any other limitations or observations from your testing experience.]**

---

## VI. Conclusion

We have presented the design, implementation, and evaluation of three custom instruction set extensions for accelerating integer matrix multiplication on a minimal 2-stage pipelined RV32I processor. The scalar baseline using standard `mul` instructions (alongside `mulh`, `mulhsu`, and `mulhu` for full M extension coverage) requires 818 cycles for a 4x4 tile multiply. Adding a packed 8-bit inner dot product instruction (`dot4`) reduces this to 194 cycles (4.2x speedup) by eliminating the innermost loop. Adding an outer product instruction (`mmatmul`) with a dedicated 16-register accumulator coprocessor further reduces the cycle count to 28 cycles (29.2x speedup) by eliminating all software loops over the tile.

The three standalone variants are synthesized to an Intel Cyclone V FPGA at 52.56 MHz. The scalar baseline occupies 1,401 ALMs and 8 DSP blocks; the dot product variant uses 1,484 ALMs and 12 DSP blocks; and the outer product variant uses 1,812 ALMs and 24 DSP blocks. Power analysis shows that the outer product variant's dynamic power (33.19 mW) is only 65% higher than the scalar baseline (20.08 mW), yet it delivers a 17.7x improvement in energy efficiency (MACs per microjoule). The outer product approach achieves 2.3 MACs per cycle on average, compared to 0.078 MACs per cycle for the scalar baseline, demonstrating that significant acceleration of DNN inference workloads is achievable with simple, area-efficient custom extensions on a minimal RISC-V core.

Future work includes extending the design to support 4-bit (INT4) data types for further density improvements, implementing the full 64x64 tiled matrix multiply with inter-tile loop control, and evaluating the design on ASIC targets for power and area characterization.

---

## Acknowledgment

The authors thank the instructors and teaching assistants of CPEN 497 at the University of British Columbia for their guidance on this project. The base processor design is derived from the RISC-V implementation published in *Digital Design and Computer Architecture: RISC-V Edition* by Harris and Harris.

---

## References

> Replace the template references in paper.tex with the following. Add more as needed.

[1] A. Waterman and K. Asanovic, "The RISC-V instruction set manual, volume I: Unprivileged ISA," RISC-V International, Dec. 2019.

[2] S. Harris and D. Harris, *Digital Design and Computer Architecture: RISC-V Edition*. Cambridge, MA: Morgan Kaufmann, 2021.

[3] N. Jouppi et al., "In-datacenter performance analysis of a tensor processing unit," in *Proc. ISCA*, 2017, pp. 1-12.

[4] RISC-V International, "RISC-V matrix extension task group," 2024. [Online]. Available: https://github.com/riscv/riscv-matrix-spec

**[TODO: Add references for the following:]**
- NVIDIA Tensor Core architecture paper or whitepaper
- ARM Helium / MVE technical reference manual
- RISC-V Packed SIMD (P extension) draft specification
- Any RISC-V matrix/DNN accelerator papers from recent ASAP, FCCM, or DAC conferences
- Intel Cyclone V FPGA device handbook (for synthesis target details)

---

## Summary of Remaining TODOs

### Figures (to create and insert)
1. **Figure 1**: Base processor block diagram (2-stage pipeline with Fetch, pipeline register, Execute)
2. **Figure 2**: mac_dot4 module block diagram (four 8x8 multipliers, adder tree, 32-bit multiplier, mode mux)
3. **Figure 3**: Outer product coprocessor block diagram (16 accumulators, 16 multipliers, memory interface)
4. **Figure 4**: Three matrix multiply decompositions side by side (scalar, inner product, outer product)
5. **Figure 5**: Bar chart — cycle count comparison (log scale)
6. **Figure 6**: Bar chart — execution time comparison
7. **Figure 7**: Grouped bar chart — ALM, register, and DSP usage across three variants

### References (to add)
8. NVIDIA Tensor Core architecture paper or whitepaper
9. ARM Helium / MVE technical reference manual
10. RISC-V Packed SIMD (P extension) draft specification
11. Any RISC-V matrix/DNN accelerator papers from recent ASAP, FCCM, or DAC conferences
12. Intel Cyclone V FPGA device handbook

### Optional
13. 64x64 benchmark measurements (to replace estimated cycle counts in Section V.D)

### Completed (data now included)
- ~~Table I: Instruction encoding summary~~ — Done (all base, M-ext, and custom instructions)
- ~~Table III: Cycle count and execution time~~ — Done (818, 194, 28 cycles)
- ~~Table IV: Per-variant FPGA resource utilization~~ — Done (ALMs, registers, memory, DSPs)
- ~~Table V: Efficiency metrics~~ — Done (MACs/cycle)
- ~~Table VI: Power dissipation~~ — Done (dynamic, static, I/O per variant)
- ~~Table VII: Energy efficiency~~ — Done (MACs/uJ)
- ~~mulh/mulhsu/mulhu support~~ — Documented in Section III.A and Table I
