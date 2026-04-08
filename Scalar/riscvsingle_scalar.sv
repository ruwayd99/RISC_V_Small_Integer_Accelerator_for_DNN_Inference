// riscvsingle.sv

// RISC-V single-cycle processor
// From Section 7.6 of Digital Design & Computer Architecture
// 27 April 2020
// David_Harris@hmc.edu 
// Sarah.Harris@unlv.edu

// run 210
// Expect simulator to print "Simulation succeeded"
// when the value 25 (0x19) is written to address 100 (0x64)

// Single-cycle implementation of RISC-V (RV32I)
// User-level Instruction Set Architecture V2.2 (May 7, 2017)
// Implements a subset of the base integer instructions:
//    lw, sw
//    add, sub, and, or, slt, 
//    addi, andi, ori, slti
//    beq
//    jal
// Exceptions, traps, and interrupts not implemented
// little-endian memory

// 31 32-bit registers x1-x31, x0 hardwired to 0
// R-Type instructions
//   add, sub, and, or, slt
//   INSTR rd, rs1, rs2
//   Instr[31:25] = funct7 (funct7b5 & opb5 = 1 for sub, 0 for others)
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode
// I-Type Instructions
//   lw, I-type ALU (addi, andi, ori, slti)
//   lw:         INSTR rd, imm(rs1)
//   I-type ALU: INSTR rd, rs1, imm (12-bit signed)
//   Instr[31:20] = imm[11:0]
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode
// S-Type Instruction
//   sw rs2, imm(rs1) (store rs2 into address specified by rs1 + immm)
//   Instr[31:25] = imm[11:5] (offset[11:5])
//   Instr[24:20] = rs2 (src)
//   Instr[19:15] = rs1 (base)
//   Instr[14:12] = funct3
//   Instr[11:7]  = imm[4:0]  (offset[4:0])
//   Instr[6:0]   = opcode
// B-Type Instruction
//   beq rs1, rs2, imm (PCTarget = PC + (signed imm x 2))
//   Instr[31:25] = imm[12], imm[10:5]
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = imm[4:1], imm[11]
//   Instr[6:0]   = opcode
// J-Type Instruction
//   jal rd, imm  (signed imm is multiplied by 2 and added to PC, rd = PC+4)
//   Instr[31:12] = imm[20], imm[10:1], imm[11], imm[19:12]
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode

//   Instruction  opcode    funct3    funct7
//   add          0110011   000       0000000
//   sub          0110011   000       0100000
//   and          0110011   111       0000000
//   or           0110011   110       0000000
//   slt          0110011   010       0000000
//   addi         0010011   000       immediate
//   andi         0010011   111       immediate
//   ori          0010011   110       immediate
//   slti         0010011   010       immediate
//   beq          1100011   000       immediate
//   lw	          0000011   010       immediate
//   sw           0100011   010       immediate
//   jal          1101111   immediate immediate

module testbench();

  localparam int OUT_BASE  = 32'd192;
  localparam int OUT_WORDS = 16;

  logic        clk;
  logic        reset;
  integer      cycle_count;
  logic [31:0] expected[0:OUT_WORDS-1];
  logic [OUT_WORDS-1:0] seen;
  logic        finished;

  logic [31:0] WriteData, DataAdr;
  logic        MemWrite;

  // instantiate device to be tested
  top dut(clk, reset, WriteData, DataAdr, MemWrite);
  
  // initialize test
  initial
    begin
      integer i;
      cycle_count = 0;
      seen = '0;
      finished = 0;
      // Dense 4x4 reference matrix (row-major):
      // [13,18,17,13,
      //  37,42,41,33,
      //  5, 4,13,10,
      //  17,12,13,0]
      expected[0]  = 32'd13;
      expected[1]  = 32'd18;
      expected[2]  = 32'd17;
      expected[3]  = 32'd13;
      expected[4]  = 32'd37;
      expected[5]  = 32'd42;
      expected[6]  = 32'd41;
      expected[7]  = 32'd33;
      expected[8]  = 32'd5;
      expected[9]  = 32'd4;
      expected[10] = 32'd13;
      expected[11] = 32'd10;
      expected[12] = 32'd17;
      expected[13] = 32'd12;
      expected[14] = 32'd13;
      expected[15] = 32'd0;
      reset <= 1; # 22; reset <= 0;
    end

  // generate clock to sequence tests
  always
    begin
      clk <= 1; # 5; clk <= 0; # 5;
    end

  always @(posedge clk)
    if (!reset)
      cycle_count <= cycle_count + 1;

  function automatic logic all_seen();
    integer i;
    begin
      all_seen = 1'b1;
      for (i = 0; i < OUT_WORDS; i = i + 1)
        if (!seen[i])
          all_seen = 1'b0;
    end
  endfunction

  task automatic fail_now(input string msg);
    begin
      $display("Simulation failed: %s", msg);
      $stop;
    end
  endtask

  task automatic check_outputs();
    integer i;
    integer word_idx;
    begin
      for (i = 0; i < OUT_WORDS; i = i + 1) begin
        word_idx = (OUT_BASE >> 2) + i;
        if (dut.dmem.RAM[word_idx] !== expected[i]) begin
          $display("Simulation failed: C[%0d] expected %0d got %0d", i, expected[i], dut.dmem.RAM[word_idx]);
          $stop;
        end
      end
      $display("Simulation succeeded in %0d cycles", cycle_count);
      $stop;
    end
  endtask

  // check writes and final outputs
  always @(negedge clk)
    begin
      int idx;
      if (MemWrite) begin
        if (DataAdr < OUT_BASE || DataAdr >= OUT_BASE + OUT_WORDS*4 || DataAdr[1:0] != 2'b00)
          fail_now("write outside output window");
        idx = (DataAdr - OUT_BASE) >> 2;
        seen[idx] = 1'b1;
      end

      if (!finished && all_seen()) begin
        finished = 1'b1;
        check_outputs();
      end

      if (cycle_count > 2000)
        fail_now("timeout before writing all 16 outputs");
    end
endmodule

module top(input  logic        clk, reset, 
           output logic [31:0] WriteData, DataAdr, 
           output logic        MemWrite);

  logic [31:0] PC, Instr, ReadData;
  
  // instantiate processor and memories
  riscvsingle rvsingle(clk, reset, PC, Instr, MemWrite, DataAdr, 
                       WriteData, ReadData);
  imem imem(PC, Instr);
  dmem dmem(clk, MemWrite, DataAdr, WriteData, ReadData);
endmodule

module riscvsingle(input  logic        clk, reset,
                   output logic [31:0] PC,
                   input  logic [31:0] Instr,
                   output logic        MemWrite,
                   output logic [31:0] ALU_MULResult, WriteData,
                   input  logic [31:0] ReadData);

  logic       ALUSrc, RegWrite, Jump, Zero;
  logic [1:0] ResultSrc, ImmSrc;
  logic [2:0] ALUControl;
  logic [31:0] InstrE; // pipeline register output (Execute stage instruction)

  // controller decodes the Execute stage instruction (InstrE), not the Fetch instruction
  controller c(InstrE[6:0], InstrE[14:12], InstrE[30], Zero,
               ResultSrc, MemWrite, PCSrc,
               ALUSrc, RegWrite, Jump,
         ImmSrc, ALUControl);
  datapath dp(clk, reset,
              ResultSrc, PCSrc,
              ALUSrc, RegWrite,
              ImmSrc, ALUControl,
              Zero, PC, Instr, InstrE,
        ALU_MULResult, WriteData, ReadData);
endmodule

module controller(input  logic [6:0] op,
                  input  logic [2:0] funct3,
                  input  logic       funct7b5,
                  input  logic       Zero,
                  output logic [1:0] ResultSrc,
                  output logic       MemWrite,
                  output logic       PCSrc, ALUSrc,
                  output logic       RegWrite, Jump,
                  output logic [1:0] ImmSrc,
          output logic [2:0] ALUControl);

  logic [1:0] ALUOp;
  logic       Branch;

  maindec md(op, funct3, ResultSrc, MemWrite, Branch,
             ALUSrc, RegWrite, Jump, ImmSrc, ALUOp);
  aludec  ad(op[5], funct3, funct7b5, ALUOp, ALUControl);


  assign PCSrc = Branch & Zero | Jump;
endmodule

module maindec(input  logic [6:0] op,
               input  logic [2:0] funct3,
               output logic [1:0] ResultSrc,
               output logic       MemWrite,
               output logic       Branch, ALUSrc,
               output logic       RegWrite, Jump,
               output logic [1:0] ImmSrc,
               output logic [1:0] ALUOp);

  logic [10:0] controls;

  assign {RegWrite, ImmSrc, ALUSrc, MemWrite,
          ResultSrc, Branch, ALUOp, Jump} = controls;

  always_comb
    case(op)
    // RegWrite_ImmSrc_ALUSrc_MemWrite_ResultSrc_Branch_ALUOp_Jump
      7'b0000011: controls = 11'b1_00_1_0_01_0_00_0; // lw
      7'b0100011: controls = 11'b0_01_1_1_00_0_00_0; // sw
      7'b0110011: controls = 11'b1_xx_0_0_00_0_10_0; // R-type
      7'b1100011: controls = 11'b0_10_0_0_00_1_01_0; // beq
      7'b0010011: controls = 11'b1_00_1_0_00_0_10_0; // I-type ALU
      7'b1101111: controls = 11'b1_11_0_0_10_0_00_1; // jal
      default:    controls = 11'bx_xx_x_x_xx_x_xx_x; // non-implemented instruction
    endcase
endmodule

module aludec(input  logic       opb5,
              input  logic [2:0] funct3,
              input  logic       funct7b5, 
              input  logic [1:0] ALUOp,
              output logic [2:0] ALUControl);

  logic  RtypeSub;
  assign RtypeSub = funct7b5 & opb5;  // TRUE for R-type subtract instruction

  always_comb
    case(ALUOp)
      2'b00:                ALUControl = 3'b000; // addition
      2'b01:                ALUControl = 3'b001; // subtraction
      default: case(funct3) // R-type or I-type ALU
                 3'b000:  if (RtypeSub) 
                            ALUControl = 3'b001; // sub
                          else          
                            ALUControl = 3'b000; // add, addi
                 3'b010:    ALUControl = 3'b101; // slt, slti
                 3'b110:    ALUControl = 3'b011; // or, ori
                 3'b111:    ALUControl = 3'b010; // and, andi
                 default:   ALUControl = 3'bxxx; // ???
               endcase
    endcase
endmodule

module datapath(input  logic        clk, reset,
                input  logic [1:0]  ResultSrc, 
                input  logic        PCSrc, ALUSrc,
                input  logic        RegWrite,
                input  logic [1:0]  ImmSrc,
                input  logic [2:0]  ALUControl,
                output logic        Zero,
                output logic [31:0] PC,
                input  logic [31:0] Instr,
                output logic [31:0] InstrE,     // Execute stage instruction (pipeline reg output)
                output logic [31:0] ALU_MULResult, WriteData,
                input  logic [31:0] ReadData);

  logic [31:0] PCNext, PCPlus4, PCTarget;
  logic [31:0] ImmExt;
  logic [31:0] SrcA, SrcB, RD2, rd_current;
  logic [31:0] Result;
  logic [31:0] PCE, PCPlus4E;  // pipeline register outputs for PC
  logic [31:0] ALUResult;
  logic [31:0] MULResult;
  logic [6:0] op;
  logic [2:0] funct3;
  logic [6:0] funct7;
  logic is_mext;

  // decode fields for multiplier
  assign op = InstrE[6:0];
  assign funct3 = InstrE[14:12];
  assign funct7 = InstrE[31:25];
  //triggered if instruction is mul
  assign is_mext = (op == 7'b0110011) && (funct7 == 7'b0000001);

  // ===== FETCH STAGE =====
  flopr #(32) pcreg(clk, reset, PCNext, PC); 
  adder       pcadd4(PC, 32'd4, PCPlus4);
  mux2 #(32)  pcmux(PCPlus4, PCTarget, PCSrc, PCNext);

    // ===== PIPELINE REGISTER: Fetch -> Execute =====
  always_ff @(posedge clk, posedge reset)
    if (reset) begin
      InstrE   <= 32'h00000013; // NOP
      PCE      <= 32'b0;
      PCPlus4E <= 32'b0;
    end else if (PCSrc) begin
      // FLUSH: branch/jump taken — the fetched instruction is wrong, replace with NOP
      InstrE   <= 32'h00000013; // NOP (addi x0, x0, 0)
      PCE      <= 32'b0;
      PCPlus4E <= 32'b0;
    end else begin
      // Normal: pass the fetched instruction to Execute
      InstrE   <= Instr;
      PCE      <= PC;
      PCPlus4E <= PCPlus4;
    end

  // ===== EXECUTE STAGE =====
  // branch/jump target uses PCE (the PC of the instruction being executed)
  adder       pcaddbranch(PCE, ImmExt, PCTarget);

  // register file logic — uses InstrE fields (Execute stage instruction)
  regfile     rf(clk, RegWrite, InstrE[19:15], InstrE[24:20],
                 InstrE[11:7], Result, SrcA, RD2, rd_current);
  extend      ext(InstrE[31:7], ImmSrc, ImmExt);

  // ALU logic
  mux2 #(32)  srcbmux(RD2, ImmExt, ALUSrc, SrcB);
  mux2 #(32)  alu_mulmux(ALUResult, MULResult, is_mext, ALU_MULResult); // sel for ALU vs multiplier result
  alu         alu(SrcA, SrcB, ALUControl, ALUResult, Zero);
  multiplier  mul(SrcA, SrcB, funct3[1:0], MULResult);
  mux4 #(32)  resultmux(ALU_MULResult, ReadData, PCPlus4E, 32'b0, ResultSrc, Result);

  assign WriteData = RD2;
endmodule
  

module regfile(input  logic        clk, 
               input  logic        we3, 
               input  logic [ 4:0] a1, a2, a3, 
               input  logic [31:0] wd3, 
               output logic [31:0] rd1, rd2, rd3);

  logic [31:0] rf[31:0];

  // three ported register file
  // read two ports combinationally (A1/RD1, A2/RD2)
  // write third port on rising edge of clock (A3/WD3/WE3)
  // register 0 hardwired to 0

  always_ff @(posedge clk)
    if (we3) rf[a3] <= wd3;	

  assign rd1 = (a1 != 0) ? rf[a1] : 0;
  assign rd2 = (a2 != 0) ? rf[a2] : 0;
  assign rd3 = (a3 != 0) ? rf[a3] : 0;
endmodule

module adder(input  [31:0] a, b,
             output [31:0] y);

  assign y = a + b;
endmodule

module extend(input  logic [31:7] instr,
              input  logic [1:0]  immsrc,
              output logic [31:0] immext);
 
  always_comb
    case(immsrc) 
               // I-type 
      2'b00:   immext = {{20{instr[31]}}, instr[31:20]};  
               // S-type (stores)
      2'b01:   immext = {{20{instr[31]}}, instr[31:25], instr[11:7]}; 
               // B-type (branches)
      2'b10:   immext = {{20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0}; 
               // J-type (jal)
      2'b11:   immext = {{12{instr[31]}}, instr[19:12], instr[20], instr[30:21], 1'b0}; 
      default: immext = 32'bx; // undefined
    endcase             
endmodule

module flopr #(parameter WIDTH = 8)
              (input  logic             clk, reset,
               input  logic [WIDTH-1:0] d, 
               output logic [WIDTH-1:0] q);

  always_ff @(posedge clk, posedge reset)
    if (reset) q <= 0;
    else       q <= d;
endmodule

module mux2 #(parameter WIDTH = 8)
             (input  logic [WIDTH-1:0] d0, d1, 
              input  logic             s, 
              output logic [WIDTH-1:0] y);

  assign y = s ? d1 : d0; 
endmodule

module mux3 #(parameter WIDTH = 8)
             (input  logic [WIDTH-1:0] d0, d1, d2,
              input  logic [1:0]       s, 
              output logic [WIDTH-1:0] y);

  assign y = s[1] ? d2 : (s[0] ? d1 : d0); 
endmodule

module mux4 #(parameter WIDTH = 8)
             (input  logic [WIDTH-1:0] d0, d1, d2, d3,
              input  logic [1:0]       s, 
              output logic [WIDTH-1:0] y);

  always_comb begin 
    case(s)
      2'b00: y = d0;
      2'b01: y = d1;
      2'b10: y = d2;
      2'b11: y = d3;
      default: y = {WIDTH{1'bx}};
    endcase
  end 
endmodule

module imem(input  logic [31:0] a,
            output logic [31:0] rd);

  logic [31:0] RAM[63:0];
  string       imem_file;

  initial
    begin
      if (!$value$plusargs("IMEM=%s", imem_file))
        imem_file = "riscvtest.txt";
      $readmemh(imem_file, RAM);
    end

  assign rd = RAM[a[31:2]]; // word aligned
endmodule

module dmem(input  logic        clk, we,
            input  logic [31:0] a, wd,
            output logic [31:0] rd);

  logic [31:0] RAM[63:0];
  string       dmem_file;

  initial
    begin
      if (!$value$plusargs("DMEM=%s", dmem_file))
        dmem_file = "benchdata.txt";
      $readmemh(dmem_file, RAM);
    end

  assign rd = RAM[a[31:2]]; // word aligned

  always_ff @(posedge clk)
    if (we) RAM[a[31:2]] <= wd;
endmodule

module alu(input  logic [31:0] a, b,
           input  logic [2:0]  alucontrol,
           output logic [31:0] result,
           output logic        zero);

  logic [31:0] condinvb, sum;
  logic        v;              // overflow
  logic        isAddSub;       // true when is add or subtract operation

  assign condinvb = alucontrol[0] ? ~b : b;
  assign sum = a + condinvb + alucontrol[0];
  assign isAddSub = ~alucontrol[2] & ~alucontrol[1] |
                    ~alucontrol[1] & alucontrol[0];

  always_comb
    case (alucontrol)
      3'b000:  result = sum;         // add
      3'b001:  result = sum;         // subtract
      3'b010:  result = a & b;       // and
      3'b011:  result = a | b;       // or
      3'b100:  result = a ^ b;       // xor
      3'b101:  result = sum[31] ^ v; // slt
      3'b110:  result = a << b[4:0]; // sll
      3'b111:  result = a >> b[4:0]; // srl
      default: result = 32'bx;
    endcase

  assign zero = (result == 32'b0);
  assign v = ~(alucontrol[0] ^ a[31] ^ b[31]) & (a[31] ^ sum[31]) & isAddSub;
  
endmodule

module multiplier(
  input  logic [31:0] a, b,
  input  logic [1:0]  mul_op,   // 00=mul, 01=mulh, 10=mulhsu, 11=mulhu
  output logic [31:0] result
);
  logic signed [63:0] product_ss;  // signed × signed
  logic signed [63:0] product_su;  // signed × unsigned
  logic        [63:0] product_uu;  // unsigned × unsigned

  assign product_ss = $signed(a) * $signed(b);
  assign product_uu = {32'b0, a} * {32'b0, b};      // zero-extend both
  assign product_su = $signed(a) * $signed({1'b0, b}); // sign-extend a, zero-extend b

  always_comb
    case (mul_op)
      2'b00: result = product_ss[31:0];   // mul  — lower 32 bits
      2'b01: result = product_ss[63:32];  // mulh — upper 32 bits (signed×signed)
      2'b10: result = product_su[63:32];  // mulhsu
      2'b11: result = product_uu[63:32];  // mulhu
      default: result = 32'bx;
    endcase
endmodule

module mac_dot4 (input  logic [31:0] a, b, //for dot4/mac accumulation
                 input  logic [31:0] accum,
                 input  logic mode, //0 for dot4, 1 for mac
                 input  logic [31:0] mul_result,
                 output logic [31:0] result);
  
   // 8-bit slices
  logic [7:0] a0, a1, a2, a3, b0, b1, b2, b3;
  assign {a3, a2, a1, a0} = a;
  assign {b3, b2, b1, b0} = b;

  // Four 8×8 multipliers (shared hardware)
  logic [15:0] p0, p1, p2, p3;
  logic [31:0] dot_sum;

  always_comb begin 
    p0 = a0 * b0;
    p1 = a1 * b1;
    p2 = a2 * b2;
    p3 = a3 * b3;
    dot_sum = {16'b0, p0} + {16'b0, p1} + {16'b0, p2} + {16'b0, p3};
    // Dot product sum
    result = mode ? accum + mul_result : accum + dot_sum;
  end

endmodule

module outer_product(
    input  logic        clk, reset,
    input  logic        do_mmatmul,
    input  logic        do_mload,
    input  logic [3:0]  c_idx,       // 4-bit index to select C[0] to C[15]
    input  logic [31:0] rs1_data,    // A vector from main regfile (rs1)
    input  logic [31:0] rs2_data,    // B vector from main regfile (rs2)
    input  logic [31:0] mem_read,    // Data from dmem (used for mload)
    output logic [31:0] c_read_data  // Data to dmem WriteData (used for mstore)
);

    // 16 individual 32-bit accumulators
    logic signed [31:0] C [15:0];

    // Unpack four 8-bit signed integers from 32-bit source registers
    logic signed [7:0] A [3:0];
    logic signed [7:0] B [3:0];

    assign {A[3], A[2], A[1], A[0]} = rs1_data;
    assign {B[3], B[2], B[1], B[0]} = rs2_data;

    // Asynchronous read for mstore (routes selected C register to dmem)
    assign c_read_data = C[c_idx];

    // Synchronous write for mload and mmatmul
    integer i, r, col;
    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            for (i = 0; i < 16; i = i + 1) C[i] <= 32'b0;
        end else if (do_mload) begin
            C[c_idx] <= mem_read;
        end else if (do_mmatmul) begin
            // 16 parallel MAC operations (outer product update)
            for (r = 0; r < 4; r = r + 1) begin
                for (col = 0; col < 4; col = col + 1) begin
                    C[(r*4) + col] <= C[(r*4) + col] + (A[r] * B[col]);
                end
            end
        end
    end
endmodule
