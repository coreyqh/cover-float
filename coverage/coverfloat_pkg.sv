
`include "../config.svh"
`include "macros.svh"


package coverfloat_pkg;


    // encodings from SoftFloat
    const logic [7:0] FLAG_INEXACT_MASK   =  8'd1;
    const logic [7:0] FLAG_UNDERFLOW_MASK =  8'd2;
    const logic [7:0] FLAG_OVERFLOW_MASK  =  8'd4;
    const logic [7:0] FLAG_INFINITE_MASK  =  8'd8;
    const logic [7:0] FLAG_INVALID_MASK   =  8'd16;

    // arbitary encoding of IBM paper operations
    // numbering scheme: bits 31:4 are major operation (pulled directly form aharoni paper)
    //                   bits 3:0 are variant operations (taylored towards riscv instrs)
    //                   coverpoint are written such that any variant covers the superset
    const logic [31:0] OP_ADD    = 32'h1_0;
    const logic [31:0] OP_SUB    = 32'h2_0;
    const logic [31:0] OP_MUL    = 32'h3_0;
    const logic [31:0] OP_DIV    = 32'h4_0;
    const logic [31:0] OP_FMA    = 32'h5_0;
    const logic [31:0] OP_FMADD  = 32'h5_1; // multiply-add
    const logic [31:0] OP_FMSUB  = 32'h5_2; // multiply-subtract
    const logic [31:0] OP_FNMADD = 32'h5_3; // negated (multiply-add)
    const logic [31:0] OP_FNMSUB = 32'h5_4; // negated (multiply-sub)
    const logic [31:0] OP_SQRT   = 32'h6_0;
    const logic [31:0] OP_REM    = 32'h7_0; // TODO: unused? Remove?
    const logic [31:0] OP_CFI    = 32'h8_0;
    const logic [31:0] OP_CFF    = 32'h9_0;
    const logic [31:0] OP_CIF    = 32'hA_0;
    const logic [31:0] OP_QC     = 32'hB_0;
    const logic [31:0] OP_FEQ    = 32'hB_1; // quiet equal
    const logic [31:0] OP_SC     = 32'hC_0;
    const logic [31:0] OP_FLT    = 32'hC_1; // signaling less than
    const logic [31:0] OP_FLE    = 32'hC_2; // signaling LT or eq
    const logic [31:0] OP_CLASS  = 32'hD_0;
    const logic [31:0] OP_MIN    = 32'hE_0;
    const logic [31:0] OP_MAX    = 32'hF_0;
    const logic [31:0] OP_CSN    = 32'h10_0; // copy sign / negate
    const logic [31:0] OP_FSGNJ  = 32'h10_1;
    const logic [31:0] OP_FSGNJN = 32'h10_2;
    const logic [31:0] OP_FSGNJX = 32'h10_3;

    // const logic [31:0] OP_

    // encodings from SoftFloat
    const logic [7:0] ROUND_NEAR_EVEN   = 8'd0;
    const logic [7:0] ROUND_MINMAG      = 8'd1;
    const logic [7:0] ROUND_MIN         = 8'd2;
    const logic [7:0] ROUND_MAX         = 8'd3;
    const logic [7:0] ROUND_NEAR_MAXMAG = 8'd4;
    // const logic [7:0] ROUND_ODD         = 8'd5;

    // format encodings
    //  {(int = 1, float = 0), (unsigned int), others => format encoding}
    const logic [7:0] FMT_INVAL  = 8'b 1_1_111111; // source unused / invalid 
    const logic [7:0] FMT_HALF   = 8'b 0_0_000000;
    const logic [7:0] FMT_SINGLE = 8'b 0_0_000001;
    const logic [7:0] FMT_DOUBLE = 8'b 0_0_000010;
    const logic [7:0] FMT_QUAD   = 8'b 0_0_000011;
    const logic [7:0] FMT_BF16   = 8'b 0_0_000100;

    const logic [7:0] FMT_INT    = 8'b 1_0_000001;
    const logic [7:0] FMT_UINT   = 8'b 1_1_000001;
    const logic [7:0] FMT_LONG   = 8'b 1_0_000010;
    const logic [7:0] FMT_ULONG  = 8'b 1_1_000010;

    
    // TODO: expand with other relvelant parameters

    // Precision (p = number of significand bits)
    const int F16_M_BITS   = 10;
    const int BF16_M_BITS  = 7;
    const int F32_M_BITS   = 23;
    const int F64_M_BITS   = 52;
    const int F128_M_BITS  = 112;

    const int F16_P   = F16_M_BITS  + 1;
    const int BF16_P  = BF16_M_BITS + 1;
    const int F32_P   = F32_M_BITS  + 1;
    const int F64_P   = F64_M_BITS  + 1;
    const int F128_P  = F128_M_BITS + 1;
    
endpackage
