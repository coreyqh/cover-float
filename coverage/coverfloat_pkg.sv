package coverfloat_pkg;

    // encodings from SoftFloat
    const logic [7:0] FLAG_INEXACT_MASK   =  8'd1;
    const logic [7:0] FLAG_UNDERFLOW_MASK =  8'd2;
    const logic [7:0] FLAG_OVERFLOW_MASK  =  8'd4;
    const logic [7:0] FLAG_INFINITE_MASK  =  8'd8;
    const logic [7:0] FLAG_INVALID_MASK   =  8'd16;

    // arbitary encoding of IBM paper operations
    const logic [31:0] OP_ADD   = 32'd1;
    const logic [31:0] OP_SUB   = 32'd2;
    const logic [31:0] OP_MUL   = 32'd3;
    const logic [31:0] OP_DIV   = 32'd4;
    const logic [31:0] OP_FMA   = 32'd5;
    const logic [31:0] OP_SQRT  = 32'd6;
    const logic [31:0] OP_REM   = 32'd7;
    const logic [31:0] OP_CFI   = 32'd8;
    const logic [31:0] OP_CIF   = 32'd9;
    const logic [31:0] OP_QC    = 32'd10;
    const logic [31:0] OP_SC    = 32'd11;
    const logic [31:0] OP_EQ    = 32'd12;
    const logic [31:0] OP_CLASS = 32'd13;
    // const logic [31:0] OP_

    // encodings from SoftFloat
    const logic [7:0] ROUND_NEAR_EVEN   = 8'd0;
    const logic [7:0] ROUND_MINMAG      = 8'd1;
    const logic [7:0] ROUND_MIN         = 8'd2;
    const logic [7:0] ROUND_MAX         = 8'd3;
    const logic [7:0] ROUND_NEAR_MAXMAG = 8'd4;
    const logic [7:0] ROUND_ODD         = 8'd5;

    // format encodings
    //  {(int = 1, float = 0), (unsigned int), others => format encoding}
    const logic [7:0] FMT_INVALL = 8'b 1_1_111111; // source unused
    const logic [7:0] FMT_HALF   = 8'b 0_0_000000;
    const logic [7:0] FMT_SINGLE = 8'b 0_0_000001;
    const logic [7:0] FMT_DOUBLE = 8'b 0_0_000010;
    const logic [7:0] FMT_QUAD   = 8'b 0_0_000011;
    const logic [7:0] FMT_BF16   = 8'b 0_0_000100;

    const logic [7:0] FMT_INT    = 8'b 1_0_000001;
    const logic [7:0] FMT_UINT   = 8'b 1_1_000001;
    const logic [7:0] FMT_LONG   = 8'b 1_0_000010;
    const logic [7:0] FMT_ULONG  = 8'b 1_1_000010;

    // float types
    typedef struct {
        logic [15:0] val;
    } bfloat16_t;

    typedef struct {
        logic [15:0] val;
    } float16_t;

    typedef struct {
        logic [31:0] val;
    } float32_t;

    typedef struct {
        logic [63:0] val;
    } float64_t;
    
    typedef struct packed {
        logic [63:0] high;
        logic [63:0] low;
    } float128_t;

    // intermediate results from extended SoftFloat

    typedef struct packed {
        bit          sign;
        logic [31:0] exp;
        logic [63:0] sig64;
        logic [63:0] sig0;
        logic [63:0] sigExtra;
    } intermResult_t;
    


endpackage