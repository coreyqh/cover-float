
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

    // Precision (p = number of significand bits + 1 implicit bits)
    parameter int F16_M_BITS   = 10;
    parameter int BF16_M_BITS  = 7;
    parameter int F32_M_BITS   = 23;
    parameter int F64_M_BITS   = 52;
    parameter int F128_M_BITS  = 112;

    parameter int F16_P   = F16_M_BITS  + 1;
    parameter int BF16_P  = BF16_M_BITS + 1;
    parameter int F32_P   = F32_M_BITS  + 1;
    parameter int F64_P   = F64_M_BITS  + 1;
    parameter int F128_P  = F128_M_BITS + 1;

    parameter int F16_E_UPPER  = 14;
    parameter int F16_E_LOWER  = 10;
    parameter int BF16_E_UPPER = 14;
    parameter int BF16_E_LOWER = 7;
    parameter int F32_E_UPPER  = 30;
    parameter int F32_E_LOWER  = 23;
    parameter int F64_E_UPPER  = 62;
    parameter int F64_E_LOWER  = 52;
    parameter int F128_E_UPPER = 126;
    parameter int F128_E_LOWER = 112;

    parameter int F16_M_UPPER  = F16_M_BITS - 1;
    parameter int BF16_M_UPPER = BF16_M_BITS - 1;
    parameter int F32_M_UPPER  = F32_M_BITS - 1;
    parameter int F64_M_UPPER  = F64_M_BITS - 1;
    parameter int F128_M_UPPER = F128_M_BITS - 1;


    // Helper functions for difficult coverpoints

    // Count leading zeros (from MSB downward)
    function automatic int count_leading_zeros (
        input logic [255:0] val,
        input int width
    );
        int i;
        begin
            count_leading_zeros = 0;
            for (i = width-1; i >= 0; i--) begin
                if (val[i] == 0)
                    count_leading_zeros++;
                else
                    break;
            end
        end
    endfunction


    // Count leading ones (from MSB downward)
    function automatic int count_leading_ones (
        input logic [255:0] val,
        input int width
    );
        int i;
        begin
            count_leading_ones = 0;
            for (i = width-1; i >= 0; i--) begin
                if (val[i] == 1)
                    count_leading_ones++;
                else
                    break;
            end
        end
    endfunction


    // Count trailing zeros (from LSB upward)
    function automatic int count_trailing_zeros (
        input logic [255:0] val,
        input int width
    );
        int i;
        begin
            count_trailing_zeros = 0;
            for (i = 0; i < width; i++) begin
                if (val[i] == 0)
                    count_trailing_zeros++;
                else
                    break;
            end
        end
    endfunction


    // Count trailing ones (from LSB upward)
    function automatic int count_trailing_ones (
        input logic [255:0] val,
        input int width
    );
        int i;
        begin
            count_trailing_ones = 0;
            for (i = 0; i < width; i++) begin
                if (val[i] == 1)
                    count_trailing_ones++;
                else
                    break;
            end
        end
    endfunction

    // Check for repeating checker pattern with optional partial final run
    // Returns signed run length:
    //   < 0 : first run is 1s
    //   > 0 : first run is 0s
    //   = 0 : invalid pattern
    function automatic int checker_run_length (
        input logic [255:0] val,
        input int width
    );
        int run_len;
        int i;
        int run_idx;
        logic first_bit;
        logic expected;

        begin
            checker_run_length = 0;

            if (width < 2)
                return 0;

            // Determine initial run length from MSB
            first_bit = val[width-1];
            run_len   = 0;

            for (i = width-1; i >= 0; i--) begin
                if (val[i] == first_bit)
                    run_len++;
                else
                    break;
            end

            // Run length must be reasonable
            if (run_len == 0 || run_len > (width >> 1))
                return 0;

            // Verify alternating pattern, allowing partial final run
            for (i = width-1; i >= 0; i--) begin
                run_idx  = (width-1 - i) / run_len;
                expected = (run_idx % 2 == 0) ? first_bit : ~first_bit;

                if (val[i] != expected)
                    return 0;
            end

            // Encode polarity in sign
            checker_run_length = first_bit ? -run_len : run_len;
        end
    endfunction

    // Find the longest contiguous run of ones in the signal
    // Returns length of longest run, or 0 if no ones
    function automatic int longest_seq_of_ones (
        input logic [255:0] val,
        input int width
    );
        int i;
        int curr_len;
        int max_len;

        begin
            longest_seq_of_ones = 0;

            if (width <= 0)
                return 0;

            curr_len = 0;
            max_len  = 0;

            // Scan from LSB to MSB (direction does not matter here)
            for (i = 0; i < width; i++) begin
                if (val[i] == 1) begin
                    curr_len++;
                    if (curr_len > max_len)
                        max_len = curr_len;
                end
                else begin
                    curr_len = 0;
                end
            end

            longest_seq_of_ones = max_len;
        end
    endfunction


endpackage
