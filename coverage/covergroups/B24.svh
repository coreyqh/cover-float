covergroup B24 (virtual coverfloat_interface CFI);
    option.per_instance = 0;

    //Rounding Mode coverpoint
    rounding_modes: coverpoint(CFI.rm){
        type_option.weight = 0;
        `include "bins_templates/rounding_modes.svh"
    }

    // Input Precision coverpoints
    F16_input_fmt: coverpoint (CFI.operandFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_input_fmt: coverpoint (CFI.operandFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_input_fmt: coverpoint (CFI.operandFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_input_fmt: coverpoint (CFI.operandFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_input_fmt: coverpoint (CFI.operandFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }

    // Sign coverpoint
    operand_sign: coverpoint $signed(get_sign(CFI.a, CFI.operandFmt)){
        type_option.weight = 0;
        bins pos = {0};
        bins neg = {1};
    }

    //Proximity To Zero Coverpoint
    proximity_to_zero: coverpoint $signed(proximity_to_zero(CFI.a, CFI.operandFmt)){
        type_option.weight = 0;
        bins zero = {1};
        bins one_quarter = {2};
        bins one_half = {3};
        bins three_quarters = {4};
        bins one = {5};
        bins one_and_one_quarter = {6};
        bins one_and_one_half = {7};
        bins one_and_three_quarters = {8};
    }

    //Result format coverpoints
    result_int32_fmt: coverpoint CFI.resultFmt {
        type_option.weight = 0;

        bins int32 = {FMT_INT};
        bins uint32 = {FMT_UINT};
    }

    result_long64_fmt: coverpoint CFI.resultFmt {
        type_option.weight = 0;

        bins int64 = {FMT_LONG};
        bins uint64 = {FMT_ULONG};
    }

    //Crosses
    //FMT_HALF
    `ifdef COVER_F16
        `ifdef COVER_INT
            B22_F16_INT:
                cross rounding_modes, F16_input_fmt, operand_sign, proximity_to_zero, result_int32_fmt;
        `endif
    `endif

    `ifdef COVER_F16
        `ifdef COVER_INT
            B22_F16_LONG:
                cross rounding_modes, F16_input_fmt, operand_sign, proximity_to_zero, result_long64_fmt;
        `endif
    `endif

    //FMT_BF16
    `ifdef COVER_BF16
        `ifdef COVER_INT
            B22_BF16_INT:
                cross rounding_modes, BF16_input_fmt, operand_sign, proximity_to_zero, result_int32_fmt;
        `endif
    `endif

    `ifdef COVER_BF16
        `ifdef COVER_INT
            B22_BF16_LONG:
                cross rounding_modes, BF16_input_fmt, operand_sign, proximity_to_zero, result_long64_fmt;
        `endif
    `endif

    //FMT_SINGLE
    `ifdef COVER_F32
        `ifdef COVER_INT
            B22_F32_INT:
                cross rounding_modes, F32_input_fmt, operand_sign, proximity_to_zero, result_int32_fmt;
        `endif
    `endif

    `ifdef COVER_F32
        `ifdef COVER_LONG
            B22_F32_LONG:
                cross rounding_modes, F32_input_fmt, operand_sign, proximity_to_zero, result_long64_fmt;
        `endif
    `endif

    //FMT_DOUBLE
    `ifdef COVER_F64
        `ifdef COVER_INT
            B22_F64_INT:
                cross rounding_modes, F64_input_fmt, operand_sign, proximity_to_zero, result_int32_fmt;
        `endif
    `endif

    `ifdef COVER_F64
        `ifdef COVER_LONG
            B22_F64_LONG:
                cross rounding_modes, F64_input_fmt, operand_sign, proximity_to_zero, result_long64_fmt;
        `endif
    `endif

    //FMT_QUAD
    `ifdef COVER_F128
        `ifdef COVER_INT
            B22_F128_INT:
                cross rounding_modes, F128_input_fmt, operand_sign, proximity_to_zero, result_int32_fmt;
        `endif
    `endif

    `ifdef COVER_F128
        `ifdef COVER_LONG
            B22_F128_LONG:
                cross rounding_modes, F128_input_fmt, operand_sign, proximity_to_zero, result_long64_fmt;
        `endif
    `endif

endgroup
