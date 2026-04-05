covergroup B22_cg (virtual coverfloat_interface CFI);
    option.per_instance = 0;

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

    //Input Exponent Coverpoint, only need 2 because unsigned/signed ints/longs have same width
    exponent_dif_int32: coverpoint $signed(get_unbiased_exponent(CFI.a, CFI.operandFmt)){
        type_option.weight = 0;

        bins less_than_neg_3 = {[$:-3]};
        bins between_neg_3_and_int_width_plus_3[] = {[-3:35]};
        bins greater_than_int_width_plus_3 = {[35:$]};
    }

    exponent_dif_long64: coverpoint $signed(get_unbiased_exponent(CFI.a, CFI.operandFmt)){
        type_option.weight = 0;

        bins less_than_neg_3 = {[$:-3]};
        bins between_neg_3_and_int_width_plus_3[] = {[-3:67]};
        bins greater_than_int_width_plus_3 = {[68:$]};
    }

    //Crosses
    //FMT_HALF
    `ifdef COVER_F16
        `ifdef COVER_INT
            B22_F16_INT:
                cross F16_input_fmt, operand_sign, result_int32_fmt, exponent_dif_int32;
        `endif
    `endif

    `ifdef COVER_F16
        `ifdef COVER_INT
            B22_F16_LONG:
                cross F16_input_fmt, operand_sign, result_long64_fmt, exponent_dif_long64;
        `endif
    `endif

    //FMT_BF16
    `ifdef COVER_BF16
        `ifdef COVER_INT
            B22_BF16_INT:
                cross BF16_input_fmt, operand_sign, result_int32_fmt, exponent_dif_int32;
        `endif
    `endif

    `ifdef COVER_BF16
        `ifdef COVER_INT
            B22_BF16_LONG:
                cross BF16_input_fmt, operand_sign, result_long64_fmt, exponent_dif_long64;
        `endif
    `endif

    //FMT_SINGLE
    `ifdef COVER_F32
        `ifdef COVER_INT
            B22_F32_INT:
                cross F32_input_fmt, operand_sign, result_int32_fmt, exponent_dif_int32;
        `endif
    `endif

    `ifdef COVER_F32
        `ifdef COVER_LONG
            B22_F32_LONG:
                cross F32_input_fmt, operand_sign, result_long64_fmt, exponent_dif_long64;
        `endif
    `endif

    //FMT_DOUBLE
    `ifdef COVER_F64
        `ifdef COVER_INT
            B22_F64_INT:
                cross F64_input_fmt, operand_sign, result_int32_fmt, exponent_dif_int32;
        `endif
    `endif

    `ifdef COVER_F64
        `ifdef COVER_LONG
            B22_F64_LONG:
                cross F64_input_fmt, operand_sign, result_long64_fmt, exponent_dif_long64;
        `endif
    `endif

    //FMT_QUAD
    `ifdef COVER_F128
        `ifdef COVER_INT
            B22_F128_INT:
                cross F128_input_fmt, operand_sign, result_int32_fmt, exponent_dif_int32;
        `endif
    `endif

    `ifdef COVER_F128
        `ifdef COVER_LONG
            B22_F128_LONG:
                cross F128_input_fmt, operand_sign, result_long64_fmt, exponent_dif_long64;
        `endif
    `endif

endgroup
