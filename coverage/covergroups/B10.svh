covergroup B10_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
     *
     * Operation helper coverpoint (Add / Sub only)
     *
     ************************************************************************/

    FP_addsub_ops: coverpoint CFI.op {
        type_option.weight = 0;
        bins add = {OP_ADD};
        bins sub = {OP_SUB};
    }

    /************************************************************************
     *
     * Result format helper coverpoints
     *
     ************************************************************************/

    F16_result_fmt: coverpoint (CFI.resultFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_result_fmt: coverpoint (CFI.resultFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_result_fmt: coverpoint (CFI.resultFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_result_fmt: coverpoint (CFI.resultFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_result_fmt: coverpoint (CFI.resultFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }

    /************************************************************************
     *
     * Shift classification helper coverpoints
     *
     ************************************************************************/

    // exponent difference = exponent(a) - exponent(b)
    // extracted directly from input operands, per format

    // TODO: re-work ranges with constants instead of these magic numbers

    F16_exp_diff: coverpoint $signed(int'(CFI.a[14:10]) - int'(CFI.b[14:10])) {
        type_option.weight = 0;

        bins small_diff = {[-(F16_M_BITS + 4) : 0]};
        bins mid_diff[] = {[-(F16_M_BITS + 4) : (F16_M_BITS + 4)]};
        bins large_diff = {[ (F16_M_BITS + 4) : $]};

    }

    BF16_exp_diff: coverpoint $signed(int'(CFI.a[14:7]) - int'(CFI.b[14:7])) {
        type_option.weight = 0;

        bins small_diff = {[-(BF16_M_BITS + 4) : 0]};
        bins mid_diff[] = {[-(BF16_M_BITS + 4) : (BF16_M_BITS + 4)]};
        bins large_diff = {[ (BF16_M_BITS + 4) : $]};

    }

    F32_exp_diff: coverpoint $signed(int'(CFI.a[30:23]) - int'(CFI.b[30:23])) {
        type_option.weight = 0;

        bins small_diff = {[-(F32_M_BITS + 4) : 0]};
        bins mid_diff[] = {[-(F32_M_BITS + 4) : (F32_M_BITS + 4)]};
        bins large_diff = {[ (F32_M_BITS + 4) : $]};

    }

    F64_exp_diff: coverpoint $signed(int'(CFI.a[62:52]) - int'(CFI.b[62:52])) {
        type_option.weight = 0;

        bins small_diff = {[-(F64_M_BITS + 4) : 0]};
        bins mid_diff[] = {[-(F64_M_BITS + 4) : (F64_M_BITS + 4)]};
        bins large_diff = {[ (F64_M_BITS + 4) : $]};

    }

    F128_exp_diff: coverpoint $signed(int'(CFI.a[126:112]) - int'(CFI.b[126:112])) {
        type_option.weight = 0;

        bins small_diff = {[-(F128_M_BITS + 4) : 0]};
        bins mid_diff[] = {[-(F128_M_BITS + 4) : (F128_M_BITS + 4)]};
        bins large_diff = {[ (F128_M_BITS + 4) : $]};

    }

    /************************************************************************
     *
     * Main crosses (precision-gated)
     *
     ************************************************************************/

    `ifdef COVER_F16
        B10_F16_addsub_shift:
            cross FP_addsub_ops, F16_exp_diff, F16_result_fmt;
    `endif

    `ifdef COVER_BF16
        B10_BF16_addsub_shift:
            cross FP_addsub_ops, BF16_exp_diff, BF16_result_fmt;
    `endif

    `ifdef COVER_F32
        B10_F32_addsub_shift:
            cross FP_addsub_ops, F32_exp_diff, F32_result_fmt;
    `endif

    `ifdef COVER_F64
        B10_F64_addsub_shift:
            cross FP_addsub_ops, F64_exp_diff, F64_result_fmt;
    `endif

    `ifdef COVER_F128
        B10_F128_addsub_shift:
            cross FP_addsub_ops, F128_exp_diff, F128_result_fmt;
    `endif

endgroup
