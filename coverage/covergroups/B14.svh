covergroup B14_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
     *
     * Multiply-Add operation helper coverpoint
     *
     ************************************************************************/

    FP_madd_ops: coverpoint CFI.op {
        type_option.weight = 0;
        bins op_fma    = {[OP_FMA : OP_FMA | 32'hF]};
        bins op_fmadd  = {OP_FMADD};
        bins op_fmsub  = {OP_FMSUB};
        bins op_fnmadd = {OP_FNMADD};
        bins op_fnmsub = {OP_FNMSUB};
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
     * Modeled as:
     *   unbiased_exp(result) - unbiased_exp(addend)
     *
     * This reflects the effective alignment shift applied to the addend
     * during the multiply-add.
     *
     ************************************************************************/

    // HALF
    F16_madd_shift: coverpoint $signed(int'(CFI.result[14:10]) - int'(CFI.c[14:10])) {
        type_option.weight = 0;

        bins small_diff = {[-(2*F16_M_BITS + 2) : 0]};
        bins mid_diff[] = {[-(2*F16_M_BITS + 1) : (F16_M_BITS + 1)]};
        bins large_diff = {[ (F16_M_BITS   + 2) : $]};

    }

    // BF16
    BF16_madd_shift: coverpoint $signed(int'(CFI.result[14:7]) - int'(CFI.c[14:7])) {
        type_option.weight = 0;

        bins small_diff = {[-(2*BF16_M_BITS + 2) : 0]};
        bins mid_diff[] = {[-(2*BF16_M_BITS + 1) : (BF16_M_BITS + 1)]};
        bins large_diff = {[ (BF16_M_BITS + 2)   : $ ] };
    }

    // SINGLE
    F32_madd_shift: coverpoint $signed(int'(CFI.result[30:23]) - int'(CFI.c[30:23])) {
        type_option.weight = 0;

        bins small_diff = {[-(2*F32_M_BITS + 2) : 0]};
        bins mid_diff[] = {[-(2*F32_M_BITS + 1) : (F32_M_BITS + 1)]};
        bins large_diff = {[ (F32_M_BITS + 2)   : $]};
    }

    // DOUBLE
    F64_madd_shift: coverpoint $signed(int'(CFI.result[62:52]) - int'(CFI.c[62:52])) {
        type_option.weight = 0;

        bins small_diff = {[-(2*F64_M_BITS + 2) : 0]};
        bins mid_diff[] = {[-(2*F64_M_BITS + 1) : (F64_M_BITS + 1)]};
        bins large_diff = {[ (F64_M_BITS + 2)   : $]};
    }

    // QUAD
    F128_madd_shift: coverpoint $signed(int'(CFI.result[126:112]) - int'(CFI.c[126:112])) {
        type_option.weight = 0;

        bins small_diff = {[-(2*F128_M_BITS + 2) : 0]};
        bins mid_diff[] = {[-(2*F128_M_BITS + 1) : (F128_M_BITS + 1)]};
        bins large_diff = {[ (F128_M_BITS + 2)   : $]};
    }

    /************************************************************************
     *
     * Main crosses (precision-gated)
     *
     ************************************************************************/

    `ifdef COVER_F16
        B14_F16_madd_shift:
            cross FP_madd_ops, F16_madd_shift, F16_result_fmt;
    `endif

    `ifdef COVER_BF16
        B14_BF16_madd_shift:
            cross FP_madd_ops, BF16_madd_shift, BF16_result_fmt;
    `endif

    `ifdef COVER_F32
        B14_F32_madd_shift:
            cross FP_madd_ops, F32_madd_shift, F32_result_fmt;
    `endif

    `ifdef COVER_F64
        B14_F64_madd_shift:
            cross FP_madd_ops, F64_madd_shift, F64_result_fmt;
    `endif

    `ifdef COVER_F128
        B14_F128_madd_shift:
            cross FP_madd_ops, F128_madd_shift, F128_result_fmt;
    `endif

endgroup
