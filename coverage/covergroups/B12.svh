covergroup B12_cg (virtual coverfloat_interface CFI);

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
     * Cancellation helper coverpoints
     *
     * cancellation = exp(result) - max(exp(a), exp(b))
     *
     ************************************************************************/

    F16_cancellation: coverpoint
        (
            $signed(int'(CFI.intermX)) -
            (
                (int'(CFI.a[14:10]) > int'(CFI.b[14:10]))
                ? int'(CFI.a[14:10])
                : int'(CFI.b[14:10])
            )
        )
    {
        type_option.weight = 0;
        bins cancel[] = {[-F16_P : 1]};
    }

    BF16_cancellation: coverpoint
        (
            $signed(int'(CFI.intermX)) -
            (
                (int'(CFI.a[14:7]) > int'(CFI.b[14:7]))
                ? int'(CFI.a[14:7])
                : int'(CFI.b[14:7])
            )
        )
    {
        type_option.weight = 0;
        bins cancel[] = {[-BF16_P : 1]};
    }

    F32_cancellation: coverpoint
        (
            $signed(int'(CFI.intermX)) -
            (
                (int'(CFI.a[30:23]) > int'(CFI.b[30:23]))
                ? int'(CFI.a[30:23])
                : int'(CFI.b[30:23])
            )
        )
    {
        type_option.weight = 0;
        bins cancel[] = {[-F32_P : 1]};
    }

    F64_cancellation: coverpoint
        (
            $signed(int'(CFI.intermX)) -
            (
                (int'(CFI.a[62:52]) > int'(CFI.b[62:52]))
                ? int'(CFI.a[62:52])
                : int'(CFI.b[62:52])
            )
        )
    {
        type_option.weight = 0;
        bins cancel[] = {[-F64_P : 1]};
    }

    F128_cancellation: coverpoint
        (
            $signed(int'(CFI.intermX)) -
            (
                (int'(CFI.a[126:112]) > int'(CFI.b[126:112]))
                ? int'(CFI.a[126:112])
                : int'(CFI.b[126:112])
            )
        )
    {
        type_option.weight = 0;
        bins cancel[] = {[-F128_P : 1]};
    }

    /************************************************************************
     *
     * Main crosses (precision-gated)
     *
     ************************************************************************/

    `ifdef COVER_F16
        B12_F16_addsub_cancel:
            cross FP_addsub_ops, F16_cancellation, F16_result_fmt;
    `endif

    `ifdef COVER_BF16
        B12_BF16_addsub_cancel:
            cross FP_addsub_ops, BF16_cancellation, BF16_result_fmt;
    `endif

    `ifdef COVER_F32
        B12_F32_addsub_cancel:
            cross FP_addsub_ops, F32_cancellation, F32_result_fmt;
    `endif

    `ifdef COVER_F64
        B12_F64_addsub_cancel:
            cross FP_addsub_ops, F64_cancellation, F64_result_fmt;
    `endif

    `ifdef COVER_F128
        B12_F128_addsub_cancel:
            cross FP_addsub_ops, F128_cancellation, F128_result_fmt;
    `endif

endgroup
