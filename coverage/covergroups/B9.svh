
covergroup B9_cg (virtual coverfloat_interface CFI);

    option.per_instance = 0;

    /************************************************************************
     *
     * Operation helper coverpoint (div, rem, sqrt, mul)
     *
     ************************************************************************/

    FP_B9_ops: coverpoint CFI.op {
        type_option.weight = 0;

        bins div  = {OP_DIV};
        // bins rem  = {OP_REM};
        bins mul  = {OP_MUL};

    }

    FP_sqrt_op: coverpoint CFI.op {
        type_option.weight = 0;

        bins sqrt = {OP_SQRT};
    }

    /************************************************************************
     *
     * Operand format helper coverpoints
     *
     ************************************************************************/

    F16_src_fmt: coverpoint (CFI.operandFmt == FMT_HALF) {
        type_option.weight = 0;
        bins f16 = {1};
    }

    BF16_src_fmt: coverpoint (CFI.operandFmt == FMT_BF16) {
        type_option.weight = 0;
        bins bf16 = {1};
    }

    F32_src_fmt: coverpoint (CFI.operandFmt == FMT_SINGLE) {
        type_option.weight = 0;
        bins f32 = {1};
    }

    F64_src_fmt: coverpoint (CFI.operandFmt == FMT_DOUBLE) {
        type_option.weight = 0;
        bins f64 = {1};
    }

    F128_src_fmt: coverpoint (CFI.operandFmt == FMT_QUAD) {
        type_option.weight = 0;
        bins f128 = {1};
    }


    /************************************************************************
     *
     * Special helper coverpoints
     *
     ************************************************************************/

    F16_sig_a_leading_zeros:   coverpoint count_leading_zeros(CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_a_leading_ones:    coverpoint count_leading_ones(CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_a_trailing_zeros:  coverpoint count_trailing_zeros(CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_a_trailing_ones:   coverpoint count_trailing_ones(CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_a_small_nbr_ones:  coverpoint $countones(CFI.a[F16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_a_small_nbr_zeros: coverpoint $countones(~CFI.a[F16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_a_checkerboard:    coverpoint checker_run_length (CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_a_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS]};
    }
    F16_sig_a_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.a[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS]};
    }


    F16_sig_b_leading_zeros:   coverpoint count_leading_zeros(CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_b_leading_ones:    coverpoint count_leading_ones(CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_b_trailing_zeros:  coverpoint count_trailing_zeros(CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_b_trailing_ones:   coverpoint count_trailing_ones(CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F16_M_BITS - 1]};
    }
    F16_sig_b_small_nbr_ones:  coverpoint $countones(CFI.b[F16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_b_small_nbr_zeros: coverpoint $countones(~CFI.b[F16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_b_checkerboard:    coverpoint checker_run_length (CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS / 2]};
    }
    F16_sig_b_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS]};
    }
    F16_sig_b_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.b[F16_M_UPPER:0], F16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F16_M_BITS]};
    }
    

    BF16_sig_a_leading_zeros:   coverpoint count_leading_zeros(CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_a_leading_ones:    coverpoint count_leading_ones(CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_a_trailing_zeros:  coverpoint count_trailing_zeros(CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_a_trailing_ones:   coverpoint count_trailing_ones(CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_a_small_nbr_ones:  coverpoint $countones(CFI.a[BF16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_a_small_nbr_zeros: coverpoint $countones(~CFI.a[BF16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_a_checkerboard:    coverpoint checker_run_length (CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_a_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS]};
    }
    BF16_sig_a_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.a[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS]};
    }


    BF16_sig_b_leading_zeros:   coverpoint count_leading_zeros(CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_b_leading_ones:    coverpoint count_leading_ones(CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_b_trailing_zeros:  coverpoint count_trailing_zeros(CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_b_trailing_ones:   coverpoint count_trailing_ones(CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : BF16_M_BITS - 1]};
    }
    BF16_sig_b_small_nbr_ones:  coverpoint $countones(CFI.b[BF16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_b_small_nbr_zeros: coverpoint $countones(~CFI.b[BF16_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_b_checkerboard:    coverpoint checker_run_length (CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS / 2]};
    }
    BF16_sig_b_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS]};
    }
    BF16_sig_b_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.b[BF16_M_UPPER:0], BF16_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : BF16_M_BITS]};
    }


    F32_sig_a_leading_zeros:   coverpoint count_leading_zeros(CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_a_leading_ones:    coverpoint count_leading_ones(CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_a_trailing_zeros:  coverpoint count_trailing_zeros(CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_a_trailing_ones:   coverpoint count_trailing_ones(CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_a_small_nbr_ones:  coverpoint $countones(CFI.a[F32_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_a_small_nbr_zeros: coverpoint $countones(~CFI.a[F32_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_a_checkerboard:    coverpoint checker_run_length (CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_a_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS]};
    }
    F32_sig_a_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.a[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS]};
    }


    F32_sig_b_leading_zeros:   coverpoint count_leading_zeros(CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_b_leading_ones:    coverpoint count_leading_ones(CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_b_trailing_zeros:  coverpoint count_trailing_zeros(CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_b_trailing_ones:   coverpoint count_trailing_ones(CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F32_M_BITS - 1]};
    }
    F32_sig_b_small_nbr_ones:  coverpoint $countones(CFI.b[F32_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_b_small_nbr_zeros: coverpoint $countones(~CFI.b[F32_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_b_checkerboard:    coverpoint checker_run_length (CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS / 2]};
    }
    F32_sig_b_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS]};
    }
    F32_sig_b_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.b[F32_M_UPPER:0], F32_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F32_M_BITS]};
    }


    F64_sig_a_leading_zeros:   coverpoint count_leading_zeros(CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_a_leading_ones:    coverpoint count_leading_ones(CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_a_trailing_zeros:  coverpoint count_trailing_zeros(CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_a_trailing_ones:   coverpoint count_trailing_ones(CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_a_small_nbr_ones:  coverpoint $countones(CFI.a[F64_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_a_small_nbr_zeros: coverpoint $countones(~CFI.a[F64_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_a_checkerboard:    coverpoint checker_run_length (CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_a_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS]};
    }
    F64_sig_a_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.a[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS]};
    }


    F64_sig_b_leading_zeros:   coverpoint count_leading_zeros(CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_b_leading_ones:    coverpoint count_leading_ones(CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_b_trailing_zeros:  coverpoint count_trailing_zeros(CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_b_trailing_ones:   coverpoint count_trailing_ones(CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F64_M_BITS - 1]};
    }
    F64_sig_b_small_nbr_ones:  coverpoint $countones(CFI.b[F64_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_b_small_nbr_zeros: coverpoint $countones(~CFI.b[F64_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_b_checkerboard:    coverpoint checker_run_length (CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS / 2]};
    }
    F64_sig_b_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS]};
    }
    F64_sig_b_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.b[F64_M_UPPER:0], F64_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F64_M_BITS]};
    }


    F128_sig_a_leading_zeros:   coverpoint count_leading_zeros(CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_a_leading_ones:    coverpoint count_leading_ones(CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_a_trailing_zeros:  coverpoint count_trailing_zeros(CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_a_trailing_ones:   coverpoint count_trailing_ones(CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_a_small_nbr_ones:  coverpoint $countones(CFI.a[F128_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F128_M_BITS / 2]};
    }
    F128_sig_a_small_nbr_zeros: coverpoint $countones(~CFI.a[F128_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F128_M_BITS / 2]};
    }
    F128_sig_a_checkerboard:    coverpoint checker_run_length (CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F128_M_BITS / 2]};
    }
    F128_sig_a_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F128_M_BITS]};
    }
    F128_sig_a_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.a[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F128_M_BITS]};
    }


    F128_sig_b_leading_zeros:   coverpoint count_leading_zeros(CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_leading_zeros[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_b_leading_ones:    coverpoint count_leading_ones(CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_leading_ones[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_b_trailing_zeros:  coverpoint count_trailing_zeros(CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_zeros[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_b_trailing_ones:   coverpoint count_trailing_ones(CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins num_trailing_ones[] = {[0 : F128_M_BITS - 1]};
    }
    F128_sig_b_small_nbr_ones:  coverpoint $countones(CFI.b[F128_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F128_M_BITS / 2]};
    }
    F128_sig_b_small_nbr_zeros: coverpoint $countones(~CFI.b[F128_M_UPPER:0]) {
        type_option.weight = 0;
        bins few_ones[] = {[0 : F128_M_BITS / 2]};
    }
    F128_sig_b_checkerboard:    coverpoint checker_run_length (CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[-F128_M_BITS / 2 : F128_M_BITS / 2]};
    }
    F128_sig_b_long_ones_seq:   coverpoint longest_seq_of_ones (CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F128_M_BITS]};
    }
    F128_sig_b_long_zeros_seq:   coverpoint longest_seq_of_ones (~CFI.b[F128_M_UPPER:0], F128_M_BITS) {
        type_option.weight = 0;
        bins checker_len[] = {[0 : F128_M_BITS]};
    }
    
    
    
    /************************************************************************
     *
     * Main crosses (precision-sorted)
     *
     ************************************************************************/
   
       `ifdef COVER_F16
        B9_F16__leading_zeros__X__leading_zeros:          cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__leading_zeros:     cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__leading_ones:           cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__leading_ones:      cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__trailing_zeros:         cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__trailing_zeros:    cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__trailing_ones:          cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__trailing_ones:     cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__small_nbr_ones:         cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__small_nbr_ones:    cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__small_nbr_zeros:        cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__small_nbr_zeros:   cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__checkerboard:           cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__checkerboard:      cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__long_ones_seq:          cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__long_ones_seq:     cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_zeros__X__long_zeros_seq:         cross FP_B9_ops,  F16_sig_a_leading_zeros,   F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__leading_zeros__X__long_zeros_seq:    cross FP_sqrt_op, F16_sig_a_leading_zeros,                              F16_src_fmt;
        B9_F16__leading_ones__X__leading_zeros:           cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__leading_zeros:      cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__leading_ones:            cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__leading_ones:       cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__trailing_zeros:          cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__trailing_zeros:     cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__trailing_ones:           cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__trailing_ones:      cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__small_nbr_ones:          cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__small_nbr_ones:     cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__small_nbr_zeros:         cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__small_nbr_zeros:    cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__checkerboard:            cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__checkerboard:       cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__long_ones_seq:           cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__long_ones_seq:      cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__leading_ones__X__long_zeros_seq:          cross FP_B9_ops,  F16_sig_a_leading_ones,    F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__leading_ones__X__long_zeros_seq:     cross FP_sqrt_op, F16_sig_a_leading_ones,                               F16_src_fmt;
        B9_F16__trailing_zeros__X__leading_zeros:         cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__leading_zeros:    cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__leading_ones:          cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__leading_ones:     cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__trailing_zeros:        cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__trailing_zeros:   cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__trailing_ones:         cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__trailing_ones:    cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__small_nbr_ones:        cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__small_nbr_ones:   cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__small_nbr_zeros:       cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__small_nbr_zeros:  cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__checkerboard:          cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__checkerboard:     cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__long_ones_seq:         cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__long_ones_seq:    cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_zeros__X__long_zeros_seq:        cross FP_B9_ops,  F16_sig_a_trailing_zeros,  F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__trailing_zeros__X__long_zeros_seq:   cross FP_sqrt_op, F16_sig_a_trailing_zeros,                             F16_src_fmt;
        B9_F16__trailing_ones__X__leading_zeros:          cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__leading_zeros:     cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__leading_ones:           cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__leading_ones:      cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__trailing_zeros:         cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__trailing_zeros:    cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__trailing_ones:          cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__trailing_ones:     cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__small_nbr_ones:         cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__small_nbr_ones:    cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__small_nbr_zeros:        cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__small_nbr_zeros:   cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__checkerboard:           cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__checkerboard:      cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__long_ones_seq:          cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__long_ones_seq:     cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__trailing_ones__X__long_zeros_seq:         cross FP_B9_ops,  F16_sig_a_trailing_ones,   F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__trailing_ones__X__long_zeros_seq:    cross FP_sqrt_op, F16_sig_a_trailing_ones,                              F16_src_fmt;
        B9_F16__small_nbr_ones__X__leading_zeros:         cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__leading_zeros:    cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__leading_ones:          cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__leading_ones:     cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__trailing_zeros:        cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__trailing_zeros:   cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__trailing_ones:         cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__trailing_ones:    cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__small_nbr_ones:        cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__small_nbr_ones:   cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__small_nbr_zeros:       cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__small_nbr_zeros:  cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__checkerboard:          cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__checkerboard:     cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__long_ones_seq:         cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__long_ones_seq:    cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_ones__X__long_zeros_seq:        cross FP_B9_ops,  F16_sig_a_small_nbr_ones,  F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_ones__X__long_zeros_seq:   cross FP_sqrt_op, F16_sig_a_small_nbr_ones,                             F16_src_fmt;
        B9_F16__small_nbr_zeros__X__leading_zeros:        cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__leading_zeros:   cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__leading_ones:         cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__leading_ones:    cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__trailing_zeros:       cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__trailing_zeros:  cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__trailing_ones:        cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__trailing_ones:   cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__small_nbr_ones:       cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__small_nbr_ones:  cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__small_nbr_zeros:      cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__small_nbr_zeros: cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__checkerboard:         cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__checkerboard:    cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__long_ones_seq:        cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__long_ones_seq:   cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__small_nbr_zeros__X__long_zeros_seq:       cross FP_B9_ops,  F16_sig_a_small_nbr_zeros, F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__small_nbr_zeros__X__long_zeros_seq:  cross FP_sqrt_op, F16_sig_a_small_nbr_zeros,                            F16_src_fmt;
        B9_F16__checkerboard__X__leading_zeros:           cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__leading_zeros:      cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__leading_ones:            cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__leading_ones:       cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__trailing_zeros:          cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__trailing_zeros:     cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__trailing_ones:           cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__trailing_ones:      cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__small_nbr_ones:          cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__small_nbr_ones:     cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__small_nbr_zeros:         cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__small_nbr_zeros:    cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__checkerboard:            cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__checkerboard:       cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__long_ones_seq:           cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__long_ones_seq:      cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__checkerboard__X__long_zeros_seq:          cross FP_B9_ops,  F16_sig_a_checkerboard,    F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__checkerboard__X__long_zeros_seq:     cross FP_sqrt_op, F16_sig_a_checkerboard,                               F16_src_fmt;
        B9_F16__long_ones_seq__X__leading_zeros:          cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__leading_zeros:     cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__leading_ones:           cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__leading_ones:      cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__trailing_zeros:         cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__trailing_zeros:    cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__trailing_ones:          cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__trailing_ones:     cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__small_nbr_ones:         cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__small_nbr_ones:    cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__small_nbr_zeros:        cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__small_nbr_zeros:   cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__checkerboard:           cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__checkerboard:      cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__long_ones_seq:          cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__long_ones_seq:     cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_ones_seq__X__long_zeros_seq:         cross FP_B9_ops,  F16_sig_a_long_ones_seq,   F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__long_ones_seq__X__long_zeros_seq:    cross FP_sqrt_op, F16_sig_a_long_ones_seq,                              F16_src_fmt;
        B9_F16__long_zeros_seq__X__leading_zeros:         cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_leading_zeros,   F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__leading_zeros:    cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__leading_ones:          cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_leading_ones,    F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__leading_ones:     cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__trailing_zeros:        cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_trailing_zeros,  F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__trailing_zeros:   cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__trailing_ones:         cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_trailing_ones,   F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__trailing_ones:    cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__small_nbr_ones:        cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_small_nbr_ones,  F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__small_nbr_ones:   cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__small_nbr_zeros:       cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_small_nbr_zeros, F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__small_nbr_zeros:  cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__checkerboard:          cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_checkerboard,    F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__checkerboard:     cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__long_ones_seq:         cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_long_ones_seq,   F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__long_ones_seq:    cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
        B9_F16__long_zeros_seq__X__long_zeros_seq:        cross FP_B9_ops,  F16_sig_a_long_zeros_seq,  F16_sig_b_long_zeros_seq,  F16_src_fmt;
        B9_F16_sqrt__long_zeros_seq__X__long_zeros_seq:   cross FP_sqrt_op, F16_sig_a_long_zeros_seq,                             F16_src_fmt;
    `endif // COVER_F16

    `ifdef COVER_BF16
        B9_BF16__leading_zeros__X__leading_zeros:          cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__leading_zeros:     cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__leading_ones:           cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__leading_ones:      cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__trailing_zeros:         cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__trailing_zeros:    cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__trailing_ones:          cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__trailing_ones:     cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__small_nbr_ones:         cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__small_nbr_ones:    cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__small_nbr_zeros:        cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__small_nbr_zeros:   cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__checkerboard:           cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__checkerboard:      cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__long_ones_seq:          cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__long_ones_seq:     cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_zeros__X__long_zeros_seq:         cross FP_B9_ops,  BF16_sig_a_leading_zeros,   BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__leading_zeros__X__long_zeros_seq:    cross FP_sqrt_op, BF16_sig_a_leading_zeros,                               BF16_src_fmt;
        B9_BF16__leading_ones__X__leading_zeros:           cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__leading_zeros:      cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__leading_ones:            cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__leading_ones:       cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__trailing_zeros:          cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__trailing_zeros:     cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__trailing_ones:           cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__trailing_ones:      cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__small_nbr_ones:          cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__small_nbr_ones:     cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__small_nbr_zeros:         cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__small_nbr_zeros:    cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__checkerboard:            cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__checkerboard:       cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__long_ones_seq:           cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__long_ones_seq:      cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__leading_ones__X__long_zeros_seq:          cross FP_B9_ops,  BF16_sig_a_leading_ones,    BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__leading_ones__X__long_zeros_seq:     cross FP_sqrt_op, BF16_sig_a_leading_ones,                                BF16_src_fmt;
        B9_BF16__trailing_zeros__X__leading_zeros:         cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__leading_zeros:    cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__leading_ones:          cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__leading_ones:     cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__trailing_zeros:        cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__trailing_zeros:   cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__trailing_ones:         cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__trailing_ones:    cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__small_nbr_ones:        cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__small_nbr_ones:   cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__small_nbr_zeros:       cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__small_nbr_zeros:  cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__checkerboard:          cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__checkerboard:     cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__long_ones_seq:         cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__long_ones_seq:    cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_zeros__X__long_zeros_seq:        cross FP_B9_ops,  BF16_sig_a_trailing_zeros,  BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_zeros__X__long_zeros_seq:   cross FP_sqrt_op, BF16_sig_a_trailing_zeros,                              BF16_src_fmt;
        B9_BF16__trailing_ones__X__leading_zeros:          cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__leading_zeros:     cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__leading_ones:           cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__leading_ones:      cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__trailing_zeros:         cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__trailing_zeros:    cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__trailing_ones:          cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__trailing_ones:     cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__small_nbr_ones:         cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__small_nbr_ones:    cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__small_nbr_zeros:        cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__small_nbr_zeros:   cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__checkerboard:           cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__checkerboard:      cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__long_ones_seq:          cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__long_ones_seq:     cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__trailing_ones__X__long_zeros_seq:         cross FP_B9_ops,  BF16_sig_a_trailing_ones,   BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__trailing_ones__X__long_zeros_seq:    cross FP_sqrt_op, BF16_sig_a_trailing_ones,                               BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__leading_zeros:         cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__leading_zeros:    cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__leading_ones:          cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__leading_ones:     cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__trailing_zeros:        cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__trailing_zeros:   cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__trailing_ones:         cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__trailing_ones:    cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__small_nbr_ones:        cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__small_nbr_ones:   cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__small_nbr_zeros:       cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__small_nbr_zeros:  cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__checkerboard:          cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__checkerboard:     cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__long_ones_seq:         cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__long_ones_seq:    cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_ones__X__long_zeros_seq:        cross FP_B9_ops,  BF16_sig_a_small_nbr_ones,  BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_ones__X__long_zeros_seq:   cross FP_sqrt_op, BF16_sig_a_small_nbr_ones,                              BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__leading_zeros:        cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__leading_zeros:   cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__leading_ones:         cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__leading_ones:    cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__trailing_zeros:       cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__trailing_zeros:  cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__trailing_ones:        cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__trailing_ones:   cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__small_nbr_ones:       cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__small_nbr_ones:  cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__small_nbr_zeros:      cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__small_nbr_zeros: cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__checkerboard:         cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__checkerboard:    cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__long_ones_seq:        cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__long_ones_seq:   cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__small_nbr_zeros__X__long_zeros_seq:       cross FP_B9_ops,  BF16_sig_a_small_nbr_zeros, BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__small_nbr_zeros__X__long_zeros_seq:  cross FP_sqrt_op, BF16_sig_a_small_nbr_zeros,                             BF16_src_fmt;
        B9_BF16__checkerboard__X__leading_zeros:           cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__leading_zeros:      cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__leading_ones:            cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__leading_ones:       cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__trailing_zeros:          cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__trailing_zeros:     cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__trailing_ones:           cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__trailing_ones:      cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__small_nbr_ones:          cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__small_nbr_ones:     cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__small_nbr_zeros:         cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__small_nbr_zeros:    cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__checkerboard:            cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__checkerboard:       cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__long_ones_seq:           cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__long_ones_seq:      cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__checkerboard__X__long_zeros_seq:          cross FP_B9_ops,  BF16_sig_a_checkerboard,    BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__checkerboard__X__long_zeros_seq:     cross FP_sqrt_op, BF16_sig_a_checkerboard,                                BF16_src_fmt;
        B9_BF16__long_ones_seq__X__leading_zeros:          cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__leading_zeros:     cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__leading_ones:           cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__leading_ones:      cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__trailing_zeros:         cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__trailing_zeros:    cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__trailing_ones:          cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__trailing_ones:     cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__small_nbr_ones:         cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__small_nbr_ones:    cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__small_nbr_zeros:        cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__small_nbr_zeros:   cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__checkerboard:           cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__checkerboard:      cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__long_ones_seq:          cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__long_ones_seq:     cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_ones_seq__X__long_zeros_seq:         cross FP_B9_ops,  BF16_sig_a_long_ones_seq,   BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__long_ones_seq__X__long_zeros_seq:    cross FP_sqrt_op, BF16_sig_a_long_ones_seq,                               BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__leading_zeros:         cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_leading_zeros,   BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__leading_zeros:    cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__leading_ones:          cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_leading_ones,    BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__leading_ones:     cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__trailing_zeros:        cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_trailing_zeros,  BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__trailing_zeros:   cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__trailing_ones:         cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_trailing_ones,   BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__trailing_ones:    cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__small_nbr_ones:        cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_small_nbr_ones,  BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__small_nbr_ones:   cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__small_nbr_zeros:       cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_small_nbr_zeros, BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__small_nbr_zeros:  cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__checkerboard:          cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_checkerboard,    BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__checkerboard:     cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__long_ones_seq:         cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_long_ones_seq,   BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__long_ones_seq:    cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
        B9_BF16__long_zeros_seq__X__long_zeros_seq:        cross FP_B9_ops,  BF16_sig_a_long_zeros_seq,  BF16_sig_b_long_zeros_seq,  BF16_src_fmt;
        B9_BF16_sqrt__long_zeros_seq__X__long_zeros_seq:   cross FP_sqrt_op, BF16_sig_a_long_zeros_seq,                              BF16_src_fmt;
    `endif // COVER_BF16

    `ifdef COVER_F32
        B9_F32__leading_zeros__X__leading_zeros:          cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__leading_zeros:     cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__leading_ones:           cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__leading_ones:      cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__trailing_zeros:         cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__trailing_zeros:    cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__trailing_ones:          cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__trailing_ones:     cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__small_nbr_ones:         cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__small_nbr_ones:    cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__small_nbr_zeros:        cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__small_nbr_zeros:   cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__checkerboard:           cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__checkerboard:      cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__long_ones_seq:          cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__long_ones_seq:     cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_zeros__X__long_zeros_seq:         cross FP_B9_ops,  F32_sig_a_leading_zeros,   F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__leading_zeros__X__long_zeros_seq:    cross FP_sqrt_op, F32_sig_a_leading_zeros,                              F32_src_fmt;
        B9_F32__leading_ones__X__leading_zeros:           cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__leading_zeros:      cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__leading_ones:            cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__leading_ones:       cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__trailing_zeros:          cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__trailing_zeros:     cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__trailing_ones:           cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__trailing_ones:      cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__small_nbr_ones:          cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__small_nbr_ones:     cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__small_nbr_zeros:         cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__small_nbr_zeros:    cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__checkerboard:            cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__checkerboard:       cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__long_ones_seq:           cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__long_ones_seq:      cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__leading_ones__X__long_zeros_seq:          cross FP_B9_ops,  F32_sig_a_leading_ones,    F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__leading_ones__X__long_zeros_seq:     cross FP_sqrt_op, F32_sig_a_leading_ones,                               F32_src_fmt;
        B9_F32__trailing_zeros__X__leading_zeros:         cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__leading_zeros:    cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__leading_ones:          cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__leading_ones:     cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__trailing_zeros:        cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__trailing_zeros:   cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__trailing_ones:         cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__trailing_ones:    cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__small_nbr_ones:        cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__small_nbr_ones:   cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__small_nbr_zeros:       cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__small_nbr_zeros:  cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__checkerboard:          cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__checkerboard:     cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__long_ones_seq:         cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__long_ones_seq:    cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_zeros__X__long_zeros_seq:        cross FP_B9_ops,  F32_sig_a_trailing_zeros,  F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__trailing_zeros__X__long_zeros_seq:   cross FP_sqrt_op, F32_sig_a_trailing_zeros,                             F32_src_fmt;
        B9_F32__trailing_ones__X__leading_zeros:          cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__leading_zeros:     cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__leading_ones:           cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__leading_ones:      cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__trailing_zeros:         cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__trailing_zeros:    cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__trailing_ones:          cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__trailing_ones:     cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__small_nbr_ones:         cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__small_nbr_ones:    cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__small_nbr_zeros:        cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__small_nbr_zeros:   cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__checkerboard:           cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__checkerboard:      cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__long_ones_seq:          cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__long_ones_seq:     cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__trailing_ones__X__long_zeros_seq:         cross FP_B9_ops,  F32_sig_a_trailing_ones,   F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__trailing_ones__X__long_zeros_seq:    cross FP_sqrt_op, F32_sig_a_trailing_ones,                              F32_src_fmt;
        B9_F32__small_nbr_ones__X__leading_zeros:         cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__leading_zeros:    cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__leading_ones:          cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__leading_ones:     cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__trailing_zeros:        cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__trailing_zeros:   cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__trailing_ones:         cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__trailing_ones:    cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__small_nbr_ones:        cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__small_nbr_ones:   cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__small_nbr_zeros:       cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__small_nbr_zeros:  cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__checkerboard:          cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__checkerboard:     cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__long_ones_seq:         cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__long_ones_seq:    cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_ones__X__long_zeros_seq:        cross FP_B9_ops,  F32_sig_a_small_nbr_ones,  F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_ones__X__long_zeros_seq:   cross FP_sqrt_op, F32_sig_a_small_nbr_ones,                             F32_src_fmt;
        B9_F32__small_nbr_zeros__X__leading_zeros:        cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__leading_zeros:   cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__leading_ones:         cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__leading_ones:    cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__trailing_zeros:       cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__trailing_zeros:  cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__trailing_ones:        cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__trailing_ones:   cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__small_nbr_ones:       cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__small_nbr_ones:  cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__small_nbr_zeros:      cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__small_nbr_zeros: cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__checkerboard:         cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__checkerboard:    cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__long_ones_seq:        cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__long_ones_seq:   cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__small_nbr_zeros__X__long_zeros_seq:       cross FP_B9_ops,  F32_sig_a_small_nbr_zeros, F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__small_nbr_zeros__X__long_zeros_seq:  cross FP_sqrt_op, F32_sig_a_small_nbr_zeros,                            F32_src_fmt;
        B9_F32__checkerboard__X__leading_zeros:           cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__leading_zeros:      cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__leading_ones:            cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__leading_ones:       cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__trailing_zeros:          cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__trailing_zeros:     cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__trailing_ones:           cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__trailing_ones:      cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__small_nbr_ones:          cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__small_nbr_ones:     cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__small_nbr_zeros:         cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__small_nbr_zeros:    cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__checkerboard:            cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__checkerboard:       cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__long_ones_seq:           cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__long_ones_seq:      cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__checkerboard__X__long_zeros_seq:          cross FP_B9_ops,  F32_sig_a_checkerboard,    F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__checkerboard__X__long_zeros_seq:     cross FP_sqrt_op, F32_sig_a_checkerboard,                               F32_src_fmt;
        B9_F32__long_ones_seq__X__leading_zeros:          cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__leading_zeros:     cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__leading_ones:           cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__leading_ones:      cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__trailing_zeros:         cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__trailing_zeros:    cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__trailing_ones:          cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__trailing_ones:     cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__small_nbr_ones:         cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__small_nbr_ones:    cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__small_nbr_zeros:        cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__small_nbr_zeros:   cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__checkerboard:           cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__checkerboard:      cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__long_ones_seq:          cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__long_ones_seq:     cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_ones_seq__X__long_zeros_seq:         cross FP_B9_ops,  F32_sig_a_long_ones_seq,   F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__long_ones_seq__X__long_zeros_seq:    cross FP_sqrt_op, F32_sig_a_long_ones_seq,                              F32_src_fmt;
        B9_F32__long_zeros_seq__X__leading_zeros:         cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_leading_zeros,   F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__leading_zeros:    cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__leading_ones:          cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_leading_ones,    F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__leading_ones:     cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__trailing_zeros:        cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_trailing_zeros,  F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__trailing_zeros:   cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__trailing_ones:         cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_trailing_ones,   F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__trailing_ones:    cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__small_nbr_ones:        cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_small_nbr_ones,  F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__small_nbr_ones:   cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__small_nbr_zeros:       cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_small_nbr_zeros, F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__small_nbr_zeros:  cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__checkerboard:          cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_checkerboard,    F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__checkerboard:     cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__long_ones_seq:         cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_long_ones_seq,   F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__long_ones_seq:    cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
        B9_F32__long_zeros_seq__X__long_zeros_seq:        cross FP_B9_ops,  F32_sig_a_long_zeros_seq,  F32_sig_b_long_zeros_seq,  F32_src_fmt;
        B9_F32_sqrt__long_zeros_seq__X__long_zeros_seq:   cross FP_sqrt_op, F32_sig_a_long_zeros_seq,                             F32_src_fmt;
    `endif // COVER_F32

    `ifdef COVER_F64
        B9_F64__leading_zeros__X__leading_zeros:          cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__leading_zeros:     cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__leading_ones:           cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__leading_ones:      cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__trailing_zeros:         cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__trailing_zeros:    cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__trailing_ones:          cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__trailing_ones:     cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__small_nbr_ones:         cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__small_nbr_ones:    cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__small_nbr_zeros:        cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__small_nbr_zeros:   cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__checkerboard:           cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__checkerboard:      cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__long_ones_seq:          cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__long_ones_seq:     cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_zeros__X__long_zeros_seq:         cross FP_B9_ops,  F64_sig_a_leading_zeros,   F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__leading_zeros__X__long_zeros_seq:    cross FP_sqrt_op, F64_sig_a_leading_zeros,                              F64_src_fmt;
        B9_F64__leading_ones__X__leading_zeros:           cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__leading_zeros:      cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__leading_ones:            cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__leading_ones:       cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__trailing_zeros:          cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__trailing_zeros:     cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__trailing_ones:           cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__trailing_ones:      cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__small_nbr_ones:          cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__small_nbr_ones:     cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__small_nbr_zeros:         cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__small_nbr_zeros:    cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__checkerboard:            cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__checkerboard:       cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__long_ones_seq:           cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__long_ones_seq:      cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__leading_ones__X__long_zeros_seq:          cross FP_B9_ops,  F64_sig_a_leading_ones,    F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__leading_ones__X__long_zeros_seq:     cross FP_sqrt_op, F64_sig_a_leading_ones,                               F64_src_fmt;
        B9_F64__trailing_zeros__X__leading_zeros:         cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__leading_zeros:    cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__leading_ones:          cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__leading_ones:     cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__trailing_zeros:        cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__trailing_zeros:   cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__trailing_ones:         cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__trailing_ones:    cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__small_nbr_ones:        cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__small_nbr_ones:   cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__small_nbr_zeros:       cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__small_nbr_zeros:  cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__checkerboard:          cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__checkerboard:     cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__long_ones_seq:         cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__long_ones_seq:    cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_zeros__X__long_zeros_seq:        cross FP_B9_ops,  F64_sig_a_trailing_zeros,  F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__trailing_zeros__X__long_zeros_seq:   cross FP_sqrt_op, F64_sig_a_trailing_zeros,                             F64_src_fmt;
        B9_F64__trailing_ones__X__leading_zeros:          cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__leading_zeros:     cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__leading_ones:           cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__leading_ones:      cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__trailing_zeros:         cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__trailing_zeros:    cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__trailing_ones:          cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__trailing_ones:     cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__small_nbr_ones:         cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__small_nbr_ones:    cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__small_nbr_zeros:        cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__small_nbr_zeros:   cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__checkerboard:           cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__checkerboard:      cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__long_ones_seq:          cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__long_ones_seq:     cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__trailing_ones__X__long_zeros_seq:         cross FP_B9_ops,  F64_sig_a_trailing_ones,   F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__trailing_ones__X__long_zeros_seq:    cross FP_sqrt_op, F64_sig_a_trailing_ones,                              F64_src_fmt;
        B9_F64__small_nbr_ones__X__leading_zeros:         cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__leading_zeros:    cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__leading_ones:          cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__leading_ones:     cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__trailing_zeros:        cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__trailing_zeros:   cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__trailing_ones:         cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__trailing_ones:    cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__small_nbr_ones:        cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__small_nbr_ones:   cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__small_nbr_zeros:       cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__small_nbr_zeros:  cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__checkerboard:          cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__checkerboard:     cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__long_ones_seq:         cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__long_ones_seq:    cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_ones__X__long_zeros_seq:        cross FP_B9_ops,  F64_sig_a_small_nbr_ones,  F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_ones__X__long_zeros_seq:   cross FP_sqrt_op, F64_sig_a_small_nbr_ones,                             F64_src_fmt;
        B9_F64__small_nbr_zeros__X__leading_zeros:        cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__leading_zeros:   cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__leading_ones:         cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__leading_ones:    cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__trailing_zeros:       cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__trailing_zeros:  cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__trailing_ones:        cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__trailing_ones:   cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__small_nbr_ones:       cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__small_nbr_ones:  cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__small_nbr_zeros:      cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__small_nbr_zeros: cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__checkerboard:         cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__checkerboard:    cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__long_ones_seq:        cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__long_ones_seq:   cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__small_nbr_zeros__X__long_zeros_seq:       cross FP_B9_ops,  F64_sig_a_small_nbr_zeros, F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__small_nbr_zeros__X__long_zeros_seq:  cross FP_sqrt_op, F64_sig_a_small_nbr_zeros,                            F64_src_fmt;
        B9_F64__checkerboard__X__leading_zeros:           cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__leading_zeros:      cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__leading_ones:            cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__leading_ones:       cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__trailing_zeros:          cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__trailing_zeros:     cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__trailing_ones:           cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__trailing_ones:      cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__small_nbr_ones:          cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__small_nbr_ones:     cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__small_nbr_zeros:         cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__small_nbr_zeros:    cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__checkerboard:            cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__checkerboard:       cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__long_ones_seq:           cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__long_ones_seq:      cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__checkerboard__X__long_zeros_seq:          cross FP_B9_ops,  F64_sig_a_checkerboard,    F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__checkerboard__X__long_zeros_seq:     cross FP_sqrt_op, F64_sig_a_checkerboard,                               F64_src_fmt;
        B9_F64__long_ones_seq__X__leading_zeros:          cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__leading_zeros:     cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__leading_ones:           cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__leading_ones:      cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__trailing_zeros:         cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__trailing_zeros:    cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__trailing_ones:          cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__trailing_ones:     cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__small_nbr_ones:         cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__small_nbr_ones:    cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__small_nbr_zeros:        cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__small_nbr_zeros:   cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__checkerboard:           cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__checkerboard:      cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__long_ones_seq:          cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__long_ones_seq:     cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_ones_seq__X__long_zeros_seq:         cross FP_B9_ops,  F64_sig_a_long_ones_seq,   F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__long_ones_seq__X__long_zeros_seq:    cross FP_sqrt_op, F64_sig_a_long_ones_seq,                              F64_src_fmt;
        B9_F64__long_zeros_seq__X__leading_zeros:         cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_leading_zeros,   F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__leading_zeros:    cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__leading_ones:          cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_leading_ones,    F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__leading_ones:     cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__trailing_zeros:        cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_trailing_zeros,  F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__trailing_zeros:   cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__trailing_ones:         cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_trailing_ones,   F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__trailing_ones:    cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__small_nbr_ones:        cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_small_nbr_ones,  F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__small_nbr_ones:   cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__small_nbr_zeros:       cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_small_nbr_zeros, F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__small_nbr_zeros:  cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__checkerboard:          cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_checkerboard,    F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__checkerboard:     cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__long_ones_seq:         cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_long_ones_seq,   F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__long_ones_seq:    cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
        B9_F64__long_zeros_seq__X__long_zeros_seq:        cross FP_B9_ops,  F64_sig_a_long_zeros_seq,  F64_sig_b_long_zeros_seq,  F64_src_fmt;
        B9_F64_sqrt__long_zeros_seq__X__long_zeros_seq:   cross FP_sqrt_op, F64_sig_a_long_zeros_seq,                             F64_src_fmt;
    `endif // COVER_F64

    `ifdef COVER_F128
        B9_F128__leading_zeros__X__leading_zeros:          cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__leading_zeros:     cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__leading_ones:           cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__leading_ones:      cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__trailing_zeros:         cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__trailing_zeros:    cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__trailing_ones:          cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__trailing_ones:     cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__small_nbr_ones:         cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__small_nbr_ones:    cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__small_nbr_zeros:        cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__small_nbr_zeros:   cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__checkerboard:           cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__checkerboard:      cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__long_ones_seq:          cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__long_ones_seq:     cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_zeros__X__long_zeros_seq:         cross FP_B9_ops,  F128_sig_a_leading_zeros,   F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__leading_zeros__X__long_zeros_seq:    cross FP_sqrt_op, F128_sig_a_leading_zeros,                               F128_src_fmt;
        B9_F128__leading_ones__X__leading_zeros:           cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__leading_zeros:      cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__leading_ones:            cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__leading_ones:       cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__trailing_zeros:          cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__trailing_zeros:     cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__trailing_ones:           cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__trailing_ones:      cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__small_nbr_ones:          cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__small_nbr_ones:     cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__small_nbr_zeros:         cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__small_nbr_zeros:    cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__checkerboard:            cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__checkerboard:       cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__long_ones_seq:           cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__long_ones_seq:      cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__leading_ones__X__long_zeros_seq:          cross FP_B9_ops,  F128_sig_a_leading_ones,    F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__leading_ones__X__long_zeros_seq:     cross FP_sqrt_op, F128_sig_a_leading_ones,                                F128_src_fmt;
        B9_F128__trailing_zeros__X__leading_zeros:         cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__leading_zeros:    cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__leading_ones:          cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__leading_ones:     cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__trailing_zeros:        cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__trailing_zeros:   cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__trailing_ones:         cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__trailing_ones:    cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__small_nbr_ones:        cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__small_nbr_ones:   cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__small_nbr_zeros:       cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__small_nbr_zeros:  cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__checkerboard:          cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__checkerboard:     cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__long_ones_seq:         cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__long_ones_seq:    cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_zeros__X__long_zeros_seq:        cross FP_B9_ops,  F128_sig_a_trailing_zeros,  F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__trailing_zeros__X__long_zeros_seq:   cross FP_sqrt_op, F128_sig_a_trailing_zeros,                              F128_src_fmt;
        B9_F128__trailing_ones__X__leading_zeros:          cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__leading_zeros:     cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__leading_ones:           cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__leading_ones:      cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__trailing_zeros:         cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__trailing_zeros:    cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__trailing_ones:          cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__trailing_ones:     cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__small_nbr_ones:         cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__small_nbr_ones:    cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__small_nbr_zeros:        cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__small_nbr_zeros:   cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__checkerboard:           cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__checkerboard:      cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__long_ones_seq:          cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__long_ones_seq:     cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__trailing_ones__X__long_zeros_seq:         cross FP_B9_ops,  F128_sig_a_trailing_ones,   F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__trailing_ones__X__long_zeros_seq:    cross FP_sqrt_op, F128_sig_a_trailing_ones,                               F128_src_fmt;
        B9_F128__small_nbr_ones__X__leading_zeros:         cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__leading_zeros:    cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__leading_ones:          cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__leading_ones:     cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__trailing_zeros:        cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__trailing_zeros:   cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__trailing_ones:         cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__trailing_ones:    cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__small_nbr_ones:        cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__small_nbr_ones:   cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__small_nbr_zeros:       cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__small_nbr_zeros:  cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__checkerboard:          cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__checkerboard:     cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__long_ones_seq:         cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__long_ones_seq:    cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_ones__X__long_zeros_seq:        cross FP_B9_ops,  F128_sig_a_small_nbr_ones,  F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_ones__X__long_zeros_seq:   cross FP_sqrt_op, F128_sig_a_small_nbr_ones,                              F128_src_fmt;
        B9_F128__small_nbr_zeros__X__leading_zeros:        cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__leading_zeros:   cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__leading_ones:         cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__leading_ones:    cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__trailing_zeros:       cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__trailing_zeros:  cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__trailing_ones:        cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__trailing_ones:   cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__small_nbr_ones:       cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__small_nbr_ones:  cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__small_nbr_zeros:      cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__small_nbr_zeros: cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__checkerboard:         cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__checkerboard:    cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__long_ones_seq:        cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__long_ones_seq:   cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__small_nbr_zeros__X__long_zeros_seq:       cross FP_B9_ops,  F128_sig_a_small_nbr_zeros, F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__small_nbr_zeros__X__long_zeros_seq:  cross FP_sqrt_op, F128_sig_a_small_nbr_zeros,                             F128_src_fmt;
        B9_F128__checkerboard__X__leading_zeros:           cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__leading_zeros:      cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__leading_ones:            cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__leading_ones:       cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__trailing_zeros:          cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__trailing_zeros:     cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__trailing_ones:           cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__trailing_ones:      cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__small_nbr_ones:          cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__small_nbr_ones:     cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__small_nbr_zeros:         cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__small_nbr_zeros:    cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__checkerboard:            cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__checkerboard:       cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__long_ones_seq:           cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__long_ones_seq:      cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__checkerboard__X__long_zeros_seq:          cross FP_B9_ops,  F128_sig_a_checkerboard,    F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__checkerboard__X__long_zeros_seq:     cross FP_sqrt_op, F128_sig_a_checkerboard,                                F128_src_fmt;
        B9_F128__long_ones_seq__X__leading_zeros:          cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__leading_zeros:     cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__leading_ones:           cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__leading_ones:      cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__trailing_zeros:         cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__trailing_zeros:    cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__trailing_ones:          cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__trailing_ones:     cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__small_nbr_ones:         cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__small_nbr_ones:    cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__small_nbr_zeros:        cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__small_nbr_zeros:   cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__checkerboard:           cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__checkerboard:      cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__long_ones_seq:          cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__long_ones_seq:     cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_ones_seq__X__long_zeros_seq:         cross FP_B9_ops,  F128_sig_a_long_ones_seq,   F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__long_ones_seq__X__long_zeros_seq:    cross FP_sqrt_op, F128_sig_a_long_ones_seq,                               F128_src_fmt;
        B9_F128__long_zeros_seq__X__leading_zeros:         cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_leading_zeros,   F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__leading_zeros:    cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__leading_ones:          cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_leading_ones,    F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__leading_ones:     cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__trailing_zeros:        cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_trailing_zeros,  F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__trailing_zeros:   cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__trailing_ones:         cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_trailing_ones,   F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__trailing_ones:    cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__small_nbr_ones:        cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_small_nbr_ones,  F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__small_nbr_ones:   cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__small_nbr_zeros:       cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_small_nbr_zeros, F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__small_nbr_zeros:  cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__checkerboard:          cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_checkerboard,    F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__checkerboard:     cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__long_ones_seq:         cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_long_ones_seq,   F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__long_ones_seq:    cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
        B9_F128__long_zeros_seq__X__long_zeros_seq:        cross FP_B9_ops,  F128_sig_a_long_zeros_seq,  F128_sig_b_long_zeros_seq,  F128_src_fmt;
        B9_F128_sqrt__long_zeros_seq__X__long_zeros_seq:   cross FP_sqrt_op, F128_sig_a_long_zeros_seq,                              F128_src_fmt;
    `endif // COVER_F128

     
endgroup
