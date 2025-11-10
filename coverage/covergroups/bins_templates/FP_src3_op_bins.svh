        // all operations where the third operand is FP

        // 1 fp input
        bins op_sqrt   = {OP_SQRT & ~32'hF};
        bins op_cfi    = {OP_CFI & ~32'hF};
        bins op_fcvtw  = {OP_FCVTW};
        bins op_fcvtwu = {OP_FCVTWU};
        bins op_fcvtl  = {OP_FCVTL};
        bins op_fcvtlu = {OP_FCVTLU};
        bins op_cff    = {OP_CFF & ~32'hF}; 
        bins op_class  = {OP_CLASS & ~32'hF};

        // 2 fp inputs
        bins op_add    = {OP_ADD & ~32'hF};
        bins op_sub    = {OP_SUB & ~32'hF};
        bins op_mul    = {OP_MUL & ~32'hF};
        bins op_div    = {OP_DIV & ~32'hF};
        bins op_rem    = {OP_REM & ~32'hF};
        bins op_qc     = {OP_QC & ~32'hF};
        bins op_feq    = {OP_FEQ};
        bins op_sc     = {OP_SC & ~32'hF};
        bins op_flt    = {OP_FLT};
        bins op_fle    = {OP_FLE};
        bins op_min    = {OP_MIN & ~32'hF};
        bins op_max    = {OP_MAX & ~32'hF};
        bins op_csn    = {OP_CSN & ~32'hF};
        bins op_fsgnj  = {OP_FSGNJ};
        bins op_fsgnjn = {OP_FSGNJN};
        bins op_fsgnjx = {OP_FSGNJX};
    
        // 3 fp inputs
        bins op_fma    = {OP_FMA & ~32'hF};
        bins op_fmadd  = {OP_FMADD};
        bins op_fmsub  = {OP_FMSUB};
        bins op_fnmadd = {OP_FNMADD};
        bins op_fnmsub = {OP_FNMSUB};
