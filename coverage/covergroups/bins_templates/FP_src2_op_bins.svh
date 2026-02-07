        // all operations where the second operand is FP

        // 2 fp inputs
        bins op_add    = {[OP_ADD : OP_ADD | 32'hF]};
        bins op_sub    = {[OP_SUB : OP_SUB | 32'hF]};
        bins op_mul    = {[OP_MUL : OP_MUL | 32'hF]};
        bins op_div    = {[OP_DIV : OP_DIV | 32'hF]};
        // bins op_rem    = {[OP_REM : OP_REM | 32'hF]};
        bins op_qc     = {[OP_QC  : OP_QC  | 32'hF]};
        bins op_feq    = {OP_FEQ};
        bins op_sc     = {[OP_SC  : OP_SC  | 32'hF]};
        bins op_flt    = {OP_FLT};
        bins op_fle    = {OP_FLE};
        bins op_min    = {[OP_MIN : OP_MIN | 32'hF]};
        bins op_max    = {[OP_MAX : OP_MAX | 32'hF]};
        bins op_csn    = {[OP_CSN : OP_CSN | 32'hF]};
        bins op_fsgnj  = {OP_FSGNJ};
        bins op_fsgnjn = {OP_FSGNJN};
        bins op_fsgnjx = {OP_FSGNJX};
        