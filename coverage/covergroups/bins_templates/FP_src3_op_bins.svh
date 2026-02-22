        // all operations where the third operand is FP

        // 3 fp inputs
        bins op_fma    = {[OP_FMA : OP_FMA | 32'hF]};
        bins op_fmadd  = {OP_FMADD};
        bins op_fmsub  = {OP_FMSUB};
        bins op_fnmadd = {OP_FNMADD};
        bins op_fnmsub = {OP_FNMSUB};
