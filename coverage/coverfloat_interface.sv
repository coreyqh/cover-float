interface coverfloat_interface; import coverfloat_pkg::*; // TODO: add params for covervector / DUT modes?

    // bit         clk;

    // bit         valid;

    bit [31:0]  op;

    bit [7:0]  rm;

    // bit [31:0]  enableBits; // legacy, not required for riscv TODO: consider having coverage based on these as a config option
    
    bit [127:0] a, b, c;
    bit [7:0]   operandFmt; 

    bit [127:0] result;
    bit [7:0]   resultFmt;

    bit         intermS;
    bit [31:0]  intermX;
    bit [191:0] intermM;

    bit [7:0]  exceptionBits;

endinterface