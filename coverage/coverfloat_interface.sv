interface coverfloat_interface; import coverfloat_pkg::*;

    // logic         clk;

    // logic         valid;

    logic [31:0]  op;

    logic [31:0]  rm;

    // logic [31:0]  enableBits; // legacy, not required for riscv TODO: consider having coverage based on these as a config option
    
    logic [127:0] a,    b,    c;
    logic [7:0]   aFmt, bFmt, cFmt; 

    logic [127:0] result;
    logic [7:0]   resultFmt;

    logic [31:0]  exceptionBits;

endinterface