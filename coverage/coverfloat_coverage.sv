import coverfloat_pkg::*;
class coverfloat_coverage;

    `INCLUDE_CGS

    virtual coverfloat_interface CFI;

    // constructor (initializes covergroups)
    function new (virtual coverfloat_interface CFI);
        this.CFI = CFI;

        `INIT_CGS

    endfunction


    function void sample();

        // Call sample functions
        `SAMPLE_CGS

    endfunction

endclass
