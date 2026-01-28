import subprocess
import random

TEST_VECTOR_WIDTH_HEX  = 144
TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS = (TEST_VECTOR_WIDTH_HEX + 8)

OP_ADD    = "00000010"
OP_SUB    = "00000020"
OP_MUL    = "00000030"
OP_DIV    = "00000040"
OP_FMA    = "00000050"
OP_FMADD  = "00000051"
OP_FMSUB  = "00000052"
OP_FNMADD = "00000053"
OP_FNMSUB = "00000054"
OP_SQRT   = "00000060"
OP_REM    = "00000070"
OP_CFI    = "00000080"
OP_FCVTW  = "00000081"
OP_FCVTWU = "00000082"
OP_FCVTL  = "00000083"
OP_FCVTLU = "00000084"
OP_CFF    = "00000090"
OP_CIF    = "000000A0"
OP_QC     = "000000B0"
OP_FEQ    = "000000B1"
OP_SC     = "000000C0"
OP_FLT    = "000000C1"
OP_FLE    = "000000C2"
OP_CLASS  = "000000D0"
OP_MIN    = "000000E0"
OP_MAX    = "000000F0"
OP_CSN    = "00000100"
OP_FSGNJ  = "00000101"
OP_FSGNJN = "00000102"
OP_FSGNJX = "00000103"

FMT_INVAL  = "FF" # 11111111
FMT_HALF   = "00" # 00000000
FMT_SINGLE = "01" # 00000001
FMT_DOUBLE = "02" # 00000010
FMT_QUAD   = "03" # 00000011
FMT_BF16   = "04" # 00000100
FMT_INT    = "81" # 10000001
FMT_UINT   = "C1" # 11000001
FMT_LONG   = "82" # 10000010
FMT_ULONG  = "C2" # 11000010

FMTS     = [FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_HALF, FMT_BF16]
INT_FMTS = [FMT_INT, FMT_UINT, FMT_LONG, FMT_ULONG]

ROUND_NEAR_EVEN   = "00"
ROUND_MINMAG      = "01"
ROUND_MIN         = "02"
ROUND_MAX         = "03"
ROUND_NEAR_MAXMAG = "04"
ROUND_ODD         = "05"

SRC1_OPS = [OP_SQRT,
            OP_CLASS]

CVT_OPS =  [OP_CFI,
            OP_CFF]

SRC2_OPS = [OP_ADD,
            OP_SUB,
            OP_MUL,
            OP_DIV,
            OP_REM,
            OP_FEQ,
            OP_FLT,
            OP_FLE,
            OP_MIN,
            OP_MAX,
            OP_FSGNJ,
            OP_FSGNJN,
            OP_FSGNJX ]

            # superset ops (no designated test)
            # OP_QC,
            # OP_SC,
            # OP_CSN,

SRC3_OPS = [OP_FMADD,
            OP_FMSUB,
            OP_FNMADD,
            OP_FNMSUB]

            # superset ops (no designated test)
            # OP_FMA,

RES_OPS = [OP_ADD,
           OP_SUB,
           OP_MUL,
           OP_DIV,
           OP_REM,
           OP_MIN,
           OP_MAX,
           OP_FSGNJ,
           OP_FSGNJN,
           OP_FSGNJX,
           OP_FMADD,
           OP_FMSUB,
           OP_FNMADD,
           OP_FNMSUB,
           OP_SQRT]
        
        #    OP_CSN,
        #    OP_FMA,

# INVERSE_OPS = {
#     OP_ADD    : OP_SUB
#     OP_SUB    : OP_ADD 
#     OP_MUL    : OP_DIV
#     OP_DIV    : OP_MUL
#     OP_REM    : 
#     OP_MIN    : 
#     OP_MAX    : 
#     OP_FSGNJ  : 
#     OP_FSGNJN : 
#     OP_FSGNJX : 
#     OP_FMADD  : 
#     OP_FMSUB  : 
#     OP_FNMADD : 
#     OP_FNMSUB : 
#     OP_SQRT   : 


# }

BASIC_TYPES = {

    FMT_SINGLE : [
        "00000000000000000000000000000000",
        "00000000000000000000000080000000",
        "0000000000000000000000003f800000",
        "000000000000000000000000bf800000",
        "0000000000000000000000003fc00000",
        "000000000000000000000000bfc00000",
        "00000000000000000000000040000000",
        "000000000000000000000000c0000000",
        "00000000000000000000000000800000",
        "00000000000000000000000080800000",
        "0000000000000000000000007f7fffff",
        "000000000000000000000000ff7fffff",
        # "00000000000000000000000000800000", TODO: NEED TO INCREMENT TO MATCH BINS
        # "0000000000000000000000007f7fffff",
        # "00000000000000000000000080800000",
        # "000000000000000000000000ff7fffff",
        "000000000000000000000000007fffff",
        "000000000000000000000000807fffff",
        "00000000000000000000000000400000",
        "00000000000000000000000080400000",
        "00000000000000000000000000000001",
        "00000000000000000000000080000001",
        # "00000000000000000000000000000001",
        # "000000000000000000000000007fffff",
        # "00000000000000000000000080000001",
        # "000000000000000000000000807fffff",
        "0000000000000000000000007f800000",
        "000000000000000000000000ff800000",
        "0000000000000000000000007fc00000",
        "0000000000000000000000007fffffff",
        "0000000000000000000000007f800001",
        "0000000000000000000000007fbfffff",
        "000000000000000000000000ffc00000",
        "000000000000000000000000ffffffff",
        "000000000000000000000000ff800001",
        "000000000000000000000000ffbfffff"
    ],
    
    FMT_DOUBLE : [
        "00000000000000000000000000000000",
        "00000000000000008000000000000000",
        "00000000000000003FF0000000000000",
        "0000000000000000BFF0000000000000",
        "00000000000000003FF8000000000000",
        "0000000000000000BFF8000000000000",
        "00000000000000004000000000000000",
        "0000000000000000c000000000000000",
        "00000000000000000010000000000000",
        "00000000000000008010000000000000",
        "00000000000000007FEFFFFFFFFFFFFF",
        "0000000000000000FFEFFFFFFFFFFFFF",
        "00000000000000000010000000000000",
        "00000000000000007FEFFFFFFFFFFFFF",
        "00000000000000008010000000000000",
        "0000000000000000FFEFFFFFFFFFFFFF",
        "0000000000000000000FFFFFFFFFFFFF",
        "0000000000000000800FFFFFFFFFFFFF",
        "00000000000000000000000000000001",
        "00000000000000008000000000000001",
        "00000000000000000000000000000001",
        "0000000000000000000FFFFFFFFFFFFF",
        "00000000000000008000000000000001",
        "0000000000000000800FFFFFFFFFFFFF",
        "00000000000000000008000000000000",
        "00000000000000008008000000000000",
        "00000000000000007FF0000000000000",
        "0000000000000000FFF0000000000000",
        "00000000000000007FF8000000000000",
        "00000000000000007FFFFFFFFFFFFFFF",
        "00000000000000007FF0000000000001",
        "00000000000000007FF7FFFFFFFFFFFF",
        "0000000000000000FFF8000000000000",
        "0000000000000000FFFFFFFFFFFFFFFF",
        "0000000000000000FFF0000000000001",
        "0000000000000000FFF7FFFFFFFFFFFF"
    ],
    
    FMT_QUAD   : [
        "00000000000000000000000000000000",
        "80000000000000000000000000000000",
        "3FFF0000000000000000000000000000",
        "BFFF0000000000000000000000000000",
        "3FFF8000000000000000000000000000",
        "BFFF8000000000000000000000000000",
        "40000000000000000000000000000000",
        "c0000000000000000000000000000000",
        "00010000000000000000000000000000",
        "80010000000000000000000000000000",
        "7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "00010000000000000000000000000000",
        "7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "80010000000000000000000000000000",
        "FFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "0000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "8000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "00000000000000000000000000000001",
        "80000000000000000000000000000001",
        "00000000000000000000000000000001",
        "0000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "80000000000000000000000000000001",
        "8000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "0000E000000000000000000000000000",
        "8000E000000000000000000000000000",
        "7FFF0000000000000000000000000000",
        "FFFF0000000000000000000000000000",
        "7FFF8000000000000000000000000000",
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "7FFF0000000000000000000000000001",
        "7FFF7FFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFF8000000000000000000000000000",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFF0000000000000000000000000001",
        "FFFF7FFFFFFFFFFFFFFFFFFFFFFFFFFF"
    ],
    
    FMT_HALF   : [
        "00000000000000000000000000000000",
        "00000000000000000000000000008000",
        "00000000000000000000000000003C00",
        "0000000000000000000000000000BC00",
        "00000000000000000000000000003E00",
        "0000000000000000000000000000BE00",
        "00000000000000000000000000004000",
        "0000000000000000000000000000C000",
        "00000000000000000000000000000400",
        "00000000000000000000000000008400",
        "00000000000000000000000000007BFF",
        "0000000000000000000000000000FBFF",
        "00000000000000000000000000000400",
        "00000000000000000000000000007BFF",
        "00000000000000000000000000008400",
        "0000000000000000000000000000FBFF",
        "000000000000000000000000000003FF",
        "000000000000000000000000000083FF",
        "00000000000000000000000000000001",
        "00000000000000000000000000008001",
        "00000000000000000000000000000001",
        "000000000000000000000000000003FF",
        "00000000000000000000000000008001",
        "000000000000000000000000000083FF",
        "00000000000000000000000000000200",
        "00000000000000000000000000008200",
        "00000000000000000000000000007C00",
        "0000000000000000000000000000FC00",
        "00000000000000000000000000007E00",
        "00000000000000000000000000007FFF",
        "00000000000000000000000000007C01",
        "00000000000000000000000000007DFF",
        "0000000000000000000000000000FE00",
        "0000000000000000000000000000FFFF",
        "0000000000000000000000000000FC01",
        "0000000000000000000000000000FDFF"
    ],
    
    FMT_BF16   : [
        "00000000000000000000000000000000",
        "00000000000000000000000000008000",
        "00000000000000000000000000003f80",
        "0000000000000000000000000000bf80",
        "00000000000000000000000000003fc0",
        "0000000000000000000000000000bfc0",
        "00000000000000000000000000004000",
        "0000000000000000000000000000c000",
        "00000000000000000000000000000080",
        "00000000000000000000000000008080",
        "00000000000000000000000000007f7f",
        "0000000000000000000000000000ff7f",
        "00000000000000000000000000000080",
        "00000000000000000000000000007f7f",
        "00000000000000000000000000008080",
        "0000000000000000000000000000ff7f",
        "0000000000000000000000000000007f",
        "0000000000000000000000000000807f",
        "00000000000000000000000000000001",
        "00000000000000000000000000008001",
        "00000000000000000000000000000001",
        "0000000000000000000000000000007f",
        "00000000000000000000000000008001",
        "0000000000000000000000000000807f",
        "00000000000000000000000000000040",
        "00000000000000000000000000008040",
        "00000000000000000000000000007f80",
        "0000000000000000000000000000ff80",
        "00000000000000000000000000007fc0",
        "00000000000000000000000000007fff",
        "00000000000000000000000000007f81",
        "00000000000000000000000000007fbf",
        "0000000000000000000000000000ffc0",
        "0000000000000000000000000000ffff",
        "0000000000000000000000000000ff81",
        "0000000000000000000000000000ffbf",
    ]
}

def coverfloat_reference(line):
    assert len(line) == TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS + 1 
    #                                                           ^~~~~~~~~ newline character
    try:
        result = subprocess.run(
            ["./build/coverfloat_reference", "-", "-", "--no-error-check"],
            input=line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout
        # print(f"OUT:  {output}")
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

    return output[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS]
    # print(output[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS], file=f)


def write1SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    # print("\n//", file=f)
    print("// 1 source operations, all basic type input combinations", file=f)
    # print("//", file=f)
    for op in SRC1_OPS:
        print(f"OP IS: {op}")
        # print(f"FMT IS: {fmt}")
        for val in BASIC_TYPES[fmt]:
            print(coverfloat_reference(f"{op}_{rm}_{val}_{32*'0'}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)

def writeCvtTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    # print("\n//", file=f)
    print("// 1 source convert operations, all basic type input and result format combinations", file=f)
    # print("//", file=f)
    for op in CVT_OPS:
        print(f"OP IS: {op}")
        # print(f"FMT IS: {fmt}")
        fmts = FMTS if op == OP_CFF else INT_FMTS
        for resultFmt in fmts:
            if resultFmt != fmt:
                for val in BASIC_TYPES[fmt]:
                    print(coverfloat_reference(f"{op}_{rm}_{val}_{32*'0'}_{32*'0'}_{fmt}_{32*'0'}_{resultFmt}_00\n"), file=f)


def write2SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    print("// 2 source operations, all basic type input combinations", file=f)
    for op in SRC2_OPS:
        print(f"OP IS: {op}")
        for val1 in BASIC_TYPES[fmt]:
            for val2 in BASIC_TYPES[fmt]:
                print(coverfloat_reference(f"{op}_{rm}_{val1}_{val2}_{32*'0'}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)


def write3SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    print("// 3 source operations, all basic type input combinations", file=f)
    for op in SRC3_OPS:
        print(f"OP IS: {op}")
        for val1 in BASIC_TYPES[fmt]:
            for val2 in BASIC_TYPES[fmt]:
                for val3 in BASIC_TYPES[fmt]:
                    print(coverfloat_reference(f"{op}_{rm}_{val1}_{val2}_{val3}_{fmt}_{32*'0'}_{fmt}_00\n"), file=f)



def main():
    with open("./tests/testvectors/B1_tv.txt", "w") as f:
        for fmt in FMTS:
            write1SrcTests(f, fmt)
            write2SrcTests(f, fmt)
            write3SrcTests(f, fmt)
            writeCvtTests (f, fmt)
            # writeResultTests(f, fmt)

if __name__ == "__main__":
    main()

"""

import subprocess
import random

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TEST_VECTOR_WIDTH_HEX  = 144
TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS = TEST_VECTOR_WIDTH_HEX + 8

ROUND_NEAR_EVEN = "00"

# Opcodes
OP_ADD    = "00000010"
OP_SUB    = "00000020"
OP_MUL    = "00000030"
OP_DIV    = "00000040"
OP_FMADD  = "00000051"
OP_FMSUB  = "00000052"
OP_FNMADD = "00000053"
OP_FNMSUB = "00000054"
OP_SQRT   = "00000060"
OP_REM    = "00000070"
OP_CFI    = "00000080"
OP_CFF    = "00000090"
OP_FEQ    = "000000B1"
OP_FLT    = "000000C1"
OP_FLE    = "000000C2"
OP_CLASS  = "000000D0"
OP_MIN    = "000000E0"
OP_MAX    = "000000F0"
OP_FSGNJ  = "00000101"
OP_FSGNJN = "00000102"
OP_FSGNJX = "00000103"

# Formats
FMT_HALF   = "00"
FMT_SINGLE = "01"
FMT_DOUBLE = "02"
FMT_QUAD   = "03"
FMT_BF16   = "04"
FMT_INT    = "81"
FMT_UINT   = "C1"
FMT_LONG   = "82"
FMT_ULONG  = "C2"

FMTS     = [FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_HALF, FMT_BF16]
INT_FMTS = [FMT_INT, FMT_UINT, FMT_LONG, FMT_ULONG]

SRC1_OPS = [OP_SQRT, OP_CLASS]
SRC2_OPS = [
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REM,
    OP_FEQ, OP_FLT, OP_FLE,
    OP_MIN, OP_MAX,
    OP_FSGNJ, OP_FSGNJN, OP_FSGNJX
]
SRC3_OPS = [OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB]
CVT_OPS  = [OP_CFI, OP_CFF]

RES_OPS = [
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_REM,
    OP_MIN, OP_MAX,
    OP_FSGNJ, OP_FSGNJN, OP_FSGNJX,
    OP_FMADD, OP_FMSUB, OP_FNMADD, OP_FNMSUB,
    OP_SQRT
]

# -----------------------------------------------------------------------------
# BASIC_TYPES (unchanged - REQUIRED)
# -----------------------------------------------------------------------------

BASIC_TYPES = {

    FMT_SINGLE : [
        "00000000000000000000000000000000",
        "00000000000000000000000080000000",
        "0000000000000000000000003f800000",
        "000000000000000000000000bf800000",
        "0000000000000000000000003fc00000",
        "000000000000000000000000bfc00000",
        "00000000000000000000000040000000",
        "000000000000000000000000c0000000",
        "00000000000000000000000000800000",
        "00000000000000000000000080800000",
        "0000000000000000000000007f7fffff",
        "000000000000000000000000ff7fffff",
        "00000000000000000000000000800000",
        "0000000000000000000000007f7fffff",
        "00000000000000000000000080800000",
        "000000000000000000000000ff7fffff",
        "000000000000000000000000007fffff",
        "000000000000000000000000807fffff",
        "00000000000000000000000000400000",
        "00000000000000000000000080400000",
        "00000000000000000000000000000001",
        "00000000000000000000000080000001",
        "00000000000000000000000000000001",
        "000000000000000000000000007fffff",
        "00000000000000000000000080000001",
        "000000000000000000000000807fffff",
        "0000000000000000000000007f800000",
        "000000000000000000000000ff800000",
        "0000000000000000000000007fc00000",
        "0000000000000000000000007fffffff",
        "0000000000000000000000007f800001",
        "0000000000000000000000007fbfffff",
        "000000000000000000000000ffc00000",
        "000000000000000000000000ffffffff",
        "000000000000000000000000ff800001",
        "000000000000000000000000ffbfffff"
    ],
    
    FMT_DOUBLE : [
        "00000000000000000000000000000000",
        "00000000000000008000000000000000",
        "00000000000000003FF0000000000000",
        "0000000000000000BFF0000000000000",
        "00000000000000003FF8000000000000",
        "0000000000000000BFF8000000000000",
        "00000000000000004000000000000000",
        "0000000000000000c000000000000000",
        "00000000000000000010000000000000",
        "00000000000000008010000000000000",
        "00000000000000007FEFFFFFFFFFFFFF",
        "0000000000000000FFEFFFFFFFFFFFFF",
        "00000000000000000010000000000000",
        "00000000000000007FEFFFFFFFFFFFFF",
        "00000000000000008010000000000000",
        "0000000000000000FFEFFFFFFFFFFFFF",
        "0000000000000000000FFFFFFFFFFFFF",
        "0000000000000000800FFFFFFFFFFFFF",
        "00000000000000000000000000000001",
        "00000000000000008000000000000001",
        "00000000000000000000000000000001",
        "0000000000000000000FFFFFFFFFFFFF",
        "00000000000000008000000000000001",
        "0000000000000000800FFFFFFFFFFFFF",
        "00000000000000000008000000000000",
        "00000000000000008008000000000000",
        "00000000000000007FF0000000000000",
        "0000000000000000FFF0000000000000",
        "00000000000000007FF8000000000000",
        "00000000000000007FFFFFFFFFFFFFFF",
        "00000000000000007FF0000000000001",
        "00000000000000007FF7FFFFFFFFFFFF",
        "0000000000000000FFF8000000000000",
        "0000000000000000FFFFFFFFFFFFFFFF",
        "0000000000000000FFF0000000000001",
        "0000000000000000FFF7FFFFFFFFFFFF"
    ],
    
    FMT_QUAD   : [
        "00000000000000000000000000000000",
        "80000000000000000000000000000000",
        "3FFF0000000000000000000000000000",
        "BFFF0000000000000000000000000000",
        "3FFF8000000000000000000000000000",
        "BFFF8000000000000000000000000000",
        "40000000000000000000000000000000",
        "c0000000000000000000000000000000",
        "00010000000000000000000000000000",
        "80010000000000000000000000000000",
        "7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "00010000000000000000000000000000",
        "7FFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "80010000000000000000000000000000",
        "FFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "0000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "8000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "00000000000000000000000000000001",
        "80000000000000000000000000000001",
        "00000000000000000000000000000001",
        "0000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "80000000000000000000000000000001",
        "8000FFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "0000E000000000000000000000000000",
        "8000E000000000000000000000000000",
        "7FFF0000000000000000000000000000",
        "FFFF0000000000000000000000000000",
        "7FFF8000000000000000000000000000",
        "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "7FFF0000000000000000000000000001",
        "7FFF7FFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFF8000000000000000000000000000",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFF0000000000000000000000000001",
        "FFFF7FFFFFFFFFFFFFFFFFFFFFFFFFFF"
    ],
    
    FMT_HALF   : [
        "00000000000000000000000000000000",
        "00000000000000000000000000008000",
        "00000000000000000000000000003C00",
        "0000000000000000000000000000BC00",
        "00000000000000000000000000003E00",
        "0000000000000000000000000000BE00",
        "00000000000000000000000000004000",
        "0000000000000000000000000000C000",
        "00000000000000000000000000000400",
        "00000000000000000000000000008400",
        "00000000000000000000000000007BFF",
        "0000000000000000000000000000FBFF",
        "00000000000000000000000000000400",
        "00000000000000000000000000007BFF",
        "00000000000000000000000000008400",
        "0000000000000000000000000000FBFF",
        "000000000000000000000000000003FF",
        "000000000000000000000000000083FF",
        "00000000000000000000000000000001",
        "00000000000000000000000000008001",
        "00000000000000000000000000000001",
        "000000000000000000000000000003FF",
        "00000000000000000000000000008001",
        "000000000000000000000000000083FF",
        "00000000000000000000000000000200",
        "00000000000000000000000000008200",
        "00000000000000000000000000007C00",
        "0000000000000000000000000000FC00",
        "00000000000000000000000000007E00",
        "00000000000000000000000000007FFF",
        "00000000000000000000000000007C01",
        "00000000000000000000000000007DFF",
        "0000000000000000000000000000FE00",
        "0000000000000000000000000000FFFF",
        "0000000000000000000000000000FC01",
        "0000000000000000000000000000FDFF"
    ],
    
    FMT_BF16   : [
        "00000000000000000000000000000000",
        "00000000000000000000000000008000",
        "00000000000000000000000000003f80",
        "0000000000000000000000000000bf80",
        "00000000000000000000000000003fc0",
        "0000000000000000000000000000bfc0",
        "00000000000000000000000000004000",
        "0000000000000000000000000000c000",
        "00000000000000000000000000000080",
        "00000000000000000000000000008080",
        "00000000000000000000000000007f7f",
        "0000000000000000000000000000ff7f",
        "00000000000000000000000000000080",
        "00000000000000000000000000007f7f",
        "00000000000000000000000000008080",
        "0000000000000000000000000000ff7f",
        "0000000000000000000000000000007f",
        "0000000000000000000000000000807f",
        "00000000000000000000000000000001",
        "00000000000000000000000000008001",
        "00000000000000000000000000000001",
        "0000000000000000000000000000007f",
        "00000000000000000000000000008001",
        "0000000000000000000000000000807f",
        "00000000000000000000000000000040",
        "00000000000000000000000000008040",
        "00000000000000000000000000007f80",
        "0000000000000000000000000000ff80",
        "00000000000000000000000000007fc0",
        "00000000000000000000000000007fff",
        "00000000000000000000000000007f81",
        "00000000000000000000000000007fbf",
        "0000000000000000000000000000ffc0",
        "0000000000000000000000000000ffff",
        "0000000000000000000000000000ff81",
        "0000000000000000000000000000ffbf",
    ]
}


# -----------------------------------------------------------------------------
# Format metadata
# -----------------------------------------------------------------------------

FMT_BITS = {
    FMT_HALF:   16,
    FMT_BF16:   16,
    FMT_SINGLE: 32,
    FMT_DOUBLE: 64,
    FMT_QUAD:   128,
}

ZERO = {fmt: "0" * 32 for fmt in FMTS}

# -----------------------------------------------------------------------------
# Reference model helpers
# -----------------------------------------------------------------------------

def run_ref(line):
    p = subprocess.run(
        ["./build/coverfloat_reference", "-", "-", "--no-error-check"],
        input=line + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    return p.stdout.strip()

def extract_result(out):
    return out.split("_")[-3]

def emit(f, line):
    out = run_ref(line)
    f.write(out[:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS] + "\n")

# -----------------------------------------------------------------------------
# Random operand generation
# -----------------------------------------------------------------------------

def random_fp(fmt):
    bits = FMT_BITS[fmt]
    val  = random.getrandbits(bits)
    return f"{val:0{bits//4}x}".rjust(32, "0")

# -----------------------------------------------------------------------------
# Triviality filtering
# -----------------------------------------------------------------------------

def is_trivial(op, fmt, a, b=None, c=None):
    z = ZERO[fmt]

    if op == OP_SQRT:
        return a == z

    if op in [OP_ADD, OP_SUB]:
        return b == z

    if op == OP_MUL:
        return b in BASIC_TYPES[fmt][2:4]  # ±1

    if op == OP_DIV:
        return b in BASIC_TYPES[fmt][2:4]

    if op in SRC3_OPS:
        return b == z or c == z

    return False

# -----------------------------------------------------------------------------
# Result-driven operand solver (FIXED)
# -----------------------------------------------------------------------------

def find_operands_for_result(op, fmt, desired, max_attempts=20, refine_steps=5):
    '''
    Find operands that produce the desired result using the reference model.
    Algorithm:
        1. Pick a random first operand (or first two for 3-operand ops).
        2. Let the reference model compute the other operand(s) to hit 'desired'.
        3. Compare actual result to desired.
        4. If mismatch, adjust the computed operand(s) slightly and retry.
        5. After max_attempts, pick a new random starting operand.
    '''
    for _ in range(max_attempts):
        if op == OP_SQRT:
            a = random_fp(fmt)
            line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{32*'0'}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00"
            result_line = run_ref(line)
            if extract_result(result_line) == desired:
                return (a,)

        elif op in SRC2_OPS:
            a = random_fp(fmt)
            # Try to iteratively refine b
            b = ZERO[fmt]
            for _ in range(refine_steps):
                line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00"
                result_line = run_ref(line)
                actual = extract_result(result_line)
                if actual == desired:
                    return (a, b)
                # Slightly adjust b toward desired (use int representation for deterministic tweak)
                b = hex((int(b, 16) + int(desired, 16)) // 2)[2:].rjust(len(b), '0')

        elif op in SRC3_OPS:
            a, b = random_fp(fmt), random_fp(fmt)
            # Try to iteratively refine c
            c = ZERO[fmt]
            for _ in range(refine_steps):
                line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{ZERO[fmt]}_{fmt}_00"
                result_line = run_ref(line)
                actual = extract_result(result_line)
                if actual == desired:
                    return (a, b, c)
                # Slightly adjust c toward desired
                c = hex((int(c, 16) + int(desired, 16)) // 2)[2:].rjust(len(c), '0')
    return None


# -----------------------------------------------------------------------------
# Exhaustive tests (unchanged behavior)
# -----------------------------------------------------------------------------

def write1SrcTests(f, fmt):
    for op in SRC1_OPS:
        for a in BASIC_TYPES[fmt]:
            emit(f, f"{op}_{ROUND_NEAR_EVEN}_{a}_{32*'0'}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00")

def write2SrcTests(f, fmt):
    for op in SRC2_OPS:
        for a in BASIC_TYPES[fmt]:
            for b in BASIC_TYPES[fmt]:
                emit(f, f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00")

def write3SrcTests(f, fmt):
    for op in SRC3_OPS:
        for a in BASIC_TYPES[fmt]:
            for b in BASIC_TYPES[fmt]:
                for c in BASIC_TYPES[fmt]:
                    emit(f, f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{ZERO[fmt]}_{fmt}_00")

def writeCvtTests(f, fmt):
    for op in CVT_OPS:
        targets = FMTS if op == OP_CFF else INT_FMTS
        for tfmt in targets:
            if tfmt == fmt:
                continue
            for a in BASIC_TYPES[fmt]:
                emit(f, f"{op}_{ROUND_NEAR_EVEN}_{a}_{32*'0'}_{32*'0'}_{fmt}_{ZERO[fmt]}_{tfmt}_00")

# -----------------------------------------------------------------------------
# Result-driven designated tests
# -----------------------------------------------------------------------------

def writeResultTests(f, fmt):
    for op in RES_OPS:
        for desired in BASIC_TYPES[fmt]:
            print(f"trying to find operands for operation {op} to get result {desired}")
            ops = find_operands_for_result(op, fmt, desired)
            if ops is None:
                print("FAILED")
                continue

            if len(ops) == 1:
                a, = ops
                line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{32*'0'}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00"
            elif len(ops) == 2:
                a, b = ops
                line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{32*'0'}_{fmt}_{ZERO[fmt]}_{fmt}_00"
            else:
                a, b, c = ops
                line = f"{op}_{ROUND_NEAR_EVEN}_{a}_{b}_{c}_{fmt}_{ZERO[fmt]}_{fmt}_00"

            emit(f, line)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    random.seed(1)
    with open("./tests/testvectors/B1_tv.txt", "w") as f:
        for fmt in FMTS:
            # write1SrcTests(f, fmt)
            # write2SrcTests(f, fmt)
            # write3SrcTests(f, fmt)
            # writeCvtTests(f, fmt)
            writeResultTests(f, fmt)

if __name__ == "__main__":
    main()
"""