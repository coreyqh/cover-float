import subprocess

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

FMTS = [FMT_SINGLE, FMT_DOUBLE, FMT_QUAD, FMT_HALF, FMT_BF16]

ROUND_NEAR_EVEN   = "00"
ROUND_MINMAG      = "01"
ROUND_MIN         = "02"
ROUND_MAX         = "03"
ROUND_NEAR_MAXMAG = "04"
ROUND_ODD         = "05"

SRC1_OPS = [OP_SQRT,
            OP_CFI,
            OP_FCVTW,
            OP_FCVTWU,
            OP_FCVTL,
            OP_FCVTLU,
            OP_CFF,
            OP_CLASS]

SRC2_OPS = [OP_ADD,
            OP_SUB,
            OP_MUL,
            OP_DIV,
            OP_REM,
            OP_QC,
            OP_FEQ,
            OP_SC,
            OP_FLT,
            OP_FLE,
            OP_MIN,
            OP_MAX,
            OP_CSN,
            OP_FSGNJ,
            OP_FSGNJN,
            OP_FSGNJX]

SRC3_OPS = [OP_FMA,
            OP_FMADD,
            OP_FMSUB,
            OP_FNMADD,
            OP_FNMSUB]

BASIC_TYPES = {

    FMT_SINGLE : [
        "00000000",
        "80000000",
        "3f800000",
        "bf800000",
        "3fc00000",
        "bfc00000",
        "40000000",
        "c0000000",
        "00800000",
        "80800000",
        "7f7fffff",
        "ff7fffff",
        "00800000",
        "7f7fffff",
        "80800000",
        "ff7fffff",
        "007fffff",
        "807fffff",
        "00000001",
        "80000001",
        "00000001",
        "007fffff",
        "80000001",
        "807fffff",
        "7f800000",
        "ff800000",
        "7fc00000",
        "7fffffff",
        "7f800001",
        "7fbfffff",
        "ffc00000",
        "ffffffff",
        "ff800001",
        "ffbfffff"
    ],
    
    FMT_DOUBLE : [
        "0000000000000000",
        "8000000000000000",
        "3FF0000000000000",
        "BFF0000000000000",
        "3FF8000000000000",
        "BFF8000000000000",
        "4000000000000000",
        "c000000000000000",
        "0010000000000000",
        "8010000000000000",
        "7FEFFFFFFFFFFFFF",
        "FFEFFFFFFFFFFFFF",
        "0010000000000000",
        "7FEFFFFFFFFFFFFF",
        "8010000000000000",
        "FFEFFFFFFFFFFFFF",
        "000FFFFFFFFFFFFF",
        "800FFFFFFFFFFFFF",
        "0000000000000001",
        "8000000000000001",
        "0000000000000001",
        "000FFFFFFFFFFFFF",
        "8000000000000001",
        "800FFFFFFFFFFFFF",
        "7FF0000000000000",
        "FFF0000000000000",
        "7FF8000000000000",
        "7FFFFFFFFFFFFFFF",
        "7FF0000000000001",
        "7FF7FFFFFFFFFFFF",
        "FFF8000000000000",
        "FFFFFFFFFFFFFFFF",
        "FFF0000000000001",
        "FFF7FFFFFFFFFFFF"
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
        "0000",
        "8000",
        "3C00",
        "BC00",
        "3E00",
        "BE00",
        "4000",
        "C000",
        "0400",
        "8400",
        "7BFF",
        "FBFF",
        "0400",
        "7BFF",
        "8400",
        "FBFF",
        "03FF",
        "83FF",
        "0001",
        "8001",
        "0001",
        "03FF",
        "8001",
        "83FF",
        "7C00",
        "FC00",
        "7E00",
        "7FFF",
        "7C01",
        "7DFF",
        "FE00",
        "FFFF",
        "FC01",
        "FDFF"
    ],
    
    FMT_BF16   : [
        "0000",
        "8000",
        "3f80",
        "bf80",
        "3fc0",
        "bfc0",
        "4000",
        "c000",
        "0080",
        "8080",
        "7f7f",
        "ff7f",
        "0080",
        "7f7f",
        "8080",
        "ff7f",
        "007f",
        "807f",
        "0001",
        "8001",
        "0001",
        "007f",
        "8001",
        "807f",
        "7f80",
        "ff80",
        "7fc0",
        "7fff",
        "7f81",
        "7fbf",
        "ffc0",
        "ffff",
        "ff81",
        "ffbf",
    ]
}


def write1SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    # print("\n//", file=f)
    print("// 1 source operations, all basic type input combinations", file=f)
    # print("//", file=f)
    for op in SRC1_OPS:
        print(f"OP IS: {op}")
        for val in BASIC_TYPES[fmt]:
            try:
                result = subprocess.run(
                    ["./build/coverfloat_reference", "-", "-", "--no-error-check"],
                    input=f"{op}_{rm}_{val}_{32*"0"}_{32*"0"}_{fmt}_{32*"0"}_{fmt}_00\n",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                output = result.stdout
            except subprocess.CalledProcessError as e:
                print("Error:", e.stderr)

            print(output[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS] + "\n", file=f)

def write2SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    print("// 2 source operations, all basic type input combinations", file=f)
    for op in SRC2_OPS:
        print(f"OP IS: {op}")
        for val1 in BASIC_TYPES[fmt]:
            for val2 in BASIC_TYPES[fmt]:
                try:
                    result = subprocess.run(
                        ["./build/coverfloat_reference", "-", "-", "--no-error-check"],
                        input=f"{op}_{rm}_{val1}_{val2}_{32*"0"}_{fmt}_{32*"0"}_{fmt}_00\n",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    output = result.stdout
                except subprocess.CalledProcessError as e:
                    print("Error:", e.stderr)

                print(output[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS] + "\n", file=f)

def write3SrcTests(f, fmt):
    
    rm = ROUND_NEAR_EVEN

    print("// 3 source operations, all basic type input combinations", file=f)
    for op in SRC3_OPS:
        print(f"OP IS: {op}")
        for val1 in BASIC_TYPES[fmt]:
            for val2 in BASIC_TYPES[fmt]:
                for val3 in BASIC_TYPES[fmt]:
                    try:
                        result = subprocess.run(
                            ["./build/coverfloat_reference", "-", "-", "--no-error-check"],
                            input=f"{op}_{rm}_{val1}_{val2}_{val3}_{fmt}_{32*"0"}_{fmt}_00\n",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True
                        )
                        output = result.stdout
                    except subprocess.CalledProcessError as e:
                        print("Error:", e.stderr)

                    print(output[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS], file=f)

def main():
    with open("./tests/testvectors/B1_tv.txt", "w") as f:
        for fmt in FMTS:
            write1SrcTests(f, fmt)
            write2SrcTests(f, fmt)
            write3SrcTests(f, fmt)
            # writeResultTests(f, fmt)

if __name__ == "__main__":
    main()