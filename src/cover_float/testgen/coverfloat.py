import ctypes
import os

filename = os.path.join(os.path.dirname(__file__), "../build/lib_coverfloat_reference.so")
try:
    lib = ctypes.CDLL(filename)
except:
    raise ImportError(f"lib_coverfloat_reference.so not found. Searched {filename}. Did you run make build?")

COVER_VECTOR_WIDTH_HEX = 213 # This mismatches coverfloat.h, not sure what is happening there
TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS = 144 + 8

def reference(line):
    cstr = line.encode('utf8') + b'\0'
    output = ctypes.create_string_buffer(COVER_VECTOR_WIDTH_HEX + 1) # One more for a null byte

    error = lib.coverfloat_runtestvector(cstr, len(cstr), output, len(output), True)
    if error == 2:
        raise ValueError(f"Coverfloat Error: Line {line} is malformed")
    elif error == 1:
        raise ValueError(f"Coverfloat Error: Testvector output doesn't match expected value")
    elif error != 0:
        raise ValueError(f"Coverfloat Error: Code {error}")
    
    # This is slicing is copied over from the old coverfloat_reference function, not sure if its necessary
    return output.raw.decode('u8').strip("\x00")[0:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS]