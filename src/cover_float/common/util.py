
def generate_test_vector(op: str, in1: int, in2: int, in3: int, fmt1: str, fmt2: str, rnd_mode: str = "00") -> str:
    zero_padding = "0" * 32
    return f"{op}_{rnd_mode}_{in1:032x}_{in2:032x}_{in3:032x}_{fmt1}_{zero_padding}_{fmt2}_00\n"

def reproducible_hash(s: str) -> int:
    """
    Return a simple hash of a string for use as a random seed.

    Python randomizes hashes by default, but we need a repeatable hash for repeatable test cases.
    """
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h





