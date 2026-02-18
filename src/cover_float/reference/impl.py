from typing import TextIO
import cover_float._reference
from cover_float.common.constants import TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS

def run_and_store_test_vector(test_vector: str, test_file: TextIO, cover_file: TextIO) -> None:
    """Run test_vector through coverfloat and store both the test vector and cover vector"""

    cover_vector = cover_float._reference.run_test_vector(test_vector)

    generated_test_vector = cover_vector[:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS]
    test_file.write(generated_test_vector + "\n")
    cover_file.write(cover_vector + "\n")

def store_cover_vector(cover_vector: str, test_file: TextIO, cover_file: TextIO) -> None:
    generated_test_vector = cover_vector[:TEST_VECTOR_WIDTH_HEX_WITH_SEPARATORS]
    test_file.write(generated_test_vector + "\n")
    cover_file.write(cover_vector)
