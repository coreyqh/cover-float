# Created By: Ryan Wolk (rwolk@hmc.edu) on 2/26/2026

import math
import random

import cover_float.common.constants as constants


class B9SignificandGenerator:
    def __init__(self, nf: int) -> None:
        """Initialize B9 Significand Generation With the Default Generators"""
        self.nf = nf
        self.lead_trail_lengths = [
            nf - 1,
            nf - 2,
            nf - 3,
            math.ceil(3 * nf / 4),
            math.ceil(nf / 2),
            math.ceil(nf / 2 - 1),
            math.floor(nf / 4),
            3,
            2,
            1,
        ]

        # 0, nf - 1, nf/2, nf/2 - 2, powers of 2, random to fill 10
        one_sparse_positions = [[0], [nf - 1], [math.ceil(nf / 2)], [math.ceil(nf / 2 - 1)]]
        position = 1
        while position < nf and len(one_sparse_positions) < 10:
            one_sparse_positions.append([position])
            position *= 2
        while len(one_sparse_positions) < min(nf, 10):
            val = [random.randint(0, nf - 1)]
            if val not in one_sparse_positions:
                one_sparse_positions.append(val)

        two_sparse_positions: list[list[int]] = []
        while len(two_sparse_positions) < 3:
            k1 = random.randint(0, nf - 1)
            k2 = random.randint(0, nf - 1)

            if abs(k1 - k2) > nf / 2 and [k1, k2] not in two_sparse_positions:
                two_sparse_positions.append([k1, k2])

        self.sparse_positions = [*one_sparse_positions, *two_sparse_positions]

        # Checkerboards: List of (run_length, offset)
        self.checkerboards = [(run_length, offset) for run_length in range(1, 3) for offset in range(0, run_length * 2)]

        # Long Runs: List of (length, positions)
        self.long_runs: list[tuple[int, int]] = []
        self.long_runs.extend(self.evenly_generated_runs(math.floor(3 * nf / 4), nf, 6))
        self.long_runs.extend(self.evenly_generated_runs(math.floor(nf / 2), nf, 7))

    @staticmethod
    def swap_ones_and_zeros(s: str) -> str:
        swap_map = str.maketrans({"1": "0", "0": "1"})
        return s.translate(swap_map)

    @staticmethod
    def evenly_generated_runs(run_length: int, total_size: int, max_n: int) -> list[tuple[int, int]]:
        ans: list[tuple[int, int]] = []
        start = 1
        end = total_size - run_length

        for i in range(max_n):
            interpolated = start + (end - start) * (i / max_n)
            ans.append((run_length, math.floor(interpolated)))

        return sorted(list(set(ans)))

    def generate_leading_and_trailing(self) -> list[str]:
        def with_n_leading_zeros(n: int) -> str:
            mantissa = "0" * n + "1" + bin(random.getrandbits(self.nf))[2:]
            return mantissa[: self.nf]

        ans = ["1" * self.nf, "0" * self.nf]
        for length in self.lead_trail_lengths:
            leading_zeros = with_n_leading_zeros(length)
            leading_ones = self.swap_ones_and_zeros(with_n_leading_zeros(length))
            trailing_zeros = with_n_leading_zeros(length)[::-1]
            trailing_ones = self.swap_ones_and_zeros(with_n_leading_zeros(length))[::-1]

            ans.append(leading_zeros)
            ans.append(leading_ones)
            ans.append(trailing_zeros)
            ans.append(trailing_ones)

        return ans

    def generate_sparse(self) -> list[str]:
        ans: list[str] = []

        for positions in self.sparse_positions:
            as_list = ["0" for _ in range(self.nf)]
            for i in positions:
                as_list[i] = "1"

            ans.append("".join(as_list))
            ans.append(self.swap_ones_and_zeros("".join(as_list)))

        return ans

    def generate_checkerboards(self) -> list[str]:
        ans: list[str] = []

        for run_length, offset in self.checkerboards:
            pattern = ("1" * run_length + "0" * run_length) * (self.nf + 2)
            ans.append(pattern[offset : offset + self.nf])

        return ans

    def generate_long_runs(self) -> list[str]:
        ans: list[str] = []

        for run_length, start in self.long_runs:
            one_run_list = list(bin(random.getrandbits(self.nf))[2:].zfill(self.nf))

            # We want the run to be exact so we pad it
            one_run_list[start - 1] = "0"
            one_run_list[start + run_length] = "0"
            for i in range(start, start + run_length):
                one_run_list[i] = "1"
            ans.append("".join(one_run_list))

            zero_run_list = list(bin(random.getrandbits(self.nf))[2:].zfill(self.nf))

            # We want the run to be exact so we pad it
            zero_run_list[start - 1] = "0"
            zero_run_list[start + run_length] = "0"
            for i in range(start, start + run_length):
                zero_run_list[i] = "1"
            ans.append("".join(zero_run_list))

        return ans

    def generate(self) -> list[str]:
        return [
            *self.generate_leading_and_trailing(),
            *self.generate_sparse(),
            *self.generate_checkerboards(),
            *self.generate_long_runs(),
        ]


def main() -> None:
    random.seed(9)

    for fmt in constants.FLOAT_FMTS:
        generator = B9SignificandGenerator(constants.MANTISSA_BITS[fmt])
        generator.generate()
