# SPDX-License-Identifier: MIT
"""Reproduce the computational verification reported in the manuscript.

The manuscript studies complete list decoding of logarithmic systematic codes
in the weighted Lee--Hamming metric.  The main test uses its GF(11) example. A
second, deliberately small GF(5) code is checked against an exhaustive scan of
the whole ambient space.  No floating-point or symbolic package is used: all
linear systems, polynomial operations and distance calculations are evaluated
exactly in the corresponding prime field.

Run this file directly with Python 3.9 or newer::

    python3 verify_article.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product, zip_longest
import random
from typing import Iterable, Iterator


Polynomial = list[int]
Word = tuple[int, ...]


def strict_zip(*iterables: Iterable[object]) -> Iterator[tuple[object, ...]]:
    """Zip equal-length iterables and fail on a length mismatch.

    This is the Python 3.9-compatible equivalent of ``zip(..., strict=True)``.
    """
    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("zip arguments have different lengths")
        yield values


def trim(poly: Polynomial, modulus: int) -> Polynomial:
    result = [coefficient % modulus for coefficient in poly]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def poly_degree(poly: Polynomial, modulus: int) -> int:
    return len(trim(poly, modulus)) - 1


def poly_sub(left: Polynomial, right: Polynomial, modulus: int) -> Polynomial:
    size = max(len(left), len(right))
    return trim(
        [
            (left[index] if index < len(left) else 0)
            - (right[index] if index < len(right) else 0)
            for index in range(size)
        ],
        modulus,
    )


def poly_scale(poly: Polynomial, scalar: int, modulus: int) -> Polynomial:
    return trim([scalar * coefficient for coefficient in poly], modulus)


def poly_divmod(
    dividend: Polynomial, divisor: Polynomial, modulus: int
) -> tuple[Polynomial, Polynomial]:
    remainder = trim(dividend, modulus)
    divisor = trim(divisor, modulus)
    if divisor == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = [0] * max(1, len(remainder) - len(divisor) + 1)
    inverse_lead = pow(divisor[-1], -1, modulus)
    while remainder != [0] and len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        coefficient = remainder[-1] * inverse_lead % modulus
        quotient[shift] = coefficient
        remainder = poly_sub(
            remainder,
            [0] * shift + poly_scale(divisor, coefficient, modulus),
            modulus,
        )
    return trim(quotient, modulus), trim(remainder, modulus)


def poly_gcd(left: Polynomial, right: Polynomial, modulus: int) -> Polynomial:
    left, right = trim(left, modulus), trim(right, modulus)
    while right != [0]:
        _, remainder = poly_divmod(left, right, modulus)
        left, right = right, remainder
    if left == [0]:
        return [0]
    return poly_scale(left, pow(left[-1], -1, modulus), modulus)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, int(value**0.5) + 1))


@dataclass(frozen=True)
class LogarithmicCode:
    """Systematic logarithmic code over Z/(Q-1) with Q prime."""

    Q: int
    g: int
    information_locators: tuple[int, ...]
    check_locators: tuple[int, ...]
    q: int = field(init=False)
    logarithms: tuple[int, ...] = field(init=False, repr=False)
    check_matrix: tuple[tuple[int, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not is_prime(self.Q):
            raise ValueError("this verifier requires a prime field")
        q = self.Q - 1
        powers: dict[int, int] = {}
        value = 1
        for exponent in range(q):
            powers.setdefault(value, exponent)
            value = value * self.g % self.Q
        if len(powers) != q:
            raise ValueError("g is not primitive")
        locators = self.information_locators + self.check_locators
        if len(set(locators)) != len(locators) or any(
            locator % self.Q == 0 for locator in locators
        ):
            raise ValueError("locators must be distinct and nonzero")

        logarithms = [-1] * self.Q
        for field_value, exponent in powers.items():
            logarithms[field_value] = exponent
        matrix = tuple(
            tuple(
                logarithms[
                    (1 - beta * pow(alpha, -1, self.Q)) % self.Q
                ]
                for alpha in self.information_locators
            )
            for beta in self.check_locators
        )
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "logarithms", tuple(logarithms))
        object.__setattr__(self, "check_matrix", matrix)

    @property
    def k(self) -> int:
        return len(self.information_locators)

    @property
    def r(self) -> int:
        return len(self.check_locators)

    def information_value(self, information: Iterable[int], point: int) -> int:
        value = 1
        for alpha, exponent in strict_zip(
            self.information_locators, information
        ):
            factor_value = (1 - point * pow(alpha, -1, self.Q)) % self.Q
            value = value * pow(factor_value, exponent, self.Q) % self.Q
        return value

    def encode(self, information: Iterable[int]) -> Word:
        information_tuple = tuple(value % self.q for value in information)
        if len(information_tuple) != self.k:
            raise ValueError("wrong information length")
        checks = tuple(
            sum(
                coefficient * symbol
                for coefficient, symbol in strict_zip(row, information_tuple)
            )
            % self.q
            for row in self.check_matrix
        )
        return information_tuple + checks


GF11_CODE = LogarithmicCode(
    Q=11,
    g=2,
    information_locators=(1, 2, 3, 4, 5, 6),
    check_locators=(7, 8, 9, 10),
)


def lee_representative(residue: int, q: int) -> int:
    """Use the positive representative for the even-alphabet antipode."""
    value = residue % q
    return value if value <= q // 2 else value - q


def weighted_distance(code: LogarithmicCode, left: Word, right: Word) -> int:
    information_weight = sum(
        abs(lee_representative(first - second, code.q))
        for first, second in strict_zip(left[: code.k], right[: code.k])
    )
    check_weight = sum(
        first != second
        for first, second in strict_zip(left[code.k :], right[code.k :])
    )
    return information_weight + 2 * check_weight


def affine_solutions(
    matrix: list[list[int]], right_hand_side: list[int], modulus: int
) -> Iterator[tuple[int, ...]]:
    """Enumerate every solution of A*x=b over a prime field."""
    if len(matrix) != len(right_hand_side):
        raise ValueError("incompatible linear-system dimensions")
    column_count = len(matrix[0]) if matrix else 0
    if any(len(row) != column_count for row in matrix):
        raise ValueError("ragged matrix")
    augmented = [
        [entry % modulus for entry in row] + [value % modulus]
        for row, value in strict_zip(matrix, right_hand_side)
    ]

    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(column_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(augmented))
                if augmented[row][column] % modulus
            ),
            None,
        )
        if selected is None:
            continue
        augmented[pivot_row], augmented[selected] = (
            augmented[selected],
            augmented[pivot_row],
        )
        inverse = pow(augmented[pivot_row][column], -1, modulus)
        augmented[pivot_row] = [
            value * inverse % modulus for value in augmented[pivot_row]
        ]
        for row in range(len(augmented)):
            if row == pivot_row:
                continue
            multiplier = augmented[row][column]
            if multiplier:
                augmented[row] = [
                    (value - multiplier * pivot_value) % modulus
                    for value, pivot_value in strict_zip(
                        augmented[row], augmented[pivot_row]
                    )
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(augmented):
            break

    if any(
        all(value == 0 for value in row[:column_count])
        and row[column_count] != 0
        for row in augmented
    ):
        return

    free_columns = [
        column for column in range(column_count) if column not in pivot_columns
    ]
    particular = [0] * column_count
    for row, column in enumerate(pivot_columns):
        particular[column] = augmented[row][column_count]

    basis: list[list[int]] = []
    for free_column in free_columns:
        vector = [0] * column_count
        vector[free_column] = 1
        for row, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -augmented[row][free_column] % modulus
        basis.append(vector)

    for parameters in product(range(modulus), repeat=len(basis)):
        solution = particular.copy()
        for parameter, vector in strict_zip(parameters, basis):
            for column, value in enumerate(vector):
                solution[column] = (
                    solution[column] + parameter * value
                ) % modulus
        yield tuple(solution)


def factor_information_polynomial(
    code: LogarithmicCode, poly: Polynomial
) -> tuple[int, ...] | None:
    """Factor a normalized polynomial over the permitted information roots."""
    remainder = trim(poly, code.Q)
    if remainder[0] != 1:
        return None
    multiplicities: list[int] = []
    for alpha in code.information_locators:
        factor_poly = [1, -pow(alpha, -1, code.Q)]
        multiplicity = 0
        while poly_degree(remainder, code.Q) > 0:
            quotient, residual = poly_divmod(remainder, factor_poly, code.Q)
            if residual != [0]:
                break
            multiplicity += 1
            remainder = quotient
        multiplicities.append(multiplicity)
    return tuple(multiplicities) if remainder == [1] else None


def ratio_samples(code: LogarithmicCode, received: Word) -> tuple[int, ...]:
    information = received[: code.k]
    checks = received[code.k :]
    return tuple(
        code.information_value(information, beta)
        * pow(pow(code.g, check, code.Q), -1, code.Q)
        % code.Q
        for beta, check in strict_zip(code.check_locators, checks)
    )


def decode_list(code: LogarithmicCode, received: Word, rho: int) -> list[Word]:
    """Return the complete weighted Lee--Hamming ball intersection.

    For each signed information-error sum nu, solve

        N(beta_j) = S_j D(beta_j),  N(0) = D(0) = 1,

    under degree bounds floor((rho+nu)/2) and floor((rho-nu)/2).
    One affine solution is sufficient even when the system is rank deficient.
    Indeed, the cross-product of two solutions has degree at most rho <= r
    and vanishes at zero and at all r check locators, hence it is identically
    zero.  Thus all solutions give the same reduced rational function.  GCD
    reduction, allowed-root factorization and an exact distance check discard
    spurious rational reconstructions.
    """
    if len(received) != code.k + code.r:
        raise ValueError("wrong received-word length")
    if not 0 <= rho <= code.r:
        raise ValueError("this decoder is verified only for 0 <= rho <= r")

    received = tuple(value % code.q for value in received)
    ratios = ratio_samples(code, received)
    candidates: set[Word] = set()

    for nu in range(-rho, rho + 1):
        numerator_bound = (rho + nu) // 2
        denominator_bound = (rho - nu) // 2
        if numerator_bound < 0 or denominator_bound < 0:
            continue

        matrix: list[list[int]] = []
        right_hand_side: list[int] = []
        for beta, ratio in strict_zip(code.check_locators, ratios):
            numerator_coefficients = [
                pow(beta, degree, code.Q)
                for degree in range(1, numerator_bound + 1)
            ]
            denominator_coefficients = [
                -ratio * pow(beta, degree, code.Q) % code.Q
                for degree in range(1, denominator_bound + 1)
            ]
            matrix.append(numerator_coefficients + denominator_coefficients)
            right_hand_side.append((ratio - 1) % code.Q)

        solution = next(
            affine_solutions(matrix, right_hand_side, code.Q), None
        )
        if solution is not None:
            numerator = [1] + list(solution[:numerator_bound])
            denominator = [1] + list(solution[numerator_bound:])
            common = poly_gcd(numerator, denominator, code.Q)
            positive_poly, positive_residual = poly_divmod(
                numerator, common, code.Q
            )
            negative_poly, negative_residual = poly_divmod(
                denominator, common, code.Q
            )
            if positive_residual != [0] or negative_residual != [0]:
                continue
            if (
                positive_poly[0] == 0
                or positive_poly[0] != negative_poly[0]
            ):
                continue
            normalization = pow(positive_poly[0], -1, code.Q)
            positive_poly = poly_scale(
                positive_poly, normalization, code.Q
            )
            negative_poly = poly_scale(
                negative_poly, normalization, code.Q
            )

            positive = factor_information_polynomial(code, positive_poly)
            negative = factor_information_polynomial(code, negative_poly)
            if positive is None or negative is None:
                continue
            if any(
                increase and decrease
                for increase, decrease in strict_zip(positive, negative)
            ):
                continue

            candidate_information = tuple(
                (symbol - increase + decrease) % code.q
                for symbol, increase, decrease in strict_zip(
                    received[: code.k], positive, negative
                )
            )
            signed_errors = tuple(
                lee_representative(observed - original, code.q)
                for observed, original in strict_zip(
                    received[: code.k], candidate_information
                )
            )
            reconstructed_errors = tuple(
                increase - decrease
                for increase, decrease in strict_zip(positive, negative)
            )
            if signed_errors != reconstructed_errors or sum(signed_errors) != nu:
                continue

            candidate = code.encode(candidate_information)
            if weighted_distance(code, received, candidate) <= rho:
                candidates.add(candidate)

    return sorted(candidates)


def canonical_signed_values(q: int, bound: int) -> tuple[int, ...]:
    positive = tuple(range(1, min(q // 2, bound) + 1))
    maximum_negative = min((q - 1) // 2, bound)
    negative = tuple(-magnitude for magnitude in range(1, maximum_negative + 1))
    return (0,) + positive + negative


def lee_error_vectors(q: int, length: int, bound: int) -> Iterator[tuple[int, ...]]:
    values = canonical_signed_values(q, bound)

    def visit(prefix: tuple[int, ...], remaining: int) -> Iterator[tuple[int, ...]]:
        if len(prefix) == length:
            yield prefix
            return
        for value in values:
            if abs(value) <= remaining:
                yield from visit(prefix + (value,), remaining - abs(value))

    yield from visit((), bound)


def corrupted_suffixes(
    suffix: Word, q: int, maximum_errors: int
) -> Iterator[Word]:
    yield suffix
    for error_count in range(1, maximum_errors + 1):
        for positions in combinations(range(len(suffix)), error_count):
            for increments in product(range(1, q), repeat=error_count):
                corrupted = list(suffix)
                for position, increment in strict_zip(positions, increments):
                    corrupted[position] = (corrupted[position] + increment) % q
                yield tuple(corrupted)


def received_words_in_ball(
    code: LogarithmicCode, codeword: Word, rho: int
) -> Iterator[Word]:
    for error in lee_error_vectors(code.q, code.k, rho):
        information_weight = sum(abs(value) for value in error)
        received_information = tuple(
            (symbol + increment) % code.q
            for symbol, increment in strict_zip(codeword[: code.k], error)
        )
        maximum_check_errors = (rho - information_weight) // 2
        yield from (
            received_information + suffix
            for suffix in corrupted_suffixes(
                codeword[code.k :], code.q, maximum_check_errors
            )
        )


def brute_force_list_full(
    code: LogarithmicCode,
    received: Word,
    rho: int,
    all_codewords: Iterable[Word],
) -> list[Word]:
    return sorted(
        codeword
        for codeword in all_codewords
        if weighted_distance(code, received, codeword) <= rho
    )


def brute_force_list_local(
    code: LogarithmicCode, received: Word, rho: int
) -> list[Word]:
    """Exact brute force restricted only by the necessary Lee bound."""
    candidates: set[Word] = set()
    for error in lee_error_vectors(code.q, code.k, rho):
        information = tuple(
            (symbol - increment) % code.q
            for symbol, increment in strict_zip(received[: code.k], error)
        )
        codeword = code.encode(information)
        if weighted_distance(code, received, codeword) <= rho:
            candidates.add(codeword)
    return sorted(candidates)


def verify_small_code_exhaustively() -> tuple[
    int, dict[int, dict[int, int]]
]:
    code = LogarithmicCode(
        Q=5,
        g=2,
        information_locators=(1, 2),
        check_locators=(3, 4),
    )
    all_codewords = [
        code.encode(information)
        for information in product(range(code.q), repeat=code.k)
    ]
    comparisons = 0
    distribution = {
        rho: {0: 0, 1: 0, 2: 0} for rho in range(code.r + 1)
    }
    for received in product(range(code.q), repeat=code.k + code.r):
        for rho in range(code.r + 1):
            decoded = decode_list(code, received, rho)
            assert decoded == brute_force_list_full(
                code, received, rho, all_codewords
            )
            if len(decoded) not in distribution[rho]:
                raise AssertionError("unexpected GF(5) list size")
            distribution[rho][len(decoded)] += 1
            comparisons += 1
    return comparisons, distribution


def verify_gf11_ball_exhaustively() -> int:
    transmitted = GF11_CODE.encode((0, 9, 1, 8, 2, 7))
    count = 0
    for received in received_words_in_ball(GF11_CODE, transmitted, rho=4):
        decoded = decode_list(GF11_CODE, received, rho=4)
        assert transmitted in decoded
        assert len(decoded) <= 2 * 4 + 1
        assert all(
            weighted_distance(GF11_CODE, received, candidate) <= 4
            for candidate in decoded
        )
        count += 1
    return count


def verify_gf11_randomly(sample_count: int = 200) -> int:
    generator = random.Random(20260723)
    for _ in range(sample_count):
        received = tuple(
            generator.randrange(GF11_CODE.q)
            for _ in range(GF11_CODE.k + GF11_CODE.r)
        )
        rho = generator.randrange(GF11_CODE.r + 1)
        decoded = decode_list(GF11_CODE, received, rho)
        assert len(decoded) <= 2 * rho + 1
        assert decoded == brute_force_list_local(GF11_CODE, received, rho)
    return sample_count


def main() -> None:
    if not __debug__:
        raise RuntimeError("run the verifier without Python's -O option")

    expected_matrix = (
        (4, 8, 9, 1, 2, 6),
        (2, 3, 1, 5, 9, 7),
        (8, 1, 6, 7, 3, 4),
        (1, 7, 4, 2, 5, 8),
    )
    assert GF11_CODE.check_matrix == expected_matrix

    information = (3, 3, 3, 3, 3, 3)
    transmitted = GF11_CODE.encode(information)
    worked_received = (4, 3, 3, 3, 2, 3, 0, 1, 7, 1)
    assert transmitted == (3, 3, 3, 3, 3, 3, 0, 1, 7, 1)
    assert decode_list(GF11_CODE, worked_received, rho=2) == [transmitted]

    zero_codeword = GF11_CODE.encode((0, 0, 0, 0, 0, 0))
    second_codeword = GF11_CODE.encode((0, 2, 0, 0, 0, 2))
    ambiguous_received = (0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    assert second_codeword == (0, 2, 0, 0, 0, 2, 8, 0, 0, 0)
    ambiguous_list = decode_list(GF11_CODE, ambiguous_received, rho=4)
    assert ambiguous_list == sorted([zero_codeword, second_codeword])
    assert weighted_distance(GF11_CODE, ambiguous_received, zero_codeword) == 2
    assert weighted_distance(GF11_CODE, ambiguous_received, second_codeword) == 4

    small_comparisons, small_distribution = verify_small_code_exhaustively()
    assert small_distribution == {
        0: {0: 240, 1: 16, 2: 0},
        1: {0: 176, 1: 80, 2: 0},
        2: {0: 32, 1: 176, 2: 48},
    }
    gf11_ball_cases = verify_gf11_ball_exhaustively()
    gf11_random_cases = verify_gf11_randomly()

    print("GF(11) check matrix =", GF11_CODE.check_matrix)
    print("unique example list size =", 1)
    print("ambiguous example list size =", len(ambiguous_list))
    print(
        "ambiguous example information vectors =",
        [candidate[: GF11_CODE.k] for candidate in ambiguous_list],
    )
    print("GF(5) list-size distribution =", small_distribution)
    print("GF(5) full-space/radius comparisons =", small_comparisons)
    print("GF(11) exhaustive radius-4 channel patterns =", gf11_ball_cases)
    print("GF(11) random exact-list comparisons =", gf11_random_cases)
    print("complete weighted list decoder verified")


if __name__ == "__main__":
    main()
