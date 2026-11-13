"""Exact small-population audit of the finite-population betting construction.

This is a finite-case implementation audit, not a replacement for the theorem.
The reference calculation uses only stdlib Fractions and independently computes
every multiplier and mixture. Core routines are used only for cross-checks.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path


FRACTIONS = tuple(Fraction(x) for x in ("0.05", "0.10", "0.20", "0.40", "0.60", "0.80", "0.95"))
HALF = Fraction(1, 2)
GUARD = Fraction(1, 10**12)


def reference_factor(x: Fraction, q: Fraction, c: Fraction) -> Fraction:
    """Independent exact-rational evaluation of the frozen rule."""
    if q <= GUARD:
        return Fraction(1)
    return Fraction(1) - c + c * x / q


def audit_population(name: str, values: tuple[Fraction, ...], alpha: Fraction,
                     core=None) -> dict:
    n = len(values)
    if not all(0 <= x <= 1 for x in values):
        raise ValueError("invalid bounded population")
    mean = sum(values, Fraction()) / n
    if mean > HALF:
        raise ValueError("this audit expects a true one-sided null")
    permutations = math.factorial(n)
    expected_sums = [Fraction() for _ in range(n)]
    ever_crossed = 0
    ever_crossed_diagnostic = 0
    max_mixture = Fraction(1)
    maximum_factor_difference = 0.0
    maximum_mixture_relative_difference = 0.0
    threshold = 1 / alpha
    diagnostic_alpha = Fraction(1, 4)
    diagnostic_threshold = 1 / diagnostic_alpha

    # Permute labelled indices, not deduplicated values: every possible random
    # permutation of the actual finite population has exactly equal weight.
    for permutation in itertools.permutations(range(n)):
        products = [Fraction(1) for _ in FRACTIONS]
        core_log_products = [0.0 for _ in FRACTIONS]
        prefix_sum = Fraction()
        crossed = False
        crossed_diagnostic = False
        for j, index in enumerate(permutation):
            x = values[index]
            q = (n * HALF - prefix_sum) / (n - j)
            factors = [reference_factor(x, q, c) for c in FRACTIONS]
            if any(factor < 0 for factor in factors):
                raise AssertionError("negative reference factor")
            products = [product * factor for product, factor in zip(products, factors)]
            mixture = sum(products, Fraction()) / len(products)
            expected_sums[j] += mixture
            max_mixture = max(max_mixture, mixture)
            crossed = crossed or mixture >= threshold
            crossed_diagnostic = crossed_diagnostic or mixture >= diagnostic_threshold

            if core is not None:
                observed = core.betting_factors(float(x), float(q)).tolist()
                for k, (exact, actual) in enumerate(zip(factors, observed)):
                    difference = abs(float(exact) - actual)
                    maximum_factor_difference = max(maximum_factor_difference, difference)
                    if not math.isfinite(actual) or actual <= 0:
                        raise AssertionError("invalid core multiplier")
                    if difference > 1e-12 * max(1.0, abs(float(exact))):
                        raise AssertionError("core multiplier differs from rational reference")
                    core_log_products[k] += math.log(actual)
                peak = max(core_log_products)
                actual_mixture = math.exp(peak) * math.fsum(
                    math.exp(value - peak) for value in core_log_products
                ) / len(core_log_products)
                relative_difference = abs(float(mixture) - actual_mixture) / max(1.0, abs(float(mixture)))
                maximum_mixture_relative_difference = max(maximum_mixture_relative_difference,
                                                           relative_difference)
                if relative_difference > 1e-11:
                    raise AssertionError("core factors accumulate to the wrong mixture")
            prefix_sum += x
        ever_crossed += int(crossed)
        ever_crossed_diagnostic += int(crossed_diagnostic)

    expectations = [total / permutations for total in expected_sums]
    if any(value > 1 for value in expectations):
        raise AssertionError("exact mean capital exceeds one under null")
    if mean == HALF and any(value != 1 for value in expectations):
        raise AssertionError("exact-null capital should have mean one in this finite case")
    crossing_probability = Fraction(ever_crossed, permutations)
    diagnostic_probability = Fraction(ever_crossed_diagnostic, permutations)
    if crossing_probability > alpha:
        raise AssertionError("anytime crossing probability exceeds alpha")
    if diagnostic_probability > diagnostic_alpha:
        raise AssertionError("diagnostic anytime crossing probability exceeds alpha")
    return {
        "case": name,
        "population": [str(x) for x in values],
        "population_mean": str(mean),
        "labelled_permutations": permutations,
        "prefix_expected_capital_exact": [str(value) for value in expectations],
        "prefix_expected_capital": [float(value) for value in expectations],
        "one_direction_alpha": float(alpha),
        "crossing_threshold": float(threshold),
        "anytime_crossings": ever_crossed,
        "anytime_crossing_probability": float(crossing_probability),
        "diagnostic_alpha": float(diagnostic_alpha),
        "diagnostic_crossings": ever_crossed_diagnostic,
        "diagnostic_crossing_probability": float(diagnostic_probability),
        "maximum_mixture_seen": float(max_mixture),
        "maximum_core_factor_absolute_difference": maximum_factor_difference,
        "maximum_core_mixture_relative_difference": maximum_mixture_relative_difference,
        "status": "passed",
    }


def audit_boundaries(core) -> dict:
    # Neutral skips depend only on the prefix-derived q, not the next score.
    neutral_cases = (0.0, -1.0, 1e-13, 1e-12)
    for q in neutral_cases:
        for x in (0.0, 0.5, 1.0):
            factors = core.betting_factors(x, q).tolist()
            if factors != [1.0] * len(FRACTIONS):
                raise AssertionError("nonneutral arithmetic guard")
    for q in (math.nextafter(1e-12, math.inf), 0.25, 0.5, 1.0, 1.5, 2.0):
        for x in (0.0, 0.5, 1.0):
            factors = core.betting_factors(x, q).tolist()
            for c, actual in zip(FRACTIONS, factors):
                expected = 1.0 - float(c) + float(c) * x / q
                if not math.isfinite(actual) or actual <= 0:
                    raise AssertionError("invalid boundary factor")
                if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError("boundary factor mismatch")
    for k in (2, 3, 6, 20):
        state = core.RaceState(6, k, alpha=0.05)
        expected = math.log(2 * (k * (k - 1) // 2) / 0.05)
        if not math.isclose(state.log_threshold, expected, rel_tol=0, abs_tol=1e-14):
            raise AssertionError("multipair/two-direction threshold mismatch")
    return {"status": "passed", "neutral_q_values": list(neutral_cases),
            "threshold_system_counts": [2, 3, 6, 20]}


def audit_controller_prefixes(core) -> dict:
    """Cross-check controller sum reconstruction and mixture aggregation."""
    values = tuple(Fraction(x) for x in ("0", "0", "0.5", "0.5", "1", "1"))
    checked = 0
    for permutation in itertools.islice(itertools.permutations(range(6)), 30):
        state = core.RaceState(6, 2)
        products = [[Fraction(1)] * len(FRACTIONS) for _ in range(2)]
        sums = [Fraction(), Fraction()]
        for n, index in enumerate(permutation):
            for side, x in enumerate((values[index], 1 - values[index])):
                q = (Fraction(3) - sums[side]) / (6 - n)
                products[side] = [product * reference_factor(x, q, c)
                                  for product, c in zip(products[side], FRACTIONS)]
                sums[side] += x
            # This choice makes the normalized paired X exactly values[index].
            state.observe_next(0, float(values[index]))
            state.observe_next(1, float(1 - values[index]))
            for side in range(2):
                actual_log_e = float(core.log_mixture(state.log_products[0, side]))
                expected_log_e = math.log(float(sum(products[side]) / len(FRACTIONS)))
                if not math.isclose(actual_log_e, expected_log_e, rel_tol=0, abs_tol=1e-12):
                    raise AssertionError("controller prefix or mixture mismatch")
                checked += 1
    return {"status": "passed", "permutations": 30,
            "directional_prefix_checks": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("--no-core-comparison", action="store_true",
                        help="run exact reference calculation without importing prototype")
    args = parser.parse_args()
    core = None
    if not args.no_core_comparison:
        import pairwise_racing as core
    alpha = Fraction(1, 20)
    exact_null = tuple(Fraction(x) for x in ("0", "0", "0.5", "0.5", "1", "1"))
    composite_null = tuple(Fraction(x) for x in ("0", "0", "0", "0.5", "0.5", "1"))
    populations = (
        ("exact_null_positive_direction", exact_null),
        ("exact_null_negative_direction", tuple(1 - value for value in exact_null)),
        ("strict_composite_null_positive_direction", composite_null),
    )
    cases = [audit_population(name, values, alpha, core) for name, values in populations]
    result = {
        "audit_type": "finite_case_exact_permutation_implementation_audit",
        "scope": "not a proof replacement, general power claim, or new MT holdout validation",
        "reference_arithmetic": "stdlib fractions.Fraction; independently derived factors and mixtures",
        "core_compared": core is not None,
        "cases": cases,
        "total_labelled_permutations": sum(case["labelled_permutations"] for case in cases),
        "boundary_checks": audit_boundaries(core) if core is not None else "not_requested",
        "controller_prefix_checks": audit_controller_prefixes(core) if core is not None else "not_requested",
        "status": "passed",
    }
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
