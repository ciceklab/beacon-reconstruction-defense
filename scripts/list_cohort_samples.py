from __future__ import annotations

import argparse
import gzip
import itertools
import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PANEL = os.path.join(ROOT, "data/chr22/integrated_call_samples_v3.20130502.ALL.panel")

POPULATION_NAMES = {
    "ACB": "African Caribbean in Barbados",
    "ASW": "African Ancestry in Southwest US",
    "BEB": "Bengali in Bangladesh",
    "CDX": "Chinese Dai in Xishuangbanna, China",
    "CEU": "Utah residents with Northern and Western European ancestry",
    "CHB": "Han Chinese in Beijing, China",
    "CHS": "Southern Han Chinese",
    "CLM": "Colombian in Medellin, Colombia",
    "ESN": "Esan in Nigeria",
    "FIN": "Finnish in Finland",
    "GBR": "British in England and Scotland",
    "GIH": "Gujarati Indian in Houston, Texas",
    "GWD": "Gambian in Western Division, The Gambia",
    "IBS": "Iberian population in Spain",
    "ITU": "Indian Telugu in the UK",
    "JPT": "Japanese in Tokyo, Japan",
    "KHV": "Kinh in Ho Chi Minh City, Vietnam",
    "LWK": "Luhya in Webuye, Kenya",
    "MSL": "Mende in Sierra Leone",
    "MXL": "Mexican Ancestry in Los Angeles, California",
    "PEL": "Peruvian in Lima, Peru",
    "PJL": "Punjabi in Lahore, Pakistan",
    "PUR": "Puerto Rican in Puerto Rico",
    "STU": "Sri Lankan Tamil in the UK",
    "TSI": "Toscani in Italy",
    "YRI": "Yoruba in Ibadan, Nigeria",
}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    pop: str
    super_pop: str
    gender: str


def load_panel(path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(path) as handle:
        header = handle.readline()
        if not header.lower().startswith("sample"):
            raise ValueError(f"{path} does not look like an integrated_call_samples panel")
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            gender = parts[3] if len(parts) > 3 else ""
            samples.append(Sample(parts[0], parts[1], parts[2], gender))
            if not samples[-1].super_pop:
                raise ValueError(f"{path}:{line_number} is missing a super-population")
    if not samples:
        raise ValueError(f"No samples in {path}")
    return samples


def candidate_cohorts(
    samples: list[Sample],
    n_rows: int,
    super_pop: str | None,
) -> list[list[Sample]]:
    
    by_super: dict[str, list[Sample]] = {}
    for sample in samples:
        by_super.setdefault(sample.super_pop, []).append(sample)
    if super_pop is not None:
        if super_pop not in by_super:
            raise ValueError(f"No {super_pop} samples in the panel")
        groups = {super_pop: by_super[super_pop]}
    else:
        groups = by_super

    candidates: list[list[Sample]] = []
    for label, group in groups.items():
        if len(group) < n_rows:
            continue
        pop_order: list[str] = []
        for sample in group:
            if sample.pop not in pop_order:
                pop_order.append(sample.pop)
        if len(group) == n_rows:
            candidates.append(group)
            continue
        for width in range(1, len(pop_order)):
            for combo in itertools.combinations(pop_order, width):
                chosen = set(combo)
                subset = [sample for sample in group if sample.pop in chosen]
                if len(subset) == n_rows:
                    candidates.append(subset)
    return candidates


def minor_allele_column(bits: np.ndarray, *, drop_flipped_doubleton: bool = False) -> np.ndarray | None:
    
    observed = np.where(bits < 0, 0, bits).astype(np.int8, copy=False)
    count = int(observed.sum())
    n_people = int(bits.shape[0])
    flipped = False
    if count * 2 > n_people:
        observed = np.where(bits < 0, 0, 1 - observed).astype(np.int8, copy=False)
        count = int(observed.sum())
        flipped = True
    if count < 2 or count * 2 > n_people:
        return None
    if drop_flipped_doubleton and flipped and count == 2:
        return None
    return observed


def _vcf_sample_index(header_line: str) -> dict[str, int]:
    fields = header_line.rstrip("\n").split("\t")
    if len(fields) < 10 or fields[0] != "#CHROM":
        raise ValueError("VCF is missing a #CHROM header")
    return {sample_id: index for index, sample_id in enumerate(fields[9:])}


def _genotype_start(line: str, n_samples: int) -> int:
    
    if not line.endswith("\n"):
        raise ValueError("Expected newline-terminated VCF records")
    start = len(line) - 1 - (n_samples * 4 - 1)
    if start < 0 or line[start + 1] not in "|/":
        raise ValueError("VCF genotype fields are not fixed-width phased GT calls")
    return start


def _carrier_bits(line: str, start: int, columns: np.ndarray) -> np.ndarray:
    bits = np.empty(columns.shape[0], dtype=np.int8)
    for row, column in enumerate(columns):
        offset = start + int(column) * 4
        left = line[offset]
        right = line[offset + 2]
        if left == "." or right == ".":
            bits[row] = -1
        elif left == "0" and right == "0":
            bits[row] = 0
        else:
            bits[row] = 1
    return bits


def kept_columns(
    vcf_path: str,
    sample_ids: list[str],
    n_columns: int,
    *,
    columns_to_check: set[int] | None = None,
) -> tuple[np.ndarray, int]:
    
    if n_columns < 1 and not columns_to_check:
        raise ValueError("n_columns must be positive")
    wanted = columns_to_check if columns_to_check is not None else set(range(n_columns))
    stored: dict[int, np.ndarray] = {}
    n_kept = 0
    with gzip.open(vcf_path, "rt") as handle:
        sample_index: dict[str, int] | None = None
        columns: np.ndarray | None = None
        n_samples = 0
        for line in handle:
            if line.startswith("#CHROM"):
                sample_index = _vcf_sample_index(line)
                n_samples = len(sample_index)
                missing = [sample_id for sample_id in sample_ids if sample_id not in sample_index]
                if missing:
                    raise ValueError(
                        f"{len(missing)} panel samples are absent from the VCF, "
                        f"including {missing[0]}"
                    )
                columns = np.array([sample_index[sample_id] for sample_id in sample_ids], dtype=np.int32)
                continue
            if line.startswith("#") or columns is None:
                continue
            start = _genotype_start(line, n_samples)
            column = minor_allele_column(_carrier_bits(line, start, columns))
            if column is None:
                continue
            if n_kept in wanted:
                stored[n_kept] = column
            n_kept += 1
            if len(stored) == len(wanted):
                break
    if n_kept == 0 or not stored:
        raise RuntimeError(f"No cohort variants were kept from {vcf_path}")
    ordered = [stored[index] for index in sorted(stored)]
    return np.column_stack(ordered), n_kept


def _carrier_bits_vectorized(line: str, start: int, columns: np.ndarray) -> np.ndarray:
    raw = np.frombuffer((line[start:-1] + "\t").encode("ascii"), dtype=np.uint8)
    alleles = raw.reshape(-1, 4)
    left = alleles[columns, 0]
    right = alleles[columns, 2]
    missing = (left == ord(".")) | (right == ord("."))
    reference = (left == ord("0")) & (right == ord("0"))
    bits = np.ones(columns.shape[0], dtype=np.int8)
    bits[reference | missing] = 0
    bits[missing] = -1
    return bits


def count_kept_variants(vcf_path: str, sample_ids: list[str], binary: np.ndarray) -> dict[str, object]:
    
    checks = {0, binary.shape[1] // 2, binary.shape[1] - 1}
    rules = (
        ("keep sites whose rarer state has two or more people", False),
        ("drop flipped doubletons (AMR matrices)", True),
    )
    kept = {drop: 0 for _, drop in rules}
    matched: dict[bool, dict[int, bool]] = {drop: {} for _, drop in rules}
    with gzip.open(vcf_path, "rt") as handle:
        columns: np.ndarray | None = None
        n_samples = 0
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#CHROM"):
                sample_index = _vcf_sample_index(line)
                n_samples = len(sample_index)
                columns = np.array([sample_index[sample_id] for sample_id in sample_ids], dtype=np.int32)
                continue
            if line.startswith("#") or columns is None:
                continue
            start = _genotype_start(line, n_samples)
            bits = _carrier_bits_vectorized(line, start, columns)
            for _label, drop in rules:
                column = minor_allele_column(bits, drop_flipped_doubleton=drop)
                if column is None:
                    continue
                seen = kept[drop]
                if seen in checks and seen < binary.shape[1]:
                    matched[drop][seen] = bool(np.array_equal(column, binary[:, seen]))
                kept[drop] = seen + 1
                if not drop and kept[drop] % 100000 == 0:
                    print(f"  kept {kept[drop]:,} variants (VCF line {line_number:,})", flush=True)
    chosen = None
    for label, drop in rules:
        columns_ok = kept[drop] == binary.shape[1] and all(matched[drop].values())
        if columns_ok and len(matched[drop]) == len(checks):
            chosen = label
            break
    return {
        "kept": kept[False],
        "kept_without_flipped_doubletons": kept[True],
        "matrix_columns": int(binary.shape[1]),
        "count_matches": chosen is not None,
        "rule": chosen,
        "column_matches": matched[False] if chosen != rules[1][0] else matched[True],
    }


def matrix_matches(binary: np.ndarray, rebuilt: np.ndarray) -> bool:
    width = min(binary.shape[1], rebuilt.shape[1])
    if width == 0:
        return False
    return bool(np.array_equal(binary[:, :width], rebuilt[:, :width]))


def write_sample_table(path: str, cohort: list[Sample], binary_path: str, vcf_path: str, panel_path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write(f"# binary\t{binary_path}\n")
        handle.write(f"# vcf\t{vcf_path}\n")
        handle.write(f"# panel\t{panel_path}\n")
        handle.write("# row_index is the row of the binary matrix\n")
        handle.write("row_index\tsample\tpop\tsuper_pop\tgender\n")
        for index, sample in enumerate(cohort):
            handle.write(
                f"{index}\t{sample.sample_id}\t{sample.pop}\t{sample.super_pop}\t{sample.gender}\n"
            )


def population_counts(cohort: list[Sample]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for sample in cohort:
        if sample.pop not in counts:
            order.append(sample.pop)
            counts[sample.pop] = 0
        counts[sample.pop] += 1
    return [(pop, counts[pop]) for pop in order]


def recover_cohort(
    binary: np.ndarray,
    samples: list[Sample],
    vcf_path: str,
    super_pop: str | None,
    check_columns: int,
) -> list[Sample]:
    n_rows = int(binary.shape[0])
    candidates = candidate_cohorts(samples, n_rows, super_pop)
    if not candidates:
        raise SystemExit(
            f"No panel cohort has {n_rows} samples"
            + (f" in {super_pop}" if super_pop else "")
            + ". The matrix rows are not a whole population or a combination of populations."
        )
    width = min(check_columns, int(binary.shape[1]))
    matches: list[list[Sample]] = []
    for cohort in candidates:
        label = "+".join(pop for pop, _ in population_counts(cohort))
        print(f"Trying {cohort[0].super_pop} {label} ({len(cohort)} people)...", flush=True)
        rebuilt, _n_kept = kept_columns(vcf_path, [sample.sample_id for sample in cohort], width)
        if matrix_matches(binary, rebuilt):
            matches.append(cohort)
            print(f"  matched the first {rebuilt.shape[1]} variant columns")
        else:
            print("  genotypes differ")
    if len(matches) != 1:
        labels = [",".join(pop for pop, _ in population_counts(cohort)) for cohort in matches]
        raise SystemExit(f"Expected one matching cohort, found {len(matches)}: {labels}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="cleaned_binary.npy, people by variant")
    parser.add_argument("--vcf", required=True, help="1000 Genomes Phase 3 genotypes VCF.gz for one chromosome")
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="integrated_call_samples panel")
    parser.add_argument("--super-pop", default=None, choices=["EUR", "AFR", "AMR", "EAS", "SAS"])
    parser.add_argument("--check-columns", type=int, default=32, help="Leading variant columns used to identify the cohort")
    parser.add_argument("--count-variants", action="store_true", help="Scan the whole VCF and check the kept-site count")
    parser.add_argument("--out", default=None, help="TSV path for row_index, sample, pop, super_pop, gender")
    args = parser.parse_args()

    binary = np.load(args.binary, mmap_mode="r")
    if binary.ndim != 2:
        raise SystemExit(f"Expected a 2-D matrix, got shape {binary.shape}")
    samples = load_panel(args.panel)
    cohort = recover_cohort(binary, samples, args.vcf, args.super_pop, args.check_columns)

    print()
    print(f"{cohort[0].super_pop}: {len(cohort)} samples in panel-file order")
    for pop, count in population_counts(cohort):
        print(f"  {pop}\t{count}\t{POPULATION_NAMES.get(pop, '')}")
    print(f"First sample: {cohort[0].sample_id} ({cohort[0].pop})")
    print(f"Last sample:  {cohort[-1].sample_id} ({cohort[-1].pop})")

    if args.count_variants:
        print("\nCounting every kept variant...")
        report = count_kept_variants(args.vcf, [sample.sample_id for sample in cohort], np.asarray(binary))
        print(
            f"Rarer-state filter kept {report['kept']:,} variants; "
            f"without flipped doubletons {report['kept_without_flipped_doubletons']:,}; "
            f"matrix has {report['matrix_columns']:,} columns"
        )
        print(f"Reproduced by: {report['rule']}")
        print(f"Column matches: {report['column_matches']}")
        if not report["count_matches"]:
            raise SystemExit("Variant filter did not reproduce the matrix")

    if args.out:
        write_sample_table(args.out, cohort, args.binary, args.vcf, args.panel)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
