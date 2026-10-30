from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseLocus:
    chromosome: int
    position: int
    rsid: str
    gene: str
    core: bool = True  


@dataclass(frozen=True)
class DiseaseDefinition:
    disease_id: str
    display_name: str
    loci: tuple[DiseaseLocus, ...]
    filler_tiers: tuple[float, float, float] = (0.35, 0.35, 0.30)
    rare_maf_max: float = 0.05
    low_freq_maf_max: float = 0.20



DISEASE_CATALOG: tuple[DiseaseDefinition, ...] = (
    DiseaseDefinition(
        "celiac_disease",
        "Celiac disease",
        (
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(6, 32_667_175, "rs4988889", "HLA-DQB1"),
            DiseaseLocus(6, 32_688_530, "rs7775228", "HLA-DQB1"),
            DiseaseLocus(6, 32_637_419, "rs2858331", "HLA-DQA1"),
            DiseaseLocus(6, 32_605_858, "rs9272723", "HLA-DQB1"),
        ),
    ),
    DiseaseDefinition(
        "type1_diabetes",
        "Type 1 diabetes",
        (
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(6, 32_605_858, "rs9272723", "HLA-DQB1"),
            DiseaseLocus(6, 32_439_700, "rs660895", "HLA-DRB1"),
            DiseaseLocus(6, 32_636_615, "rs2647044", "HLA-DQB1"),
            DiseaseLocus(6, 32_637_419, "rs2858331", "HLA-DQA1"),
        ),
    ),
    DiseaseDefinition(
        "rheumatoid_arthritis",
        "Rheumatoid arthritis",
        (
            DiseaseLocus(6, 32_439_700, "rs660895", "HLA-DRB1"),
            DiseaseLocus(6, 32_605_858, "rs9272723", "HLA-DQB1"),
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(6, 32_752_730, "rs9272219", "HLA-DQA1"),
            DiseaseLocus(1, 114_377_568, "rs2476601", "PTPN22", core=False),
        ),
    ),
    DiseaseDefinition(
        "multiple_sclerosis",
        "Multiple sclerosis",
        (
            DiseaseLocus(6, 32_336_615, "rs3135388", "HLA-DRB1"),
            DiseaseLocus(6, 32_439_700, "rs660895", "HLA-DRB1"),
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(6, 32_605_858, "rs9272723", "HLA-DQB1"),
        ),
    ),
    DiseaseDefinition(
        "crohn_disease",
        "Crohn's disease",
        (
            DiseaseLocus(6, 32_605_858, "rs9272723", "HLA-DQB1"),
            DiseaseLocus(6, 32_637_419, "rs424232", "HLA-DQA1"),
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(1, 67_352_868, "rs2476601", "PADI4", core=False),
        ),
    ),
    DiseaseDefinition(
        "schizophrenia",
        "Schizophrenia",
        (
            DiseaseLocus(22, 19_951_271, "rs4680", "COMT"),
            DiseaseLocus(22, 18_916_703, "rs4815670", "PRODH"),
            DiseaseLocus(22, 19_755_000, "rs5746647", "TBX1"),
            DiseaseLocus(22, 19_680_000, "rs6006493", "DGCR8"),
            DiseaseLocus(6, 32_313_790, "rs3131379", "MSH5", core=False),
        ),
    ),
    DiseaseDefinition(
        "coronary_artery_disease",
        "Coronary artery disease",
        (
            DiseaseLocus(1, 55_524_308, "rs11591147", "PCSK9"),
            DiseaseLocus(1, 55_503_335, "rs112605746", "PCSK9"),
            DiseaseLocus(1, 109_274_968, "rs4977574", "CDKN2B-AS1"),
            DiseaseLocus(1, 21_207_800, "rs693", "APOB"),
            DiseaseLocus(9, 22_099_757, "rs1333049", "CDKN2B-AS1", core=False),
        ),
    ),
    DiseaseDefinition(
        "alzheimers_disease",
        "Alzheimer's disease",
        (
            DiseaseLocus(1, 207_685_786, "rs6656401", "CR1"),
            DiseaseLocus(1, 207_796_065, "rs3818361", "CR1"),
            DiseaseLocus(1, 226_893_093, "rs744373", "BIN1", core=False),
            DiseaseLocus(19, 45_409_186, "rs429358", "APOE", core=False),
        ),
    ),
    DiseaseDefinition(
        "systemic_lupus",
        "Systemic lupus erythematosus",
        (
            DiseaseLocus(6, 32_313_790, "rs3131379", "MSH5"),
            DiseaseLocus(6, 32_657_781, "rs2187668", "HLA-DQA1"),
            DiseaseLocus(6, 32_752_730, "rs9272219", "HLA-DQA1"),
            DiseaseLocus(1, 114_377_568, "rs2476601", "PTPN22", core=False),
        ),
    ),
    DiseaseDefinition(
        "breast_cancer",
        "Breast cancer",
        (
            DiseaseLocus(1, 121_479_514, "rs11249433", "FGFR2", core=False),
            DiseaseLocus(1, 240_369_495, "rs16857609", "CASC16", core=False),
            DiseaseLocus(10, 12_344_770, "rs2981579", "FGFR2", core=False),
            DiseaseLocus(16, 52_581_667, "rs3803662", "TOX3", core=False),
        ),
    ),
)

DISEASE_BY_ID: dict[str, DiseaseDefinition] = {d.disease_id: d for d in DISEASE_CATALOG}
DISEASE_IDS: tuple[str, ...] = tuple(d.disease_id for d in DISEASE_CATALOG)
