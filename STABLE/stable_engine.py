"""
STABLE Rule Engine — Standardized Therapy Audit-Based Logic and Estimates
=========================================================================
Two-layer prescription verification system for cardiovascular medications.

Layer 1 (Steps 1-9): Automated binary pass/fail checks
Layer 2 (Step 10):   Safety information display for clinical review

Priority order: Dose → Renal → Pregnancy → Frequency
(Naseralallah 2023: dose most error-prone; Shehab 2017: OR 6.02 renal)
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Enums & Data Classes
# ──────────────────────────────────────────────

class Verdict(Enum):
    PASS = "Pass"
    FLAG = "Flag"       # under-dose or minor concern → caution
    REJECT = "Reject"   # over max, contraindicated, etc.
    SKIP = "Skip"       # insufficient data to check


class DosePattern(Enum):
    FIXED_SINGLE = "Fixed-Single"
    FIXED_RANGE = "Fixed-Range"
    WEIGHT_SINGLE = "Weight-Single"
    WEIGHT_RANGE = "Weight-Range"
    IU_SINGLE = "IU-Single"
    IU_WEIGHT = "IU-Weight"
    UNIT_SINGLE = "Unit-Single"
    FDC = "FDC"
    NONE = "None"


@dataclass
class StepResult:
    step: int
    name: str
    verdict: Verdict
    detail: str
    data: dict = field(default_factory=dict)


@dataclass
class Prescription:
    """What the prescriber wrote."""
    drug_name: str
    indication: str = ""
    age_group: str = "Adult"
    route: str = ""
    dose_value: float = 0.0          # prescribed dose (mg, IU, or Unit)
    dose_unit: str = "mg"            # "mg" | "IU" | "Unit"
    patient_weight_kg: float = 0.0   # 0 = not provided
    frequency: float = 0.0           # times per period
    timing: str = "Day"              # Day | Hour | Minute | Week | Month | Dose
    patient_crcl: float = -1.0       # -1 = not provided
    is_pregnant: bool = False
    patient_egfr: float = -1.0       # -1 = not provided


@dataclass
class AuditReport:
    prescription: Prescription
    steps: list[StepResult] = field(default_factory=list)
    final_verdict: Verdict = Verdict.PASS
    safety_info: dict = field(default_factory=dict)

    @property
    def has_reject(self) -> bool:
        return any(s.verdict == Verdict.REJECT for s in self.steps)

    @property
    def has_flag(self) -> bool:
        return any(s.verdict == Verdict.FLAG for s in self.steps)


# ──────────────────────────────────────────────
# Column name constants (Sheet3)
# ──────────────────────────────────────────────

COL = {
    "drug": "Generic",
    "drug_class": "Drug Class",
    "brand": "Brand Name",
    "weight": "Weight",
    "age": "Age Group",
    "indication": "Indication",
    "dose_type": "Type of Doses",
    "route": "Route",
    "dose_mg_single": "Direct Dose mg(Single Strength)",
    "dose_mg_min": "Min Direct Dose mg(Multiple Strength)",
    "dose_mg_max": "Max  Direct Dose mg(Multiple Strength)",
    "dose_wt_single": "Dose Per Weight mg(Single Strength)",
    "dose_wt_min": "Dose Per Weight mg(Min Multiple Strength)",
    "dose_wt_max": "Dose Per Weight mg(Max Multiple Strength)",
    "dose_iu": "Direct Dose(IU)",
    "dose_iu_wt": "Dose Per weight(IU)",
    "dose_unit": "Single Dose(Unit)",
    "freq_single": "Single Frequency",
    "freq_min": "Min Frequency ",
    "freq_max": "Max Frequency",
    "timing": "Timing",
    "duration": "Duration",
    "administration": "Administration",
    "instruction": "Instruction",
    "contraindication": "Contradiction",
    "warning": "Warning",
    "pregnancy": "Pregnancy Category",
    "ddi": "DDI",
    "disease_interaction": "Disease Interactions",
    "serious_effect": "Serious Effect",
    "adverse_effect": "Adverse Effect",
}

RENAL_COL = {
    "drug": "Generic",
    "indication": "Indication",
    "crcl_text": "CrCl(mL/min)",
    "crcl_min": "Min CrCl",
    "crcl_max": "Max CrCl",
    "egfr_text": "eGFR",
    "gfr_min": "Min GFR",
    "gfr_max": "Max GFR",
    "dose_type": "Type of Doses",
    "dose_single": "Direct Dose mg (Single Strength)",
    "dose_min": "Min Direct Dose (Multiple Strength)",
    "dose_max": "Max Direct Dose (Multiple Strength)",
    "dose_wt_single": "Dose Per Weight (Single Strength)",
    "freq_single": "Single Frequency",
    "freq_min": "Min Frequency (Multiple)",
    "freq_max": "Max Frequency (Multiple)",
    "freq_unit": "Frequency Unit",
    "instruction": "Instruction",
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def safe_float(val) -> Optional[float]:
    """Try to parse a float from mixed-type cell values."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Strip trailing units: "5 mg" → "5"
    s = re.sub(r'\s*(mg|kg|iu|unit|mcg|g).*$', '', s, flags=re.IGNORECASE)
    # Handle FDC text like "25/100" — return None, handled separately
    if '/' in s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def is_fdc_value(val) -> bool:
    """Check if a cell contains an FDC dose like '25/100'."""
    if val is None:
        return False
    s = str(val).strip()
    return bool(re.match(r'^\d+\.?\d*/\d+\.?\d*$', s))


def parse_fdc_doses(val) -> list[float]:
    """Parse '25/100' → [25.0, 100.0]."""
    s = str(val).strip()
    parts = s.split('/')
    return [float(p) for p in parts if p]


def normalize_str(s: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


def parse_crcl_condition(text) -> tuple[Optional[str], Optional[float]]:
    """
    Parse CrCl text like '<10', '>40', '≥30 ', '5', '10-20' etc.
    Returns (operator, value) or (None, None).
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None, None
    s = str(text).strip()
    # Range like '10-20' — not handled here, use Min/Max CrCl columns
    if '-' in s and not s.startswith('<') and not s.startswith('>'):
        return None, None
    m = re.match(r'([<>≤≥]=?)\s*(\d+\.?\d*)', s)
    if m:
        return m.group(1), float(m.group(2))
    # Plain number
    try:
        return '=', float(s)
    except ValueError:
        return None, None


def crcl_matches(patient_crcl: float, op: str, val: float,
                 min_crcl=None, max_crcl=None) -> bool:
    """Check if patient CrCl falls into a renal dose bracket."""
    # Use min/max if available
    if min_crcl is not None and max_crcl is not None:
        return min_crcl <= patient_crcl <= max_crcl
    if min_crcl is not None:
        return patient_crcl >= min_crcl
    if max_crcl is not None:
        return patient_crcl <= max_crcl
    # Fall back to operator
    ops = {
        '<': patient_crcl < val,
        '<=': patient_crcl <= val,
        '≤': patient_crcl <= val,
        '>': patient_crcl > val,
        '>=': patient_crcl >= val,
        '≥': patient_crcl >= val,
        '=': abs(patient_crcl - val) < 0.5,
    }
    return ops.get(op, False)


# ──────────────────────────────────────────────
# STABLE Engine
# ──────────────────────────────────────────────

class STABLEEngine:
    """
    10-step, 2-layer cardiovascular prescription audit engine.
    Loads the STABLE dataset (Sheet3 + Renal Dose) from an Excel file.
    """

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self._load_data()

    # ── Data loading ──────────────────────────

    def _load_data(self):
        xls = pd.ExcelFile(self.excel_path)

        # Main dosing sheet
        self.df = pd.read_excel(xls, sheet_name="Sheet3")
        self.df[COL["drug"]] = self.df[COL["drug"]].astype(str).str.strip()
        self.df[COL["dose_type"]] = self.df[COL["dose_type"]].astype(str).str.strip()
        self.df[COL["route"]] = self.df[COL["route"]].astype(str).str.strip()
        self.df[COL["age"]] = self.df[COL["age"]].astype(str).str.strip()
        self.df[COL["indication"]] = self.df[COL["indication"]].astype(str).str.strip()
        self.df[COL["timing"]] = self.df[COL["timing"]].astype(str).str.strip()

        # Renal sheet
        if "Renal Dose" in xls.sheet_names:
            self.renal_df = pd.read_excel(xls, sheet_name="Renal Dose")
            self.renal_df[RENAL_COL["drug"]] = (
                self.renal_df[RENAL_COL["drug"]].astype(str).str.strip()
            )
        else:
            self.renal_df = pd.DataFrame()

        # Build drug index for fast lookup
        self.drug_names = sorted(
            self.df[COL["drug"]].dropna().unique().tolist()
        )
        self.drug_names_lower = {normalize_str(d): d for d in self.drug_names}

    # ── Public API ────────────────────────────

    def audit(self, rx: Prescription) -> AuditReport:
        """Run the full 10-step audit on a prescription."""
        report = AuditReport(prescription=rx)

        # Step 1 — Drug lookup
        s1 = self._step1_drug_lookup(rx)
        report.steps.append(s1)
        if s1.verdict == Verdict.REJECT:
            report.final_verdict = Verdict.REJECT
            return report
        matched_drug = s1.data["matched_drug"]

        # Step 2 — Context resolution
        s2 = self._step2_context_resolution(rx, matched_drug)
        report.steps.append(s2)
        matched_rows = s2.data.get("matched_rows", pd.DataFrame())

        # Step 3 — Dose pattern detection
        s3 = self._step3_dose_pattern(rx, matched_rows)
        report.steps.append(s3)

        # Step 4 — Multi-row dose resolution
        s4 = self._step4_dose_resolution(rx, matched_rows)
        report.steps.append(s4)

        # Step 5 — Dose boundary check
        s5 = self._step5_dose_boundary(rx, s4.data)
        report.steps.append(s5)

        # Step 6 — Renal adjustment check
        s6 = self._step6_renal(rx, matched_drug)
        report.steps.append(s6)

        # Step 7 — Pregnancy check
        s7 = self._step7_pregnancy(rx, matched_drug)
        report.steps.append(s7)

        # Step 8 — Frequency & timing verification
        s8 = self._step8_frequency(rx, matched_rows)
        report.steps.append(s8)

        # Step 9 — Final verdict
        s9 = self._step9_verdict(report.steps)
        report.steps.append(s9)
        report.final_verdict = s9.verdict

        # Step 10 — Safety display (Layer 2)
        s10 = self._step10_safety_display(matched_drug)
        report.steps.append(s10)
        report.safety_info = s10.data

        return report

    def get_drug_list(self) -> list[str]:
        return self.drug_names

    def get_indications_for_drug(self, drug: str) -> list[str]:
        mask = self.df[COL["drug"]].str.lower() == drug.strip().lower()
        return sorted(
            self.df.loc[mask, COL["indication"]]
            .dropna().unique().tolist()
        )

    def get_routes_for_drug(self, drug: str) -> list[str]:
        mask = self.df[COL["drug"]].str.lower() == drug.strip().lower()
        return sorted(
            self.df.loc[mask, COL["route"]]
            .dropna().unique().tolist()
        )

    def get_age_groups_for_drug(self, drug: str) -> list[str]:
        mask = self.df[COL["drug"]].str.lower() == drug.strip().lower()
        vals = self.df.loc[mask, COL["age"]].dropna().unique().tolist()
        return sorted(set(v for v in vals if v and v.lower() != 'nan'))

    # ── Step 1: Drug Lookup ───────────────────

    def _step1_drug_lookup(self, rx: Prescription) -> StepResult:
        query = normalize_str(rx.drug_name)
        if not query:
            return StepResult(1, "Drug Lookup", Verdict.REJECT,
                              "No drug name provided.")

        # Exact match
        if query in self.drug_names_lower:
            matched = self.drug_names_lower[query]
            return StepResult(1, "Drug Lookup", Verdict.PASS,
                              f"Exact match: {matched}",
                              {"matched_drug": matched})

        # Partial / fuzzy match
        candidates = [
            d for key, d in self.drug_names_lower.items()
            if query in key or key in query
        ]
        if len(candidates) == 1:
            return StepResult(1, "Drug Lookup", Verdict.PASS,
                              f"Partial match: {candidates[0]}",
                              {"matched_drug": candidates[0]})
        if len(candidates) > 1:
            return StepResult(1, "Drug Lookup", Verdict.FLAG,
                              f"Ambiguous: {candidates}. Using first match.",
                              {"matched_drug": candidates[0],
                               "alternatives": candidates})

        return StepResult(1, "Drug Lookup", Verdict.REJECT,
                          f"Drug '{rx.drug_name}' not found in STABLE dataset.",
                          {"matched_drug": None})

    # ── Step 2: Context Resolution ────────────

    def _step2_context_resolution(self, rx: Prescription,
                                   matched_drug: str) -> StepResult:
        """
        Filter rows by drug + indication + age_group + route.
        Cascading fallback: try all four, then drop route, then drop age,
        then drop indication.
        """
        base = self.df[self.df[COL["drug"]].str.lower() == matched_drug.lower()]

        if base.empty:
            return StepResult(2, "Context Resolution", Verdict.REJECT,
                              f"No rows for {matched_drug}.",
                              {"matched_rows": pd.DataFrame()})

        filters = [
            ("drug+indication+age+route",
             self._filter(base, rx.indication, rx.age_group, rx.route)),
            ("drug+indication+age",
             self._filter(base, rx.indication, rx.age_group, "")),
            ("drug+indication+route",
             self._filter(base, rx.indication, "", rx.route)),
            ("drug+indication",
             self._filter(base, rx.indication, "", "")),
            ("drug+age+route",
             self._filter(base, "", rx.age_group, rx.route)),
            ("drug+age",
             self._filter(base, "", rx.age_group, "")),
            ("drug only",
             base),
        ]

        for label, subset in filters:
            if not subset.empty:
                return StepResult(
                    2, "Context Resolution", Verdict.PASS,
                    f"Matched {len(subset)} rows via {label}.",
                    {"matched_rows": subset, "match_level": label}
                )

        return StepResult(2, "Context Resolution", Verdict.FLAG,
                          "No context match; using all drug rows.",
                          {"matched_rows": base, "match_level": "drug only"})

    def _filter(self, df: pd.DataFrame, indication: str,
                age: str, route: str) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        if indication:
            mask &= df[COL["indication"]].str.lower() == indication.strip().lower()
        if age:
            mask &= df[COL["age"]].str.lower() == age.strip().lower()
        if route:
            mask &= df[COL["route"]].str.lower() == route.strip().lower()
        return df[mask]

    # ── Step 3: Dose Pattern Detection ────────

    def _step3_dose_pattern(self, rx: Prescription,
                             rows: pd.DataFrame) -> StepResult:
        if rows.empty:
            return StepResult(3, "Dose Pattern Detection", Verdict.SKIP,
                              "No rows to detect pattern from.",
                              {"pattern": DosePattern.NONE})

        # Check which columns have data across matched rows
        has = lambda c: rows[c].apply(safe_float).notna().any()
        has_fdc = any(
            rows[c].apply(is_fdc_value).any()
            for c in [COL["dose_mg_single"], COL["dose_mg_min"], COL["dose_mg_max"]]
        )

        if has_fdc:
            pattern = DosePattern.FDC
            detail = "Fixed-dose combination detected. Dose stored as text ratio."
        elif rx.dose_unit == "IU":
            if has(COL["dose_iu_wt"]):
                pattern = DosePattern.IU_WEIGHT
                detail = "IU weight-based dosing."
            elif has(COL["dose_iu"]):
                pattern = DosePattern.IU_SINGLE
                detail = "IU fixed dosing."
            else:
                pattern = DosePattern.NONE
                detail = "IU requested but no IU data found."
        elif rx.dose_unit == "Unit":
            pattern = DosePattern.UNIT_SINGLE
            detail = "Unit-based dosing."
        elif has(COL["dose_wt_single"]) or has(COL["dose_wt_min"]):
            if rx.patient_weight_kg <= 0:
                return StepResult(
                    3, "Dose Pattern Detection", Verdict.FLAG,
                    "Weight-based dosing detected but patient weight not provided.",
                    {"pattern": DosePattern.WEIGHT_SINGLE,
                     "weight_required": True}
                )
            if has(COL["dose_wt_min"]) and has(COL["dose_wt_max"]):
                pattern = DosePattern.WEIGHT_RANGE
                detail = "Weight-based range dosing (mg/kg)."
            else:
                pattern = DosePattern.WEIGHT_SINGLE
                detail = "Weight-based single dosing (mg/kg)."
        elif has(COL["dose_mg_min"]) and has(COL["dose_mg_max"]):
            pattern = DosePattern.FIXED_RANGE
            detail = "Fixed-dose range (mg)."
        elif has(COL["dose_mg_single"]):
            pattern = DosePattern.FIXED_SINGLE
            detail = "Fixed-dose single value (mg)."
        else:
            pattern = DosePattern.NONE
            detail = "No dose data found in matched rows."

        return StepResult(3, "Dose Pattern Detection", Verdict.PASS,
                          detail, {"pattern": pattern})

    # ── Step 4: Multi-Row Dose Resolution ─────

    def _step4_dose_resolution(self, rx: Prescription,
                                rows: pd.DataFrame) -> StepResult:
        """
        Across all matched rows, derive:
        - floor (from Initial / Loading dose types)
        - ceiling (from Maximum dose type)
        - maintenance range
        """
        if rows.empty:
            return StepResult(4, "Multi-Row Dose Resolution", Verdict.SKIP,
                              "No rows available.",
                              {"floor": None, "ceiling": None})

        resolved = {"floor": None, "ceiling": None,
                    "maintenance_min": None, "maintenance_max": None,
                    "floor_type": None, "ceiling_type": None}

        dose_types = rows[COL["dose_type"]].str.strip().str.lower()

        # --- Extract floor (Initial / Loading) ---
        floor_mask = dose_types.isin(["initial", "loading"])
        floor_rows = rows[floor_mask]
        if not floor_rows.empty:
            floor_vals = self._extract_dose_values(floor_rows, rx)
            if floor_vals:
                resolved["floor"] = min(floor_vals)
                resolved["floor_type"] = "Initial/Loading"

        # --- Extract ceiling (Maximum) ---
        max_mask = dose_types == "maximum"
        max_rows = rows[max_mask]
        if not max_rows.empty:
            max_vals = self._extract_dose_values(max_rows, rx)
            if max_vals:
                resolved["ceiling"] = max(max_vals)
                resolved["ceiling_type"] = "Maximum"

        # --- Maintenance range ---
        maint_mask = dose_types.isin(["maintenance", "regular"])
        maint_rows = rows[maint_mask]
        if not maint_rows.empty:
            maint_vals = self._extract_dose_values(maint_rows, rx)
            if maint_vals:
                resolved["maintenance_min"] = min(maint_vals)
                resolved["maintenance_max"] = max(maint_vals)

        # If no ceiling from Maximum rows, use the highest value across all rows
        if resolved["ceiling"] is None:
            all_vals = self._extract_dose_values(rows, rx)
            if all_vals:
                resolved["ceiling"] = max(all_vals)
                resolved["ceiling_type"] = "Derived (max across all rows)"

        # If no floor, use the lowest value across initial/regular rows
        if resolved["floor"] is None:
            starter_mask = dose_types.isin(["initial", "loading", "regular", "maintenance"])
            starter_rows = rows[starter_mask]
            if not starter_rows.empty:
                starter_vals = self._extract_dose_values(starter_rows, rx)
                if starter_vals:
                    resolved["floor"] = min(starter_vals)
                    resolved["floor_type"] = "Derived (min across starting rows)"

        detail_parts = []
        if resolved["floor"] is not None:
            detail_parts.append(f"Floor: {resolved['floor']} mg ({resolved['floor_type']})")
        if resolved["ceiling"] is not None:
            detail_parts.append(f"Ceiling: {resolved['ceiling']} mg ({resolved['ceiling_type']})")
        if resolved["maintenance_min"] is not None:
            detail_parts.append(
                f"Maintenance: {resolved['maintenance_min']}-{resolved['maintenance_max']} mg"
            )

        verdict = Verdict.PASS if detail_parts else Verdict.SKIP
        detail = "; ".join(detail_parts) if detail_parts else "No dose values resolved."
        return StepResult(4, "Multi-Row Dose Resolution", verdict, detail, resolved)

    def _extract_dose_values(self, rows: pd.DataFrame,
                              rx: Prescription) -> list[float]:
        """Pull all numeric dose values from a set of rows, applying weight if needed."""
        vals: list[float] = []

        if rx.dose_unit == "IU":
            for _, r in rows.iterrows():
                v = safe_float(r.get(COL["dose_iu"]))
                if v is not None:
                    vals.append(v)
                v_wt = safe_float(r.get(COL["dose_iu_wt"]))
                if v_wt is not None and rx.patient_weight_kg > 0:
                    vals.append(v_wt * rx.patient_weight_kg)
            return vals

        if rx.dose_unit == "Unit":
            for _, r in rows.iterrows():
                v = safe_float(r.get(COL["dose_unit"]))
                if v is not None:
                    vals.append(v)
            return vals

        # mg-based
        for _, r in rows.iterrows():
            for col in [COL["dose_mg_single"], COL["dose_mg_min"],
                        COL["dose_mg_max"]]:
                v = safe_float(r.get(col))
                if v is not None:
                    vals.append(v)
            # Weight-based columns
            for col in [COL["dose_wt_single"], COL["dose_wt_min"],
                        COL["dose_wt_max"]]:
                v = safe_float(r.get(col))
                if v is not None and rx.patient_weight_kg > 0:
                    vals.append(v * rx.patient_weight_kg)

        return vals

    # ── Step 5: Dose Boundary Check ───────────

    def _step5_dose_boundary(self, rx: Prescription,
                              resolved: dict) -> StepResult:
        floor_val = resolved.get("floor")
        ceiling_val = resolved.get("ceiling")
        prescribed = rx.dose_value

        if prescribed <= 0:
            return StepResult(5, "Dose Boundary Check", Verdict.SKIP,
                              "No prescribed dose to check.")

        if floor_val is None and ceiling_val is None:
            return StepResult(5, "Dose Boundary Check", Verdict.SKIP,
                              "No floor or ceiling resolved from dataset.")

        detail_parts = []
        verdict = Verdict.PASS

        # Over maximum → REJECT
        if ceiling_val is not None and prescribed > ceiling_val:
            verdict = Verdict.REJECT
            detail_parts.append(
                f"OVER MAX: prescribed {prescribed} mg > ceiling {ceiling_val} mg."
            )

        # Under floor → FLAG (not reject; could be titration)
        if floor_val is not None and prescribed < floor_val:
            if verdict != Verdict.REJECT:
                verdict = Verdict.FLAG
            detail_parts.append(
                f"UNDER MIN: prescribed {prescribed} mg < floor {floor_val} mg."
            )

        if not detail_parts:
            range_str = ""
            if floor_val is not None and ceiling_val is not None:
                range_str = f" (range: {floor_val}–{ceiling_val} mg)"
            elif ceiling_val is not None:
                range_str = f" (max: {ceiling_val} mg)"
            detail_parts.append(
                f"Prescribed {prescribed} mg is within bounds{range_str}."
            )

        return StepResult(5, "Dose Boundary Check", verdict,
                          " ".join(detail_parts),
                          {"floor": floor_val, "ceiling": ceiling_val,
                           "prescribed": prescribed})

    # ── Step 6: Renal Adjustment Check ────────

    def _step6_renal(self, rx: Prescription,
                      matched_drug: str) -> StepResult:
        if rx.patient_crcl < 0 and rx.patient_egfr < 0:
            return StepResult(6, "Renal Adjustment Check", Verdict.SKIP,
                              "Patient CrCl/eGFR not provided.")

        if self.renal_df.empty:
            return StepResult(6, "Renal Adjustment Check", Verdict.SKIP,
                              "No renal dose sheet loaded.")

        renal_rows = self.renal_df[
            self.renal_df[RENAL_COL["drug"]].str.lower() == matched_drug.lower()
        ]
        if renal_rows.empty:
            return StepResult(6, "Renal Adjustment Check", Verdict.PASS,
                              f"No renal adjustment required for {matched_drug}.",
                              {"renal_drug": False})

        # Find the bracket that matches patient CrCl
        patient_val = rx.patient_crcl if rx.patient_crcl >= 0 else rx.patient_egfr
        matched_bracket = None

        for _, row in renal_rows.iterrows():
            op, val = parse_crcl_condition(row.get(RENAL_COL["crcl_text"]))
            min_c = safe_float(row.get(RENAL_COL["crcl_min"]))
            max_c = safe_float(row.get(RENAL_COL["crcl_max"]))
            if op is not None and crcl_matches(patient_val, op, val, min_c, max_c):
                matched_bracket = row
                break
            # Also check eGFR columns
            if rx.patient_egfr >= 0:
                gfr_op, gfr_val = parse_crcl_condition(row.get(RENAL_COL["egfr_text"]))
                gfr_min = safe_float(row.get(RENAL_COL["gfr_min"]))
                gfr_max = safe_float(row.get(RENAL_COL["gfr_max"]))
                if gfr_op is not None and crcl_matches(rx.patient_egfr, gfr_op, gfr_val,
                                                        gfr_min, gfr_max):
                    matched_bracket = row
                    break

        if matched_bracket is None:
            return StepResult(
                6, "Renal Adjustment Check", Verdict.FLAG,
                f"Drug has renal data but CrCl {patient_val} didn't match any bracket. "
                f"Available brackets: {renal_rows[RENAL_COL['crcl_text']].tolist()}",
                {"renal_drug": True, "bracket_matched": False}
            )

        # Extract the adjusted dose from the bracket
        adj_dose = safe_float(matched_bracket.get(RENAL_COL["dose_single"]))
        adj_min = safe_float(matched_bracket.get(RENAL_COL["dose_min"]))
        adj_max = safe_float(matched_bracket.get(RENAL_COL["dose_max"]))
        instruction = matched_bracket.get(RENAL_COL["instruction"], "")
        dose_type = str(matched_bracket.get(RENAL_COL["dose_type"], ""))

        renal_ceiling = adj_max or adj_dose
        detail_parts = [
            f"Renal bracket matched (CrCl {matched_bracket.get(RENAL_COL['crcl_text'])}).",
            f"Renal-adjusted dose ({dose_type}): "
        ]
        if adj_dose is not None:
            detail_parts.append(f"{adj_dose} mg")
        if adj_min is not None and adj_max is not None:
            detail_parts.append(f"{adj_min}–{adj_max} mg")
        if instruction and str(instruction).lower() != 'nan':
            detail_parts.append(f" Instruction: {instruction}")

        verdict = Verdict.PASS
        if renal_ceiling is not None and rx.dose_value > 0 and rx.dose_value > renal_ceiling:
            verdict = Verdict.REJECT
            detail_parts.append(
                f" EXCEEDS renal-adjusted max: {rx.dose_value} mg > {renal_ceiling} mg."
            )

        return StepResult(6, "Renal Adjustment Check", verdict,
                          " ".join(detail_parts),
                          {"renal_drug": True, "bracket_matched": True,
                           "renal_ceiling": renal_ceiling,
                           "renal_dose_type": dose_type})

    # ── Step 7: Pregnancy Check ───────────────

    def _step7_pregnancy(self, rx: Prescription,
                          matched_drug: str) -> StepResult:
        if not rx.is_pregnant:
            return StepResult(7, "Pregnancy Check", Verdict.SKIP,
                              "Patient not pregnant or pregnancy status not specified.")

        drug_rows = self.df[
            self.df[COL["drug"]].str.lower() == matched_drug.lower()
        ]
        categories = drug_rows[COL["pregnancy"]].dropna().unique()
        categories = [c.strip() for c in categories if str(c).strip().lower() != 'nan']

        if not categories:
            return StepResult(7, "Pregnancy Check", Verdict.FLAG,
                              f"No pregnancy category data for {matched_drug}.",
                              {"categories": []})

        cat = categories[0]  # Safety data is per-drug, same across rows

        if cat in ("D", "Z"):
            return StepResult(
                7, "Pregnancy Check", Verdict.REJECT,
                f"Category {cat}: {matched_drug} is contraindicated in pregnancy.",
                {"categories": categories, "action": "contraindicated"}
            )
        if cat == "C":
            return StepResult(
                7, "Pregnancy Check", Verdict.FLAG,
                f"Category C: {matched_drug} — use only if benefit outweighs risk.",
                {"categories": categories, "action": "caution"}
            )

        return StepResult(
            7, "Pregnancy Check", Verdict.PASS,
            f"Category {cat}: {matched_drug} is generally considered safe in pregnancy.",
            {"categories": categories, "action": "safe"}
        )

    # ── Step 8: Frequency & Timing ────────────

    def _step8_frequency(self, rx: Prescription,
                          rows: pd.DataFrame) -> StepResult:
        if rx.frequency <= 0:
            return StepResult(8, "Frequency & Timing Verification", Verdict.SKIP,
                              "No frequency provided in prescription.")

        if rows.empty:
            return StepResult(8, "Frequency & Timing Verification", Verdict.SKIP,
                              "No matched rows to verify frequency.")

        # Collect expected frequencies across matched rows
        expected_freqs: list[float] = []
        expected_ranges: list[tuple[float, float]] = []

        for _, r in rows.iterrows():
            sf = safe_float(r.get(COL["freq_single"]))
            fmin = safe_float(r.get(COL["freq_min"]))
            fmax = safe_float(r.get(COL["freq_max"]))

            if sf is not None:
                expected_freqs.append(sf)
            if fmin is not None and fmax is not None:
                expected_ranges.append((fmin, fmax))
            elif fmin is not None:
                expected_freqs.append(fmin)
            elif fmax is not None:
                expected_freqs.append(fmax)

        # Check timing match
        row_timings = rows[COL["timing"]].dropna().unique()
        row_timings = [t.strip() for t in row_timings if str(t).lower() != 'nan']
        timing_ok = True
        if row_timings and rx.timing:
            timing_ok = rx.timing.strip().lower() in [t.lower() for t in row_timings]

        # Check frequency value
        freq_ok = False
        if expected_freqs and rx.frequency in expected_freqs:
            freq_ok = True
        for fmin, fmax in expected_ranges:
            if fmin <= rx.frequency <= fmax:
                freq_ok = True
                break

        # If no data at all, skip
        if not expected_freqs and not expected_ranges:
            return StepResult(8, "Frequency & Timing Verification", Verdict.SKIP,
                              "No frequency data in matched rows.")

        verdict = Verdict.PASS
        detail_parts = []

        if not freq_ok:
            verdict = Verdict.FLAG
            all_expected = expected_freqs + [
                f"{lo}-{hi}" for lo, hi in expected_ranges
            ]
            detail_parts.append(
                f"Frequency {rx.frequency}/{rx.timing} not in expected "
                f"values: {all_expected}."
            )
        else:
            detail_parts.append(f"Frequency {rx.frequency}/{rx.timing} matches dataset.")

        if not timing_ok:
            verdict = Verdict.FLAG
            detail_parts.append(
                f"Timing '{rx.timing}' not in expected: {row_timings}."
            )

        return StepResult(8, "Frequency & Timing Verification", verdict,
                          " ".join(detail_parts),
                          {"expected_freqs": expected_freqs,
                           "expected_ranges": expected_ranges})

    # ── Step 9: Final Verdict ─────────────────

    def _step9_verdict(self, steps: list[StepResult]) -> StepResult:
        """Any REJECT → Reject; any FLAG → Caution; all PASS/SKIP → Accept."""
        rejects = [s for s in steps if s.verdict == Verdict.REJECT]
        flags = [s for s in steps if s.verdict == Verdict.FLAG]

        if rejects:
            reasons = "; ".join(f"Step {s.step} ({s.name}): {s.detail}" for s in rejects)
            return StepResult(9, "Final Verdict", Verdict.REJECT,
                              f"REJECT — {len(rejects)} critical issue(s). {reasons}")
        if flags:
            reasons = "; ".join(f"Step {s.step} ({s.name}): {s.detail}" for s in flags)
            return StepResult(9, "Final Verdict", Verdict.FLAG,
                              f"CAUTION — {len(flags)} concern(s). {reasons}")

        return StepResult(9, "Final Verdict", Verdict.PASS,
                          "ACCEPT — All checks passed.")

    # ── Step 10: Safety Display (Layer 2) ─────

    def _step10_safety_display(self, matched_drug: str) -> StepResult:
        """Retrieve all safety fields for clinical review."""
        drug_rows = self.df[
            self.df[COL["drug"]].str.lower() == matched_drug.lower()
        ]
        if drug_rows.empty:
            return StepResult(10, "Safety Information Display", Verdict.SKIP,
                              "No data for safety display.", {})

        def _first_nonempty(col_key: str) -> str:
            vals = drug_rows[COL[col_key]].dropna().unique()
            vals = [str(v).strip() for v in vals if str(v).strip().lower() != 'nan']
            return vals[0] if vals else ""

        safety = {
            "drug": matched_drug,
            "drug_class": _first_nonempty("drug_class"),
            "contraindications": _first_nonempty("contraindication"),
            "warnings": _first_nonempty("warning"),
            "ddi": _first_nonempty("ddi"),
            "disease_interactions": _first_nonempty("disease_interaction"),
            "serious_effects": _first_nonempty("serious_effect"),
            "adverse_effects": _first_nonempty("adverse_effect"),
            "pregnancy_category": _first_nonempty("pregnancy"),
        }

        return StepResult(10, "Safety Information Display", Verdict.PASS,
                          "Safety information retrieved for clinical review.",
                          safety)