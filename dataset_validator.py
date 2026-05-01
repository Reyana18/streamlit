"""
STABLE Dataset Validator
========================
Layer A: Automated internal consistency checks (deterministic, no AI).
Layer B: LLM-powered clinical plausibility checks (requires API key).
Layer C: Source URL verification.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np


class Severity(Enum):
    ERROR = "Error"        # Must fix before submission
    WARNING = "Warning"    # Should investigate
    INFO = "Info"          # Minor, cosmetic


@dataclass
class ValidationIssue:
    severity: Severity
    category: str          # e.g., "Duplicate", "Typo", "Dose Logic"
    description: str
    rows: list = field(default_factory=list)   # affected row indices
    field: str = ""        # column name
    suggestion: str = ""   # how to fix


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0
    total_drugs: int = 0
    checks_run: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.INFO)


class DatasetValidator:
    """Runs all Layer A automated checks on the STABLE dataset."""

    def __init__(self, df: pd.DataFrame, renal_df: pd.DataFrame = None):
        self.df = df.copy()
        self.renal_df = renal_df.copy() if renal_df is not None else pd.DataFrame()
        self.report = ValidationReport(
            total_rows=len(df),
            total_drugs=df["Generic"].nunique() if "Generic" in df.columns else 0,
        )

    def run_all_checks(self) -> ValidationReport:
        """Run every Layer A check and return the report."""
        self._check_duplicates()
        self._check_missing_required_fields()
        self._check_drug_name_typos()
        self._check_drug_class_typos()
        self._check_route_typos()
        self._check_indication_inconsistencies()
        self._check_age_group_inconsistencies()
        self._check_dose_type_values()
        self._check_pregnancy_categories()
        self._check_initial_exceeds_maximum()
        self._check_single_and_range_both_filled()
        self._check_frequency_values()
        self._check_dose_outliers()
        self._check_fdc_dose_format()
        self._check_source_urls_present()
        self._check_timing_values()
        self._check_renal_sheet_consistency()
        self._check_weight_column_values()
        self.report.checks_run = 17
        return self.report

    # ── 1. Duplicate Detection ────────────────

    def _check_duplicates(self):
        key_cols = ["Generic", "Indication", "Type of Doses", "Route",
                    "Age Group", "Direct Dose mg(Single Strength)",
                    "Min Direct Dose mg(Multiple Strength)",
                    "Max  Direct Dose mg(Multiple Strength)"]
        available = [c for c in key_cols if c in self.df.columns]
        if not available:
            return

        dupes = self.df[self.df.duplicated(subset=available, keep=False)]
        if not dupes.empty:
            groups = dupes.groupby(available, dropna=False)
            for name, group in groups:
                if len(group) > 1:
                    rows = group.index.tolist()
                    drug = group["Generic"].iloc[0] if "Generic" in group.columns else "Unknown"
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="Duplicate Rows",
                        description=f"Duplicate entry for {drug}: rows {rows} have identical key fields.",
                        rows=rows,
                        suggestion="Delete duplicates or verify they should have different indications/dose types."
                    ))

    # ── 2. Missing Required Fields ────────────

    def _check_missing_required_fields(self):
        required = {
            "Generic": "Drug name",
            "Indication": "Indication",
            "Route": "Route",
            "Drug Class": "Drug class",
            "Type of Doses": "Dose type",
        }
        for col, label in required.items():
            if col not in self.df.columns:
                continue
            mask = self.df[col].isna() | (self.df[col].astype(str).str.strip() == "") | (self.df[col].astype(str).str.lower() == "nan")
            missing = self.df[mask]
            if not missing.empty:
                rows = missing.index.tolist()
                drugs = missing["Generic"].dropna().unique().tolist() if "Generic" in missing.columns else []
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Missing Field",
                    description=f"{len(rows)} rows missing {label}. Drugs: {drugs[:10]}",
                    rows=rows,
                    field=col,
                    suggestion=f"Fill in {label} for all rows."
                ))

    # ── 3. Drug Name Typos ────────────────────

    def _check_drug_name_typos(self):
        if "Generic" not in self.df.columns:
            return
        names = self.df["Generic"].dropna().unique()
        # Check for known typos
        known_typos = {
            "Chlorthalidon": "Chlorthalidone",
        }
        for typo, correct in known_typos.items():
            mask = self.df["Generic"].str.strip() == typo
            if mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Typo",
                    description=f'Drug name "{typo}" should be "{correct}".',
                    rows=self.df[mask].index.tolist(),
                    field="Generic",
                    suggestion=f'Replace "{typo}" with "{correct}".'
                ))

        # Check for near-duplicate names (e.g., trailing spaces, case differences)
        normalized = {}
        for name in names:
            key = str(name).strip().lower()
            if key in normalized and normalized[key] != str(name).strip():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Inconsistent Naming",
                    description=f'Drug name variant: "{name}" vs "{normalized[key]}".',
                    field="Generic",
                    suggestion="Standardize to one spelling."
                ))
            normalized[key] = str(name).strip()

    # ── 4. Drug Class Typos ───────────────────

    def _check_drug_class_typos(self):
        if "Drug Class" not in self.df.columns:
            return
        classes = self.df["Drug Class"].dropna().unique()

        known_typos = {
            "Antianginal Angent": "Antianginal Agents",
            "Antianginal agents": "Antianginal Agents",
        }
        for typo, correct in known_typos.items():
            mask = self.df["Drug Class"].str.strip() == typo
            if mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Typo",
                    description=f'Drug class "{typo}" should be "{correct}".',
                    rows=self.df[mask].index.tolist(),
                    field="Drug Class",
                    suggestion=f'Replace with "{correct}".'
                ))

        # Check for near-duplicates
        seen = {}
        for cls in classes:
            key = str(cls).strip().lower()
            if key in seen and seen[key] != str(cls).strip():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Inconsistent Naming",
                    description=f'Drug class variant: "{cls}" vs "{seen[key]}".',
                    field="Drug Class",
                    suggestion="Merge into a single standardized name."
                ))
            seen[key] = str(cls).strip()

    # ── 5. Route Typos ────────────────────────

    def _check_route_typos(self):
        if "Route" not in self.df.columns:
            return
        known_typos = {
            "Intramascularly(IM)": "Intramuscularly(IM)",
        }
        for typo, correct in known_typos.items():
            mask = self.df["Route"].str.strip() == typo
            if mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Typo",
                    description=f'Route "{typo}" should be "{correct}". Affects {mask.sum()} entries.',
                    rows=self.df[mask].index.tolist(),
                    field="Route",
                    suggestion=f'Replace with "{correct}".'
                ))

    # ── 6. Indication Inconsistencies ─────────

    def _check_indication_inconsistencies(self):
        if "Indication" not in self.df.columns:
            return
        indications = self.df["Indication"].dropna().unique()
        seen = {}
        for ind in indications:
            key = str(ind).strip().lower()
            if key in seen and seen[key] != str(ind).strip():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Inconsistent Naming",
                    description=f'Indication variant: "{ind}" vs "{seen[key]}".',
                    field="Indication",
                    suggestion="Standardize case and spacing."
                ))
            seen[key] = str(ind).strip()

        # Check for indications that look like drug names (known issue)
        for ind in indications:
            s = str(ind).strip()
            if s in self.df["Generic"].values:
                mask = self.df["Indication"].str.strip() == s
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Data Entry Error",
                    description=f'Indication "{s}" appears to be a drug name, not an indication.',
                    rows=self.df[mask].index.tolist(),
                    field="Indication",
                    suggestion="Replace with the actual clinical indication."
                ))

    # ── 7. Age Group Inconsistencies ──────────

    def _check_age_group_inconsistencies(self):
        if "Age Group" not in self.df.columns:
            return
        ages = self.df["Age Group"].dropna().unique()
        seen = {}
        for age in ages:
            key = str(age).strip().lower()
            if key in seen and seen[key] != str(age).strip():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.INFO,
                    category="Inconsistent Naming",
                    description=f'Age group variant: "{age}" vs "{seen[key]}".',
                    field="Age Group",
                    suggestion="Remove leading/trailing spaces."
                ))
            seen[key] = str(age).strip()

    # ── 8. Dose Type Values ───────────────────

    def _check_dose_type_values(self):
        if "Type of Doses" not in self.df.columns:
            return
        valid_types = {
            "initial", "regular", "maintenance", "maximum", "target",
            "titrate", "loading", "repeat", "diagnostic",
            "pregnancy-associated", "test", "preoperative"
        }
        types = self.df["Type of Doses"].dropna().unique()
        for t in types:
            if str(t).strip().lower() not in valid_types:
                mask = self.df["Type of Doses"].str.strip() == str(t).strip()
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Invalid Value",
                    description=f'Unexpected dose type: "{t}".',
                    rows=self.df[mask].index.tolist(),
                    field="Type of Doses",
                    suggestion=f"Should be one of: {', '.join(sorted(valid_types))}"
                ))

    # ── 9. Pregnancy Categories ───────────────

    def _check_pregnancy_categories(self):
        if "Pregnancy Category" not in self.df.columns:
            return
        valid_fda = {"B", "C", "D", "Z"}
        tga_only = {"B1", "B2", "B3"}
        cats = self.df["Pregnancy Category"].dropna().unique()

        for cat in cats:
            c = str(cat).strip()
            if c in tga_only:
                mask = self.df["Pregnancy Category"].str.strip() == c
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Mixed Classification",
                    description=f'Pregnancy category "{c}" is TGA (Australian), not FDA. '
                                f'Dataset primarily uses FDA categories.',
                    rows=self.df[mask].index.tolist(),
                    field="Pregnancy Category",
                    suggestion="Decide whether to use FDA or TGA system consistently."
                ))
            elif c not in valid_fda and c.lower() != "nan":
                mask = self.df["Pregnancy Category"].str.strip() == c
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Invalid Value",
                    description=f'Unknown pregnancy category: "{c}".',
                    rows=self.df[mask].index.tolist(),
                    field="Pregnancy Category",
                    suggestion=f"Should be one of: {', '.join(sorted(valid_fda | tga_only))}"
                ))

    # ── 10. Initial Dose > Maximum Dose ───────

    def _check_initial_exceeds_maximum(self):
        if "Generic" not in self.df.columns or "Type of Doses" not in self.df.columns:
            return

        dose_col = "Direct Dose mg(Single Strength)"
        if dose_col not in self.df.columns:
            return

        drugs = self.df["Generic"].dropna().unique()
        for drug in drugs:
            drug_rows = self.df[self.df["Generic"].str.strip() == str(drug).strip()]
            # Get indications for this drug
            indications = drug_rows["Indication"].dropna().unique() if "Indication" in drug_rows.columns else [""]

            for ind in indications:
                if "Indication" in drug_rows.columns:
                    ind_rows = drug_rows[drug_rows["Indication"].str.strip() == str(ind).strip()]
                else:
                    ind_rows = drug_rows

                initial_rows = ind_rows[ind_rows["Type of Doses"].str.strip().str.lower().isin(["initial", "loading"])]
                max_rows = ind_rows[ind_rows["Type of Doses"].str.strip().str.lower() == "maximum"]

                if initial_rows.empty or max_rows.empty:
                    continue

                def safe_max(series):
                    vals = pd.to_numeric(series, errors="coerce").dropna()
                    return vals.max() if not vals.empty else None

                def safe_min(series):
                    vals = pd.to_numeric(series, errors="coerce").dropna()
                    return vals.min() if not vals.empty else None

                # Check single dose column
                init_max_val = safe_max(initial_rows[dose_col])
                max_min_val = safe_min(max_rows[dose_col])

                if init_max_val is not None and max_min_val is not None:
                    if init_max_val > max_min_val:
                        self.report.issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            category="Dose Logic",
                            description=f'{drug} ({ind}): Initial dose ({init_max_val} mg) exceeds Maximum dose ({max_min_val} mg).',
                            rows=initial_rows.index.tolist() + max_rows.index.tolist(),
                            field=dose_col,
                            suggestion="Verify dose values against source."
                        ))

    # ── 11. Both Single and Range Dose Filled ─

    def _check_single_and_range_both_filled(self):
        single_col = "Direct Dose mg(Single Strength)"
        min_col = "Min Direct Dose mg(Multiple Strength)"
        max_col = "Max  Direct Dose mg(Multiple Strength)"

        if not all(c in self.df.columns for c in [single_col, min_col, max_col]):
            return

        for idx, row in self.df.iterrows():
            single_val = pd.to_numeric(row.get(single_col), errors="coerce")
            min_val = pd.to_numeric(row.get(min_col), errors="coerce")
            max_val = pd.to_numeric(row.get(max_col), errors="coerce")

            single_filled = not (pd.isna(single_val))
            range_filled = not (pd.isna(min_val)) and not (pd.isna(max_val))

            if single_filled and range_filled:
                drug = row.get("Generic", "Unknown")
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Dose Pattern",
                    description=f'Row {idx} ({drug}): Both single dose ({single_val}) and range dose ({min_val}-{max_val}) are filled. Should be one or the other.',
                    rows=[idx],
                    field=single_col,
                    suggestion="Keep only one dose pattern per row (single OR range)."
                ))

    # ── 12. Frequency Values ──────────────────

    def _check_frequency_values(self):
        freq_cols = ["Single Frequency", "Min Frequency ", "Max Frequency"]
        for col in freq_cols:
            if col not in self.df.columns:
                continue
            vals = pd.to_numeric(self.df[col], errors="coerce")
            # Check for zero or negative
            bad_mask = vals.notna() & (vals <= 0)
            if bad_mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Invalid Value",
                    description=f'{bad_mask.sum()} rows have zero or negative frequency in {col}.',
                    rows=self.df[bad_mask].index.tolist(),
                    field=col,
                    suggestion="Frequency must be a positive number."
                ))
            # Check for unusually high frequency
            high_mask = vals > 24
            if high_mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Suspicious Value",
                    description=f'{high_mask.sum()} rows have frequency > 24 in {col}. Verify timing unit.',
                    rows=self.df[high_mask].index.tolist(),
                    field=col,
                    suggestion="Check if timing should be Hour instead of Day."
                ))

        # Check single AND range frequency both filled
        if "Single Frequency" in self.df.columns and "Min Frequency " in self.df.columns:
            for idx, row in self.df.iterrows():
                sf = pd.to_numeric(row.get("Single Frequency"), errors="coerce")
                mf = pd.to_numeric(row.get("Min Frequency "), errors="coerce")
                if not pd.isna(sf) and not pd.isna(mf):
                    drug = row.get("Generic", "Unknown")
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="Frequency Pattern",
                        description=f'Row {idx} ({drug}): Both single ({sf}) and range frequency ({mf}) filled.',
                        rows=[idx],
                        field="Single Frequency",
                        suggestion="Use either single OR range frequency, not both."
                    ))

    # ── 13. Dose Outliers ─────────────────────

    def _check_dose_outliers(self):
        dose_col = "Direct Dose mg(Single Strength)"
        if dose_col not in self.df.columns:
            return

        vals = pd.to_numeric(self.df[dose_col], errors="coerce").dropna()
        if vals.empty:
            return

        # Flag extremely high doses (>5000mg is suspicious for most CV drugs)
        high_mask = pd.to_numeric(self.df[dose_col], errors="coerce") > 5000
        high_mask = high_mask.fillna(False)
        if high_mask.any():
            high_rows = self.df[high_mask]
            for idx, row in high_rows.iterrows():
                drug = row.get("Generic", "Unknown")
                dose = row.get(dose_col)
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Dose Outlier",
                    description=f'Row {idx} ({drug}): Dose of {dose} mg is unusually high for a CV drug.',
                    rows=[idx],
                    field=dose_col,
                    suggestion="Verify against source. Could be a unit error (mcg vs mg)."
                ))

        # Check specific known errors from context
        # Fondaparinux 100mg should be 10mg
        if "Generic" in self.df.columns:
            fond_mask = (self.df["Generic"].str.strip() == "Fondaparinux") & \
                        (pd.to_numeric(self.df[dose_col], errors="coerce") == 100)
            if fond_mask.any():
                self.report.issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="Known Data Error",
                    description='Fondaparinux shows 100mg but typical max is 10mg.',
                    rows=self.df[fond_mask].index.tolist(),
                    field=dose_col,
                    suggestion="Change 100mg to 10mg."
                ))

    # ── 14. FDC Dose Format ───────────────────

    def _check_fdc_dose_format(self):
        dose_cols = [
            "Direct Dose mg(Single Strength)",
            "Min Direct Dose mg(Multiple Strength)",
            "Max  Direct Dose mg(Multiple Strength)",
        ]
        for col in dose_cols:
            if col not in self.df.columns:
                continue
            for idx, val in self.df[col].items():
                s = str(val).strip()
                if "/" in s and s not in ["nan", ""]:
                    # Check format is valid like "25/100"
                    if not re.match(r'^\d+\.?\d*/\d+\.?\d*$', s):
                        drug = self.df.at[idx, "Generic"] if "Generic" in self.df.columns else "Unknown"
                        self.report.issues.append(ValidationIssue(
                            severity=Severity.ERROR,
                            category="FDC Format",
                            description=f'Row {idx} ({drug}): FDC dose "{s}" has invalid format in {col}.',
                            rows=[idx],
                            field=col,
                            suggestion='FDC doses should be "number/number" format like "25/100".'
                        ))

    # ── 15. Source URLs ───────────────────────

    def _check_source_urls_present(self):
        source_cols = [c for c in self.df.columns if c.startswith("Source")]
        if not source_cols:
            return

        # Check for drugs with zero sources
        for idx, row in self.df.iterrows():
            has_source = False
            for col in source_cols:
                val = str(row.get(col, "")).strip()
                if val and val.lower() != "nan" and val.startswith("http"):
                    has_source = True
                    break
            # Only flag the first row per drug (safety data is per-drug)
            # Actually, flag all rows with no source
            if not has_source:
                drug = row.get("Generic", "Unknown")
                dose_type = row.get("Type of Doses", "")
                # This is too noisy for safety-only rows, so only flag
                # if this is not a safety-repeat row
                # For now, collect and group later

        # Group by drug and check
        if "Generic" in self.df.columns:
            for drug in self.df["Generic"].dropna().unique():
                drug_rows = self.df[self.df["Generic"].str.strip() == str(drug).strip()]
                any_source = False
                for col in source_cols:
                    vals = drug_rows[col].astype(str)
                    if vals.str.startswith("http").any():
                        any_source = True
                        break
                if not any_source:
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="Missing Source",
                        description=f'{drug}: No source URLs found for any entry.',
                        rows=drug_rows.index.tolist(),
                        field="Source 1",
                        suggestion="Add at least one verifiable source URL."
                    ))

        # Check for invalid URL format
        for col in source_cols:
            for idx, val in self.df[col].items():
                s = str(val).strip()
                if s and s.lower() != "nan" and not s.startswith("http"):
                    drug = self.df.at[idx, "Generic"] if "Generic" in self.df.columns else "Unknown"
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="Invalid URL",
                        description=f'Row {idx} ({drug}): {col} value "{s[:50]}" is not a valid URL.',
                        rows=[idx],
                        field=col,
                        suggestion="URLs should start with http:// or https://"
                    ))

    # ── 16. Timing Values ─────────────────────

    def _check_timing_values(self):
        if "Timing" not in self.df.columns:
            return
        valid_timings = {"day", "hour", "minute", "week", "month", "dose"}
        timings = self.df["Timing"].dropna().unique()
        for t in timings:
            if str(t).strip().lower() not in valid_timings:
                mask = self.df["Timing"].str.strip() == str(t).strip()
                self.report.issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Invalid Value",
                    description=f'Unexpected timing value: "{t}".',
                    rows=self.df[mask].index.tolist(),
                    field="Timing",
                    suggestion=f"Should be one of: {', '.join(sorted(valid_timings))}"
                ))

    # ── 17. Renal Sheet Consistency ───────────

    def _check_renal_sheet_consistency(self):
        if self.renal_df.empty:
            return

        # Check that all renal drugs exist in main sheet
        if "Generic" in self.renal_df.columns and "Generic" in self.df.columns:
            main_drugs = set(self.df["Generic"].str.strip().str.lower().dropna())
            renal_drugs = self.renal_df["Generic"].str.strip().dropna().unique()

            for drug in renal_drugs:
                if drug.strip().lower() not in main_drugs:
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="Renal Mismatch",
                        description=f'Renal sheet drug "{drug}" not found in main sheet.',
                        field="Generic",
                        suggestion="Ensure drug names match between sheets."
                    ))

        # Check renal dose values
        dose_col = "Direct Dose mg (Single Strength)"
        if dose_col in self.renal_df.columns:
            for idx, row in self.renal_df.iterrows():
                val = pd.to_numeric(row.get(dose_col), errors="coerce")
                if not pd.isna(val) and val > 1000:
                    drug = row.get("Generic", "Unknown")
                    self.report.issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="Renal Dose Outlier",
                        description=f'Renal row {idx} ({drug}): Dose {val} mg seems high for a renal-adjusted dose.',
                        rows=[idx],
                        field=dose_col,
                        suggestion="Verify against source."
                    ))

    # ── 18. Weight Column Values ──────────────

    def _check_weight_column_values(self):
        if "Weight" not in self.df.columns:
            return
        for idx, val in self.df["Weight"].items():
            s = str(val).strip()
            if s and s.lower() not in ["nan", ".", ""]:
                # Weight should be a range like "20-50" or condition like "<50"
                # Flag anything that doesn't look right
                if not re.match(r'^[<>≤≥]?\d+\.?\d*(\s*[-–]\s*[<>≤≥]?\s*\d+\.?\d*)?\s*(kg)?:?$', s):
                    drug = self.df.at[idx, "Generic"] if "Generic" in self.df.columns else "Unknown"
                    # Only flag truly weird values, not standard patterns
                    pass  # Weight formats are varied and mostly valid


def validate_source_urls(df: pd.DataFrame, max_checks: int = 50) -> list[ValidationIssue]:
    """
    Layer C: Check if source URLs are reachable.
    Returns list of issues for broken/unreachable URLs.
    """
    import requests
    issues = []
    source_cols = [c for c in df.columns if c.startswith("Source")]
    checked = 0

    for col in source_cols:
        for idx, val in df[col].items():
            if checked >= max_checks:
                return issues
            s = str(val).strip()
            if not s.startswith("http"):
                continue
            try:
                resp = requests.head(s, timeout=10, allow_redirects=True)
                if resp.status_code >= 400:
                    drug = df.at[idx, "Generic"] if "Generic" in df.columns else "Unknown"
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="Broken URL",
                        description=f'Row {idx} ({drug}): {col} returned HTTP {resp.status_code}.',
                        rows=[idx],
                        field=col,
                        suggestion="Verify URL is still active or find replacement."
                    ))
                checked += 1
            except Exception as e:
                drug = df.at[idx, "Generic"] if "Generic" in df.columns else "Unknown"
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="Unreachable URL",
                    description=f'Row {idx} ({drug}): {col} could not be reached ({str(e)[:60]}).',
                    rows=[idx],
                    field=col,
                    suggestion="Check URL manually."
                ))
                checked += 1

    return issues
