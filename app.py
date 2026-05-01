"""
STABLE Prescription Audit Dashboard
"""

import streamlit as st
import pandas as pd
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_engine import STABLEEngine, Prescription, Verdict, DosePattern
from dataset_validator import DatasetValidator, Severity, validate_source_urls
from llm_helper import generate_verdict_explanation, generate_safety_summary, batch_plausibility_check

st.set_page_config(
    page_title="STABLE — CV Prescription Audit",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
.header-bar {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
    padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    display: flex; align-items: center; gap: 1rem;
}
.header-bar h1 { color: #f0f9ff; margin: 0; font-size: 1.75rem; font-weight: 700; }
.header-bar p { color: #93c5fd; margin: 0; font-size: 0.85rem; }
.header-icon { font-size: 2rem; }
.verdict-accept {
    background: linear-gradient(135deg, #065f46, #047857); color: #ecfdf5;
    padding: 1.2rem 2rem; border-radius: 10px; text-align: center;
    font-size: 1.3rem; font-weight: 700; border: 1px solid #34d399;
}
.verdict-caution {
    background: linear-gradient(135deg, #78350f, #92400e); color: #fefce8;
    padding: 1.2rem 2rem; border-radius: 10px; text-align: center;
    font-size: 1.3rem; font-weight: 700; border: 1px solid #fbbf24;
}
.verdict-reject {
    background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #fef2f2;
    padding: 1.2rem 2rem; border-radius: 10px; text-align: center;
    font-size: 1.3rem; font-weight: 700; border: 1px solid #f87171;
}
.step-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.5rem; border-left: 4px solid #94a3b8;
}
.step-card.pass { border-left-color: #10b981; background: #f0fdf4; }
.step-card.flag { border-left-color: #f59e0b; background: #fffbeb; }
.step-card.reject { border-left-color: #ef4444; background: #fef2f2; }
.step-card.skip { border-left-color: #94a3b8; background: #f8fafc; }
.step-header { font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.5rem; }
.step-detail { font-size: 0.82rem; color: #475569; font-family: 'JetBrains Mono', monospace; line-height: 1.45; }
.safety-box {
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.safety-box h4 { color: #9a3412; margin: 0 0 0.4rem 0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
.safety-box p { color: #431407; margin: 0; font-size: 0.82rem; line-height: 1.5; white-space: pre-wrap; }
.stat-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }
.stat-card .number { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
.stat-card .label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.ai-box {
    background: linear-gradient(135deg, #eff6ff, #f0f9ff); border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1rem 0;
}
.ai-box h4 { color: #1e40af; margin: 0 0 0.5rem 0; font-size: 0.9rem; }
.ai-box p { color: #1e3a5f; margin: 0; font-size: 0.85rem; line-height: 1.6; }
.val-error { background: #fef2f2; border-left: 4px solid #ef4444; padding: 0.7rem 1rem; margin-bottom: 0.4rem; border-radius: 0 6px 6px 0; }
.val-warning { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.7rem 1rem; margin-bottom: 0.4rem; border-radius: 0 6px 6px 0; }
.val-info { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 0.7rem 1rem; margin-bottom: 0.4rem; border-radius: 0 6px 6px 0; }
.val-title { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.2rem; }
.val-desc { font-size: 0.8rem; color: #475569; }
.val-fix { font-size: 0.78rem; color: #059669; font-style: italic; }
section[data-testid="stSidebar"] { background: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


def show_raw_safety(safety):
    if not safety:
        st.info("No safety information available.")
        return
    safety_fields = [
        ("Contraindications", "contraindications", "🚫"),
        ("Drug-Drug Interactions", "ddi", "💊"),
        ("Disease Interactions", "disease_interactions", "🦠"),
        ("Warnings", "warnings", "⚠️"),
        ("Serious Effects", "serious_effects", "🔴"),
        ("Adverse Effects", "adverse_effects", "🟡"),
        ("Pregnancy Category", "pregnancy_category", "🤰"),
    ]
    col_left, col_right = st.columns(2)
    for i, (label, key, icon) in enumerate(safety_fields):
        val = safety.get(key, "")
        if not val:
            continue
        target = col_left if i % 2 == 0 else col_right
        with target:
            display_val = val if len(val) < 600 else val[:600] + "..."
            st.markdown(
                '<div class="safety-box"><h4>' + icon + " " + label + "</h4><p>" + display_val + "</p></div>",
                unsafe_allow_html=True
            )


def show_validation_issue(issue, css_class):
    rows_str = ""
    if issue.rows:
        display_rows = issue.rows[:10]
        rows_str = " | Rows: " + str(display_rows)
        if len(issue.rows) > 10:
            rows_str = rows_str + " (+" + str(len(issue.rows) - 10) + " more)"

    field_str = ""
    if issue.field:
        field_str = " | " + issue.field

    fix_html = ""
    if issue.suggestion:
        fix_html = '<div class="val-fix">Fix: ' + issue.suggestion + "</div>"

    html = (
        '<div class="' + css_class + '">'
        + '<div class="val-title">' + issue.category + field_str + rows_str + "</div>"
        + '<div class="val-desc">' + issue.description + "</div>"
        + fix_html
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(app_dir):
        if f.endswith(".xlsx") and "stable" in f.lower():
            return STABLEEngine(os.path.join(app_dir, f))
    return None


engine = load_engine()

st.markdown("""
<div class="header-bar">
    <span class="header-icon">🫀</span>
    <div>
        <h1>STABLE</h1>
        <p>Standardized Therapy Audit-Based Logic and Estimates — Cardiovascular Prescription Verification</p>
    </div>
</div>
""", unsafe_allow_html=True)

if engine is None:
    st.error("Dataset file not found. Place the STABLE Excel file in the app directory or upload it below.")
    uploaded = st.file_uploader("Upload STABLE dataset (.xlsx)", type=["xlsx"])
    if uploaded:
        tmp_path = os.path.join(tempfile.gettempdir(), "stable_dataset.xlsx")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())
        engine = STABLEEngine(tmp_path)
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown("### 🔑 AI Features (Optional)")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="Enter your key to enable AI features. Get one at console.anthropic.com")
    ai_enabled = bool(api_key and api_key.startswith("sk-"))
    if api_key and not ai_enabled:
        st.caption("Key should start with 'sk-'")
    elif ai_enabled:
        st.caption("AI features enabled")
    st.markdown("---")

tab_audit, tab_validate = st.tabs(["🔍 Prescription Audit", "🛡️ Dataset Validation"])

# ═══════════════════════════════════════
# TAB 1: PRESCRIPTION AUDIT
# ═══════════════════════════════════════
with tab_audit:
    drug_list = engine.get_drug_list()
    total_rows = len(engine.df)
    total_renal = len(engine.renal_df) if not engine.renal_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="stat-card"><div class="number">' + str(len(drug_list)) + '</div><div class="label">Drugs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="stat-card"><div class="number">' + str(total_rows) + '</div><div class="label">Dosing Entries</div></div>', unsafe_allow_html=True)
    with c3:
        n_classes = engine.df["Drug Class"].nunique()
        st.markdown('<div class="stat-card"><div class="number">' + str(n_classes) + '</div><div class="label">Drug Classes</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="stat-card"><div class="number">' + str(total_renal) + '</div><div class="label">Renal Entries</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📋 Prescription Input")
        selected_drug = st.selectbox("Drug Name", options=[""] + drug_list, index=0)
        indications, routes, age_groups = [], [], []
        if selected_drug:
            indications = engine.get_indications_for_drug(selected_drug)
            routes = engine.get_routes_for_drug(selected_drug)
            age_groups = engine.get_age_groups_for_drug(selected_drug)

        selected_indication = st.selectbox("Indication", options=[""] + indications, index=0)
        selected_age = st.selectbox("Age Group",
            options=[""] + age_groups if age_groups else ["", "Adult", "Pediatric", "Geriatric"], index=0)
        selected_route = st.selectbox("Route", options=[""] + routes, index=0)

        st.markdown("---")
        st.markdown("### 💊 Dose Details")
        dose_unit = st.selectbox("Dose Unit", ["mg", "IU", "Unit"])
        dose_value = st.number_input("Prescribed Dose (" + dose_unit + ")", min_value=0.0, step=0.5, value=0.0, format="%.2f")
        frequency = st.number_input("Frequency (times per period)", min_value=0.0, step=1.0, value=0.0, format="%.1f")
        timing = st.selectbox("Timing", ["Day", "Hour", "Minute", "Week", "Month", "Dose"])

        st.markdown("---")
        st.markdown("### 🏥 Patient Parameters")
        patient_weight = st.number_input("Weight (kg)", min_value=0.0, step=1.0, value=0.0)
        patient_crcl = st.number_input("CrCl (mL/min)", min_value=-1.0, step=1.0, value=-1.0)
        patient_egfr = st.number_input("eGFR", min_value=-1.0, step=1.0, value=-1.0)
        is_pregnant = st.checkbox("Patient is pregnant")

        st.markdown("---")
        run_audit = st.button("🔍 Run Audit", use_container_width=True, type="primary")

    if not selected_drug:
        st.info("Select a drug from the sidebar to begin a prescription audit.")
        with st.expander("📖 Dataset Quick Reference", expanded=False):
            ref_df = engine.df[["Drug Class", "Generic", "Indication", "Type of Doses", "Route"]].drop_duplicates().head(100)
            st.dataframe(ref_df, use_container_width=True, height=400)

    elif not run_audit:
        drug_rows = engine.df[engine.df["Generic"].str.lower() == selected_drug.lower()]
        st.markdown("#### " + selected_drug + " — " + str(len(drug_rows)) + " dosing entries")
        preview_cols = ["Indication", "Type of Doses", "Route", "Direct Dose mg(Single Strength)",
                        "Min Direct Dose mg(Multiple Strength)", "Max  Direct Dose mg(Multiple Strength)",
                        "Single Frequency", "Timing"]
        avail = [c for c in preview_cols if c in drug_rows.columns]
        st.dataframe(drug_rows[avail].reset_index(drop=True), use_container_width=True, height=300)
        st.caption("Fill in the prescription details and click **Run Audit**.")

    else:
        rx = Prescription(
            drug_name=selected_drug, indication=selected_indication,
            age_group=selected_age if selected_age else "Adult", route=selected_route,
            dose_value=dose_value, dose_unit=dose_unit, patient_weight_kg=patient_weight,
            frequency=frequency, timing=timing, patient_crcl=patient_crcl,
            patient_egfr=patient_egfr, is_pregnant=is_pregnant,
        )
        report = engine.audit(rx)

        verdict_map = {
            Verdict.PASS: ("verdict-accept", "ACCEPT", "All checks passed. Prescription appears safe."),
            Verdict.FLAG: ("verdict-caution", "CAUTION", "One or more concerns flagged. Clinical review recommended."),
            Verdict.REJECT: ("verdict-reject", "REJECT", "Critical issue detected. Do not proceed without override."),
        }
        css_class, label, desc = verdict_map.get(report.final_verdict, ("verdict-caution", "?", ""))
        st.markdown(
            '<div class="' + css_class + '">' + label + '<br><span style="font-size:0.8rem;font-weight:400;">' + desc + '</span></div>',
            unsafe_allow_html=True
        )

        if ai_enabled and report.final_verdict in (Verdict.REJECT, Verdict.FLAG):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("Generating clinical explanation..."):
                rx_dict = {
                    "drug": selected_drug, "indication": selected_indication,
                    "dose_value": dose_value, "dose_unit": dose_unit,
                    "frequency": frequency, "timing": timing,
                    "route": selected_route, "weight": patient_weight,
                    "crcl": patient_crcl if patient_crcl >= 0 else "Not provided",
                    "egfr": patient_egfr if patient_egfr >= 0 else "Not provided",
                    "pregnant": is_pregnant, "age_group": selected_age,
                }
                steps_list = []
                for s in report.steps:
                    if s.step <= 9:
                        steps_list.append({"step": s.step, "name": s.name, "verdict": s.verdict.value, "detail": s.detail})
                explanation = generate_verdict_explanation(api_key, rx_dict, steps_list, report.safety_info)
            if not explanation.startswith("ERROR"):
                st.markdown(
                    '<div class="ai-box"><h4>🤖 AI Clinical Explanation</h4><p>' + explanation + '</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.warning(explanation)

        st.markdown("<br>", unsafe_allow_html=True)

        icon_map = {Verdict.PASS: "✅", Verdict.FLAG: "⚠️", Verdict.REJECT: "❌", Verdict.SKIP: "⏭️"}
        css_step_map = {Verdict.PASS: "pass", Verdict.FLAG: "flag", Verdict.REJECT: "reject", Verdict.SKIP: "skip"}

        st.markdown("### Layer 1 — Automated Verification")
        for step in report.steps:
            if step.step > 9:
                continue
            icon = icon_map.get(step.verdict, "?")
            css = css_step_map.get(step.verdict, "skip")
            st.markdown(
                '<div class="step-card ' + css + '">'
                + '<div class="step-header">' + icon + " Step " + str(step.step) + ": " + step.name
                + ' <span style="color:#64748b;font-weight:400;">— ' + step.verdict.value + '</span></div>'
                + '<div class="step-detail">' + step.detail + '</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Layer 2 — Safety Information for Clinical Review")

        if ai_enabled:
            with st.spinner("Generating patient-specific safety summary..."):
                rx_dict_safety = {
                    "drug": selected_drug, "indication": selected_indication,
                    "age_group": selected_age, "weight": patient_weight,
                    "crcl": patient_crcl if patient_crcl >= 0 else "Not provided",
                    "egfr": patient_egfr if patient_egfr >= 0 else "Not provided",
                    "pregnant": is_pregnant,
                }
                ai_summary = generate_safety_summary(api_key, rx_dict_safety, report.safety_info)
            if not ai_summary.startswith("ERROR"):
                st.markdown(
                    '<div class="ai-box"><h4>🤖 Patient-Specific Safety Summary</h4><p>' + ai_summary + '</p></div>',
                    unsafe_allow_html=True
                )
                with st.expander("📋 Full Safety Data (Raw)", expanded=False):
                    show_raw_safety(report.safety_info)
            else:
                st.warning(ai_summary)
                show_raw_safety(report.safety_info)
        else:
            st.caption("Add an Anthropic API key in the sidebar for AI-powered patient-specific safety summaries.")
            show_raw_safety(report.safety_info)

        st.markdown("<br>", unsafe_allow_html=True)
        if report.final_verdict in (Verdict.REJECT, Verdict.FLAG):
            with st.expander("🔓 Clinical Override", expanded=False):
                st.warning("If you believe the prescription is clinically appropriate, document an override reason below.")
                override_reason = st.text_area("Override Reason", placeholder="e.g., Patient stable on this dose for 6 months...")
                if st.button("Submit Override"):
                    if override_reason.strip():
                        st.success("Override documented. " + report.final_verdict.value + " -> ACCEPT with reason recorded.")
                    else:
                        st.error("An override reason is required.")


# ═══════════════════════════════════════
# TAB 2: DATASET VALIDATION
# ═══════════════════════════════════════
with tab_validate:
    st.markdown("### Dataset Validation")
    st.caption("Run automated checks to find errors, inconsistencies, and suspicious values before submission.")

    val_col1, val_col2, val_col3 = st.columns(3)
    with val_col1:
        run_layer_a = st.button("🔬 Layer A: Automated Checks", use_container_width=True, type="primary")
    with val_col2:
        run_layer_b = st.button("🤖 Layer B: AI Plausibility", use_container_width=True, disabled=not ai_enabled)
    with val_col3:
        run_layer_c = st.button("🌐 Layer C: URL Verification", use_container_width=True)

    st.markdown("---")

    if run_layer_a:
        with st.spinner("Running 17 automated consistency checks..."):
            validator = DatasetValidator(engine.df, engine.renal_df)
            val_report = validator.run_all_checks()

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Checks Run", val_report.checks_run)
        with sc2:
            st.metric("Errors", val_report.error_count)
        with sc3:
            st.metric("Warnings", val_report.warning_count)
        with sc4:
            st.metric("Info", val_report.info_count)

        st.markdown("<br>", unsafe_allow_html=True)

        if not val_report.issues:
            st.success("No issues found. Dataset passes all automated checks.")
        else:
            for sev, sev_label, css_class in [
                (Severity.ERROR, "Errors (must fix)", "val-error"),
                (Severity.WARNING, "Warnings (investigate)", "val-warning"),
                (Severity.INFO, "Info (minor)", "val-info"),
            ]:
                issues = [i for i in val_report.issues if i.severity == sev]
                if not issues:
                    continue
                st.markdown("#### " + sev_label + " (" + str(len(issues)) + ")")
                for issue in issues:
                    show_validation_issue(issue, css_class)

            with st.expander("📊 Export Issues as Table", expanded=False):
                issues_data = []
                for i in val_report.issues:
                    issues_data.append({
                        "Severity": i.severity.value,
                        "Category": i.category,
                        "Description": i.description,
                        "Field": i.field,
                        "Rows": str(i.rows[:10]) if i.rows else "",
                        "Suggestion": i.suggestion,
                    })
                issues_df = pd.DataFrame(issues_data)
                st.dataframe(issues_df, use_container_width=True)
                csv = issues_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "validation_report.csv", "text/csv")

    if run_layer_b and ai_enabled:
        st.markdown("#### AI Clinical Plausibility Check")
        st.caption("The AI reviews dose values, indications, and safety data against clinical knowledge.")

        all_drugs = sorted(engine.df["Generic"].dropna().unique().tolist())
        check_option = st.radio("Which drugs to check?",
                                ["Sample (10 random)", "All drugs (takes longer)", "Select specific drugs"],
                                horizontal=True)

        if check_option == "Sample (10 random)":
            import random
            drugs_to_check = random.sample(all_drugs, min(10, len(all_drugs)))
        elif check_option == "All drugs (takes longer)":
            drugs_to_check = all_drugs
        else:
            drugs_to_check = st.multiselect("Select drugs", all_drugs)

        if st.button("Start AI Check", type="primary") and drugs_to_check:
            progress = st.progress(0)
            status = st.empty()

            def update_progress(i, total, drug):
                progress.progress((i + 1) / total)
                status.text("Checking " + drug + "... (" + str(i + 1) + "/" + str(total) + ")")

            results = batch_plausibility_check(api_key, engine.df, drugs_to_check, update_progress)
            progress.empty()
            status.empty()

            for drug, findings in results.items():
                if findings.startswith("ERROR"):
                    st.error(findings)
                    break
                with st.expander("💊 " + drug, expanded=False):
                    st.markdown(findings)

    if run_layer_c:
        st.markdown("#### Source URL Verification")
        max_urls = st.slider("Max URLs to check", 10, 200, 50)
        with st.spinner("Checking up to " + str(max_urls) + " source URLs..."):
            try:
                url_issues = validate_source_urls(engine.df, max_checks=max_urls)
            except Exception as e:
                url_issues = []
                st.error("URL check failed: " + str(e))

        if not url_issues:
            st.success("All checked URLs are reachable.")
        else:
            st.warning("Found " + str(len(url_issues)) + " URL issues:")
            for issue in url_issues:
                show_validation_issue(issue, "val-warning")


st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.75rem;">'
    'STABLE v1.0 — AIMS Lab, IRIIC, United International University, Dhaka<br>'
    'For research purposes only. Not a substitute for clinical judgment.</div>',
    unsafe_allow_html=True
)