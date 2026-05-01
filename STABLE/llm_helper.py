"""
STABLE LLM Helper
==================
Uses the Anthropic API for:
1. Verdict explanation in plain clinical language
2. Patient-contextualized safety summary
3. Layer B clinical plausibility checks on dataset
"""

import json
import requests


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def _call_claude(api_key: str, system: str, user_msg: str, max_tokens: int = 1500) -> str:
    """Call the Anthropic API and return the text response."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    }
    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            return "ERROR: Invalid API key. Please check your Anthropic API key."
        return f"ERROR: API request failed ({resp.status_code}): {str(e)}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def generate_verdict_explanation(api_key: str, prescription: dict,
                                  steps: list[dict], safety: dict) -> str:
    """
    Generate a plain-language clinical explanation of the audit verdict.
    Tells the doctor what went wrong and what to do about it.
    """
    system = (
        "You are a clinical pharmacology assistant. Given a prescription audit result "
        "from the STABLE cardiovascular prescription verification system, write a clear, "
        "actionable clinical explanation. Be specific about dose numbers, what the limits "
        "are, and what the prescriber should consider. Keep it to 3-5 sentences. "
        "Do not use the word 'crucial' or 'comprehensive'. Write plainly."
    )

    steps_text = "\n".join(
        f"Step {s['step']} ({s['name']}): {s['verdict']} - {s['detail']}"
        for s in steps
    )

    user_msg = f"""Prescription:
- Drug: {prescription.get('drug', 'N/A')}
- Indication: {prescription.get('indication', 'N/A')}
- Dose: {prescription.get('dose_value', 0)} {prescription.get('dose_unit', 'mg')}
- Frequency: {prescription.get('frequency', 0)} per {prescription.get('timing', 'Day')}
- Route: {prescription.get('route', 'N/A')}
- Patient weight: {prescription.get('weight', 'Not provided')} kg
- CrCl: {prescription.get('crcl', 'Not provided')} mL/min
- Pregnant: {prescription.get('pregnant', False)}

Audit Results:
{steps_text}

Pregnancy Category: {safety.get('pregnancy_category', 'N/A')}

Write a clinical explanation of this verdict for the prescribing physician."""

    return _call_claude(api_key, system, user_msg)


def generate_safety_summary(api_key: str, prescription: dict,
                             safety: dict) -> str:
    """
    Generate a patient-contextualized safety summary.
    Filters and prioritizes safety information relevant to this specific patient.
    """
    system = (
        "You are a clinical pharmacology assistant. Given a drug's safety profile and "
        "a specific patient's context, write a prioritized safety summary. Put the most "
        "relevant warnings first based on the patient's characteristics (age, renal "
        "function, pregnancy status, etc.). Skip safety items that are not relevant to "
        "this patient. Keep it concise: 4-6 bullet points maximum. "
        "Do not use the word 'crucial' or 'comprehensive'."
    )

    user_msg = f"""Patient Context:
- Age group: {prescription.get('age_group', 'Adult')}
- Weight: {prescription.get('weight', 'Not provided')} kg
- CrCl: {prescription.get('crcl', 'Not provided')} mL/min
- eGFR: {prescription.get('egfr', 'Not provided')}
- Pregnant: {prescription.get('pregnant', False)}
- Indication: {prescription.get('indication', 'N/A')}

Drug: {safety.get('drug', 'N/A')} ({safety.get('drug_class', 'N/A')})

Full Safety Profile:
- Contraindications: {safety.get('contraindications', 'None listed')[:800]}
- Warnings: {safety.get('warnings', 'None listed')[:800]}
- Drug-Drug Interactions: {safety.get('ddi', 'None listed')[:800]}
- Disease Interactions: {safety.get('disease_interactions', 'None listed')[:500]}
- Serious Effects: {safety.get('serious_effects', 'None listed')[:500]}
- Adverse Effects: {safety.get('adverse_effects', 'None listed')[:500]}
- Pregnancy Category: {safety.get('pregnancy_category', 'N/A')}

Write a prioritized safety summary relevant to THIS specific patient."""

    return _call_claude(api_key, system, user_msg)


def check_clinical_plausibility(api_key: str, drug_data: list[dict],
                                 drug_name: str) -> str:
    """
    Layer B: LLM checks whether dose values, indications, pregnancy category,
    and other data for a drug are clinically plausible.
    """
    system = (
        "You are a clinical pharmacology expert reviewing a cardiovascular drug dataset "
        "for accuracy. For the given drug data, check whether:\n"
        "1. Dose values are clinically reasonable for each indication\n"
        "2. The indications listed are standard for this drug\n"
        "3. The pregnancy category matches known classifications\n"
        "4. The routes are appropriate\n"
        "5. Any dose values seem like likely data entry errors\n\n"
        "Report ONLY issues you find. If everything looks correct, say so briefly. "
        "Be specific about which rows or values are suspicious and what the correct "
        "value should be. Format as a numbered list of findings."
    )

    # Limit data sent to avoid token overflow
    data_str = json.dumps(drug_data[:20], indent=2, default=str)

    user_msg = f"""Review the following dataset entries for {drug_name}:

{data_str}

Check clinical plausibility of doses, indications, pregnancy category, and routes."""

    return _call_claude(api_key, system, user_msg, max_tokens=2000)


def batch_plausibility_check(api_key: str, df, drugs: list[str],
                              progress_callback=None) -> dict[str, str]:
    """
    Run clinical plausibility checks on multiple drugs.
    Returns dict of drug_name -> findings.
    """
    results = {}
    for i, drug in enumerate(drugs):
        if progress_callback:
            progress_callback(i, len(drugs), drug)

        drug_rows = df[df["Generic"].str.strip() == drug]
        cols_to_send = [
            "Generic", "Indication", "Type of Doses", "Route", "Age Group",
            "Direct Dose mg(Single Strength)",
            "Min Direct Dose mg(Multiple Strength)",
            "Max  Direct Dose mg(Multiple Strength)",
            "Dose Per Weight mg(Single Strength)",
            "Single Frequency", "Timing",
            "Pregnancy Category",
        ]
        available_cols = [c for c in cols_to_send if c in drug_rows.columns]
        data = drug_rows[available_cols].to_dict(orient="records")

        result = check_clinical_plausibility(api_key, data, drug)
        results[drug] = result

        if result.startswith("ERROR"):
            break  # Stop on API errors

    return results
