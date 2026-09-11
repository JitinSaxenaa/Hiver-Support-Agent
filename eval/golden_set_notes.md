# Golden Evaluation Set Methodology & Annotation Notes

This document details the sampling strategy, labeling guidelines, annotation protocol, and operational constraints behind the creation of eval/golden_set.csv.

---

## 1. Candidate Sampling & Split Hygiene

To ensure zero evaluation leakage and preserve statistical validity:
1. **Total Grounding Corpus:** 25,000 time-stratified customer-brand reply pairs for @AppleSupport were extracted from the Kaggle dataset.
2. **Strict Split Isolation:**
   - **Retrieval Index Split:** Rows 0 to 19,999 (20,000 pairs) are reserved strictly for the retrieval embedding corpus (data/processed/retrieval_index_corpus.csv).
   - **Training/Validation Pool:** Rows 20,000 to 22,499 (2,500 pairs) are reserved for training classical baseline classifiers (TF-IDF + Logistic Regression).
   - **Evaluation Candidate Pool:** Rows 22,500 to 24,999 (2,500 pairs) are the strictly held-out evaluation reserve. Neither the vector store nor the TF-IDF model has ever seen or indexed these examples.
3. **Stratified Selection:**
   - From the 2,500 held-out candidate pool, 250 candidate tweets were initially screened across the 8 intent buckets using high-precision lexical and semantic filters.
   - 200 final examples were selected and individually verified, ensuring representation for every class in the taxonomy.

---

## 2. Intent Distribution in Golden Set (N = 200)

| Intent Category | Count | Proportion |
| :--- | :--- | :--- |
| software_update_os | 57 | 28.5% |
| attery_power_charging | 37 | 18.5% |
| ccount_appleid_icloud | 26 | 13.0% |
| illing_subscription_store | 20 | 10.0% |
| 
etwork_connectivity | 20 | 10.0% |
| hardware_display_audio | 19 | 9.5% |
| other | 15 | 7.5% |
| general_complaint_store | 6 | 3.0% |
| **Total** | **200** | **100.0%** |

---

## 3. Annotation Protocol & Operational Rules

### A. Intent Resolution
- **Compound Clauses:** If a customer mentions an OS update and battery drain simultaneously (e.g. \"*Updated to iOS 11 and now my battery is draining 50% per hour*\"), the primary failure symptom was prioritized (attery_power_charging over general software_update_os) when diagnostic action is required.
- **Short Snippets:** Messages with $\le 4$ words lacking technical context (e.g. \"Thanks\", \"Done\") were labeled as other.

### B. Auto-Handle vs. Escalate Ground Truth
- **High Risk / Mandatory Escalation:**
  - Any request involving credit card charges, refund disputes, or AppleCare billing (illing_subscription_store).
  - Any inquiry involving locked Apple IDs, two-factor authentication bypass, or password recovery (ccount_appleid_icloud).
  - Mentions of legal threats, lawsuits, fraud accusations, or severe customer outrage.
  - Physical hardware damage requiring an in-person Genius Bar store appointment.
- **Auto-Handle Eligibility:**
  - Well-defined technical troubleshooting flows: iOS version checks, reboot instructions, network reset settings, battery health diagnostics.
  - Conversational acknowledgments or basic product usage questions.

### C. Reference Reply Direction
- Each example contains a structured 1-sentence reference specification of what an ideal brand response must accomplish (e.g. device model confirmation, step-by-step navigation path, official Apple portal link, or invitation to private DM).

---

## 4. Single-Annotator Disclosure & Limitations

> [!WARNING]
> **Single-Annotator Bias:**
> The entire golden set of 200 examples was reviewed and labeled by a single annotator over approximately 2.5 hours. While deterministic labeling rules were applied consistently, a single annotator introduces systematic bias:
> 1. Ambiguous boundary cases (such as the boundary between a \"store complaint\" vs \"hardware repair delay\") reflect individual interpretation.
> 2. The escalation threshold reflects an intentionally conservative risk-averse stance (recall-oriented on escalations).
> 3. True inter-annotator agreement (e.g. Fleiss' Kappa across multiple independent human labelers) is not measured on this dataset.
