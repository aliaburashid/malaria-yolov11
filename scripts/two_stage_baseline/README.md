# Two-stage baseline (Option A)

Scripts for the two-stage pipeline: **YOLO (detector only) → crop → CNN (classifier)**. Numbered steps keep the order obvious.

- **Step 1:** Check 27k cell dataset — `step1_check_cell_images.py`
- **Step 2:** Train Stage-2 classifier on 27k cells — `step2_train_classifier_27k.py` (to add)
- **Step 3:** Two-stage inference (detect → crop → classify) — `step3_two_stage_inference.py` (to add)
- **Step 4:** Evaluate end-to-end vs two-stage — `step4_evaluate_two_stage.py` (to add)

See `docs/OPTION_A_TWO_STAGE.md` for the full guide.

Run from **project root**, e.g.:
`python3 scripts/two_stage_baseline/step1_check_cell_images.py`
