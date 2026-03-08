# Two-stage baseline scripts

| Script | Purpose |
|--------|---------|
| step1_check_cell_images.py | Check 27k cell dataset is present |
| step2_train_classifier_27k.py | Train classifier on 27k cells |
| step2b_finetune_classifier_thinsmear.py | Fine-tune classifier on thin-smear GT crops |
| step3_two_stage_inference.py | YOLO detect → crop → CNN classify |
| step4_evaluate_two_stage.py | Evaluate two-stage pipeline metrics |
