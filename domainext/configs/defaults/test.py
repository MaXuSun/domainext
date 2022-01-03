from yacs.config import CfgNode as CN

TEST = CN()
TEST.EVALUATOR = "Classification"
TEST.PER_CLASS_RESULT = False
TEST.COMPUTE_CMAT = False            # Compute confusion matrix, which will be saved to $OUTPUT_DIR/cmat.pt
TEST.NO_TEST = False                 # If NO_TEST=True, no testing will be conducted
TEST.SPLIT = "test"                  # Use test or val set for FINAL evaluation
TEST.FINAL_MODEL = "last_step"       # Which model to test after training. Either last_step or best_val