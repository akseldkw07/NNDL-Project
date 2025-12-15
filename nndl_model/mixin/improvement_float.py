from tqdm.contrib.logging import logging_redirect_tqdm

from .abc_nn import ABCNN

# Dictionary mapping metric names to whether improvement is indicated by a smaller value
IMPROVE_IS_LT = {"eval_loss"}


class CheckImprovementFloatMixin(ABCNN):
    def _improved(self, eval: dict[str, float]):
        min_improvement = self.training_config.improvement_tol
        ret = {}

        for key, val in eval.items():
            if key in IMPROVE_IS_LT:
                ret[key] = val < (self.eval_best[f"best_{key}"] - min_improvement)
            else:
                ret[key] = val > (self.eval_best[f"best_{key}"] + min_improvement)

        return ret

    def _on_improvement(self, improved: dict[str, bool], eval: dict[str, float], epochs_no_improve: int) -> int:
        if not any(improved.values()):
            return epochs_no_improve + 1

        for key, val in improved.items():
            if val:
                self.eval_best[f"best_{key}"] = eval[key]

        if self.training_config.save_on_improvement:
            self.save_weights()

        if self.training_config.log_on_improvement:
            # Create concise improvement message
            improved_metrics = {k: f"{eval[k]:.4f}" for k, v in improved.items() if v}
            with logging_redirect_tqdm([self.logger]):
                self.logger.info(f"✓ Improvement: {improved_metrics} - counter reset")
        return 0
