import os
import time

from transformers import TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback


class QuietProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


def configure_training_progress(trainer):
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    if not trainer.args.disable_tqdm:
        trainer.add_callback(QuietProgressCallback())


class MilestoneCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps, adapter_dir):
        self.save_steps = save_steps
        self.adapter_dir = adapter_dir

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control

        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            checkpoint_dir = os.path.join(
                self.adapter_dir, f"milestone-{state.global_step}"
            )
            max_retries = 5
            retry_delay = 60

            for attempt in range(1, max_retries + 1):
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(checkpoint_dir)
                    break
                except Exception as e:
                    print(f"WARNING: Milestone save attempt {attempt}/{max_retries} "
                          f"failed at step {state.global_step}: {e}")
                    if attempt < max_retries:
                        print(f"  Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        print(f"ERROR: All {max_retries} attempts failed. "
                              f"Stopping training to prevent losing checkpoint at step {state.global_step}.")
                        raise
        return control

