import accelerate


class MyAccelerator(accelerate.Accelerator):
    """Thin wrapper for accelerate.Accelerator."""

    def __repr__(self):
        return (
            f"Accelerator(\n"
            f"  state={self.state}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps:.6f}, \n"
            f"  split_batches={self.split_batches}, \n"
            f"  step_scheduler_with_optimizer={self.step_scheduler_with_optimizer},\n"
            f")"
        )

    def unwrap_optimizer(self, optimizer: accelerate.accelerator.AcceleratedOptimizer):
        return optimizer.optimizer
