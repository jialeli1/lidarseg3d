import time

from .hook import Hook


class IterTimerHook(Hook):
    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        runner.log_buffer.update({"data_time": time.time() - self.t})

    def after_iter(self, runner):
        runner.log_buffer.update({"time": time.time() - self.t})
        self.t = time.time()

    def after_data_to_device(self, runner):
        runner.log_buffer.update({"transfer_time": time.time() - self.t})

    def after_forward(self, runner):
        runner.log_buffer.update({"forward_time": time.time() - self.t})

    def after_parse_loss(self, runner):
        runner.log_buffer.update({"loss_parse_time": time.time() - self.t})
        self.t1 = time.time()

    def after_grad_bp(self, runner):
        # 因为after_iter发生在
        # 这里的计时起点是调用after_parse_loss的结束时刻
        runner.log_buffer.update({"backward_time": time.time() - self.t1})
