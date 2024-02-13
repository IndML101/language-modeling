import math


class LRPolicyAdam(object):
    def __init__(self, d_model: int = 128):
        self.d_model = d_model

    def __call__(self, epoch):
        warmup_steps = 10
        return (
            1
            / math.sqrt(self.d_model)
            * min(1 / math.sqrt(epoch + 1), (epoch + 1) / math.pow(warmup_steps, 3 / 2))
        )


class LRPolicyAdam2(object):
    def __init__(self, d_model: int = 128):
        self.d_model = d_model

    def __call__(self, epoch):
        warmup_steps = 10
        return (
            1
            / math.sqrt(self.d_model)
            * min(1 / math.sqrt(epoch + 1), (epoch + 1) / math.pow(warmup_steps, 3 / 2))
        )
