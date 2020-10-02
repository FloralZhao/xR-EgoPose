class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, num=1):
        self.sum += val * num
        self.val = val
        self.count += num
        self.avg = self.sum / self.count if self.count!=0 else 0.0