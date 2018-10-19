from main import main
from utils import Util

util = Util()
batch_sizes = [32, 46, 128, 256]
learning_rates = [0.001, 0.005, 0.01, 0.1, 1]

for b in batch_sizes:
    util.set_cfg_param('batch_size', b)
    for lr in learning_rates:
        util.set_cfg_param('learning_rate', lr)
        main()