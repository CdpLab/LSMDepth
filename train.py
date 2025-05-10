from __future__ import absolute_import, division, print_function

from options import LightOptions
from trainer import Trainer

options = LightOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()