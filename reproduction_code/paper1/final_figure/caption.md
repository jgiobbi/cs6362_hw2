**Figure 1 reproduction (PyTorch).** (a) Training loss of the small Inception on CIFAR10: true labels, random labels and shuffled pixels
(single runs from the Phase 1 baseline), and random-pixel and Gaussian inputs with true labels (mean over 2 seeds, 250-step bins).
Random pixels are shown for two permutation variants: the same permutation across channels (colour statistics survive) and a permutation of the flattened
C*H*W vector. (b) Time to overfit, defined as the first epoch with train accuracy >= 99%, relative to the same seed's p=0 run. (c) Test error at the epoch where
train error <= 0.1% and online loss < 0.1. (b),(c): mean +/- sample std over 5 seeds (Inception: decay 0.95; AlexNet, MLP: decay 0.99; PyTorch default init;
SGD momentum 0.9, lr 0.1 / 0.01, batch 128, no weight decay, dropout or augmentation). Hollow markers are single-seed round-1 runs at p=0.2, 0.6.
Dashed lines are the paper's curves read by eye from its Figure 1 (about +-0.02 in (c), about +-0.1 in (b)); they are for visual reference only.
