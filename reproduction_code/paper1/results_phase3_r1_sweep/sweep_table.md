## Sweep table (slowdowns normalized by each run's own p=0; '>=' = not reached, lower bound)

| setting | model | p0_test_err | paper_p0 | p0_epochs | p1_slow_acc99 | p1_slow_acc99.9 | p1_slow_loss0.1 | p1_hit_cap | p1_train_err | p1_test_err |
|---|---|---|---|---|---|---|---|---|---|---|
| d0.95_default | inception | 0.1290 | 0.1425 | 17 | 3.29 | 2.76 | 4.09 | False | 0.0004 | 0.8975 |
| d0.95_default | alexnet | 0.2268 | 0.2393 | 14 | 2.79 | 3.14 | 3.25 | False | 0.0010 | 0.9009 |
| d0.95_default | mlp | 0.4780 | 0.4949 | 42 | 2.93 | 3.57>= | 5.17>= | True | 0.0079 | 0.9041 |
| d0.95_glorot | inception | 0.1419 | 0.1425 | 14 | 3.00 | 2.86 | 3.80 | False | 0.0002 | 0.8953 |
| d0.95_glorot | alexnet | 0.2375 | 0.2393 | 13 | 2.58 | 2.69 | 2.82 | False | 0.0008 | 0.9021 |
| d0.95_glorot | mlp | 0.4833 | 0.4949 | 38 | 2.72 | 3.95>= | 5.77>= | True | 0.0072 | 0.8972 |
| d0.99_default | alexnet | 0.2319 | 0.2393 | 23 | 2.71 | 2.17 | 3.00 | False | 0.0005 | 0.8996 |
| d0.99_default | mlp | 0.4846 | 0.4949 | 35 | 1.97 | 2.00 | 2.03 | False | 0.0007 | 0.9011 |
| d0.99_glorot | alexnet | 0.2308 | 0.2393 | 24 | 2.94 | 2.17 | 3.42 | False | 0.0005 | 0.8992 |
| d0.99_glorot | mlp | 0.4878 | 0.4949 | 33 | 2.03 | 2.06 | 2.07 | False | 0.0009 | 0.9027 |
| d0.999_default | alexnet | 0.2347 | 0.2393 | 31 | 3.67 | 9.68>= | 3.21 | True | 0.0059 | 0.9036 |
| d0.999_default | mlp | 0.4897 | 0.4949 | 51 | 2.88 | 2.96 | 3.05 | False | 0.0008 | 0.9003 |
| d0.999_glorot | alexnet | 0.2421 | 0.2393 | 40 | 15.79>= | 7.50>= | 5.77 | True | 0.0604 | 0.9032 |
| d0.999_glorot | mlp | 0.4860 | 0.4949 | 47 | 3.09 | 3.02 | 3.27 | False | 0.0010 | 0.8977 |

Paper approx. p=1 slowdown (by eye from Fig. 1b, used only for scoring): {'inception': 3.6, 'alexnet': 1.5, 'mlp': 2.0}

## Selection scores (AlexNet + MLP; lower = closer to paper; arbitrary units, see score_settings)

| rank | setting | score | alexnet_err_term | alexnet_slow_p1 | alexnet_slow_term | mlp_err_term | mlp_slow_p1 | mlp_slow_term |
|---|---|---|---|---|---|---|---|---|
| 1 | d0.99_default | 1.534 | 0.148 | 2.71 | 2.644 | 0.206 | 1.97 | 0.071 |
| 2 | d0.99_glorot | 1.699 | 0.170 | 2.94 | 3.012 | 0.142 | 2.03 | 0.074 |
| 3 | d0.95_glorot | 2.041 | 0.036 | 2.58 | 2.436 | 0.232 | 2.72 | 1.378 |
| 4 | d0.95_default | 2.534 | 0.250 | 2.79 | 2.774 | 0.338 | 2.93 | 1.705 |
| 5 | d0.999_default | 2.918 | 0.092 | 3.67 | 4.006 | 0.104 | 2.88 | 1.634 |
| 6 | d0.999_glorot | 6.367 | 0.056 | 15.79>= | 10.549 | 0.178 | 3.09 | 1.951 |
