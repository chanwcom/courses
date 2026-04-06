# HW01

## Please use Python 3 and PyTorch version > 2.4.

### Due Date June 3rd, 2025. Please send your code to chanwcom@korea.ac.kr

To check whether the unit test passes, please run the following command:

```
python ctc_loss_test.py
```
Instructions:
### 1. Implement the **calculate_log_label_prob** method so that the unit test passes.

Note that the log_label_prob is log ((\hat{y}_t))_{c_l}).

### 2. Implement the **calculate_alpha_betacalculate_alpha_beta** method so that the unit test passes.

Avoid using for-loops if possible. However, in calculating the forward-backward variables, we cannot completely remove for-loops.
