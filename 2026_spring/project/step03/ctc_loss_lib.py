"""A module implementing utilities for sequence losses."""

# pylint: disable=no-member, invalid-name, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"
# Standard imports
import enum

# Third-party imports
import numpy as np
import torch

torch._dynamo.config.capture_scalar_outputs = True

LOG_0 = torch.tensor(np.log(1e-307)).type(torch.float32)
#LOG_0 = torch.tensor(np.log(np.finfo(np.float64).tiny).astype(np.float32))

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """Applies sequence masking.

    This implementation is based on the following website.
    https://discuss.pytorch.org/t/pytorch-equivalent-for-tf-sequence-mask/39036/3

    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    return mask.type(dtype)


def to_blank_augmented_labels(
        inputs: dict, blank_index: int=0, boundary_blanks: bool=True,
        update_non_blank_token_index: bool=True) -> dict:  # yapf: disable
    """Expands the input sequence with blank labels.

    The blank symbol is inserted at the beginning and at the end of the
    sequences, as well as between labels. If boundary_blanks is False, then
    blank labels are not inserted at the beginning and the end of the sequence.

    Args:
        inputs: A dict containing the input sequence.
            SEQ_DATA: A sparse tensor containing ground truth values.
                The shape is (batch_size, sequence_length).
            SEQ_LEN: A tensor of rank one containing the length of each
                ground truth sequence. The shape is (batch_size).
        blank_index:
            An integer for the blank label in the CTC loss.
        boundary_blanks:
            A boolean flag to insert labels at the boundaries of the sequence.
        unpdate_non_blank_token_index:
            A boolean flag to update non-blank token indices.
                When the blank token index is added, we may need to update the
                indices of non-blank tokens to make a room for the blank token
                index to avoid having conflicting indices. If this issue has
                been already taken care of, then set this flag to False. In
                fine tuning the Wav2Vec2.0 huggingface model, this flag needs
                to be set False.
    Returns:
        A dictionary containing a blank augmented sequence.
    """
    assert isinstance(inputs, dict)
    assert {"SEQ_DATA", "SEQ_LEN"} <= inputs.keys()

    # If some values are larger than blank_index, then those values are added
    # by one to make a room for the blank index.
    ids = torch.where(inputs["SEQ_DATA"] >= blank_index)
    updated_data = inputs["SEQ_DATA"].clone().detach()
    if update_non_blank_token_index:
        updated_data[ids] = inputs["SEQ_DATA"][ids] + 1

    output = {}
    # Creates a tensor filled with blank values.
    blank_tensor = torch.full(inputs["SEQ_DATA"].shape, fill_value=blank_index)

    # updated_data is interleaved with the blank tensor using "stacking" and
    # "reshaping".
    if boundary_blanks:
        data = torch.stack((blank_tensor, updated_data), axis=2)
        data = torch.reshape(data, (updated_data.shape[0], -1))

        # Concatenates a zero at the end of the sequence.
        padded = torch.full((updated_data.shape[0], 1), fill_value=blank_index)
        data = torch.concat((data, padded), axis=1)

        # If boundary_blanks are not used, then the length is 2 * L + 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] + 1
    else:
        data = torch.stack((updated_data, blank_tensor), axis=2)
        data = torch.reshape(data, (updated_data.shape[0], -1))
        data = data[:, :-1]

        # If boundary_blanks are not used, then the length is 2 * L - 1.
        output["SEQ_LEN"] = 2 * inputs["SEQ_LEN"] - 1

    mask = sequence_mask(output["SEQ_LEN"],
                         maxlen=data.shape[1],
                         dtype=data.dtype)
    output["SEQ_DATA"] = data * mask


    return output


def label_trans_allowance_table(labels, labels_len):
    """Constructs a table containing the label transition allowance flags.

    We assume that label_seq should contain "blank labels" described in the
    original CTC paper.
    The shape of the returned tensor is (batch_size, max_seq_len, max_seq_len).
    The transition rule is as follows:

    Depending on whether the transition from the i-th label to the j-th label
    in the label sequence is allowed,
      a(b, i, j) = 0,         if this transition is allowed.
      a[b, i, j] = LOG_0:     if this transition is not allowed.

    Args:
        label_seq: A dictionary containing a batch of label sequences.
            * "DATA": A tensor containing label sequences.
                The shape is (batch_size, max_seq_length). Note that the data
                should follow the blank label rule, which states that "blank"
                labels should be interleaved with real labels. In addition to
                this, blank symbols are prepended and appended to the sequence.
            * "SEQ_LEN": A tensor containing the length of each label sequence.
                The shape is (batch_size).
    Returns:
        A tensor containing flags whether transitions are allowed.
            The shape is (batch_size, max_label_seq_len, max_seq_len)
    """
    max_seq_len = torch.max(labels_len)
    l = torch.arange(max_seq_len, dtype=torch.int32)

    # Indices corresponding to i -> i.
    indices0 = torch.stack([l, l], axis=1)

    # Indices corresponding to i -> i + 1.
    indices1 = torch.stack([l[:-1], l[:-1] + 1], axis=1)

    # Indices corresponding to i -> i + 2.
    indices2 = torch.stack([l[:-2], l[:-2] + 2], axis=1)

    # Constructs the transition table.
    indices = torch.concat([indices0, indices1, indices2], axis=0)
    values = torch.zeros([indices.shape[0]])

    trans_table = torch.full(size=(max_seq_len, max_seq_len), fill_value=LOG_0)
    trans_table[torch.unbind(indices, axis=1)] = 0

    batch_size = labels.shape[0]
    trans_table = torch.tile(torch.unsqueeze(trans_table, axis=0),
                             [batch_size, 1, 1])

    # Detects repeats and blank to blank transitions.
    #
    # These cases can be detected by checking whether y[l] == y[l + 2].
    indices = torch.where(labels[:, :-2] == labels[:, 2:])
    indices = [indices[0], indices[1], indices[1] + 2]
    trans_table[indices] = LOG_0

    return trans_table


def calculate_log_label_prob(labels, softmax_output):
    """Calculates log({\hat{y}_t}_{c_l}).

    This calculates the log probability of each label in the label sequence
    c_l 0 <= l <= L-1 predicted by the model at time t. The returned value is
    a three-dimensional tensor, where value is stored in (b, t, l) where b is
    the batch index, t is the time index, and l ls the label sequence index.

    Args:
        labels: A tensor containing a batch of ground-truth label sequences.
            Note that this label sequence should already include blank labels.
            The shape is given by (batch_size, max_labels_len).
        softmax_output: The output of the model.
            The shape is given by:
            (batch_size, max_seq_len, num_classes).

    Returns:
        The shape is (batch, max_logit_len, max_labels_len).
    """
    max_logit_len = softmax_output.shape[1]
    labels = torch.tile(torch.unsqueeze(labels, dim=1), (1, max_logit_len, 1))

    return torch.log(torch.gather(input=softmax_output, dim=2, index=labels))


def _calculate_unnormalized_log_seq_prob(log_alpha, accum_log_seq_prob_sum,
                                         logit_len, target_len):
    # In alpha calculation, the log probabilty is normalized to prevent
    # over-flowing and under-flowing. This effect is compensated here.
    # log_p_ctc = log
    batch_size = log_alpha.shape[0]
    batch_index = torch.arange(batch_size, dtype=torch.int32)

    final_log_alpha0 = log_alpha[batch_index, logit_len - 1, target_len - 1]
    final_log_alpha1 = log_alpha[batch_index, logit_len - 1, target_len - 2]

    # max(alpha_{T-1,L-1}, alpha_{T-1,L})
    #
    # TODO(chanwcom)
    # There is an issue with the following statement.
    # It should be  addition rather than max.
    # alpha_{T-1,L-2}, alpha_{T-1,L-1}
    # log(Exp(log_alpha_{T-1, L-2}) + Exp(log_alpha_{T-1, L-1}))
    final_log_alpha = torch.max(final_log_alpha0, final_log_alpha1)

    # Finds the accumulated log seq probability at the last time index.
    final_accum = accum_log_seq_prob_sum[batch_index, logit_len - 1]

    return final_log_alpha + final_accum


def calculate_alpha_beta(label_trans_table, log_target_probs, target_lens,
                         logit_lens):
    """Calculates the alpha and beta variables required for CTC computation.

    Note that the definition of beta variable is somewhat different from the
    original CTC paper. This equation will be explained in my future paper.
    TODO(chanwcom): Adds the paper link.

    Args:
        label_trans_table: A tensor containing the transition tables. The shape
          is (batch_size, max_target_len, max_target_len).
        log_target_probs: A tensor of log posterior probabilities of each
          target token. The shape is (batch_size, max_logit_len,
          max_target_len). Mathematically, it is given by: log (p_{[m]}(y_l |
          x)).
        target_lens: A tensor containing the target lengths. The shape is
          (batch_size).
        logit_lens: A tensor containing the logit lengths. The shape is
          (batch_size).

    Returns:
        log_alpha: The calculated forward variable tensor.
        log_beta: The calculated backward variable tensor.
        log_seq_prob_final: The final unnormalized sequence log probabilities.
    """
    batch_size = log_target_probs.shape[0]
    device = log_target_probs.device
    dtype = log_target_probs.dtype
    max_target_len = torch.max(target_lens)
    max_logit_len = torch.max(logit_lens)

    # Initialize log_alpha and log_beta arrays.
    log_alpha = torch.full((batch_size, max_logit_len, max_target_len),
                           fill_value=LOG_0,
                           device=device,
                           dtype=dtype)
    log_beta = torch.full((batch_size, max_logit_len, max_target_len),
                          fill_value=LOG_0,
                          device=device,
                          dtype=dtype)

    # Initialize alpha variable at the first step without one_hot ops.
    prev_log_alpha = torch.full((batch_size, max_target_len),
                                fill_value=LOG_0,
                                device=device,
                                dtype=dtype)
    prev_log_alpha[:, 0] = 0.0

    accum_log_alpha_max = torch.zeros((batch_size, max_logit_len),
                                      device=device,
                                      dtype=dtype)
    prev_log_alpha_max = torch.zeros((batch_size,), device=device, dtype=dtype)

    # Forward Pass: Calculates log_alpha recursively.
    total_logit_timesteps = log_target_probs.shape[1]
    for t in range(total_logit_timesteps):
        # Trans table broadcast: [B, L, 1] + [B, L, L] -> [B, L, L]
        log_alpha[:, t, :] = (
            torch.logsumexp(
                prev_log_alpha.unsqueeze(2) + label_trans_table, dim=1) +
            log_target_probs[:, t, :])

        # Normalize the log sequence prob to prevent underflow.
        log_alpha_max = torch.max(
            log_alpha[:, t, :], dim=1, keepdim=True).values
        log_alpha[:, t, :] -= log_alpha_max

        # Accumulate the maximum scaling factors.
        accum_log_alpha_max[:, t] = (
            prev_log_alpha_max + log_alpha_max.squeeze(-1))
        prev_log_alpha_max = accum_log_alpha_max[:, t]
        prev_log_alpha = log_alpha[:, t, :]

    # Initialize beta variable using target length info.
    initial_log_beta = torch.full((batch_size, max_target_len),
                                  fill_value=LOG_0,
                                  device=device,
                                  dtype=dtype)
    batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)
    target_indices = (target_lens - 1).to(torch.long)
    initial_log_beta[(batch_indices, target_indices)] = 0.0
    prev_log_beta = initial_log_beta

    # Prepare time mask for padding frames. Shape: [B, T, 1]
    time_mask = sequence_mask(
        logit_lens, maxlen=max_logit_len, dtype=dtype).unsqueeze(2)

    # Backward Pass: Calculates log_beta recursively.
    next_log_target_probs = torch.zeros((batch_size, max_target_len),
                                        device=device,
                                        dtype=dtype)
    for t in range(total_logit_timesteps - 1, -1, -1):
        log_beta[:, t, :] = torch.logsumexp(
            (prev_log_beta + next_log_target_probs).unsqueeze(1) +
            label_trans_table,
            dim=2)

        next_log_target_probs = log_target_probs[:, t, :]

        # Normalize the log beta prob.
        log_beta_max = torch.max(
            log_beta[:, t, :], dim=1, keepdim=True).values
        log_beta[:, t, :] -= log_beta_max

        # Reset log_beta for padded frames and re-initialize boundary.
        current_mask = time_mask[:, t, :]
        log_beta[:, t, :] = ((log_beta[:, t, :] * current_mask) +
                             (initial_log_beta * (1.0 - current_mask)))
        prev_log_beta = log_beta[:, t, :]

    # Final Sequence Masking: Vectorized masking outside the loop.
    label_mask = sequence_mask(
        target_lens, maxlen=max_target_len, dtype=dtype).unsqueeze(1)

    final_valid_mask = time_mask * label_mask  # Shape: [B, T, L]
    log_alpha = torch.where(final_valid_mask == 1.0, log_alpha, LOG_0)
    log_beta = torch.where(final_valid_mask == 1.0, log_beta, LOG_0)

    # Calculate final unnormalized sequence probability.
    log_seq_prob_final = _calculate_unnormalized_log_seq_prob(
        log_alpha, accum_log_alpha_max, logit_lens, target_lens)

    return log_alpha, log_beta, log_seq_prob_final
