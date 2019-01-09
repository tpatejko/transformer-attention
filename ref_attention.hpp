#pragma once

#include "operators.hpp"

tensor ref_attention_module(const tensor& q, const tensor& k, const tensor& v,
                            size_t batch_size, size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
  tensor split_q = split_heads(q, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_k = split_heads(k, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_v = split_heads(v, {batch_size, max_seq_len, n_head, d_model/n_head});

  tensor scaled_q = scale(split_q, 1 / std::sqrt(d_key));
  tensor qk = matmul<matmul_op>(scaled_q, split_k, false, true);
  //tensor softmaxed_qk = softmax(qk);
  tensor multiheads = matmul<matmul_op>(qk, split_v, false, false);

  return combine_heads(multiheads);
}
