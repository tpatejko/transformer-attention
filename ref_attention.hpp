#pragma once

#include "operators.hpp"

tensor split_heads(const tensor& src, const std::vector<size_t>& dst_dims) {
  return transpose(reshape(src, dst_dims), {0, 2, 1, 3});
}

tensor combine_heads(const tensor& src) {
  auto t = transpose(src, {0, 2, 1, 3});
  return reshape(t, {t.shape(0), t.shape(1), t.shape(2) * t.shape(3)});
}

tensor split_mkldnn_heads(const tensor& src, const std::vector<size_t>& dst_dims) {
  return transpose_mkldnn(reshape(src, dst_dims), {0, 2, 1, 3});
}

tensor combine_mkldnn_heads(const tensor& src) {
  auto t = transpose_mkldnn(src, {0, 2, 1, 3});
  return reshape(t, {t.shape(0), t.shape(1), t.shape(2) * t.shape(3)});
}

tensor scaled_dot_product_attention(const tensor& q, const tensor& k, const tensor& v,
                                    size_t d_key) {
  tensor scaled_q = scale(q, 1 / std::sqrt(d_key));
  tensor qk = matmul<matmul_op>(scaled_q, k, false, true);
  tensor softmax_qk = softmax(qk);
  tensor dropout_qk = dropout(softmax_qk, 0.5);
  tensor multiheads = matmul<matmul_op>(dropout_qk, v, false, false);

  return multiheads;
}

tensor ref_attention_module(const tensor& q, const tensor& k, const tensor& v,
                            size_t batch_size, size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
  tensor split_q = split_heads(q, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_k = split_heads(k, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_v = split_heads(v, {batch_size, max_seq_len, n_head, d_model/n_head});

  tensor multiheads = scaled_dot_product_attention(split_q, split_k, split_v, d_key);

  return combine_heads(multiheads);
}

tensor ref_mkldnn_attention_module(const tensor& q, const tensor& k, const tensor& v,
                                   size_t batch_size, size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
  tensor split_q = split_mkldnn_heads(q, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_k = split_mkldnn_heads(k, {batch_size, max_seq_len, n_head, d_model/n_head});
  tensor split_v = split_mkldnn_heads(v, {batch_size, max_seq_len, n_head, d_model/n_head});

  tensor multiheads = scaled_dot_product_attention(split_q, split_k, split_v, d_key);
  
  return combine_mkldnn_heads(multiheads);
}
