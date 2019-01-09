#pragma once

#include "operators.hpp"

class qk_matmul_op {
 public:
  tensor operator()(const tensor& a, const tensor& b,
                    size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
    tensor c({max_seq_len, n_head*max_seq_len});

    for (int i = 0; i < n_head; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  max_seq_len, max_seq_len, d_key,
                  1.0,
                  a.ptr() + i*d_key, n_head*d_key,
                  b.ptr() + i*d_key, n_head*d_key,
                  0.0,
                  c.ptr() + i*max_seq_len, n_head*max_seq_len);
    }

    return c;
  }
};

class qkv_matmul_op {
 public:
  tensor operator()(const tensor& a, const tensor& b,
                    size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
    tensor c({max_seq_len, n_head*d_key});

    for (int i = 0; i < n_head; i++) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  max_seq_len, d_key, max_seq_len,
                  1.0,
                  a.ptr() + i*max_seq_len, n_head*max_seq_len,
                  b.ptr() + i*d_key, n_head*d_key,
                  0.0,
                  c.ptr() + i*d_key, n_head*d_key);
    }

    return c;
  }

};


tensor opt_attention_module(const tensor& q, const tensor& k, const tensor& v,
                            size_t max_seq_len, size_t n_head, size_t d_model, size_t d_key) {
  tensor scaled_q = scale(q, 1 / std::sqrt(d_key));
  tensor qk = matmul<qk_matmul_op>(scaled_q, k, max_seq_len, n_head, d_model, d_key);
  tensor combined_heads = matmul<qkv_matmul_op>(qk, v, max_seq_len, n_head, d_model, d_key);
  return combined_heads;
}
