#include <iostream>
#include <random>
#include <algorithm>
#include <memory>
#include <functional>
#include <numeric>

#include "tensor.hpp"
#include "operators.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

//constexpr size_t batch_size = 32;
constexpr size_t max_seq_len = 12;
constexpr size_t n_head = 8;
constexpr size_t d_key = 64;
constexpr size_t d_value = 64;
constexpr size_t d_model = n_head * d_key;
constexpr float threshold = 1e-7;

tensor ref_attention_module(const tensor& q, const tensor& k, const tensor& v) {
  tensor split_q = split_heads(q, {max_seq_len, n_head, d_model/n_head});
  tensor split_k = split_heads(k, {max_seq_len, n_head, d_model/n_head});
  tensor split_v = split_heads(v, {max_seq_len, n_head, d_model/n_head});

  tensor scaled_q = scale(split_q, 1 / std::sqrt(d_key));
  tensor qk = matmul<matmul_op>(scaled_q, split_k, false, true);
  //tensor softmaxed_qk = softmax(qk);
  tensor multiheads = matmul<matmul_op>(qk, split_v, false, false);

  return combine_heads(multiheads);
}

tensor new_attention_module(const tensor& q, const tensor& k, const tensor& v) {
  tensor scaled_q = scale(q, 1 / std::sqrt(d_key));
  tensor qk = matmul<new_matmul1_op>(scaled_q, k, false, true);
  tensor combined_heads = matmul<new_matmul2_op>(qk, v, false, false);
  return combined_heads;
}

bool are_same(float a, float b) {
    return std::fabs(a - b) <= threshold;
}

int main() {
    std::random_device rd;
    std::mt19937 e{rd()};
    std::uniform_real_distribution<float> dist{0, 1};
  
    tensor q({max_seq_len, d_model});
    tensor k({max_seq_len, d_model});
    tensor v({max_seq_len, d_model});

    for (int i = 0; i < q.size(); i++) {
        q.ptr()[i] = dist(e);
    }

    for (int i = 0; i < k.size(); i++) {
        k.ptr()[i] = dist(e);
    }

    for (int i = 0; i < v.size(); i++) {
        v.ptr()[i] = dist(e);
    }

    auto combined_heads = ref_attention_module(q, k, v);
    auto new_combined_heads = new_attention_module(q, k, v);

    for (size_t m = 0; m < max_seq_len; m++) {
        for (size_t d = 0; d < d_model; d++) {
            auto combined_head = combined_heads.value({m, d});
            if (are_same(q.value({m, d}), combined_head)) {
                std::cout << "Incorrect with q\n";
                std::cout << q.value({m, d}) << " " << combined_head << std::endl;
                std::terminate();
            } else if (are_same(k.value({m, d}), combined_head)) {
                std::cout << "Incorrect with k\n";
                std::terminate();
            } else if (are_same(v.value({m, d}), combined_head)) {
                std::cout << "Incorrect with v\n";
                std::cout << v.value({m, d}) << " " << combined_head << std::endl;
                std::terminate();
            }
        }
    }

  if (combined_heads.dims().size() != new_combined_heads.dims().size()) {
    std::cout << "Sizes incorrect\n";
    std::terminate();
  }

  if (combined_heads.dims() != new_combined_heads.dims()) {
    std::cout << "Dims incorrect\n";
    std::copy(std::begin(combined_heads.dims()), std::end(combined_heads.dims()),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << "\n";
    std::copy(std::begin(new_combined_heads.dims()), std::end(new_combined_heads.dims()),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::terminate();
  }

  for (size_t m = 0; m < max_seq_len; m++) {
    for (size_t d = 0; d < d_model; d++) {
      auto c = combined_heads.value({m, d});
      auto n = new_combined_heads.value({m, d});

      if (!std::isnormal(c) || !are_same(c, n)) {
        std::cout << "Error: " << m << " " << d << " "
                  << c << " " << n
                  << std::endl;
        std::terminate();
      }
    }
  }
  std::cout << "Done\n";
  
  return 0;
}
