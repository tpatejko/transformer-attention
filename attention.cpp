#include <iostream>
#include <random>
#include <algorithm>
#include <memory>
#include <functional>
#include <numeric>

#include "ref_attention.hpp"
#include "opt_attention.hpp"

constexpr size_t batch_size = 32;
constexpr size_t max_seq_len = 256;
constexpr size_t n_head = 8;
constexpr size_t d_key = 64;
constexpr size_t d_value = 64;
constexpr size_t d_model = n_head * d_key;
constexpr float threshold = 1e-7;

bool are_same(float a, float b) {
  return std::fabs(a - b) <= threshold;
}

int main() {
  std::random_device rd;
  std::mt19937 e{rd()};
  std::uniform_real_distribution<float> dist{0, 1};

  tensor q({batch_size, max_seq_len, d_model});
  tensor k({batch_size, max_seq_len, d_model});
  tensor v({batch_size, max_seq_len, d_model});

  for (int i = 0; i < q.size(); i++) {
    q.ptr()[i] = dist(e);
  }

  for (int i = 0; i < k.size(); i++) {
    k.ptr()[i] = dist(e);
  }

  for (int i = 0; i < v.size(); i++) {
    v.ptr()[i] = dist(e);
  }

  auto combined_heads = ref_attention_module(q, k, v, batch_size, max_seq_len, n_head, d_model, d_key);
  auto opt_combined_heads = opt_attention_module(q, k, v, batch_size, max_seq_len, n_head, d_model, d_key);

/*
  for (size_t b = 0; b < batch_size; b++) {
    for (size_t m = 0; m < max_seq_len; m++) {
      for (size_t d = 0; d < d_model; d++) {
        auto combined_head = combined_heads.value({b, m, d});
        if (are_same(q.value({b, m, d}), combined_head)) {
          std::cout << "Incorrect with q\n";
          std::cout << b << " " << m << " " << d << "\n";
          std::cout << q.value({b, m, d}) << " " << combined_head << std::endl;
          std::terminate();
        } else if (are_same(k.value({b, m, d}), combined_head)) {
          std::cout << "Incorrect with k\n";
          std::cout << b << " " << m << " " << d << "\n";
          std::terminate();
        } else if (are_same(v.value({b, m, d}), combined_head)) {
          std::cout << "Incorrect with v\n";
          std::cout << b << " " << m << " " << d << "\n";
          std::cout << v.value({b, m, d}) << " " << combined_head << std::endl;
          std::terminate();
        }
      }
    }
  }
*/
  if (combined_heads.dims().size() != opt_combined_heads.dims().size()) {
    std::cout << "Sizes incorrect\n";
    std::terminate();
  }

  if (combined_heads.dims() != opt_combined_heads.dims()) {
    std::cout << "Dims incorrect\n";
    std::copy(std::begin(combined_heads.dims()), std::end(combined_heads.dims()),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << "\n";
    std::copy(std::begin(opt_combined_heads.dims()), std::end(opt_combined_heads.dims()),
              std::ostream_iterator<size_t>(std::cout, " "));
    std::terminate();
  }

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t m = 0; m < max_seq_len; m++) {
      for (size_t d = 0; d < d_model; d++) {
        auto c = combined_heads.value({b, m, d});
        auto n = opt_combined_heads.value({b, m, d});

        if (!std::isnormal(c) || !are_same(c, n)) {
          std::cout << "Error: " << m << " " << d << " "
                    << c << " " << n
                    << std::endl;
          std::terminate();
        }
      }
    }
  }
  std::cout << "Done\n";

  return 0;
}
