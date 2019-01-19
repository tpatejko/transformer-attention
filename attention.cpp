#include <iostream>
#include <random>
#include <algorithm>
#include <memory>
#include <functional>
#include <numeric>
#include <chrono>

#include <gflags/gflags.h>

#include "ref_attention.hpp"
#include "opt_attention.hpp"

DEFINE_uint64(iterations, 100, "Number of repetitions");
DEFINE_uint64(batch_size, 32, "Batch size");
DEFINE_uint64(max_seq_len, 256, "Max sequence length");
DEFINE_uint64(n_head, 8, "Number of heads");
DEFINE_uint64(d_key, 64, "Number of keys");
DEFINE_uint64(d_value, 64, "Number of values");
DEFINE_bool(validate, false, "Validate correctness");

constexpr float threshold = 1e-7;

using millisec = std::chrono::duration<double, std::milli>;
using experiment_time = millisec;

template<typename func_t, typename... arg_ts>
experiment_time run_single(func_t f, arg_ts&&... args) {
  auto s = std::chrono::high_resolution_clock::now();
  f(std::forward<arg_ts>(args)...);
  auto e = std::chrono::high_resolution_clock::now();

  return e - s;
}

template<typename func_t, typename... arg_ts>
std::vector<experiment_time> run_iterations(size_t iterations, func_t func, arg_ts&&... args) {
  std::vector<experiment_time> times;

  for (size_t i = 0; i < iterations; i++) {
    auto t = run_single(func, std::forward<arg_ts>(args)...);
    times.push_back(t);
  }

  return times;
}

experiment_time total_time(const std::vector<experiment_time>& measurements) {
  return std::accumulate(std::begin(measurements),
                         std::end(measurements),
                         experiment_time::zero(),
                         std::plus<experiment_time>());
}

experiment_time average_time(const std::vector<experiment_time>& measurements) {
  auto total = total_time(measurements);
  return total / measurements.size();
}

template<typename func_t, typename... arg_ts>
experiment_time measure_average(size_t iterations, func_t f, arg_ts&&... args) {
  return average_time(run_iterations(iterations, f, std::forward<arg_ts>(args)...));
}

bool are_same(float a, float b) {
  return std::fabs(a - b) <= threshold;
}

void validate(const tensor& q, const tensor& k, const tensor& v,
              size_t batch_size, size_t max_seq_len, size_t n_head,
              size_t d_model, size_t d_key) {
  auto combined_heads = ref_attention_module(q, k, v, batch_size, max_seq_len, n_head, d_model, d_key);
  auto opt_combined_heads = opt_attention_module(q, k, v, batch_size, max_seq_len, n_head, d_model, d_key);

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
  std::cout << "Correct\n";
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto iterations = FLAGS_iterations;
  auto batch_size = FLAGS_batch_size;
  auto max_seq_len = FLAGS_max_seq_len;
  auto n_head = FLAGS_n_head;
  auto d_key = FLAGS_d_key;
  auto d_value = FLAGS_d_value;
  size_t d_model = n_head * d_key;

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

  {
    std::cout << "Average reference time: " << measure_average(iterations, ref_attention_module, q, k, v, batch_size, max_seq_len, n_head, d_model, d_key).count() << " milliseconds" << std::endl;
    std::cout << "Average mkldnn reference time: " << measure_average(iterations, ref_mkldnn_attention_module, q, k, v, batch_size, max_seq_len, n_head, d_model, d_key).count() << " milliseconds" << std::endl;

    std::cout << "Average optimized time: " << measure_average(iterations, opt_attention_module, q, k, v, batch_size, max_seq_len, n_head, d_model, d_key).count() << " milliseconds" << std::endl;
  }

  {
    if (FLAGS_validate) {
      validate(q, k, v, batch_size, max_seq_len, n_head, d_model, d_key);
    }
  }

  return 0;
}
