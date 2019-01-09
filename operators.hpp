#pragma once

#include <algorithm>
#include <numeric>
#include <iterator>

#include <unsupported/Eigen/CXX11/Tensor>

#include "tensor.hpp"
#include "matmul.hpp"

tensor reshape(const tensor& src, const std::vector<size_t>& dims) {
  tensor dst(dims);

  if (src.size() != dst.size()) {
    std::cout << "Incorrect sizes" << std::endl;
  } else {
    std::copy_n(src.ptr(), src.size(), dst.ptr());
  }

  return dst;
}

template<size_t rank>
void shuffle_for_rank(const tensor& src, tensor& dst, const std::vector<size_t>& orders) {
  std::array<size_t, rank> src_dims;
  std::array<size_t, rank> dst_dims;

  std::copy_n(std::begin(src.dims()), rank, std::begin(src_dims));
  std::copy_n(std::begin(dst.dims()), rank, std::begin(dst_dims));

  Eigen::TensorMap<Eigen::Tensor<float, rank, Eigen::RowMajor, size_t>> eigen_src{src.ptr(), src_dims};
  Eigen::TensorMap<Eigen::Tensor<float, rank, Eigen::RowMajor, size_t>> eigen_dst{dst.ptr(), dst_dims};

  eigen_dst = eigen_src.shuffle(orders);
}

tensor transpose(const tensor& src, const std::vector<size_t>& orders) {
  std::vector<size_t> dst_dims(orders.size());

  std::transform(std::begin(orders), std::end(orders), std::begin(dst_dims),
      [&src](auto o) -> size_t {
        return src.dims()[o];
      });

  tensor dst{dst_dims};

  switch (dst.rank()) {
    case 1:
      shuffle_for_rank<1>(src, dst, orders);
      break;
    case 2:
      shuffle_for_rank<2>(src, dst, orders);
      break;
    case 3:
      shuffle_for_rank<3>(src, dst, orders);
      break;
    case 4:
      shuffle_for_rank<4>(src, dst, orders);
      break;
  }

  return dst;
}

tensor split_heads(const tensor& src, const std::vector<size_t>& dst_dims) {
  return transpose(reshape(src, dst_dims), {0, 2, 1, 3});
}

tensor combine_heads(const tensor& src) {
  auto t = transpose(src, {0, 2, 1, 3});
  return reshape(t, {t.shape(0), t.shape(1), t.shape(2) * t.shape(3)});
}

tensor scale(const tensor& src, float scale) {
  tensor dst{src.dims()};

  auto src_ptr = src.ptr();
  auto dst_ptr = dst.ptr();

  for (size_t i = 0; i < src.size(); i++) {
    dst_ptr[i] = scale * src_ptr[i];
  }

  return dst;
}

class softmax_op {
 public:
  tensor operator()(const tensor& x) {
    tensor y{x.dims()};

    auto axis = x.rank() - 1;
    auto x_dims = x.dims();

    size_t first_dim = std::accumulate(std::begin(x_dims), std::next(std::begin(x_dims), axis), 1, std::multiplies<size_t>());
    size_t second_dim = std::accumulate(std::next(std::begin(x_dims), axis), std::end(x_dims), 1, std::multiplies<size_t>());

    size_t batch_dim = 0;
    size_t class_dim = 1;

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, size_t>> logits{x.ptr(), {first_dim, second_dim}};
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, size_t>> softmax{y.ptr(), {first_dim, second_dim}};

    auto batch_size = logits.dimension(batch_dim);
    auto num_classes = logits.dimension(class_dim);

    Eigen::DSizes<size_t, 1> along_class{class_dim};
    Eigen::DSizes<size_t, 2> batch_by_one{batch_size, 1};
    Eigen::DSizes<size_t, 2> one_by_class{1, num_classes};

    auto value_clip = [](const float& x) -> float {
      float threshold = -64.0;
      return x < threshold ? threshold : x;
    };

    auto shifted_logits = (logits - logits.maximum(along_class)
        .eval()
        .reshape(batch_by_one)
        .broadcast(one_by_class))
        .unaryExpr(value_clip);

    softmax = shifted_logits.exp();
    softmax = (softmax * softmax.sum(along_class)
                                .inverse()
                                .eval()
                                .reshape(batch_by_one)
                                .broadcast(one_by_class));

    return y;
  }
};

tensor softmax(const tensor& x) {
  softmax_op s;
  return s(x);
}

template<size_t rank>
void dropout_for_rank(const tensor& x, tensor& y, float dropout_prob) {
  std::array<size_t, rank> x_dims;
  std::array<size_t, rank> y_dims;

  std::copy_n(std::begin(x.dims()), rank, std::begin(x_dims));
  std::copy_n(std::begin(y.dims()), rank, std::begin(y_dims));

  Eigen::TensorMap<Eigen::Tensor<float, rank, Eigen::RowMajor, size_t>> eigen_x{x.ptr(), x_dims};
  Eigen::TensorMap<Eigen::Tensor<float, rank, Eigen::RowMajor, size_t>> eigen_y{y.ptr(), y_dims};

  eigen_y = eigen_x * (1.0f - dropout_prob);
}

tensor dropout(const tensor& x, float dropout_prob) {
  tensor y{x.dims()};

  switch (x.rank()) {
    case 1:
      dropout_for_rank<1>(x, y, dropout_prob);
      break;
    case 2:
      dropout_for_rank<2>(x, y, dropout_prob);
      break;
    case 3:
      dropout_for_rank<3>(x, y, dropout_prob);
      break;
    case 4:
      dropout_for_rank<4>(x, y, dropout_prob);
      break;
  };

  return y;
}
