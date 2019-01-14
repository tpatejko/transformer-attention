#pragma once

#include <vector>
#include <mkl_cblas.h>

struct mat_descriptor {
  size_t height;
  size_t width;
  size_t stride{0};
  size_t batch_size{0};
  bool transposed;
};

mat_descriptor create_matrix_descriptor(const std::vector<size_t>& dims, bool transposed) {
  mat_descriptor desc;
  size_t rank = dims.size();

  if (rank == 2) {
    desc.height = dims[0];
    desc.width = dims[1];
  } else {
    desc.batch_size = 1;

    for (size_t i = 0; i < dims.size() - 2; ++i) {
      desc.batch_size *= dims[i];
    }

    desc.width = *dims.rbegin();
    desc.height = *(dims.rbegin() + 1);

    desc.stride = desc.height * desc.width;
  }

  if (transposed) {
    std::swap(desc.height, desc.width);
  }

  desc.transposed = transposed;

  return desc;
}


class matmul_op {
  void sgemm(CBLAS_TRANSPOSE a_transposed, CBLAS_TRANSPOSE b_transposed,
             int batch_size, int m, int n, int k,
             float alpha,
             int a_stride, const float* a, int lda,
             int b_stride, const float* b, int ldb,
             float beta,
             int c_stride, float* c, int ldc) {
    std::vector<const float*> a_arrays(batch_size);
    std::vector<const float*> b_arrays(batch_size);
    std::vector<float*> c_arrays(batch_size);

    for (int i = 0; i < batch_size; i++) {
      a_arrays[i] = a + i * a_stride;
      b_arrays[i] = b + i * b_stride;
      c_arrays[i] = c + i * c_stride;
    }

    cblas_sgemm_batch(CblasRowMajor, &a_transposed, &b_transposed,
          &m, &n, &k,
          &alpha,
          a_arrays.data(), &lda,
          b_arrays.data(), &ldb, &beta,
          c_arrays.data(), &ldc,
          1, &batch_size);
  }

protected:
  using transposes_t = std::tuple<CBLAS_TRANSPOSE, CBLAS_TRANSPOSE>;
  using parameters_t = std::tuple<int, int, int, int, int, int>;

  transposes_t transposed(const mat_descriptor& dims_a,
                          const mat_descriptor& dims_b) {
    CBLAS_TRANSPOSE a_transposed = dims_a.transposed ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE b_transposed = dims_b.transposed ? CblasTrans : CblasNoTrans;

    return std::make_tuple(a_transposed, b_transposed);
  }

  parameters_t parameters(const mat_descriptor& dims_a,
                          const mat_descriptor& dims_b) {
    int m = dims_a.height;
    int n = dims_b.width;
    int k = dims_a.width;

    int lda = dims_a.transposed ? m : k;
    int ldb = dims_b.transposed ? k : n;
    int ldc = n;

    return std::make_tuple(m, n, k, lda, ldb, ldc);
  }

private:
  void matmul(const mat_descriptor& dims_a, const float* a,
        const mat_descriptor& dims_b, const float* b,
        float* c,
        float alpha, float beta) {
    auto [a_transposed, b_transposed] = transposed(dims_a, dims_b);

    auto [m, n, k, lda, ldb, ldc] = parameters(dims_a, dims_b);

    int batch_size = dims_a.batch_size;

    sgemm(a_transposed, b_transposed, batch_size, m, n, k,
          alpha,
          dims_a.stride, a, lda,
          dims_b.stride, b, ldb,
          beta,
          m * n, c, ldc);
  }

 public:
  tensor operator()(const tensor& a, const tensor& b, bool transpose_a, bool transpose_b) {
    auto tensor_rank = a.rank();

    auto a_desc = create_matrix_descriptor(a.dims(), transpose_a);
    auto b_desc = create_matrix_descriptor(b.dims(), transpose_b);

    tensor c({a.dims()[0], a.dims()[1], a_desc.height, b_desc.width});

    matmul(a_desc, a.ptr(), b_desc, b.ptr(), c.ptr(), 1.0, 0.0);

    return c;
  }
};

template<typename matmul_t, typename... arg_ts>
tensor matmul(const tensor& a, const tensor& b, arg_ts... args) {
  matmul_t m;
  if constexpr (sizeof...(arg_ts) != 0) {
    return m(a, b, args...);
  } else {
    return m(a, b);
  }

}
