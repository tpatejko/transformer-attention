#include <iostream>
#include <random>
#include <algorithm>
#include <memory>
#include <functional>
#include <numeric>

#include <unsupported/Eigen/CXX11/Tensor>

constexpr size_t batch_size = 32;
constexpr size_t max_seq_len = 256;
constexpr size_t n_head = 8;
constexpr size_t d_key = 64;
constexpr size_t d_value = 64;
constexpr size_t d_model = n_head * d_key;

template<size_t _rank>
class tensor {
public:
  tensor(const std::array<size_t, _rank>& dims)
    : _dims{dims}
    , _data{new float[size()]}
  { }

  static constexpr size_t rank() {
    return _rank;
  }
  
  size_t size() const {
    return std::accumulate(std::begin(_dims), std::end(_dims), 1, std::multiplies<size_t>());
  }

  std::array<size_t, _rank> dims() const {
    return _dims;
  }
    
  float* ptr() const {
    return _data.get();
  }

  float value(const std::array<size_t, _rank>& ids) const {
    std::array<size_t, _rank> partial_offsets;
    partial_offsets[0] = 1;

    std::partial_sum(std::rbegin(_dims), std::rend(_dims)-1, std::begin(partial_offsets)+1,
		     std::multiplies<size_t>());
    
    std::reverse(std::begin(partial_offsets), std::end(partial_offsets));
    auto idx = std::inner_product(std::begin(ids), std::end(ids), std::begin(partial_offsets), 0,
				  std::plus<size_t>(),
				  std::multiplies<size_t>());

    return _data[idx];	      
  }
  
  size_t shape(size_t i) const {
    return _dims[i];
  }
  
private:
  std::array<size_t, _rank> _dims;
  std::unique_ptr<float[]> _data;
};


template<typename dst_t, typename src_t, typename dims_t = std::array<size_t, dst_t::rank()>>
dst_t reshape(const src_t& src, const dims_t& dims) {
  dst_t dst{dims};

  if (src.size() != dst.size()) {
    std::cout << "Incorrect sizes" << std::endl;
  } else {
    std::copy_n(src.ptr(), src.size(), dst.ptr());
  }

  return dst;
}

template<typename tensor_t>
tensor_t transpose(const tensor_t& src, const std::array<size_t, tensor_t::rank()>& orders) {
  std::array<size_t, tensor_t::rank()> dst_dims;

  std::transform(std::begin(orders), std::end(orders), std::begin(dst_dims),
		 [&src](auto o) -> size_t {
		   return src.dims()[o];
		 });

  tensor dst{dst_dims};
    
  Eigen::TensorMap<Eigen::Tensor<float, tensor_t::rank(), Eigen::RowMajor, size_t>> eigen_src{src.ptr(), src.dims()};
  Eigen::TensorMap<Eigen::Tensor<float, tensor_t::rank(), Eigen::RowMajor, size_t>> eigen_dst{dst.ptr(), dst.dims()};

  eigen_dst = eigen_src.shuffle(orders);

  return dst;
}

template<typename dst_tensor_t, typename src_tensor_t>
dst_tensor_t split_heads(const src_tensor_t& src, size_t n_heads) {
  return transpose(reshape<dst_tensor_t>(src, {batch_size, max_seq_len, n_heads, src.shape(src_tensor_t::rank()-1) / n_heads}),
		   {0, 2, 1, 3});
}

template<typename dst_tensor_t, typename src_tensor_t>
dst_tensor_t combine_heads(const src_tensor_t& src) {
  
  auto t = transpose(src, {0, 2, 1, 3});
  return reshape<dst_tensor_t>(t, {t.shape(0), t.shape(1), t.shape(2) * t.shape(3)});
}

class matmul_op {

};

template<typename tensor_t>
tensor_t scale(const tensor_t& src, float scale) {
  tensor_t dst{src.dims()};

  auto src_ptr = src.ptr();
  auto dst_ptr = dst.ptr();

  for (size_t i = 0; i < src.size(); i++) {
    dst_ptr[i] = scale * src_ptr[i];
  }
}

int main() {
  std::random_device rd;
  std::mt19937 e{rd()};
  std::uniform_real_distribution<float> dist{0, 1};
  
  tensor<3> q{{batch_size, max_seq_len, d_model}};
  tensor<3> k{{batch_size, max_seq_len, d_model}};
  tensor<3> v{{batch_size, max_seq_len, d_model}};

  for (int i = 0; i < q.size(); i++) {
    q.ptr()[i] = dist(e);
  }

  for (int i = 0; i < k.size(); i++) {
    k.ptr()[i] = dist(e);
  }

  for (int i = 0; i < v.size(); i++) {
    v.ptr()[i] = dist(e);
  }

  tensor<4> split_q = split_heads<tensor<4>>(q, n_head);
  tensor<4> split_k = split_heads<tensor<4>>(k, n_head);
  tensor<4> split_v = split_heads<tensor<4>>(v, n_head);

  tensor<3> combined_q = combine_heads<tensor<3>>(split_q);

  for (size_t bs = 0; bs < batch_size; bs++) {
    for (size_t m = 0; m < max_seq_len; m++) {
      for (size_t d = 0; d < d_model; d++) {
	if (q.value({bs, m, d}) != combined_q.value({bs, m, d})) {
	  std::cout << "Incorrect q\n";
	}
      }
    }
  }

  for (size_t bs = 0; bs < batch_size; bs++) {
    for (size_t m = 0; m < max_seq_len; m++) {
      for (size_t d = 0; d < n_head; d++) {
	for (size_t k = 0; k < d_key; k++) {
	  auto s = split_q.value({bs, d, m, k});
	  auto r = q.value({bs, m, d*d_key+k});
	  
	  if (s != r) {
	    std::cout << "Incorrect q1\n";
	    std::cout << bs << " " << m << " " << d << std::endl;
	    std::cout << s << " " << r << std::endl;
	    std::terminate();
	  }
	}
      }
    }
  }

  return 0;
}
