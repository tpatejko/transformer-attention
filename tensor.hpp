#pragma once

#include <algorithm>
#include <memory>

class tensor {
public:
  tensor(const std::vector<size_t>& dims)
  : _dims(dims)
  , _data{new float[size()]}
  { }

  size_t rank() const {
    return _dims.size();
  }

  size_t size() const {
    return std::accumulate(std::begin(_dims), std::end(_dims), 1, std::multiplies<size_t>());
  }

  std::vector<size_t> dims() const {
    return _dims;
  }

  float* ptr() const {
    return _data.get();
  }

  float value(const std::vector<size_t>& ids) const {
    std::vector<size_t> partial_offsets(rank());
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
  std::vector<size_t> _dims;
  std::unique_ptr<float[]> _data;
};
