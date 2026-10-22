#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
கட்டமைப்பு ggml_zdnn_pool {
    கட்டமைப்பு buffer {
        வெறுமை * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    வெறுமை * alloc(size_t size, size_t * actual_size);
    வெறுமை   விடுவி(வெறுமை * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
கட்டமைப்பு ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = இயல்பு;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        எனில் (ptr != nullptr) {
            pool->விடுவி(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        இது->pool = &pool;
        ptr = (T *) pool.alloc(size * அளவு(T), &actual_size);
        திரும்பு ptr;
    }

    T * get() {
        திரும்பு ptr;
    }

    ggml_zdnn_pool_alloc(மாறிலி ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(மாறிலி ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
