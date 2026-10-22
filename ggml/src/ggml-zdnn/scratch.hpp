#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#شامل "ggml.h"

#شامل <cstddef>
#شامل <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ساختار ggml_zdnn_pool {
    ساختار buffer {
        تهی * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    تهی * alloc(size_t size, size_t * actual_size);
    تهی   آزاد(تهی * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
ساختار ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = پیشفرض;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        اگر (ptr != nullptr) {
            pool->آزاد(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        این->pool = &pool;
        ptr = (T *) pool.alloc(size * اندازه(T), &actual_size);
        بازگشت ptr;
    }

    T * get() {
        بازگشت ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ثابت ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
