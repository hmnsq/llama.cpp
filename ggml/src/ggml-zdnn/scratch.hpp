#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
संरचना ggml_zdnn_pool {
    संरचना buffer {
        शून्य * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    शून्य * alloc(size_t size, size_t * actual_size);
    शून्य   मुक्त(शून्य * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
संरचना ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = स्वभाव;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        सचे (ptr != nullptr) {
            pool->मुक्त(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        एतत्->pool = &pool;
        ptr = (T *) pool.alloc(size * परिमाण(T), &actual_size);
        पटिनिवत्त ptr;
    }

    T * get() {
        पटिनिवत्त ptr;
    }

    ggml_zdnn_pool_alloc(ध्रुव ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ध्रुव ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
