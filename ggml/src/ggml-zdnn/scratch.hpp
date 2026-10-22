#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
संरचना ggml_zdnn_pool {
    संरचना buffer {
        शून्य * ptr  = nullptr;
        माप_प्रकार size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    शून्य * alloc(माप_प्रकार size, माप_प्रकार * actual_size);
    शून्य   मुक्त(शून्य * ptr, माप_प्रकार size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
ढाँचा <typename T>
संरचना ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    माप_प्रकार actual_size = 0;

    ggml_zdnn_pool_alloc() = स्वभाव;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, माप_प्रकार size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        यदि (ptr != nullptr) {
            pool->मुक्त(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, माप_प्रकार size) {
        GGML_ASSERT(ptr == nullptr);
        एतत्->pool = &pool;
        ptr = (T *) pool.alloc(size * परिमाण(T), &actual_size);
        प्रत्यागम ptr;
    }

    T * get() {
        प्रत्यागम ptr;
    }

    ggml_zdnn_pool_alloc(ध्रुव ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & चालक=(ध्रुव ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & चालक=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
