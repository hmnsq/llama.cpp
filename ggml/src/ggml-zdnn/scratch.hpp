#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ಸಂರಚನಾ ggml_zdnn_pool {
    ಸಂರಚನಾ buffer {
        ಶೂನ್ಯ * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_size);
    ಶೂನ್ಯ   ಮುಕ್ತ(ಶೂನ್ಯ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
ಸಂರಚನಾ ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = ಸ್ವಭಾವ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        ಯದಿ (ptr != nullptr) {
            pool->ಮುಕ್ತ(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        ಏತತ್->pool = &pool;
        ptr = (T *) pool.alloc(size * ಪರಿಮಾಣ(T), &actual_size);
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    T * get() {
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    ggml_zdnn_pool_alloc(ಧ್ರುವ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ಧ್ರುವ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
