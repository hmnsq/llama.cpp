#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
સંરચના ggml_zdnn_pool {
    સંરચના buffer {
        શૂન્ય * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    શૂન્ય * alloc(size_t size, size_t * actual_size);
    શૂન્ય   મુક્ત(શૂન્ય * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
સંરચના ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = મૂળભૂત;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        જો (ptr != nullptr) {
            pool->મુક્ત(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        આ->pool = &pool;
        ptr = (T *) pool.alloc(size * કદ(T), &actual_size);
        પાછા ptr;
    }

    T * get() {
        પાછા ptr;
    }

    ggml_zdnn_pool_alloc(અચલ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(અચલ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
