#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ସଂରଚନା ggml_zdnn_pool {
    ସଂରଚନା buffer {
        ଶୂନ୍ୟ * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    ଶୂନ୍ୟ * alloc(size_t size, size_t * actual_size);
    ଶୂନ୍ୟ   ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
ସଂରଚନା ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        ଯଦି (ptr != nullptr) {
            pool->ମୁକ୍ତ(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        ଏହି->pool = &pool;
        ptr = (T *) pool.alloc(size * ଆକାର(T), &actual_size);
        ଫେରନ୍ତୁ ptr;
    }

    T * get() {
        ଫେରନ୍ତୁ ptr;
    }

    ggml_zdnn_pool_alloc(ଅଚଳ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ଅଚଳ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
