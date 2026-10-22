#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#অন্তর্ভাৱ "ggml.h"

#অন্তর্ভাৱ <cstddef>
#অন্তর্ভাৱ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
সংরচনা ggml_zdnn_pool {
    সংরচনা buffer {
        শূন্য * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    শূন্য * alloc(size_t size, size_t * actual_size);
    শূন্য   মুক্ত(শূন্য * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
সংরচনা ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = স্ৱভাৱ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        যদি (ptr != nullptr) {
            pool->মুক্ত(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        এতত্->pool = &pool;
        ptr = (T *) pool.alloc(size * পরিমাণ(T), &actual_size);
        ঘূৰাই ptr;
    }

    T * get() {
        ঘূৰাই ptr;
    }

    ggml_zdnn_pool_alloc(ধ্রুৱ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ধ্রুৱ ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
