#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
কাঠামো ggml_zdnn_pool {
    কাঠামো বাফার {
        শূন্য * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<বাফার> cached;

    ~ggml_zdnn_pool();

    শূন্য * alloc(size_t size, size_t * actual_আকার);
    শূন্য   মুক্ত(শূন্য * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
কাঠামো ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_আকার = 0;

    ggml_zdnn_pool_alloc() = পূর্বনির্ধারিত;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        যদি (ptr != nullptr) {
            pool->মুক্ত(ptr, actual_আকার);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        এই->pool = &pool;
        ptr = (T *) pool.alloc(size * আকার(T), &actual_আকার);
        ফেরত ptr;
    }

    T * get() {
        ফেরত ptr;
    }

    ggml_zdnn_pool_alloc(ধ্রুবক ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(ধ্রুবক ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
