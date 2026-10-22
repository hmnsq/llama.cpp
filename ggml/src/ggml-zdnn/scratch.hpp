#اگر_غیر_تعریف_شدہ GGML_ZDNN_SCRATCH_HPP
#تعریف_کرو GGML_ZDNN_SCRATCH_HPP

#شامل "ggml.h"

#شامل <cstddef>
#شامل <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ساخت ggml_zdnn_pool {
    ساخت بفر {
        خالی * ptr  = خالی_اشاریہ;
        size_t size = 0;
    };

    std::vector<بفر> cached;

    ~ggml_zdnn_pool();

    خالی * alloc(size_t size, size_t * actual_سائز);
    خالی   آزاد(خالی * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
سانچہ <قسم_نام T>
ساخت ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = خالی_اشاریہ;
    T * ptr = خالی_اشاریہ;
    size_t actual_سائز = 0;

    ggml_zdnn_pool_alloc() = طےشدہ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        اگر (ptr != خالی_اشاریہ) {
            pool->آزاد(ptr, actual_سائز);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == خالی_اشاریہ);
        یہ->pool = &pool;
        ptr = (T *) pool.alloc(size * حجم(T), &actual_سائز);
        واپس ptr;
    }

    T * get() {
        واپس ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = حذف;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = حذف;
    ggml_zdnn_pool_alloc & عامل=(ثابت ggml_zdnn_pool_alloc &) = حذف;
    ggml_zdnn_pool_alloc & عامل=(ggml_zdnn_pool_alloc &&) = حذف;
};

#اگر_ختم  // GGML_ZDNN_SCRATCH_HPP
