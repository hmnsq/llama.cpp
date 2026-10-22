#جے_بنا_تعریف GGML_ZDNN_SCRATCH_HPP
#تعریف_کرو GGML_ZDNN_SCRATCH_HPP

#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
بنية ggml_zdnn_pool {
    بنية بفر {
        فراغ * ptr  = خالی_سوچک;
        size_t size = 0;
    };

    std::vector<بفر> cached;

    ~ggml_zdnn_pool();

    فراغ * alloc(size_t size, size_t * actual_ناپ);
    فراغ   حرر(فراغ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
سانچا <قسم_ناں T>
بنية ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = خالی_سوچک;
    T * ptr = خالی_سوچک;
    size_t actual_ناپ = 0;

    ggml_zdnn_pool_alloc() = افتراضي;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        إذا (ptr != خالی_سوچک) {
            pool->حرر(ptr, actual_ناپ);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == خالی_سوچک);
        هذا->pool = &pool;
        ptr = (T *) pool.alloc(size * حجم(T), &actual_ناپ);
        أرجع ptr;
    }

    T * get() {
        أرجع ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = مٹاؤ;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = مٹاؤ;
    ggml_zdnn_pool_alloc & عامل=(ثابت ggml_zdnn_pool_alloc &) = مٹاؤ;
    ggml_zdnn_pool_alloc & عامل=(ggml_zdnn_pool_alloc &&) = مٹاؤ;
};

#جے_انت  // GGML_ZDNN_SCRATCH_HPP
