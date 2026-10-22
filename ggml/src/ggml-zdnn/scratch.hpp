#اگر_تعریف_نشده GGML_ZDNN_SCRATCH_HPP
#تعریف_کن GGML_ZDNN_SCRATCH_HPP

#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
بنية ggml_zdnn_pool {
    بنية میانگیر {
        فراغ * ptr  = اشاره_تهی;
        size_t size = 0;
    };

    std::vector<میانگیر> cached;

    ~ggml_zdnn_pool();

    فراغ * alloc(size_t size, size_t * actual_اندازه);
    فراغ   حرر(فراغ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
قالب <نام_نوع T>
بنية ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = اشاره_تهی;
    T * ptr = اشاره_تهی;
    size_t actual_اندازه = 0;

    ggml_zdnn_pool_alloc() = افتراضي;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        إذا (ptr != اشاره_تهی) {
            pool->حرر(ptr, actual_اندازه);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == اشاره_تهی);
        هذا->pool = &pool;
        ptr = (T *) pool.alloc(size * حجم(T), &actual_اندازه);
        أرجع ptr;
    }

    T * get() {
        أرجع ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = حذف;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = حذف;
    ggml_zdnn_pool_alloc & عملگر=(ثابت ggml_zdnn_pool_alloc &) = حذف;
    ggml_zdnn_pool_alloc & عملگر=(ggml_zdnn_pool_alloc &&) = حذف;
};

#پایان_اگر  // GGML_ZDNN_SCRATCH_HPP
