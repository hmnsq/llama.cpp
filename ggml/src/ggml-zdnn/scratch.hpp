#که_نه_تعریف_شوی GGML_ZDNN_SCRATCH_HPP
#تعریف_کړه GGML_ZDNN_SCRATCH_HPP

#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
بنية ggml_zdnn_pool {
    بنية بفر {
        فراغ * ptr  = تش_اشاره;
        size_t size = 0;
    };

    std::vector<بفر> cached;

    ~ggml_zdnn_pool();

    فراغ * alloc(size_t size, size_t * actual_اندازه);
    فراغ   حرر(فراغ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
کينډۍ <ډول_نوم T>
بنية ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = تش_اشاره;
    T * ptr = تش_اشاره;
    size_t actual_اندازه = 0;

    ggml_zdnn_pool_alloc() = افتراضي;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        إذا (ptr != تش_اشاره) {
            pool->حرر(ptr, actual_اندازه);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == تش_اشاره);
        هذا->pool = &pool;
        ptr = (T *) pool.alloc(size * حجم(T), &actual_اندازه);
        أرجع ptr;
    }

    T * get() {
        أرجع ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = ړنګول;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = ړنګول;
    ggml_zdnn_pool_alloc & چلوونکی=(ثابت ggml_zdnn_pool_alloc &) = ړنګول;
    ggml_zdnn_pool_alloc & چلوونکی=(ggml_zdnn_pool_alloc &&) = ړنګول;
};

#د_که_پای  // GGML_ZDNN_SCRATCH_HPP
