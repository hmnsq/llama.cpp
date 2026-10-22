#جيڪڏهن_وصف_نه_ٿيل GGML_ZDNN_SCRATCH_HPP
#وصف_ڪريو GGML_ZDNN_SCRATCH_HPP

#ضمن "ggml.h"

#ضمن <cstddef>
#ضمن <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
بنية ggml_zdnn_pool {
    بنية بفر {
        فراغ * ptr  = خالي_اشارو;
        size_t size = 0;
    };

    std::vector<بفر> cached;

    ~ggml_zdnn_pool();

    فراغ * alloc(size_t size, size_t * actual_سائيز);
    فراغ   حرر(فراغ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
سانچو <قسم_نالو T>
بنية ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = خالي_اشارو;
    T * ptr = خالي_اشارو;
    size_t actual_سائيز = 0;

    ggml_zdnn_pool_alloc() = افتراضي;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        إذا (ptr != خالي_اشارو) {
            pool->حرر(ptr, actual_سائيز);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == خالي_اشارو);
        هذا->pool = &pool;
        ptr = (T *) pool.alloc(size * حجم(T), &actual_سائيز);
        أرجع ptr;
    }

    T * get() {
        أرجع ptr;
    }

    ggml_zdnn_pool_alloc(ثابت ggml_zdnn_pool_alloc &) = ميٽيو;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = ميٽيو;
    ggml_zdnn_pool_alloc & عامل=(ثابت ggml_zdnn_pool_alloc &) = ميٽيو;
    ggml_zdnn_pool_alloc & عامل=(ggml_zdnn_pool_alloc &&) = ميٽيو;
};

#جيڪڏهن_پورو  // GGML_ZDNN_SCRATCH_HPP
