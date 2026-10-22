#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_ZDNN_SCRATCH_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ZDNN_SCRATCH_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ਬਣਤਰ ggml_zdnn_pool {
    ਬਣਤਰ ਬਫਰ {
        ਖਾਲੀ * ptr  = ਖਾਲੀ_ਸੂਚਕ;
        size_t size = 0;
    };

    std::vector<ਬਫਰ> cached;

    ~ggml_zdnn_pool();

    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ);
    ਖਾਲੀ   ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = ਖਾਲੀ_ਸੂਚਕ;
    T * ptr = ਖਾਲੀ_ਸੂਚਕ;
    size_t actual_ਮਾਪ = 0;

    ggml_zdnn_pool_alloc() = ਮੂਲ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        ਜੇ (ptr != ਖਾਲੀ_ਸੂਚਕ) {
            pool->ਮੁਕਤ(ptr, actual_ਮਾਪ);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == ਖਾਲੀ_ਸੂਚਕ);
        ਇਹ->pool = &pool;
        ptr = (T *) pool.alloc(size * ਆਕਾਰ(T), &actual_ਮਾਪ);
        ਵਾਪਸ ptr;
    }

    T * get() {
        ਵਾਪਸ ptr;
    }

    ggml_zdnn_pool_alloc(ਅਚਲ ggml_zdnn_pool_alloc &) = ਮਿਟਾਓ;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = ਮਿਟਾਓ;
    ggml_zdnn_pool_alloc & ਸੰਚਾਲਕ=(ਅਚਲ ggml_zdnn_pool_alloc &) = ਮਿਟਾਓ;
    ggml_zdnn_pool_alloc & ਸੰਚਾਲਕ=(ggml_zdnn_pool_alloc &&) = ਮਿਟਾਓ;
};

#ਜੇ_ਅੰਤ  // GGML_ZDNN_SCRATCH_HPP
