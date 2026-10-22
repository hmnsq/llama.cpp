#ifndef GGML_ZDNN_SCRATCH_HPP
#define GGML_ZDNN_SCRATCH_HPP

#അന്തര്ഭാവ "ggml.h"

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ഘടന ggml_zdnn_pool {
    ഘടന buffer {
        ശൂന്യം * ptr  = nullptr;
        size_t size = 0;
    };

    std::vector<buffer> cached;

    ~ggml_zdnn_pool();

    ശൂന്യം * alloc(size_t size, size_t * actual_size);
    ശൂന്യം   സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
template <typename T>
ഘടന ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_zdnn_pool_alloc() = സ്ഥിരസ്ഥിതി;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        എങ്കിൽ (ptr != nullptr) {
            pool->സ്വതന്ത്രം(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == nullptr);
        ഇത്->pool = &pool;
        ptr = (T *) pool.alloc(size * വലുപ്പം(T), &actual_size);
        തിരികെ ptr;
    }

    T * get() {
        തിരികെ ptr;
    }

    ggml_zdnn_pool_alloc(സ്ഥിരം ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = delete;
    ggml_zdnn_pool_alloc & operator=(സ്ഥിരം ggml_zdnn_pool_alloc &) = delete;
    ggml_zdnn_pool_alloc & operator=(ggml_zdnn_pool_alloc &&) = delete;
};

#endif  // GGML_ZDNN_SCRATCH_HPP
