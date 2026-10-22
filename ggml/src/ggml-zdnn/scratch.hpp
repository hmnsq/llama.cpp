#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_ZDNN_SCRATCH_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ZDNN_SCRATCH_HPP

#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
ರಚನೆ ggml_zdnn_pool {
    ರಚನೆ ಬಫರ್ {
        ಶೂನ್ಯ * ptr  = ಶೂನ್ಯಸೂಚಕ;
        size_t size = 0;
    };

    std::vector<ಬಫರ್> cached;

    ~ggml_zdnn_pool();

    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ);
    ಶೂನ್ಯ   ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ರಚನೆ ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = ಶೂನ್ಯಸೂಚಕ;
    T * ptr = ಶೂನ್ಯಸೂಚಕ;
    size_t actual_ಗಾತ್ರ = 0;

    ggml_zdnn_pool_alloc() = ಪೂರ್ವನಿಯೋಜಿತ;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        ಆದರೆ (ptr != ಶೂನ್ಯಸೂಚಕ) {
            pool->ಬಿಡುಗಡೆ(ptr, actual_ಗಾತ್ರ);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == ಶೂನ್ಯಸೂಚಕ);
        ಇದು->pool = &pool;
        ptr = (T *) pool.alloc(size * ಗಾತ್ರ(T), &actual_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ptr;
    }

    T * get() {
        ಹಿಂತಿರುಗಿ ptr;
    }

    ggml_zdnn_pool_alloc(ಸ್ಥಿರಾಂಕ ggml_zdnn_pool_alloc &) = ಅಳಿಸು;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = ಅಳಿಸು;
    ggml_zdnn_pool_alloc & ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ ggml_zdnn_pool_alloc &) = ಅಳಿಸು;
    ggml_zdnn_pool_alloc & ಕ್ರಿಯಾಕಾರಕ=(ggml_zdnn_pool_alloc &&) = ಅಳಿಸು;
};

#ಆದರೆ_ಅಂತ್ಯ  // GGML_ZDNN_SCRATCH_HPP
