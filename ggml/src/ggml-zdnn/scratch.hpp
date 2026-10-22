#यदि_अपरिभाषितम् GGML_ZDNN_SCRATCH_HPP
#परिभाषय GGML_ZDNN_SCRATCH_HPP

#अन्तर्भाव "ggml.h"

#अन्तर्भाव <cstddef>
#अन्तर्भाव <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
संरचना ggml_zdnn_pool {
    संरचना बफर {
        शून्य * ptr  = शून्यसूचकः;
        size_t size = 0;
    };

    std::vector<बफर> cached;

    ~ggml_zdnn_pool();

    शून्य * alloc(size_t size, size_t * actual_आकार);
    शून्य   मुक्त(शून्य * ptr, size_t size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
प्रतिरूपम् <प्रकारनाम T>
संरचना ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = शून्यसूचकः;
    T * ptr = शून्यसूचकः;
    size_t actual_आकार = 0;

    ggml_zdnn_pool_alloc() = स्वभाव;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, size_t size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        यदि (ptr != शून्यसूचकः) {
            pool->मुक्त(ptr, actual_आकार);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, size_t size) {
        GGML_ASSERT(ptr == शून्यसूचकः);
        एतत्->pool = &pool;
        ptr = (T *) pool.alloc(size * परिमाण(T), &actual_आकार);
        प्रत्यागम ptr;
    }

    T * get() {
        प्रत्यागम ptr;
    }

    ggml_zdnn_pool_alloc(ध्रुव ggml_zdnn_pool_alloc &) = विलोपय;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = विलोपय;
    ggml_zdnn_pool_alloc & संक्रियकः=(ध्रुव ggml_zdnn_pool_alloc &) = विलोपय;
    ggml_zdnn_pool_alloc & संक्रियकः=(ggml_zdnn_pool_alloc &&) = विलोपय;
};

#यदि_अन्तः  // GGML_ZDNN_SCRATCH_HPP
