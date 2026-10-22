#अगर_अघोषित GGML_ZDNN_SCRATCH_HPP
#घोषणा GGML_ZDNN_SCRATCH_HPP

#समावेश "ggml.h"

#समावेश <cstddef>
#समावेश <vector>

// 4K aligned host buffers that are cached for reuse, see ggml_cuda_pool_leg
काष्ठा ggml_zdnn_pool {
    काष्ठा बफर {
        व्योम * ptr  = nullptr;
        माप_प्रकार size = 0;
    };

    std::vector<बफर> cached;

    ~ggml_zdnn_pool();

    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार);
    व्योम   मुक्त(व्योम * ptr, माप_प्रकार size);
};

// a buffer of the pool that goes back to it at the end of its scope, see ggml_cuda_pool_alloc
ढाँचा <typename T>
काष्ठा ggml_zdnn_pool_alloc {
    ggml_zdnn_pool * pool = nullptr;
    T * ptr = nullptr;
    माप_प्रकार actual_आकार = 0;

    ggml_zdnn_pool_alloc() = शेष;

    ggml_zdnn_pool_alloc(ggml_zdnn_pool & pool, माप_प्रकार size) {
        alloc(pool, size);
    }

    ~ggml_zdnn_pool_alloc() {
        अगर (ptr != nullptr) {
            pool->मुक्त(ptr, actual_आकार);
        }
    }

    // size is in number of elements
    T * alloc(ggml_zdnn_pool & pool, माप_प्रकार size) {
        GGML_ASSERT(ptr == nullptr);
        यह->pool = &pool;
        ptr = (T *) pool.alloc(size * माप(T), &actual_आकार);
        वापस ptr;
    }

    T * लो() {
        वापस ptr;
    }

    ggml_zdnn_pool_alloc(स्थिर ggml_zdnn_pool_alloc &) = मिटाओ;
    ggml_zdnn_pool_alloc(ggml_zdnn_pool_alloc &&) = मिटाओ;
    ggml_zdnn_pool_alloc & चालक=(स्थिर ggml_zdnn_pool_alloc &) = मिटाओ;
    ggml_zdnn_pool_alloc & चालक=(ggml_zdnn_pool_alloc &&) = मिटाओ;
};

#पूर्ण_अगर  // GGML_ZDNN_SCRATCH_HPP
