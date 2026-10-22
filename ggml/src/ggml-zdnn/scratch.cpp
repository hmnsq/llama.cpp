#समावेश "ggml.h"
#समावेश "scratch.hpp"

#समावेश <algorithm>
#समावेश <cstdint>
#समावेश <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    क्रम (स्थिर बफर & b : cached) {
        std::मुक्त(b.ptr);
    }
}

व्योम * ggml_zdnn_pool::alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) {
    int64_t ibest = -1;
    क्रम (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        अगर (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    अगर (ibest >= 0) {
        व्योम * ptr   = cached[ibest].ptr;
        *actual_आकार = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        वापस ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    अगर (!cached.empty()) {
        स्थिर स्वतः largest = std::max_element(cached.begin(), cached.end(), [](स्थिर बफर & a, स्थिर बफर & b) { वापस a.size < b.size; });
        std::मुक्त(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_आकार = GGML_PAD(std::max<माप_प्रकार>(1, (माप_प्रकार)(1.05 * size)), 4096);
    व्योम * ptr   = std::aligned_alloc(4096, *actual_आकार);
    GGML_ASSERT(ptr != nullptr);
    वापस ptr;
}

व्योम ggml_zdnn_pool::मुक्त(व्योम * ptr, माप_प्रकार size) {
    cached.push_back({ ptr, size });
}
