#अन्तर्भाव "ggml.h"
#अन्तर्भाव "scratch.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    साठी (स्थिरांक buffer & b : cached) {
        std::मुक्त(b.ptr);
    }
}

शून्य * ggml_zdnn_pool::alloc(माप_प्रकार size, माप_प्रकार * actual_size) {
    int64_t ibest = -1;
    साठी (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        जर (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    जर (ibest >= 0) {
        शून्य * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        परत ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    जर (!cached.empty()) {
        स्थिरांक स्वतः largest = std::max_element(cached.begin(), cached.end(), [](स्थिरांक buffer & a, स्थिरांक buffer & b) { परत a.size < b.size; });
        std::मुक्त(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<माप_प्रकार>(1, (माप_प्रकार)(1.05 * size)), 4096);
    शून्य * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    परत ptr;
}

शून्य ggml_zdnn_pool::मुक्त(शून्य * ptr, माप_प्रकार size) {
    cached.push_back({ ptr, size });
}
