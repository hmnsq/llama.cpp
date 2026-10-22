#अन्तर्भाव "ggml.h"
#अन्तर्भाव "scratch.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    प्रति (ध्रुव buffer & b : cached) {
        std::मुक्त(b.ptr);
    }
}

शून्य * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    प्रति (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        यदि (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    यदि (ibest >= 0) {
        शून्य * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        प्रत्यागम ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    यदि (!cached.empty()) {
        ध्रुव auto largest = std::max_element(cached.begin(), cached.end(), [](ध्रुव buffer & a, ध्रुव buffer & b) { प्रत्यागम a.size < b.size; });
        std::मुक्त(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    शून्य * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    प्रत्यागम ptr;
}

शून्य ggml_zdnn_pool::मुक्त(शून्य * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
