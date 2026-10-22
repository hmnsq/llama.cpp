#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "scratch.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    ಪ್ರತಿ (ಧ್ರುವ buffer & b : cached) {
        std::ಮುಕ್ತ(b.ptr);
    }
}

ಶೂನ್ಯ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    ಪ್ರತಿ (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ಯದಿ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ಯದಿ (ibest >= 0) {
        ಶೂನ್ಯ * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ಯದಿ (!cached.empty()) {
        ಧ್ರುವ auto largest = std::max_element(cached.begin(), cached.end(), [](ಧ್ರುವ buffer & a, ಧ್ರುವ buffer & b) { ಪ್ರತ್ಯಾಗಮ a.size < b.size; });
        std::ಮುಕ್ತ(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    ಶೂನ್ಯ * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    ಪ್ರತ್ಯಾಗಮ ptr;
}

ಶೂನ್ಯ ggml_zdnn_pool::ಮುಕ್ತ(ಶೂನ್ಯ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
