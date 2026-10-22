#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "scratch.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಬಫರ್ & b : cached) {
        std::ಬಿಡುಗಡೆ(b.ptr);
    }
}

ಶೂನ್ಯ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_ಗಾತ್ರ) {
    int64_t ibest = -1;
    ಗಾಗಿ (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ಆದರೆ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ಆದರೆ (ibest >= 0) {
        ಶೂನ್ಯ * ptr   = cached[ibest].ptr;
        *actual_ಗಾತ್ರ = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ಹಿಂತಿರುಗಿ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ಆದರೆ (!cached.ಖಾಲಿಗಳು()) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ಅತಿದೊಡ್ಡ = std::max_element(cached.begin(), cached.end(), [](ಸ್ಥಿರಾಂಕ ಬಫರ್ & a, ಸ್ಥಿರಾಂಕ ಬಫರ್ & b) { ಹಿಂತಿರುಗಿ a.size < b.size; });
        std::ಬಿಡುಗಡೆ(ಅತಿದೊಡ್ಡ->ptr);
        cached.erase(ಅತಿದೊಡ್ಡ);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_ಗಾತ್ರ = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    ಶೂನ್ಯ * ptr   = std::aligned_alloc(4096, *actual_ಗಾತ್ರ);
    GGML_ASSERT(ptr != ಶೂನ್ಯಸೂಚಕ);
    ಹಿಂತಿರುಗಿ ptr;
}

ಶೂನ್ಯ ggml_zdnn_pool::ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
