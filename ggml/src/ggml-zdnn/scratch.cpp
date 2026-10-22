#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "scratch.hpp"

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    ପାଇଁ (ଅଚଳ buffer & b : cached) {
        std::ମୁକ୍ତ(b.ptr);
    }
}

ଶୂନ୍ୟ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    ପାଇଁ (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ଯଦି (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ଯଦି (ibest >= 0) {
        ଶୂନ୍ୟ * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ଫେରନ୍ତୁ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ଯଦି (!cached.empty()) {
        ଅଚଳ auto largest = std::max_element(cached.begin(), cached.end(), [](ଅଚଳ buffer & a, ଅଚଳ buffer & b) { ଫେରନ୍ତୁ a.size < b.size; });
        std::ମୁକ୍ତ(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    ଶୂନ୍ୟ * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    ଫେରନ୍ତୁ ptr;
}

ଶୂନ୍ୟ ggml_zdnn_pool::ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
