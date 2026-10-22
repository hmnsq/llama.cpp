#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "scratch.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    জন্য (ধ্রুবক বাফার & b : cached) {
        std::মুক্ত(b.ptr);
    }
}

শূন্য * ggml_zdnn_pool::alloc(size_t size, size_t * actual_আকার) {
    int64_t ibest = -1;
    জন্য (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        যদি (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    যদি (ibest >= 0) {
        শূন্য * ptr   = cached[ibest].ptr;
        *actual_আকার = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ফেরত ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    যদি (!cached.empty()) {
        ধ্রুবক auto largest = std::max_element(cached.begin(), cached.end(), [](ধ্রুবক বাফার & a, ধ্রুবক বাফার & b) { ফেরত a.size < b.size; });
        std::মুক্ত(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_আকার = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    শূন্য * ptr   = std::aligned_alloc(4096, *actual_আকার);
    GGML_ASSERT(ptr != nullptr);
    ফেরত ptr;
}

শূন্য ggml_zdnn_pool::মুক্ত(শূন্য * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
