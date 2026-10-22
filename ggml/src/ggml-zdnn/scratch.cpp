#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "scratch.hpp"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    বাবে (ধ্রুৱ buffer & b : cached) {
        std::মুক্ত(b.ptr);
    }
}

শূন্য * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    বাবে (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        যদি (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    যদি (ibest >= 0) {
        শূন্য * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ঘূৰাই ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    যদি (!cached.empty()) {
        ধ্রুৱ auto largest = std::max_element(cached.begin(), cached.end(), [](ধ্রুৱ buffer & a, ধ্রুৱ buffer & b) { ঘূৰাই a.size < b.size; });
        std::মুক্ত(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    শূন্য * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    ঘূৰাই ptr;
}

শূন্য ggml_zdnn_pool::মুক্ত(শূন্য * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
