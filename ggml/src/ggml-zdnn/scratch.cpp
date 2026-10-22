#شامل "ggml.h"
#شامل "scratch.hpp"

#شامل <algorithm>
#شامل <cstdint>
#شامل <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    برای (ثابت buffer & b : cached) {
        std::آزاد(b.ptr);
    }
}

تهی * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    برای (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        اگر (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    اگر (ibest >= 0) {
        تهی * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        بازگشت ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    اگر (!cached.empty()) {
        ثابت auto largest = std::max_element(cached.begin(), cached.end(), [](ثابت buffer & a, ثابت buffer & b) { بازگشت a.size < b.size; });
        std::آزاد(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    تهی * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    بازگشت ptr;
}

تهی ggml_zdnn_pool::آزاد(تهی * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
