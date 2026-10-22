#شامل "ggml.h"
#شامل "scratch.hpp"

#شامل <algorithm>
#شامل <cstdint>
#شامل <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    برائے (ثابت بفر & b : cached) {
        std::آزاد(b.ptr);
    }
}

خالی * ggml_zdnn_pool::alloc(size_t size, size_t * actual_سائز) {
    int64_t ibest = -1;
    برائے (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        اگر (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    اگر (ibest >= 0) {
        خالی * ptr   = cached[ibest].ptr;
        *actual_سائز = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        واپس ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    اگر (!cached.خالی_خانے()) {
        ثابت خودکار سب_سے_بڑا = std::max_element(cached.begin(), cached.end(), [](ثابت بفر & a, ثابت بفر & b) { واپس a.size < b.size; });
        std::آزاد(سب_سے_بڑا->ptr);
        cached.erase(سب_سے_بڑا);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_سائز = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    خالی * ptr   = std::aligned_alloc(4096, *actual_سائز);
    GGML_ASSERT(ptr != خالی_اشاریہ);
    واپس ptr;
}

خالی ggml_zdnn_pool::آزاد(خالی * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
