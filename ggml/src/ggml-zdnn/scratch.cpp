#ضمن "ggml.h"
#ضمن "scratch.hpp"

#ضمن <algorithm>
#ضمن <cstdint>
#ضمن <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    لكل (ثابت میانگیر & b : cached) {
        std::حرر(b.ptr);
    }
}

فراغ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_اندازه) {
    int64_t ibest = -1;
    لكل (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        إذا (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    إذا (ibest >= 0) {
        فراغ * ptr   = cached[ibest].ptr;
        *actual_اندازه = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        أرجع ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    إذا (!cached.خالی()) {
        ثابت خودکار بزرگترین = std::max_element(cached.begin(), cached.end(), [](ثابت میانگیر & a, ثابت میانگیر & b) { أرجع a.size < b.size; });
        std::حرر(بزرگترین->ptr);
        cached.erase(بزرگترین);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_اندازه = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    فراغ * ptr   = std::aligned_alloc(4096, *actual_اندازه);
    GGML_ASSERT(ptr != اشاره_تهی);
    أرجع ptr;
}

فراغ ggml_zdnn_pool::حرر(فراغ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
