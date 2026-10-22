#ضمن "ggml.h"
#ضمن "scratch.hpp"

#ضمن <algorithm>
#ضمن <cstdint>
#ضمن <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    لكل (ثابت buffer & b : cached) {
        std::حرر(b.ptr);
    }
}

فراغ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    لكل (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        𐤀𐤌 (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    𐤀𐤌 (ibest >= 0) {
        فراغ * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        أرجع ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    𐤀𐤌 (!cached.empty()) {
        ثابت auto largest = std::max_element(cached.begin(), cached.end(), [](ثابت buffer & a, ثابت buffer & b) { أرجع a.size < b.size; });
        std::حرر(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    فراغ * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    أرجع ptr;
}

فراغ ggml_zdnn_pool::حرر(فراغ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
