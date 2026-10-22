#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "scratch.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    ਲਈ (ਅਚਲ ਬਫਰ & b : cached) {
        std::ਮੁਕਤ(b.ptr);
    }
}

ਖਾਲੀ * ggml_zdnn_pool::alloc(size_t size, size_t * actual_ਮਾਪ) {
    int64_t ibest = -1;
    ਲਈ (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ਜੇ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ਜੇ (ibest >= 0) {
        ਖਾਲੀ * ptr   = cached[ibest].ptr;
        *actual_ਮਾਪ = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ਵਾਪਸ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ਜੇ (!cached.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ ਸਭ_ਤੋਂ_ਵੱਡਾ = std::max_element(cached.begin(), cached.end(), [](ਅਚਲ ਬਫਰ & a, ਅਚਲ ਬਫਰ & b) { ਵਾਪਸ a.size < b.size; });
        std::ਮੁਕਤ(ਸਭ_ਤੋਂ_ਵੱਡਾ->ptr);
        cached.erase(ਸਭ_ਤੋਂ_ਵੱਡਾ);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_ਮਾਪ = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    ਖਾਲੀ * ptr   = std::aligned_alloc(4096, *actual_ਮਾਪ);
    GGML_ASSERT(ptr != ਖਾਲੀ_ਸੂਚਕ);
    ਵਾਪਸ ptr;
}

ਖਾਲੀ ggml_zdnn_pool::ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
