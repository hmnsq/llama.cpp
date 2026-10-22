#include "ggml.h"
#include "scratch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    ꦏꦁꦒꦺꦴ (const buffer & b : cached) {
        std::free(b.ptr);
    }
}

void * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    ꦏꦁꦒꦺꦴ (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ꦪꦼꦤ꧀ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ꦪꦼꦤ꧀ (ibest >= 0) {
        void * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        ꦮꦭꦶ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ꦪꦼꦤ꧀ (!cached.empty()) {
        const auto largest = std::max_element(cached.begin(), cached.end(), [](const buffer & a, const buffer & b) { ꦮꦭꦶ a.size < b.size; });
        std::free(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    void * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    ꦮꦭꦶ ptr;
}

void ggml_zdnn_pool::free(void * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
