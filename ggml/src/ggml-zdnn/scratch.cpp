#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "scratch.hpp"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    માટે (અચલ buffer & b : cached) {
        std::મુક્ત(b.ptr);
    }
}

શૂન્ય * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    માટે (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        જો (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    જો (ibest >= 0) {
        શૂન્ય * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        પાછા ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    જો (!cached.empty()) {
        અચલ auto largest = std::max_element(cached.begin(), cached.end(), [](અચલ buffer & a, અચલ buffer & b) { પાછા a.size < b.size; });
        std::મુક્ત(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    શૂન્ય * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    પાછા ptr;
}

શૂન્ય ggml_zdnn_pool::મુક્ત(શૂન્ય * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
