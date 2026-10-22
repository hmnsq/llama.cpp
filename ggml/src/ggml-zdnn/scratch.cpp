#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "scratch.hpp"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    க்கு (மாறிலி buffer & b : cached) {
        std::விடுவி(b.ptr);
    }
}

வெறுமை * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    க்கு (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        எனில் (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    எனில் (ibest >= 0) {
        வெறுமை * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        திரும்பு ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    எனில் (!cached.empty()) {
        மாறிலி auto largest = std::max_element(cached.begin(), cached.end(), [](மாறிலி buffer & a, மாறிலி buffer & b) { திரும்பு a.size < b.size; });
        std::விடுவி(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    வெறுமை * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    திரும்பு ptr;
}

வெறுமை ggml_zdnn_pool::விடுவி(வெறுமை * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
