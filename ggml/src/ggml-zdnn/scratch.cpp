#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "scratch.hpp"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    വേണ്ടി (സ്ഥിരം buffer & b : cached) {
        std::സ്വതന്ത്രം(b.ptr);
    }
}

ശൂന്യം * ggml_zdnn_pool::alloc(size_t size, size_t * actual_size) {
    int64_t ibest = -1;
    വേണ്ടി (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        എങ്കിൽ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    എങ്കിൽ (ibest >= 0) {
        ശൂന്യം * ptr   = cached[ibest].ptr;
        *actual_size = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        തിരികെ ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    എങ്കിൽ (!cached.empty()) {
        സ്ഥിരം auto largest = std::max_element(cached.begin(), cached.end(), [](സ്ഥിരം buffer & a, സ്ഥിരം buffer & b) { തിരികെ a.size < b.size; });
        std::സ്വതന്ത്രം(largest->ptr);
        cached.erase(largest);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_size = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    ശൂന്യം * ptr   = std::aligned_alloc(4096, *actual_size);
    GGML_ASSERT(ptr != nullptr);
    തിരികെ ptr;
}

ശൂന്യം ggml_zdnn_pool::സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
