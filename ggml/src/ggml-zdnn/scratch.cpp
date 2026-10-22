#అన్తర్భావ "ggml.h"
#అన్తర్భావ "scratch.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>

ggml_zdnn_pool::~ggml_zdnn_pool() {
    కొరకు (స్థిరరాశి బఫరు & b : cached) {
        std::విడుదల(b.ptr);
    }
}

శూన్య * ggml_zdnn_pool::alloc(size_t size, size_t * actual_కొలత) {
    int64_t ibest = -1;
    కొరకు (int64_t i = 0; i < (int64_t)cached.size(); i++) {
        ఒకవేళ (cached[i].size >= size && (ibest < 0 || cached[i].size < cached[ibest].size)) {
            ibest = i;
        }
    }

    ఒకవేళ (ibest >= 0) {
        శూన్య * ptr   = cached[ibest].ptr;
        *actual_కొలత = cached[ibest].size;
        cached.erase(cached.begin() + ibest);
        తిరిగి ptr;
    }

    // every cached buffer is too small, free the largest so the pool does not grow while sizes grow with the context
    ఒకవేళ (!cached.ఖాళీలు()) {
        స్థిరరాశి స్వయంచాలక అతిపెద్ద = std::max_element(cached.begin(), cached.end(), [](స్థిరరాశి బఫరు & a, స్థిరరాశి బఫరు & b) { తిరిగి a.size < b.size; });
        std::విడుదల(అతిపెద్ద->ptr);
        cached.erase(అతిపెద్ద);
    }

    // 5% more, so slowly growing sizes do not allocate every time
    *actual_కొలత = GGML_PAD(std::max<size_t>(1, (size_t)(1.05 * size)), 4096);
    శూన్య * ptr   = std::aligned_alloc(4096, *actual_కొలత);
    GGML_ASSERT(ptr != శూన్యసూచిక);
    తిరిగి ptr;
}

శూన్య ggml_zdnn_pool::విడుదల(శూన్య * ptr, size_t size) {
    cached.push_back({ ptr, size });
}
