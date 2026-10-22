#অন্তর্ভাৱ "ggml-threading.h"
#অন্তর্ভাৱ <mutex>

std::mutex ggml_critical_section_mutex;

শূন্য ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

শূন্য ggml_critical_section_end(শূন্য) {
    ggml_critical_section_mutex.unlock();
}
