#অন্তর্ভাব "ggml-threading.h"
#অন্তর্ভাব <mutex>

std::mutex ggml_critical_section_mutex;

শূন্য ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

শূন্য ggml_critical_section_end(শূন্য) {
    ggml_critical_section_mutex.unlock();
}
