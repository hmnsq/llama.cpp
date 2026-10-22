#شامل "ggml-threading.h"
#شامل <mutex>

std::mutex ggml_critical_section_mutex;

تهی ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

تهی ggml_critical_section_end(تهی) {
    ggml_critical_section_mutex.unlock();
}
