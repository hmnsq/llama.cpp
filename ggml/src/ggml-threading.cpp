#அந்தர்பாவ "ggml-threading.h"
#அந்தர்பாவ <mutex>

std::mutex ggml_critical_section_mutex;

வெறுமை ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

வெறுமை ggml_critical_section_end(வெறுமை) {
    ggml_critical_section_mutex.unlock();
}
