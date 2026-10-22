#അന്തര്ഭാവ "ggml-threading.h"
#അന്തര്ഭാവ <mutex>

std::mutex ggml_critical_section_mutex;

ശൂന്യം ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

ശൂന്യം ggml_critical_section_end(ശൂന്യം) {
    ggml_critical_section_mutex.unlock();
}
