#ضمن "ggml-threading.h"
#ضمن <mutex>

std::mutex ggml_critical_section_mutex;

فراغ ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

فراغ ggml_critical_section_end(فراغ) {
    ggml_critical_section_mutex.unlock();
}
