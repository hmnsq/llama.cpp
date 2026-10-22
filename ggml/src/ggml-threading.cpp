#समावेश "ggml-threading.h"
#समावेश <mutex>

std::mutex ggml_critical_section_mutex;

व्योम ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

व्योम ggml_critical_section_end(व्योम) {
    ggml_critical_section_mutex.unlock();
}
