#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ <mutex>

std::mutex ggml_critical_section_mutex;

ಶೂನ್ಯ ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

ಶೂನ್ಯ ggml_critical_section_end(ಶೂನ್ಯ) {
    ggml_critical_section_mutex.unlock();
}
