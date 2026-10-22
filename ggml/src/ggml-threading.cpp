#ضمن "ggml-threading.h"
#ضمن <mutex>

std::mutex ggml_critical_section_mutex;

فراغ ggml_critical_section_شروعات() {
    ggml_critical_section_mutex.تالو();
}

فراغ ggml_critical_section_پڇاڙي(فراغ) {
    ggml_critical_section_mutex.unlock();
}
