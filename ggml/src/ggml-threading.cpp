#ضمن "ggml-threading.h"
#ضمن <mutex>

std::mutex ggml_critical_section_mutex;

فراغ ggml_critical_section_پیل() {
    ggml_critical_section_mutex.قلف();
}

فراغ ggml_critical_section_پای_ټکی(فراغ) {
    ggml_critical_section_mutex.unlock();
}
