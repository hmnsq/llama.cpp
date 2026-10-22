#ضمن "ggml-threading.h"
#ضمن <mutex>

std::mutex ggml_critical_section_mutex;

فراغ ggml_critical_section_شروع() {
    ggml_critical_section_mutex.جندرا();
}

فراغ ggml_critical_section_سرا(فراغ) {
    ggml_critical_section_mutex.unlock();
}
