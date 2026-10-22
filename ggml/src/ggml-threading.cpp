#અન્તર્ભાવ "ggml-threading.h"
#અન્તર્ભાવ <mutex>

std::mutex ggml_critical_section_mutex;

શૂન્ય ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

શૂન્ય ggml_critical_section_end(શૂન્ય) {
    ggml_critical_section_mutex.unlock();
}
