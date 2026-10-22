#شامل "ggml-threading.h"
#شامل <mutex>

std::mutex ggml_critical_section_mutex;

خالی ggml_critical_section_آغاز() {
    ggml_critical_section_mutex.تالا();
}

خالی ggml_critical_section_انتہا(خالی) {
    ggml_critical_section_mutex.unlock();
}
