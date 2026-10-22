#अन्तर्भाव "ggml-threading.h"
#अन्तर्भाव <mutex>

std::mutex ggml_critical_section_mutex;

शून्य ggml_critical_section_start() {
    ggml_critical_section_mutex.तालकम्();
}

शून्य ggml_critical_section_अन्तः(शून्य) {
    ggml_critical_section_mutex.unlock();
}
