#ಅನ್ತರ್ಭಾವ "ggml-threading.h"
#ಅನ್ತರ್ಭಾವ <mutex>

std::mutex ggml_critical_section_mutex;

ಶೂನ್ಯ ggml_critical_section_ಆರಂಭ() {
    ggml_critical_section_mutex.ಬೀಗ();
}

ಶೂನ್ಯ ggml_critical_section_ಕೊನೆ(ಶೂನ್ಯ) {
    ggml_critical_section_mutex.unlock();
}
