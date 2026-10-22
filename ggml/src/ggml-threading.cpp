#ଅନ୍ତର୍ଭାଵ "ggml-threading.h"
#ଅନ୍ତର୍ଭାଵ <mutex>

std::mutex ggml_critical_section_mutex;

ଶୂନ୍ୟ ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

ଶୂନ୍ୟ ggml_critical_section_end(ଶୂନ୍ୟ) {
    ggml_critical_section_mutex.unlock();
}
