#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-threading.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>

std::mutex ggml_critical_section_mutex;

ਖਾਲੀ ggml_critical_section_ਸ਼ੁਰੂ() {
    ggml_critical_section_mutex.ਤਾਲਾ();
}

ਖਾਲੀ ggml_critical_section_ਸਿਰਾ(ਖਾਲੀ) {
    ggml_critical_section_mutex.unlock();
}
