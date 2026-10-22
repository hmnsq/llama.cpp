#అన్తర్భావ "ggml-threading.h"
#అన్తర్భావ <mutex>

std::mutex ggml_critical_section_mutex;

శూన్య ggml_critical_section_ప్రారంభం() {
    ggml_critical_section_mutex.తాళం();
}

శూన్య ggml_critical_section_చివర(శూన్య) {
    ggml_critical_section_mutex.unlock();
}
