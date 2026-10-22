#अन्तर्भाव "memtrace.hpp"

#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdio>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>

constexpr size_t MIB = 1024 * 1024;

स्थिर ध्रुव वर्ण * mem_type_name(ggml_sycl_mem_type type) {
    विकल्प (type) {
        पक्ष GGML_SYCL_MEM_बफर:   प्रत्यागम "buffer";
        पक्ष GGML_SYCL_MEM_POOL_LEG: प्रत्यागम "pool_leg";
        पक्ष GGML_SYCL_MEM_POOL_VMM: प्रत्यागम "pool_vmm";
        पक्ष GGML_SYCL_MEM_ASYNC:    प्रत्यागम "async";
        पक्ष GGML_SYCL_MEM_प्रत्यक्षम्:   प्रत्यागम "direct";
        स्वभाव:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (पूर्णाङ्क) type);
    }
}

संरचना mem_tracker {
    std::mutex mutex;
    std::unordered_map<ध्रुव शून्य *, std::pair<ggml_sycl_mem_type, size_t>> जीवितम्_by_ptr;
    size_t जीवितम्[GGML_SYCL_MEM_TYPE_गणनम्] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_गणनम्] = {};
    size_t योग_जीवितम् = 0;
    size_t योग_peak = 0;
    size_t चरमः_logged_peak = 0;
};

स्थिर mem_tracker & get_tracker() {
    स्थिर mem_tracker t;
    प्रत्यागम t;
}

स्थिर size_t चरणः_अष्टकानि() {
    ध्रुव पूर्णाङ्क mib = g_ggml_sycl_memtrace_चरणः > 0 ? g_ggml_sycl_memtrace_चरणः : 64;
    प्रत्यागम (size_t) mib * MIB;
}

स्थिर शून्य report_sites_locked() {
    mem_tracker & t = get_tracker();
    प्रति (पूर्णाङ्क i = 0; i < GGML_SYCL_MEM_TYPE_गणनम्; i++) {
        यदि (t.peak[i] == 0) {
            अनुवर्त;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.जीवितम्[i] / MIB, t.peak[i] / MIB);
    }
}

स्थिर शून्य report_locked(ध्रुव वर्ण * चिह्नकम्) {
    mem_tracker & t = get_tracker();

    ध्रुव size_t allocated = t.योग_जीवितम् / MIB;
    ध्रुव size_t buffers   = t.जीवितम्[GGML_SYCL_MEM_बफर] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  चिह्नकम्, allocated, buffers, allocated - buffers, t.योग_peak / MIB);
    report_sites_locked();
}

स्थिर शून्य log_event_locked(ध्रुव वर्ण * op, ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t अष्टकानि) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().योग_जीवितम् / MIB, op, mem_type_name(type),
                  (द्विगुण) अष्टकानि / MIB, ptr);
}

तार्किक ggml_sycl_memtrace_enabled() {
    प्रत्यागम g_ggml_sycl_memtrace > 0;
}

शून्य ggml_sycl_memtrace_सङ्कलय(ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t अष्टकानि) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    GGML_ASSERT(ptr != शून्यसूचकः);
    GGML_ASSERT(अष्टकानि != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> तालकम्(t.mutex);

    स्वचालित it = t.जीवितम्_by_ptr.find(ptr);
    यदि (it != t.जीवितम्_by_ptr.end()) {
        t.जीवितम्[it->second.first] -= it->second.second;
        t.योग_जीवितम् -= it->second.second;
    }

    t.जीवितम्_by_ptr[ptr] = { type, अष्टकानि };
    t.जीवितम्[type] += अष्टकानि;
    t.योग_जीवितम् += अष्टकानि;

    यदि (t.जीवितम्[type] > t.peak[type]) {
        t.peak[type] = t.जीवितम्[type];
    }
    यदि (t.योग_जीवितम् > t.योग_peak) {
        t.योग_peak = t.योग_जीवितम्;
    }

    यदि (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, अष्टकानि);
    }

    स्थिर ध्रुव size_t चरणः = चरणः_अष्टकानि();
    यदि (t.योग_peak >= t.चरमः_logged_peak + चरणः) {
        t.चरमः_logged_peak = t.योग_peak;
        वर्ण चिह्नकम्[96];
        std::snprintf(चिह्नकम्, परिमाण(चिह्नकम्), "peak grew (+%zu MiB from %s)", अष्टकानि / MIB,
                      mem_type_name(type));
        report_locked(चिह्नकम्);
    }
}

शून्य ggml_sycl_memtrace_del(ध्रुव शून्य * ptr) {
    यदि (!ggml_sycl_memtrace_enabled() || ptr == शून्यसूचकः) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> तालकम्(t.mutex);

    स्वचालित it = t.जीवितम्_by_ptr.find(ptr);
    यदि (it == t.जीवितम्_by_ptr.end()) {
        प्रत्यागम;
    }
    ध्रुव ggml_sycl_mem_type type = it->second.first;
    ध्रुव size_t अष्टकानि = it->second.second;
    t.जीवितम्[type] -= अष्टकानि;
    t.योग_जीवितम् -= अष्टकानि;
    t.जीवितम्_by_ptr.erase(it);

    यदि (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, अष्टकानि);
    }
}

शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t अष्टकानि) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_चिह्नकम् " alloc FAILED: %9.3f MiB %s\n",
                   (द्विगुण) अष्टकानि / MIB, mem_type_name(type));
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> तालकम्(t.mutex);
    report_locked("at allocation failure");
}

शून्य ggml_sycl_memtrace_report(ध्रुव वर्ण * चिह्नकम्) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> तालकम्(t.mutex);
    report_locked(चिह्नकम्);
}

स्थिर तार्किक device_स्मृतिः_is_dedicated(पूर्णाङ्क device) {
    यदि (device < 0 || device >= ggml_sycl_info().device_गणनम्) {
        प्रत्यागम असत्य;
    }
    ध्रुव sycl_device_info & info = ggml_sycl_info().devices[device];
    प्रत्यागम info.l0_device_type_valid && info.l0_discrete_gpu;
}

शून्य ggml_sycl_memtrace_report_device(ध्रुव वर्ण * चिह्नकम्, पूर्णाङ्क device, size_t dev_free, size_t dev_योग) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> तालकम्(t.mutex);

    ध्रुव size_t अन्तर्_use    = dev_योग > dev_free ? dev_योग - dev_free : 0;
    ध्रुव size_t योग     = dev_योग / MIB;
    ध्रुव size_t freed     = dev_free / MIB;
    ध्रुव size_t allocated = t.योग_जीवितम् / MIB;
    ध्रुव size_t buffers   = t.जीवितम्[GGML_SYCL_MEM_बफर] / MIB;
    ध्रुव size_t peak      = t.योग_peak / MIB;

    यदि (अन्तर्_use >= t.योग_जीवितम् && device_स्मृतिः_is_dedicated(device) && योग >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      चिह्नकम्, योग, freed, allocated, buffers, allocated - buffers,
                      योग - freed - allocated, peak);
    } अन्यथा {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      चिह्नकम्, योग, freed, अन्तर्_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
