#अन्तर्भाव "memtrace.hpp"

#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdio>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>

constexpr size_t MIB = 1024 * 1024;

स्थिर ध्रुव वर्ण * mem_type_name(ggml_sycl_mem_type type) {
    विकल्प (type) {
        पक्ष GGML_SYCL_MEM_BUFFER:   पटिनिवत्त "buffer";
        पक्ष GGML_SYCL_MEM_POOL_LEG: पटिनिवत्त "pool_leg";
        पक्ष GGML_SYCL_MEM_POOL_VMM: पटिनिवत्त "pool_vmm";
        पक्ष GGML_SYCL_MEM_ASYNC:    पटिनिवत्त "async";
        पक्ष GGML_SYCL_MEM_DIRECT:   पटिनिवत्त "direct";
        स्वभाव:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (पूर्णाङ्क) type);
    }
}

संरचना mem_tracker {
    std::mutex mutex;
    std::unordered_map<ध्रुव शून्य *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_गणनम्] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_गणनम्] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

स्थिर mem_tracker & get_tracker() {
    स्थिर mem_tracker t;
    पटिनिवत्त t;
}

स्थिर size_t step_bytes() {
    ध्रुव पूर्णाङ्क mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    पटिनिवत्त (size_t) mib * MIB;
}

स्थिर शून्य report_sites_locked() {
    mem_tracker & t = get_tracker();
    अत्थाय (पूर्णाङ्क i = 0; i < GGML_SYCL_MEM_TYPE_गणनम्; i++) {
        सचे (t.peak[i] == 0) {
            अनुवर्त;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

स्थिर शून्य report_locked(ध्रुव वर्ण * tag) {
    mem_tracker & t = get_tracker();

    ध्रुव size_t allocated = t.total_live / MIB;
    ध्रुव size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

स्थिर शून्य log_event_locked(ध्रुव वर्ण * op, ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (द्विगुण) bytes / MIB, ptr);
}

तार्किक ggml_sycl_memtrace_enabled() {
    पटिनिवत्त g_ggml_sycl_memtrace > 0;
}

शून्य ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ध्रुव शून्य * ptr, size_t bytes) {
    सचे (!ggml_sycl_memtrace_enabled()) {
        पटिनिवत्त;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    सचे (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    सचे (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    सचे (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    सचे (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    स्थिर ध्रुव size_t step = step_bytes();
    सचे (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        वर्ण tag[96];
        std::snprintf(tag, परिमाण(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

शून्य ggml_sycl_memtrace_del(ध्रुव शून्य * ptr) {
    सचे (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        पटिनिवत्त;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    सचे (it == t.live_by_ptr.end()) {
        पटिनिवत्त;
    }
    ध्रुव ggml_sycl_mem_type type = it->second.first;
    ध्रुव size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    सचे (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (द्विगुण) bytes / MIB, mem_type_name(type));
    सचे (!ggml_sycl_memtrace_enabled()) {
        पटिनिवत्त;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

शून्य ggml_sycl_memtrace_report(ध्रुव वर्ण * tag) {
    सचे (!ggml_sycl_memtrace_enabled()) {
        पटिनिवत्त;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

स्थिर तार्किक device_memory_is_dedicated(पूर्णाङ्क device) {
    सचे (device < 0 || device >= ggml_sycl_info().device_गणनम्) {
        पटिनिवत्त मिच्छा;
    }
    ध्रुव sycl_device_info & info = ggml_sycl_info().devices[device];
    पटिनिवत्त info.l0_device_type_valid && info.l0_discrete_gpu;
}

शून्य ggml_sycl_memtrace_report_device(ध्रुव वर्ण * tag, पूर्णाङ्क device, size_t dev_free, size_t dev_total) {
    सचे (!ggml_sycl_memtrace_enabled()) {
        पटिनिवत्त;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ध्रुव size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ध्रुव size_t total     = dev_total / MIB;
    ध्रुव size_t freed     = dev_free / MIB;
    ध्रुव size_t allocated = t.total_live / MIB;
    ध्रुव size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ध्रुव size_t peak      = t.total_peak / MIB;

    सचे (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } अञ्ञथा {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
