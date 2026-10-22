#अन्तर्भाव "memtrace.hpp"

#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdio>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>

constexpr माप_प्रकार MIB = 1024 * 1024;

स्थिर ध्रुव वर्ण * mem_type_name(ggml_sycl_mem_type type) {
    विकल्प (type) {
        पक्ष GGML_SYCL_MEM_BUFFER:   प्रत्यागम "buffer";
        पक्ष GGML_SYCL_MEM_POOL_LEG: प्रत्यागम "pool_leg";
        पक्ष GGML_SYCL_MEM_POOL_VMM: प्रत्यागम "pool_vmm";
        पक्ष GGML_SYCL_MEM_ASYNC:    प्रत्यागम "async";
        पक्ष GGML_SYCL_MEM_DIRECT:   प्रत्यागम "direct";
        स्वभाव:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (पूर्णाङ्क) type);
    }
}

संरचना mem_tracker {
    std::mutex mutex;
    std::unordered_map<ध्रुव शून्य *, std::pair<ggml_sycl_mem_type, माप_प्रकार>> live_by_ptr;
    माप_प्रकार live[GGML_SYCL_MEM_TYPE_गिनती] = {};
    माप_प्रकार peak[GGML_SYCL_MEM_TYPE_गिनती] = {};
    माप_प्रकार total_live = 0;
    माप_प्रकार total_peak = 0;
    माप_प्रकार last_logged_peak = 0;
};

स्थिर mem_tracker & get_tracker() {
    स्थिर mem_tracker t;
    प्रत्यागम t;
}

स्थिर माप_प्रकार step_bytes() {
    ध्रुव पूर्णाङ्क mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    प्रत्यागम (माप_प्रकार) mib * MIB;
}

स्थिर शून्य report_sites_locked() {
    mem_tracker & t = get_tracker();
    प्रति (पूर्णाङ्क i = 0; i < GGML_SYCL_MEM_TYPE_गिनती; i++) {
        यदि (t.peak[i] == 0) {
            अनुवर्त;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

स्थिर शून्य report_locked(ध्रुव वर्ण * tag) {
    mem_tracker & t = get_tracker();

    ध्रुव माप_प्रकार allocated = t.total_live / MIB;
    ध्रुव माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

स्थिर शून्य log_event_locked(ध्रुव वर्ण * op, ggml_sycl_mem_type type, ध्रुव शून्य * ptr, माप_प्रकार bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (द्विगुण) bytes / MIB, ptr);
}

तार्किक ggml_sycl_memtrace_enabled() {
    प्रत्यागम g_ggml_sycl_memtrace > 0;
}

शून्य ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ध्रुव शून्य * ptr, माप_प्रकार bytes) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    यदि (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    यदि (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    यदि (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    यदि (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    स्थिर ध्रुव माप_प्रकार step = step_bytes();
    यदि (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        वर्ण tag[96];
        std::snprintf(tag, परिमाण(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

शून्य ggml_sycl_memtrace_del(ध्रुव शून्य * ptr) {
    यदि (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    यदि (it == t.live_by_ptr.end()) {
        प्रत्यागम;
    }
    ध्रुव ggml_sycl_mem_type type = it->second.first;
    ध्रुव माप_प्रकार bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    यदि (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, माप_प्रकार bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (द्विगुण) bytes / MIB, mem_type_name(type));
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

शून्य ggml_sycl_memtrace_report(ध्रुव वर्ण * tag) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

स्थिर तार्किक device_memory_is_dedicated(पूर्णाङ्क device) {
    यदि (device < 0 || device >= ggml_sycl_info().device_गिनती) {
        प्रत्यागम असत्य;
    }
    ध्रुव sycl_device_info & info = ggml_sycl_info().devices[device];
    प्रत्यागम info.l0_device_type_valid && info.l0_discrete_gpu;
}

शून्य ggml_sycl_memtrace_report_device(ध्रुव वर्ण * tag, पूर्णाङ्क device, माप_प्रकार dev_free, माप_प्रकार dev_total) {
    यदि (!ggml_sycl_memtrace_enabled()) {
        प्रत्यागम;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ध्रुव माप_प्रकार in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ध्रुव माप_प्रकार total     = dev_total / MIB;
    ध्रुव माप_प्रकार freed     = dev_free / MIB;
    ध्रुव माप_प्रकार allocated = t.total_live / MIB;
    ध्रुव माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ध्रुव माप_प्रकार peak      = t.total_peak / MIB;

    यदि (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } अन्यथा {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
