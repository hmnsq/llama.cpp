#अन्तर्भाव "memtrace.hpp"

#अन्तर्भाव "common.hpp"
#अन्तर्भाव "ggml-impl.h"

#अन्तर्भाव <cstdio>
#अन्तर्भाव <mutex>
#अन्तर्भाव <unordered_map>

constexpr माप_प्रकार MIB = 1024 * 1024;

स्थिर स्थिरांक अक्षर * mem_type_name(ggml_sycl_mem_type type) {
    निवड (type) {
        स्थिती GGML_SYCL_MEM_BUFFER:   परत "buffer";
        स्थिती GGML_SYCL_MEM_POOL_LEG: परत "pool_leg";
        स्थिती GGML_SYCL_MEM_POOL_VMM: परत "pool_vmm";
        स्थिती GGML_SYCL_MEM_ASYNC:    परत "async";
        स्थिती GGML_SYCL_MEM_DIRECT:   परत "direct";
        पूर्वनिर्धारित:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (पूर्णांक) type);
    }
}

संरचना mem_tracker {
    std::mutex mutex;
    std::unordered_map<स्थिरांक शून्य *, std::pair<ggml_sycl_mem_type, माप_प्रकार>> live_by_ptr;
    माप_प्रकार live[GGML_SYCL_MEM_TYPE_मोजणी] = {};
    माप_प्रकार peak[GGML_SYCL_MEM_TYPE_मोजणी] = {};
    माप_प्रकार total_live = 0;
    माप_प्रकार total_peak = 0;
    माप_प्रकार last_logged_peak = 0;
};

स्थिर mem_tracker & get_tracker() {
    स्थिर mem_tracker t;
    परत t;
}

स्थिर माप_प्रकार step_bytes() {
    स्थिरांक पूर्णांक mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    परत (माप_प्रकार) mib * MIB;
}

स्थिर शून्य report_sites_locked() {
    mem_tracker & t = get_tracker();
    साठी (पूर्णांक i = 0; i < GGML_SYCL_MEM_TYPE_मोजणी; i++) {
        जर (t.peak[i] == 0) {
            सुरूठेवा;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

स्थिर शून्य report_locked(स्थिरांक अक्षर * tag) {
    mem_tracker & t = get_tracker();

    स्थिरांक माप_प्रकार allocated = t.total_live / MIB;
    स्थिरांक माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

स्थिर शून्य log_event_locked(स्थिरांक अक्षर * op, ggml_sycl_mem_type type, स्थिरांक शून्य * ptr, माप_प्रकार bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (द्विगुण) bytes / MIB, ptr);
}

तार्किक ggml_sycl_memtrace_enabled() {
    परत g_ggml_sycl_memtrace > 0;
}

शून्य ggml_sycl_memtrace_add(ggml_sycl_mem_type type, स्थिरांक शून्य * ptr, माप_प्रकार bytes) {
    जर (!ggml_sycl_memtrace_enabled()) {
        परत;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    जर (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    जर (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    जर (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    जर (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    स्थिर स्थिरांक माप_प्रकार step = step_bytes();
    जर (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        अक्षर tag[96];
        std::snprintf(tag, आकार(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

शून्य ggml_sycl_memtrace_del(स्थिरांक शून्य * ptr) {
    जर (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        परत;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    जर (it == t.live_by_ptr.end()) {
        परत;
    }
    स्थिरांक ggml_sycl_mem_type type = it->second.first;
    स्थिरांक माप_प्रकार bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    जर (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

शून्य ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, माप_प्रकार bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (द्विगुण) bytes / MIB, mem_type_name(type));
    जर (!ggml_sycl_memtrace_enabled()) {
        परत;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

शून्य ggml_sycl_memtrace_report(स्थिरांक अक्षर * tag) {
    जर (!ggml_sycl_memtrace_enabled()) {
        परत;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

स्थिर तार्किक device_memory_is_dedicated(पूर्णांक device) {
    जर (device < 0 || device >= ggml_sycl_info().device_मोजणी) {
        परत असत्य;
    }
    स्थिरांक sycl_device_info & info = ggml_sycl_info().devices[device];
    परत info.l0_device_type_valid && info.l0_discrete_gpu;
}

शून्य ggml_sycl_memtrace_report_device(स्थिरांक अक्षर * tag, पूर्णांक device, माप_प्रकार dev_free, माप_प्रकार dev_total) {
    जर (!ggml_sycl_memtrace_enabled()) {
        परत;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्थिरांक माप_प्रकार in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    स्थिरांक माप_प्रकार total     = dev_total / MIB;
    स्थिरांक माप_प्रकार freed     = dev_free / MIB;
    स्थिरांक माप_प्रकार allocated = t.total_live / MIB;
    स्थिरांक माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    स्थिरांक माप_प्रकार peak      = t.total_peak / MIB;

    जर (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
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
