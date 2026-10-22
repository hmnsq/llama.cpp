#समावेश "memtrace.hpp"

#समावेश "common.hpp"
#समावेश "ggml-impl.h"

#समावेश <cstdio>
#समावेश <mutex>
#समावेश <unordered_map>

constexpr माप_प्रकार MIB = 1024 * 1024;

अटल स्थिर अक्षर * mem_type_name(ggml_sycl_mem_type type) {
    चयन (type) {
        हाल GGML_SYCL_MEM_बफर:   वापस "buffer";
        हाल GGML_SYCL_MEM_POOL_LEG: वापस "pool_leg";
        हाल GGML_SYCL_MEM_POOL_VMM: वापस "pool_vmm";
        हाल GGML_SYCL_MEM_ASYNC:    वापस "async";
        हाल GGML_SYCL_MEM_DIRECT:   वापस "direct";
        शेष:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (पूर्णांक) type);
    }
}

काष्ठा mem_tracker {
    std::mutex mutex;
    std::unordered_map<स्थिर व्योम *, std::pair<ggml_sycl_mem_type, माप_प्रकार>> live_by_ptr;
    माप_प्रकार live[GGML_SYCL_MEM_TYPE_गिनती] = {};
    माप_प्रकार peak[GGML_SYCL_MEM_TYPE_गिनती] = {};
    माप_प्रकार योग_live = 0;
    माप_प्रकार योग_peak = 0;
    माप_प्रकार last_logged_peak = 0;
};

अटल mem_tracker & get_tracker() {
    अटल mem_tracker t;
    वापस t;
}

अटल माप_प्रकार step_bytes() {
    स्थिर पूर्णांक mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    वापस (माप_प्रकार) mib * MIB;
}

अटल व्योम report_sites_locked() {
    mem_tracker & t = get_tracker();
    क्रम (पूर्णांक i = 0; i < GGML_SYCL_MEM_TYPE_गिनती; i++) {
        अगर (t.peak[i] == 0) {
            जारी;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

अटल व्योम report_locked(स्थिर अक्षर * tag) {
    mem_tracker & t = get_tracker();

    स्थिर माप_प्रकार allocated = t.योग_live / MIB;
    स्थिर माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_बफर] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.योग_peak / MIB);
    report_sites_locked();
}

अटल व्योम log_event_locked(स्थिर अक्षर * op, ggml_sycl_mem_type type, स्थिर व्योम * ptr, माप_प्रकार bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().योग_live / MIB, op, mem_type_name(type),
                  (द्विगुन) bytes / MIB, ptr);
}

bool ggml_sycl_memtrace_enabled() {
    वापस g_ggml_sycl_memtrace > 0;
}

व्योम ggml_sycl_memtrace_add(ggml_sycl_mem_type type, स्थिर व्योम * ptr, माप_प्रकार bytes) {
    अगर (!ggml_sycl_memtrace_enabled()) {
        वापस;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    अगर (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.योग_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.योग_live += bytes;

    अगर (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    अगर (t.योग_live > t.योग_peak) {
        t.योग_peak = t.योग_live;
    }

    अगर (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    अटल स्थिर माप_प्रकार step = step_bytes();
    अगर (t.योग_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.योग_peak;
        अक्षर tag[96];
        std::snprintf(tag, माप(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

व्योम ggml_sycl_memtrace_del(स्थिर व्योम * ptr) {
    अगर (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        वापस;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्वतः it = t.live_by_ptr.find(ptr);
    अगर (it == t.live_by_ptr.end()) {
        वापस;
    }
    स्थिर ggml_sycl_mem_type type = it->second.first;
    स्थिर माप_प्रकार bytes = it->second.second;
    t.live[type] -= bytes;
    t.योग_live -= bytes;
    t.live_by_ptr.erase(it);

    अगर (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

व्योम ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, माप_प्रकार bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (द्विगुन) bytes / MIB, mem_type_name(type));
    अगर (!ggml_sycl_memtrace_enabled()) {
        वापस;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

व्योम ggml_sycl_memtrace_report(स्थिर अक्षर * tag) {
    अगर (!ggml_sycl_memtrace_enabled()) {
        वापस;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

अटल bool device_memory_is_dedicated(पूर्णांक device) {
    अगर (device < 0 || device >= ggml_sycl_info().device_गिनती) {
        वापस false;
    }
    स्थिर sycl_device_info & info = ggml_sycl_info().devices[device];
    वापस info.l0_device_type_valid && info.l0_discrete_gpu;
}

व्योम ggml_sycl_memtrace_report_device(स्थिर अक्षर * tag, पूर्णांक device, माप_प्रकार dev_free, माप_प्रकार dev_योग) {
    अगर (!ggml_sycl_memtrace_enabled()) {
        वापस;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    स्थिर माप_प्रकार in_use    = dev_योग > dev_free ? dev_योग - dev_free : 0;
    स्थिर माप_प्रकार योग     = dev_योग / MIB;
    स्थिर माप_प्रकार freed     = dev_free / MIB;
    स्थिर माप_प्रकार allocated = t.योग_live / MIB;
    स्थिर माप_प्रकार buffers   = t.live[GGML_SYCL_MEM_बफर] / MIB;
    स्थिर माप_प्रकार peak      = t.योग_peak / MIB;

    अगर (in_use >= t.योग_live && device_memory_is_dedicated(device) && योग >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, योग, freed, allocated, buffers, allocated - buffers,
                      योग - freed - allocated, peak);
    } अन्यथा {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, योग, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
