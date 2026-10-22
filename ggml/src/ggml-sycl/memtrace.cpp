#ଅନ୍ତର୍ଭାଵ "memtrace.hpp"

#ଅନ୍ତର୍ଭାଵ "common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

#ଅନ୍ତର୍ଭାଵ <cstdio>
#ଅନ୍ତର୍ଭାଵ <mutex>
#ଅନ୍ତର୍ଭାଵ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * mem_type_name(ggml_sycl_mem_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_SYCL_MEM_BUFFER:   ଫେରନ୍ତୁ "buffer";
        ସ୍ଥିତି GGML_SYCL_MEM_POOL_LEG: ଫେରନ୍ତୁ "pool_leg";
        ସ୍ଥିତି GGML_SYCL_MEM_POOL_VMM: ଫେରନ୍ତୁ "pool_vmm";
        ସ୍ଥିତି GGML_SYCL_MEM_ASYNC:    ଫେରନ୍ତୁ "async";
        ସ୍ଥିତି GGML_SYCL_MEM_DIRECT:   ଫେରନ୍ତୁ "direct";
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) type);
    }
}

ସଂରଚନା mem_tracker {
    std::mutex mutex;
    std::unordered_map<ଅଚଳ ଶୂନ୍ୟ *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_ଗଣନା] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ଗଣନା] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

ସ୍ଥିର mem_tracker & get_tracker() {
    ସ୍ଥିର mem_tracker t;
    ଫେରନ୍ତୁ t;
}

ସ୍ଥିର size_t step_bytes() {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    ଫେରନ୍ତୁ (size_t) mib * MIB;
}

ସ୍ଥିର ଶୂନ୍ୟ report_sites_locked() {
    mem_tracker & t = get_tracker();
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_SYCL_MEM_TYPE_ଗଣନା; i++) {
        ଯଦି (t.peak[i] == 0) {
            ଜାରି;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ report_locked(ଅଚଳ ଅକ୍ଷର * tag) {
    mem_tracker & t = get_tracker();

    ଅଚଳ size_t allocated = t.total_live / MIB;
    ଅଚଳ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

ସ୍ଥିର ଶୂନ୍ୟ log_event_locked(ଅଚଳ ଅକ୍ଷର * op, ggml_sycl_mem_type type, ଅଚଳ ଶୂନ୍ୟ * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (ଦ୍ୱିଗୁଣ) bytes / MIB, ptr);
}

ତାର୍କିକ ggml_sycl_memtrace_enabled() {
    ଫେରନ୍ତୁ g_ggml_sycl_memtrace > 0;
}

ଶୂନ୍ୟ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ଅଚଳ ଶୂନ୍ୟ * ptr, size_t bytes) {
    ଯଦି (!ggml_sycl_memtrace_enabled()) {
        ଫେରନ୍ତୁ;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    ଯଦି (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    ଯଦି (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    ଯଦି (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    ଯଦି (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    ସ୍ଥିର ଅଚଳ size_t step = step_bytes();
    ଯଦି (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        ଅକ୍ଷର tag[96];
        std::snprintf(tag, ଆକାର(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

ଶୂନ୍ୟ ggml_sycl_memtrace_del(ଅଚଳ ଶୂନ୍ୟ * ptr) {
    ଯଦି (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        ଫେରନ୍ତୁ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    ଯଦି (it == t.live_by_ptr.end()) {
        ଫେରନ୍ତୁ;
    }
    ଅଚଳ ggml_sycl_mem_type type = it->second.first;
    ଅଚଳ size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    ଯଦି (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

ଶୂନ୍ୟ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (ଦ୍ୱିଗୁଣ) bytes / MIB, mem_type_name(type));
    ଯଦି (!ggml_sycl_memtrace_enabled()) {
        ଫେରନ୍ତୁ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

ଶୂନ୍ୟ ggml_sycl_memtrace_report(ଅଚଳ ଅକ୍ଷର * tag) {
    ଯଦି (!ggml_sycl_memtrace_enabled()) {
        ଫେରନ୍ତୁ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

ସ୍ଥିର ତାର୍କିକ device_memory_is_dedicated(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଯଦି (device < 0 || device >= ggml_sycl_info().device_ଗଣନା) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଅଚଳ sycl_device_info & info = ggml_sycl_info().devices[device];
    ଫେରନ୍ତୁ info.l0_device_type_valid && info.l0_discrete_gpu;
}

ଶୂନ୍ୟ ggml_sycl_memtrace_report_device(ଅଚଳ ଅକ୍ଷର * tag, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t dev_free, size_t dev_total) {
    ଯଦି (!ggml_sycl_memtrace_enabled()) {
        ଫେରନ୍ତୁ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ଅଚଳ size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ଅଚଳ size_t total     = dev_total / MIB;
    ଅଚଳ size_t freed     = dev_free / MIB;
    ଅଚଳ size_t allocated = t.total_live / MIB;
    ଅଚଳ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ଅଚଳ size_t peak      = t.total_peak / MIB;

    ଯଦି (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } ଅନ୍ୟଥା {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
