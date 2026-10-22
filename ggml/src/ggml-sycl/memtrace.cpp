#அந்தர்பாவ "memtrace.hpp"

#அந்தர்பாவ "common.hpp"
#அந்தர்பாவ "ggml-impl.h"

#அந்தர்பாவ <cstdio>
#அந்தர்பாவ <mutex>
#அந்தர்பாவ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

நிலையான மாறிலி எழுத்து * mem_type_name(ggml_sycl_mem_type type) {
    தேர்வு (type) {
        நிலை GGML_SYCL_MEM_BUFFER:   திரும்பு "buffer";
        நிலை GGML_SYCL_MEM_POOL_LEG: திரும்பு "pool_leg";
        நிலை GGML_SYCL_MEM_POOL_VMM: திரும்பு "pool_vmm";
        நிலை GGML_SYCL_MEM_ASYNC:    திரும்பு "async";
        நிலை GGML_SYCL_MEM_DIRECT:   திரும்பு "direct";
        இயல்பு:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (முழு) type);
    }
}

கட்டமைப்பு mem_tracker {
    std::mutex mutex;
    std::unordered_map<மாறிலி வெறுமை *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_மொத்தம்] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_மொத்தம்] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

நிலையான mem_tracker & get_tracker() {
    நிலையான mem_tracker t;
    திரும்பு t;
}

நிலையான size_t step_bytes() {
    மாறிலி முழு mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    திரும்பு (size_t) mib * MIB;
}

நிலையான வெறுமை report_sites_locked() {
    mem_tracker & t = get_tracker();
    க்கு (முழு i = 0; i < GGML_SYCL_MEM_TYPE_மொத்தம்; i++) {
        எனில் (t.peak[i] == 0) {
            தொடர்;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

நிலையான வெறுமை report_locked(மாறிலி எழுத்து * tag) {
    mem_tracker & t = get_tracker();

    மாறிலி size_t allocated = t.total_live / MIB;
    மாறிலி size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

நிலையான வெறுமை log_event_locked(மாறிலி எழுத்து * op, ggml_sycl_mem_type type, மாறிலி வெறுமை * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (இரட்டை) bytes / MIB, ptr);
}

தருக்க ggml_sycl_memtrace_enabled() {
    திரும்பு g_ggml_sycl_memtrace > 0;
}

வெறுமை ggml_sycl_memtrace_add(ggml_sycl_mem_type type, மாறிலி வெறுமை * ptr, size_t bytes) {
    எனில் (!ggml_sycl_memtrace_enabled()) {
        திரும்பு;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    எனில் (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    எனில் (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    எனில் (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    எனில் (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    நிலையான மாறிலி size_t step = step_bytes();
    எனில் (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        எழுத்து tag[96];
        std::snprintf(tag, அளவு(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

வெறுமை ggml_sycl_memtrace_del(மாறிலி வெறுமை * ptr) {
    எனில் (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        திரும்பு;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    எனில் (it == t.live_by_ptr.end()) {
        திரும்பு;
    }
    மாறிலி ggml_sycl_mem_type type = it->second.first;
    மாறிலி size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    எனில் (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

வெறுமை ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (இரட்டை) bytes / MIB, mem_type_name(type));
    எனில் (!ggml_sycl_memtrace_enabled()) {
        திரும்பு;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

வெறுமை ggml_sycl_memtrace_report(மாறிலி எழுத்து * tag) {
    எனில் (!ggml_sycl_memtrace_enabled()) {
        திரும்பு;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

நிலையான தருக்க device_memory_is_dedicated(முழு device) {
    எனில் (device < 0 || device >= ggml_sycl_info().device_மொத்தம்) {
        திரும்பு பொய்;
    }
    மாறிலி sycl_device_info & info = ggml_sycl_info().devices[device];
    திரும்பு info.l0_device_type_valid && info.l0_discrete_gpu;
}

வெறுமை ggml_sycl_memtrace_report_device(மாறிலி எழுத்து * tag, முழு device, size_t dev_free, size_t dev_total) {
    எனில் (!ggml_sycl_memtrace_enabled()) {
        திரும்பு;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    மாறிலி size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    மாறிலி size_t total     = dev_total / MIB;
    மாறிலி size_t freed     = dev_free / MIB;
    மாறிலி size_t allocated = t.total_live / MIB;
    மாறிலி size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    மாறிலி size_t peak      = t.total_peak / MIB;

    எனில் (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } இல்லையெனில் {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
