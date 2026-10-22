#അന്തര്ഭാവ "memtrace.hpp"

#അന്തര്ഭാവ "common.hpp"
#അന്തര്ഭാവ "ggml-impl.h"

#അന്തര്ഭാവ <cstdio>
#അന്തര്ഭാവ <mutex>
#അന്തര്ഭാവ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

സ്ഥിതം സ്ഥിരം അക്ഷരം * mem_type_name(ggml_sycl_mem_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_SYCL_MEM_BUFFER:   തിരികെ "buffer";
        അവസ്ഥ GGML_SYCL_MEM_POOL_LEG: തിരികെ "pool_leg";
        അവസ്ഥ GGML_SYCL_MEM_POOL_VMM: തിരികെ "pool_vmm";
        അവസ്ഥ GGML_SYCL_MEM_ASYNC:    തിരികെ "async";
        അവസ്ഥ GGML_SYCL_MEM_DIRECT:   തിരികെ "direct";
        സ്ഥിരസ്ഥിതി:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (പൂർണ്ണസംഖ്യ) type);
    }
}

ഘടന mem_tracker {
    std::mutex mutex;
    std::unordered_map<സ്ഥിരം ശൂന്യം *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_എണ്ണം] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_എണ്ണം] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

സ്ഥിതം mem_tracker & get_tracker() {
    സ്ഥിതം mem_tracker t;
    തിരികെ t;
}

സ്ഥിതം size_t step_bytes() {
    സ്ഥിരം പൂർണ്ണസംഖ്യ mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    തിരികെ (size_t) mib * MIB;
}

സ്ഥിതം ശൂന്യം report_sites_locked() {
    mem_tracker & t = get_tracker();
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_SYCL_MEM_TYPE_എണ്ണം; i++) {
        എങ്കിൽ (t.peak[i] == 0) {
            തുടരുക;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

സ്ഥിതം ശൂന്യം report_locked(സ്ഥിരം അക്ഷരം * tag) {
    mem_tracker & t = get_tracker();

    സ്ഥിരം size_t allocated = t.total_live / MIB;
    സ്ഥിരം size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

സ്ഥിതം ശൂന്യം log_event_locked(സ്ഥിരം അക്ഷരം * op, ggml_sycl_mem_type type, സ്ഥിരം ശൂന്യം * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (ഇരട്ട) bytes / MIB, ptr);
}

തർക്കം ggml_sycl_memtrace_enabled() {
    തിരികെ g_ggml_sycl_memtrace > 0;
}

ശൂന്യം ggml_sycl_memtrace_add(ggml_sycl_mem_type type, സ്ഥിരം ശൂന്യം * ptr, size_t bytes) {
    എങ്കിൽ (!ggml_sycl_memtrace_enabled()) {
        തിരികെ;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    എങ്കിൽ (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    എങ്കിൽ (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    എങ്കിൽ (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    എങ്കിൽ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    സ്ഥിതം സ്ഥിരം size_t step = step_bytes();
    എങ്കിൽ (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        അക്ഷരം tag[96];
        std::snprintf(tag, വലുപ്പം(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

ശൂന്യം ggml_sycl_memtrace_del(സ്ഥിരം ശൂന്യം * ptr) {
    എങ്കിൽ (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        തിരികെ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    എങ്കിൽ (it == t.live_by_ptr.end()) {
        തിരികെ;
    }
    സ്ഥിരം ggml_sycl_mem_type type = it->second.first;
    സ്ഥിരം size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    എങ്കിൽ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

ശൂന്യം ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (ഇരട്ട) bytes / MIB, mem_type_name(type));
    എങ്കിൽ (!ggml_sycl_memtrace_enabled()) {
        തിരികെ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

ശൂന്യം ggml_sycl_memtrace_report(സ്ഥിരം അക്ഷരം * tag) {
    എങ്കിൽ (!ggml_sycl_memtrace_enabled()) {
        തിരികെ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

സ്ഥിതം തർക്കം device_memory_is_dedicated(പൂർണ്ണസംഖ്യ device) {
    എങ്കിൽ (device < 0 || device >= ggml_sycl_info().device_എണ്ണം) {
        തിരികെ അസത്യം;
    }
    സ്ഥിരം sycl_device_info & info = ggml_sycl_info().devices[device];
    തിരികെ info.l0_device_type_valid && info.l0_discrete_gpu;
}

ശൂന്യം ggml_sycl_memtrace_report_device(സ്ഥിരം അക്ഷരം * tag, പൂർണ്ണസംഖ്യ device, size_t dev_free, size_t dev_total) {
    എങ്കിൽ (!ggml_sycl_memtrace_enabled()) {
        തിരികെ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    സ്ഥിരം size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    സ്ഥിരം size_t total     = dev_total / MIB;
    സ്ഥിരം size_t freed     = dev_free / MIB;
    സ്ഥിരം size_t allocated = t.total_live / MIB;
    സ്ഥിരം size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    സ്ഥിരം size_t peak      = t.total_peak / MIB;

    എങ്കിൽ (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } അല്ലെങ്കിൽ {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
