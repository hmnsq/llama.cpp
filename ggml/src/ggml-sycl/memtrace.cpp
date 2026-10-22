#ضمن "memtrace.hpp"

#ضمن "common.hpp"
#ضمن "ggml-impl.h"

#ضمن <cstdio>
#ضمن <mutex>
#ضمن <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ساكن ثابت حرف * mem_type_name(ggml_sycl_mem_type type) {
    بدل (type) {
        حالة GGML_SYCL_MEM_BUFFER:   أرجع "buffer";
        حالة GGML_SYCL_MEM_POOL_LEG: أرجع "pool_leg";
        حالة GGML_SYCL_MEM_POOL_VMM: أرجع "pool_vmm";
        حالة GGML_SYCL_MEM_ASYNC:    أرجع "async";
        حالة GGML_SYCL_MEM_DIRECT:   أرجع "direct";
        افتراضي:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحيح) type);
    }
}

بنية mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت فراغ *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_عدد] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_عدد] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

ساكن mem_tracker & get_tracker() {
    ساكن mem_tracker t;
    أرجع t;
}

ساكن size_t step_bytes() {
    ثابت صحيح mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    أرجع (size_t) mib * MIB;
}

ساكن فراغ report_sites_locked() {
    mem_tracker & t = get_tracker();
    لكل (صحيح i = 0; i < GGML_SYCL_MEM_TYPE_عدد; i++) {
        إذا (t.peak[i] == 0) {
            استمر;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

ساكن فراغ report_locked(ثابت حرف * tag) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.total_live / MIB;
    ثابت size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

ساكن فراغ log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (مزدوج) bytes / MIB, ptr);
}

منطقي ggml_sycl_memtrace_enabled() {
    أرجع g_ggml_sycl_memtrace > 0;
}

فراغ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t bytes) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    إذا (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    إذا (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    إذا (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    ساكن ثابت size_t step = step_bytes();
    إذا (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        حرف tag[96];
        std::snprintf(tag, حجم(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr) {
    إذا (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    إذا (it == t.live_by_ptr.end()) {
        أرجع;
    }
    ثابت ggml_sycl_mem_type type = it->second.first;
    ثابت size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (مزدوج) bytes / MIB, mem_type_name(type));
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

فراغ ggml_sycl_memtrace_report(ثابت حرف * tag) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

ساكن منطقي device_memory_is_dedicated(صحيح device) {
    إذا (device < 0 || device >= ggml_sycl_info().device_عدد) {
        أرجع خطأ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    أرجع info.l0_device_type_valid && info.l0_discrete_gpu;
}

فراغ ggml_sycl_memtrace_report_device(ثابت حرف * tag, صحيح device, size_t dev_free, size_t dev_total) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ثابت size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ثابت size_t total     = dev_total / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.total_live / MIB;
    ثابت size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ثابت size_t peak      = t.total_peak / MIB;

    إذا (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } وإلا {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
