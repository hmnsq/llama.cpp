#ಅನ್ತರ್ಭಾವ "memtrace.hpp"

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * mem_type_name(ggml_sycl_mem_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_SYCL_MEM_BUFFER:   ಪ್ರತ್ಯಾಗಮ "buffer";
        ಪಕ್ಷ GGML_SYCL_MEM_POOL_LEG: ಪ್ರತ್ಯಾಗಮ "pool_leg";
        ಪಕ್ಷ GGML_SYCL_MEM_POOL_VMM: ಪ್ರತ್ಯಾಗಮ "pool_vmm";
        ಪಕ್ಷ GGML_SYCL_MEM_ASYNC:    ಪ್ರತ್ಯಾಗಮ "async";
        ಪಕ್ಷ GGML_SYCL_MEM_DIRECT:   ಪ್ರತ್ಯಾಗಮ "direct";
        ಸ್ವಭಾವ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (ಪೂರ್ಣಾಙ್ಕ) type);
    }
}

ಸಂರಚನಾ mem_tracker {
    std::mutex mutex;
    std::unordered_map<ಧ್ರುವ ಶೂನ್ಯ *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

ಸ್ಥಿರ mem_tracker & get_tracker() {
    ಸ್ಥಿರ mem_tracker t;
    ಪ್ರತ್ಯಾಗಮ t;
}

ಸ್ಥಿರ size_t step_bytes() {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    ಪ್ರತ್ಯಾಗಮ (size_t) mib * MIB;
}

ಸ್ಥಿರ ಶೂನ್ಯ report_sites_locked() {
    mem_tracker & t = get_tracker();
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ; i++) {
        ಯದಿ (t.peak[i] == 0) {
            ಅನುವರ್ತ;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ report_locked(ಧ್ರುವ ವರ್ಣ * tag) {
    mem_tracker & t = get_tracker();

    ಧ್ರುವ size_t allocated = t.total_live / MIB;
    ಧ್ರುವ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

ಸ್ಥಿರ ಶೂನ್ಯ log_event_locked(ಧ್ರುವ ವರ್ಣ * op, ggml_sycl_mem_type type, ಧ್ರುವ ಶೂನ್ಯ * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (ದ್ವಿಗುಣ) bytes / MIB, ptr);
}

ತಾರ್ಕಿಕ ggml_sycl_memtrace_enabled() {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_memtrace > 0;
}

ಶೂನ್ಯ ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ಧ್ರುವ ಶೂನ್ಯ * ptr, size_t bytes) {
    ಯದಿ (!ggml_sycl_memtrace_enabled()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    ಯದಿ (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    ಯದಿ (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    ಯದಿ (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    ಯದಿ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    ಸ್ಥಿರ ಧ್ರುವ size_t step = step_bytes();
    ಯದಿ (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        ವರ್ಣ tag[96];
        std::snprintf(tag, ಪರಿಮಾಣ(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

ಶೂನ್ಯ ggml_sycl_memtrace_del(ಧ್ರುವ ಶೂನ್ಯ * ptr) {
    ಯದಿ (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    ಯದಿ (it == t.live_by_ptr.end()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ ggml_sycl_mem_type type = it->second.first;
    ಧ್ರುವ size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    ಯದಿ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

ಶೂನ್ಯ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (ದ್ವಿಗುಣ) bytes / MIB, mem_type_name(type));
    ಯದಿ (!ggml_sycl_memtrace_enabled()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

ಶೂನ್ಯ ggml_sycl_memtrace_report(ಧ್ರುವ ವರ್ಣ * tag) {
    ಯದಿ (!ggml_sycl_memtrace_enabled()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ device_memory_is_dedicated(ಪೂರ್ಣಾಙ್ಕ device) {
    ಯದಿ (device < 0 || device >= ggml_sycl_info().device_ಎಣಿಕೆ) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಧ್ರುವ sycl_device_info & info = ggml_sycl_info().devices[device];
    ಪ್ರತ್ಯಾಗಮ info.l0_device_type_valid && info.l0_discrete_gpu;
}

ಶೂನ್ಯ ggml_sycl_memtrace_report_device(ಧ್ರುವ ವರ್ಣ * tag, ಪೂರ್ಣಾಙ್ಕ device, size_t dev_free, size_t dev_total) {
    ಯದಿ (!ggml_sycl_memtrace_enabled()) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ಧ್ರುವ size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ಧ್ರುವ size_t total     = dev_total / MIB;
    ಧ್ರುವ size_t freed     = dev_free / MIB;
    ಧ್ರುವ size_t allocated = t.total_live / MIB;
    ಧ್ರುವ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ಧ್ರುವ size_t peak      = t.total_peak / MIB;

    ಯದಿ (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } ಅನ್ಯಥಾ {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
