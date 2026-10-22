#ಅನ್ತರ್ಭಾವ "memtrace.hpp"

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * mem_type_name(ggml_sycl_mem_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_SYCL_MEM_ಬಫರ್:   ಹಿಂತಿರುಗಿ "buffer";
        ಸ್ಥಿತಿ GGML_SYCL_MEM_POOL_LEG: ಹಿಂತಿರುಗಿ "pool_leg";
        ಸ್ಥಿತಿ GGML_SYCL_MEM_POOL_VMM: ಹಿಂತಿರುಗಿ "pool_vmm";
        ಸ್ಥಿತಿ GGML_SYCL_MEM_ASYNC:    ಹಿಂತಿರುಗಿ "async";
        ಸ್ಥಿತಿ GGML_SYCL_MEM_ನೇರ:   ಹಿಂತಿರುಗಿ "direct";
        ಪೂರ್ವನಿಯೋಜಿತ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (ಪೂರ್ಣಾಂಕ) type);
    }
}

ರಚನೆ mem_tracker {
    std::mutex mutex;
    std::unordered_map<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, std::pair<ggml_sycl_mem_type, size_t>> ಜೀವಂತ_by_ptr;
    size_t ಜೀವಂತ[GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ] = {};
    size_t ಒಟ್ಟು_ಜೀವಂತ = 0;
    size_t ಒಟ್ಟು_peak = 0;
    size_t ಕೊನೆಯ_logged_peak = 0;
};

ಸ್ಥಿರ mem_tracker & get_tracker() {
    ಸ್ಥಿರ mem_tracker t;
    ಹಿಂತಿರುಗಿ t;
}

ಸ್ಥಿರ size_t ಹೆಜ್ಜೆ_ಬೈಟುಗಳು() {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mib = g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ > 0 ? g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ : 64;
    ಹಿಂತಿರುಗಿ (size_t) mib * MIB;
}

ಸ್ಥಿರ ಶೂನ್ಯ report_sites_locked() {
    mem_tracker & t = get_tracker();
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_SYCL_MEM_TYPE_ಎಣಿಕೆ; i++) {
        ಆದರೆ (t.peak[i] == 0) {
            ಮುಂದುವರಿ;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.ಜೀವಂತ[i] / MIB, t.peak[i] / MIB);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ report_locked(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್) {
    mem_tracker & t = get_tracker();

    ಸ್ಥಿರಾಂಕ size_t allocated = t.ಒಟ್ಟು_ಜೀವಂತ / MIB;
    ಸ್ಥಿರಾಂಕ size_t buffers   = t.ಜೀವಂತ[GGML_SYCL_MEM_ಬಫರ್] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  ಟ್ಯಾಗ್, allocated, buffers, allocated - buffers, t.ಒಟ್ಟು_peak / MIB);
    report_sites_locked();
}

ಸ್ಥಿರ ಶೂನ್ಯ log_event_locked(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * op, ggml_sycl_mem_type type, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr, size_t ಬೈಟುಗಳು) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().ಒಟ್ಟು_ಜೀವಂತ / MIB, op, mem_type_name(type),
                  (ದ್ವಿಗುಣ) ಬೈಟುಗಳು / MIB, ptr);
}

ತಾರ್ಕಿಕ ggml_sycl_memtrace_enabled() {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_memtrace > 0;
}

ಶೂನ್ಯ ggml_sycl_memtrace_ಕೂಡಿಸು(ggml_sycl_mem_type type, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr, size_t ಬೈಟುಗಳು) {
    ಆದರೆ (!ggml_sycl_memtrace_enabled()) {
        ಹಿಂತಿರುಗಿ;
    }
    GGML_ASSERT(ptr != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(ಬೈಟುಗಳು != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ಬೀಗ(t.mutex);

    ಸ್ವಯಂಚಾಲಿತ it = t.ಜೀವಂತ_by_ptr.find(ptr);
    ಆದರೆ (it != t.ಜೀವಂತ_by_ptr.end()) {
        t.ಜೀವಂತ[it->second.ಮೊದಲ] -= it->second.second;
        t.ಒಟ್ಟು_ಜೀವಂತ -= it->second.second;
    }

    t.ಜೀವಂತ_by_ptr[ptr] = { type, ಬೈಟುಗಳು };
    t.ಜೀವಂತ[type] += ಬೈಟುಗಳು;
    t.ಒಟ್ಟು_ಜೀವಂತ += ಬೈಟುಗಳು;

    ಆದರೆ (t.ಜೀವಂತ[type] > t.peak[type]) {
        t.peak[type] = t.ಜೀವಂತ[type];
    }
    ಆದರೆ (t.ಒಟ್ಟು_ಜೀವಂತ > t.ಒಟ್ಟು_peak) {
        t.ಒಟ್ಟು_peak = t.ಒಟ್ಟು_ಜೀವಂತ;
    }

    ಆದರೆ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, ಬೈಟುಗಳು);
    }

    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t ಹೆಜ್ಜೆ = ಹೆಜ್ಜೆ_ಬೈಟುಗಳು();
    ಆದರೆ (t.ಒಟ್ಟು_peak >= t.ಕೊನೆಯ_logged_peak + ಹೆಜ್ಜೆ) {
        t.ಕೊನೆಯ_logged_peak = t.ಒಟ್ಟು_peak;
        ಅಕ್ಷರ ಟ್ಯಾಗ್[96];
        std::snprintf(ಟ್ಯಾಗ್, ಗಾತ್ರ(ಟ್ಯಾಗ್), "peak grew (+%zu MiB from %s)", ಬೈಟುಗಳು / MIB,
                      mem_type_name(type));
        report_locked(ಟ್ಯಾಗ್);
    }
}

ಶೂನ್ಯ ggml_sycl_memtrace_del(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ptr) {
    ಆದರೆ (!ggml_sycl_memtrace_enabled() || ptr == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ಬೀಗ(t.mutex);

    ಸ್ವಯಂಚಾಲಿತ it = t.ಜೀವಂತ_by_ptr.find(ptr);
    ಆದರೆ (it == t.ಜೀವಂತ_by_ptr.end()) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ggml_sycl_mem_type type = it->second.ಮೊದಲ;
    ಸ್ಥಿರಾಂಕ size_t ಬೈಟುಗಳು = it->second.second;
    t.ಜೀವಂತ[type] -= ಬೈಟುಗಳು;
    t.ಒಟ್ಟು_ಜೀವಂತ -= ಬೈಟುಗಳು;
    t.ಜೀವಂತ_by_ptr.erase(it);

    ಆದರೆ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, ಬೈಟುಗಳು);
    }
}

ಶೂನ್ಯ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t ಬೈಟುಗಳು) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " alloc FAILED: %9.3f MiB %s\n",
                   (ದ್ವಿಗುಣ) ಬೈಟುಗಳು / MIB, mem_type_name(type));
    ಆದರೆ (!ggml_sycl_memtrace_enabled()) {
        ಹಿಂತಿರುಗಿ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ಬೀಗ(t.mutex);
    report_locked("at allocation failure");
}

ಶೂನ್ಯ ggml_sycl_memtrace_report(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್) {
    ಆದರೆ (!ggml_sycl_memtrace_enabled()) {
        ಹಿಂತಿರುಗಿ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ಬೀಗ(t.mutex);
    report_locked(ಟ್ಯಾಗ್);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ device_ಸ್ಮರಣೆ_is_dedicated(ಪೂರ್ಣಾಂಕ device) {
    ಆದರೆ (device < 0 || device >= ggml_sycl_info().device_ಎಣಿಕೆ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಸ್ಥಿರಾಂಕ sycl_device_info & info = ggml_sycl_info().devices[device];
    ಹಿಂತಿರುಗಿ info.l0_device_type_valid && info.l0_discrete_gpu;
}

ಶೂನ್ಯ ggml_sycl_memtrace_report_device(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಟ್ಯಾಗ್, ಪೂರ್ಣಾಂಕ device, size_t dev_free, size_t dev_ಒಟ್ಟು) {
    ಆದರೆ (!ggml_sycl_memtrace_enabled()) {
        ಹಿಂತಿರುಗಿ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ಬೀಗ(t.mutex);

    ಸ್ಥಿರಾಂಕ size_t ಒಳ_use    = dev_ಒಟ್ಟು > dev_free ? dev_ಒಟ್ಟು - dev_free : 0;
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು     = dev_ಒಟ್ಟು / MIB;
    ಸ್ಥಿರಾಂಕ size_t freed     = dev_free / MIB;
    ಸ್ಥಿರಾಂಕ size_t allocated = t.ಒಟ್ಟು_ಜೀವಂತ / MIB;
    ಸ್ಥಿರಾಂಕ size_t buffers   = t.ಜೀವಂತ[GGML_SYCL_MEM_ಬಫರ್] / MIB;
    ಸ್ಥಿರಾಂಕ size_t peak      = t.ಒಟ್ಟು_peak / MIB;

    ಆದರೆ (ಒಳ_use >= t.ಒಟ್ಟು_ಜೀವಂತ && device_ಸ್ಮರಣೆ_is_dedicated(device) && ಒಟ್ಟು >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      ಟ್ಯಾಗ್, ಒಟ್ಟು, freed, allocated, buffers, allocated - buffers,
                      ಒಟ್ಟು - freed - allocated, peak);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      ಟ್ಯಾಗ್, ಒಟ್ಟು, freed, ಒಳ_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
