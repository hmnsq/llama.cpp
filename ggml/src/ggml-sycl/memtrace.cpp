#ਅਨ੍ਤਰ੍ਭਾਵ "memtrace.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * mem_type_name(ggml_sycl_mem_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_SYCL_MEM_ਬਫਰ:   ਵਾਪਸ "buffer";
        ਹਾਲਤ GGML_SYCL_MEM_POOL_LEG: ਵਾਪਸ "pool_leg";
        ਹਾਲਤ GGML_SYCL_MEM_POOL_VMM: ਵਾਪਸ "pool_vmm";
        ਹਾਲਤ GGML_SYCL_MEM_ASYNC:    ਵਾਪਸ "async";
        ਹਾਲਤ GGML_SYCL_MEM_ਸਿੱਧਾ:   ਵਾਪਸ "direct";
        ਮੂਲ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (ਪੂਰਨ) type);
    }
}

ਬਣਤਰ mem_tracker {
    std::mutex mutex;
    std::unordered_map<ਅਚਲ ਖਾਲੀ *, std::pair<ggml_sycl_mem_type, size_t>> ਜੀਵਤ_by_ptr;
    size_t ਜੀਵਤ[GGML_SYCL_MEM_TYPE_ਗਿਣਤੀ] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ਗਿਣਤੀ] = {};
    size_t ਕੁੱਲ_ਜੀਵਤ = 0;
    size_t ਕੁੱਲ_peak = 0;
    size_t ਆਖਰੀ_logged_peak = 0;
};

ਸਥਿਰ mem_tracker & get_tracker() {
    ਸਥਿਰ mem_tracker t;
    ਵਾਪਸ t;
}

ਸਥਿਰ size_t ਕਦਮ_ਬਾਈਟਾਂ() {
    ਅਚਲ ਪੂਰਨ mib = g_ggml_sycl_memtrace_ਕਦਮ > 0 ? g_ggml_sycl_memtrace_ਕਦਮ : 64;
    ਵਾਪਸ (size_t) mib * MIB;
}

ਸਥਿਰ ਖਾਲੀ report_sites_locked() {
    mem_tracker & t = get_tracker();
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_SYCL_MEM_TYPE_ਗਿਣਤੀ; i++) {
        ਜੇ (t.peak[i] == 0) {
            ਜਾਰੀ;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.ਜੀਵਤ[i] / MIB, t.peak[i] / MIB);
    }
}

ਸਥਿਰ ਖਾਲੀ report_locked(ਅਚਲ ਅੱਖਰ * ਟੈਗ) {
    mem_tracker & t = get_tracker();

    ਅਚਲ size_t allocated = t.ਕੁੱਲ_ਜੀਵਤ / MIB;
    ਅਚਲ size_t buffers   = t.ਜੀਵਤ[GGML_SYCL_MEM_ਬਫਰ] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  ਟੈਗ, allocated, buffers, allocated - buffers, t.ਕੁੱਲ_peak / MIB);
    report_sites_locked();
}

ਸਥਿਰ ਖਾਲੀ log_event_locked(ਅਚਲ ਅੱਖਰ * op, ggml_sycl_mem_type type, ਅਚਲ ਖਾਲੀ * ptr, size_t ਬਾਈਟਾਂ) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().ਕੁੱਲ_ਜੀਵਤ / MIB, op, mem_type_name(type),
                  (ਦੁੱਗਣਾ) ਬਾਈਟਾਂ / MIB, ptr);
}

ਤਾਰਕਿਕ ggml_sycl_memtrace_enabled() {
    ਵਾਪਸ g_ggml_sycl_memtrace > 0;
}

ਖਾਲੀ ggml_sycl_memtrace_ਜੋੜੋ(ggml_sycl_mem_type type, ਅਚਲ ਖਾਲੀ * ptr, size_t ਬਾਈਟਾਂ) {
    ਜੇ (!ggml_sycl_memtrace_enabled()) {
        ਵਾਪਸ;
    }
    GGML_ASSERT(ptr != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(ਬਾਈਟਾਂ != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ਤਾਲਾ(t.mutex);

    ਸਵੈਚਾਲਿਤ it = t.ਜੀਵਤ_by_ptr.find(ptr);
    ਜੇ (it != t.ਜੀਵਤ_by_ptr.end()) {
        t.ਜੀਵਤ[it->second.ਪਹਿਲਾ] -= it->second.second;
        t.ਕੁੱਲ_ਜੀਵਤ -= it->second.second;
    }

    t.ਜੀਵਤ_by_ptr[ptr] = { type, ਬਾਈਟਾਂ };
    t.ਜੀਵਤ[type] += ਬਾਈਟਾਂ;
    t.ਕੁੱਲ_ਜੀਵਤ += ਬਾਈਟਾਂ;

    ਜੇ (t.ਜੀਵਤ[type] > t.peak[type]) {
        t.peak[type] = t.ਜੀਵਤ[type];
    }
    ਜੇ (t.ਕੁੱਲ_ਜੀਵਤ > t.ਕੁੱਲ_peak) {
        t.ਕੁੱਲ_peak = t.ਕੁੱਲ_ਜੀਵਤ;
    }

    ਜੇ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, ਬਾਈਟਾਂ);
    }

    ਸਥਿਰ ਅਚਲ size_t ਕਦਮ = ਕਦਮ_ਬਾਈਟਾਂ();
    ਜੇ (t.ਕੁੱਲ_peak >= t.ਆਖਰੀ_logged_peak + ਕਦਮ) {
        t.ਆਖਰੀ_logged_peak = t.ਕੁੱਲ_peak;
        ਅੱਖਰ ਟੈਗ[96];
        std::snprintf(ਟੈਗ, ਆਕਾਰ(ਟੈਗ), "peak grew (+%zu MiB from %s)", ਬਾਈਟਾਂ / MIB,
                      mem_type_name(type));
        report_locked(ਟੈਗ);
    }
}

ਖਾਲੀ ggml_sycl_memtrace_del(ਅਚਲ ਖਾਲੀ * ptr) {
    ਜੇ (!ggml_sycl_memtrace_enabled() || ptr == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ਤਾਲਾ(t.mutex);

    ਸਵੈਚਾਲਿਤ it = t.ਜੀਵਤ_by_ptr.find(ptr);
    ਜੇ (it == t.ਜੀਵਤ_by_ptr.end()) {
        ਵਾਪਸ;
    }
    ਅਚਲ ggml_sycl_mem_type type = it->second.ਪਹਿਲਾ;
    ਅਚਲ size_t ਬਾਈਟਾਂ = it->second.second;
    t.ਜੀਵਤ[type] -= ਬਾਈਟਾਂ;
    t.ਕੁੱਲ_ਜੀਵਤ -= ਬਾਈਟਾਂ;
    t.ਜੀਵਤ_by_ptr.erase(it);

    ਜੇ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, ਬਾਈਟਾਂ);
    }
}

ਖਾਲੀ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t ਬਾਈਟਾਂ) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_ਟੈਗ " alloc FAILED: %9.3f MiB %s\n",
                   (ਦੁੱਗਣਾ) ਬਾਈਟਾਂ / MIB, mem_type_name(type));
    ਜੇ (!ggml_sycl_memtrace_enabled()) {
        ਵਾਪਸ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ਤਾਲਾ(t.mutex);
    report_locked("at allocation failure");
}

ਖਾਲੀ ggml_sycl_memtrace_report(ਅਚਲ ਅੱਖਰ * ਟੈਗ) {
    ਜੇ (!ggml_sycl_memtrace_enabled()) {
        ਵਾਪਸ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ਤਾਲਾ(t.mutex);
    report_locked(ਟੈਗ);
}

ਸਥਿਰ ਤਾਰਕਿਕ device_ਸਮ੍ਰਿਤੀ_is_dedicated(ਪੂਰਨ device) {
    ਜੇ (device < 0 || device >= ggml_sycl_info().device_ਗਿਣਤੀ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਅਚਲ sycl_device_info & info = ggml_sycl_info().devices[device];
    ਵਾਪਸ info.l0_device_type_valid && info.l0_discrete_gpu;
}

ਖਾਲੀ ggml_sycl_memtrace_report_device(ਅਚਲ ਅੱਖਰ * ਟੈਗ, ਪੂਰਨ device, size_t dev_free, size_t dev_ਕੁੱਲ) {
    ਜੇ (!ggml_sycl_memtrace_enabled()) {
        ਵਾਪਸ;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> ਤਾਲਾ(t.mutex);

    ਅਚਲ size_t ਵਿੱਚ_use    = dev_ਕੁੱਲ > dev_free ? dev_ਕੁੱਲ - dev_free : 0;
    ਅਚਲ size_t ਕੁੱਲ     = dev_ਕੁੱਲ / MIB;
    ਅਚਲ size_t freed     = dev_free / MIB;
    ਅਚਲ size_t allocated = t.ਕੁੱਲ_ਜੀਵਤ / MIB;
    ਅਚਲ size_t buffers   = t.ਜੀਵਤ[GGML_SYCL_MEM_ਬਫਰ] / MIB;
    ਅਚਲ size_t peak      = t.ਕੁੱਲ_peak / MIB;

    ਜੇ (ਵਿੱਚ_use >= t.ਕੁੱਲ_ਜੀਵਤ && device_ਸਮ੍ਰਿਤੀ_is_dedicated(device) && ਕੁੱਲ >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      ਟੈਗ, ਕੁੱਲ, freed, allocated, buffers, allocated - buffers,
                      ਕੁੱਲ - freed - allocated, peak);
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      ਟੈਗ, ਕੁੱਲ, freed, ਵਿੱਚ_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
