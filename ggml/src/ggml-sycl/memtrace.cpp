#ضمن "memtrace.hpp"

#ضمن "common.hpp"
#ضمن "ggml-impl.h"

#ضمن <cstdio>
#ضمن <mutex>
#ضمن <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ساكن ثابت حرف * mem_type_name(ggml_sycl_mem_type type) {
    بدل (type) {
        حالة GGML_SYCL_MEM_بفر:   أرجع "buffer";
        حالة GGML_SYCL_MEM_POOL_LEG: أرجع "pool_leg";
        حالة GGML_SYCL_MEM_POOL_VMM: أرجع "pool_vmm";
        حالة GGML_SYCL_MEM_ASYNC:    أرجع "async";
        حالة GGML_SYCL_MEM_سدھا:   أرجع "direct";
        افتراضي:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحيح) type);
    }
}

بنية mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت فراغ *, std::pair<ggml_sycl_mem_type, size_t>> جیوندا_by_ptr;
    size_t جیوندا[GGML_SYCL_MEM_TYPE_گنتی] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_گنتی] = {};
    size_t کل_جیوندا = 0;
    size_t کل_peak = 0;
    size_t آخری_logged_peak = 0;
};

ساكن mem_tracker & get_tracker() {
    ساكن mem_tracker t;
    أرجع t;
}

ساكن size_t قدم_بائٹاں() {
    ثابت صحيح mib = g_ggml_sycl_memtrace_قدم > 0 ? g_ggml_sycl_memtrace_قدم : 64;
    أرجع (size_t) mib * MIB;
}

ساكن فراغ report_sites_locked() {
    mem_tracker & t = get_tracker();
    لكل (صحيح i = 0; i < GGML_SYCL_MEM_TYPE_گنتی; i++) {
        إذا (t.peak[i] == 0) {
            استمر;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٹیگ "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.جیوندا[i] / MIB, t.peak[i] / MIB);
    }
}

ساكن فراغ report_locked(ثابت حرف * ٹیگ) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.کل_جیوندا / MIB;
    ثابت size_t buffers   = t.جیوندا[GGML_SYCL_MEM_بفر] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٹیگ " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  ٹیگ, allocated, buffers, allocated - buffers, t.کل_peak / MIB);
    report_sites_locked();
}

ساكن فراغ log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائٹاں) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٹیگ " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().کل_جیوندا / MIB, op, mem_type_name(type),
                  (مزدوج) بائٹاں / MIB, ptr);
}

منطقي ggml_sycl_memtrace_enabled() {
    أرجع g_ggml_sycl_memtrace > 0;
}

فراغ ggml_sycl_memtrace_جوڑو(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائٹاں) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    GGML_ASSERT(ptr != خالی_سوچک);
    GGML_ASSERT(بائٹاں != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> جندرا(t.mutex);

    خودکار it = t.جیوندا_by_ptr.find(ptr);
    إذا (it != t.جیوندا_by_ptr.end()) {
        t.جیوندا[it->second.پہلا] -= it->second.second;
        t.کل_جیوندا -= it->second.second;
    }

    t.جیوندا_by_ptr[ptr] = { type, بائٹاں };
    t.جیوندا[type] += بائٹاں;
    t.کل_جیوندا += بائٹاں;

    إذا (t.جیوندا[type] > t.peak[type]) {
        t.peak[type] = t.جیوندا[type];
    }
    إذا (t.کل_جیوندا > t.کل_peak) {
        t.کل_peak = t.کل_جیوندا;
    }

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, بائٹاں);
    }

    ساكن ثابت size_t قدم = قدم_بائٹاں();
    إذا (t.کل_peak >= t.آخری_logged_peak + قدم) {
        t.آخری_logged_peak = t.کل_peak;
        حرف ٹیگ[96];
        std::snprintf(ٹیگ, حجم(ٹیگ), "peak grew (+%zu MiB from %s)", بائٹاں / MIB,
                      mem_type_name(type));
        report_locked(ٹیگ);
    }
}

فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr) {
    إذا (!ggml_sycl_memtrace_enabled() || ptr == خالی_سوچک) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> جندرا(t.mutex);

    خودکار it = t.جیوندا_by_ptr.find(ptr);
    إذا (it == t.جیوندا_by_ptr.end()) {
        أرجع;
    }
    ثابت ggml_sycl_mem_type type = it->second.پہلا;
    ثابت size_t بائٹاں = it->second.second;
    t.جیوندا[type] -= بائٹاں;
    t.کل_جیوندا -= بائٹاں;
    t.جیوندا_by_ptr.erase(it);

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, بائٹاں);
    }
}

فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائٹاں) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_ٹیگ " alloc FAILED: %9.3f MiB %s\n",
                   (مزدوج) بائٹاں / MIB, mem_type_name(type));
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> جندرا(t.mutex);
    report_locked("at allocation failure");
}

فراغ ggml_sycl_memtrace_report(ثابت حرف * ٹیگ) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> جندرا(t.mutex);
    report_locked(ٹیگ);
}

ساكن منطقي device_حافظہ_is_dedicated(صحيح device) {
    إذا (device < 0 || device >= ggml_sycl_info().device_گنتی) {
        أرجع خطأ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    أرجع info.l0_device_type_valid && info.l0_discrete_gpu;
}

فراغ ggml_sycl_memtrace_report_device(ثابت حرف * ٹیگ, صحيح device, size_t dev_free, size_t dev_کل) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> جندرا(t.mutex);

    ثابت size_t وچ_use    = dev_کل > dev_free ? dev_کل - dev_free : 0;
    ثابت size_t کل     = dev_کل / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.کل_جیوندا / MIB;
    ثابت size_t buffers   = t.جیوندا[GGML_SYCL_MEM_بفر] / MIB;
    ثابت size_t peak      = t.کل_peak / MIB;

    إذا (وچ_use >= t.کل_جیوندا && device_حافظہ_is_dedicated(device) && کل >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٹیگ " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      ٹیگ, کل, freed, allocated, buffers, allocated - buffers,
                      کل - freed - allocated, peak);
    } وإلا {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٹیگ " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      ٹیگ, کل, freed, وچ_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
