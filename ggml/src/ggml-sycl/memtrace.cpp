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
        حالة GGML_SYCL_MEM_سڌو:   أرجع "direct";
        افتراضي:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحيح) type);
    }
}

بنية mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت فراغ *, std::pair<ggml_sycl_mem_type, size_t>> جيئرو_by_ptr;
    size_t جيئرو[GGML_SYCL_MEM_TYPE_ڳڻپ] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ڳڻپ] = {};
    size_t ڪل_جيئرو = 0;
    size_t ڪل_peak = 0;
    size_t آخري_logged_peak = 0;
};

ساكن mem_tracker & get_tracker() {
    ساكن mem_tracker t;
    أرجع t;
}

ساكن size_t قدم_بائيٽون() {
    ثابت صحيح mib = g_ggml_sycl_memtrace_قدم > 0 ? g_ggml_sycl_memtrace_قدم : 64;
    أرجع (size_t) mib * MIB;
}

ساكن فراغ report_sites_locked() {
    mem_tracker & t = get_tracker();
    لكل (صحيح i = 0; i < GGML_SYCL_MEM_TYPE_ڳڻپ; i++) {
        إذا (t.peak[i] == 0) {
            استمر;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.جيئرو[i] / MIB, t.peak[i] / MIB);
    }
}

ساكن فراغ report_locked(ثابت حرف * ٽيگ) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.ڪل_جيئرو / MIB;
    ثابت size_t buffers   = t.جيئرو[GGML_SYCL_MEM_بفر] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  ٽيگ, allocated, buffers, allocated - buffers, t.ڪل_peak / MIB);
    report_sites_locked();
}

ساكن فراغ log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائيٽون) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().ڪل_جيئرو / MIB, op, mem_type_name(type),
                  (مزدوج) بائيٽون / MIB, ptr);
}

منطقي ggml_sycl_memtrace_enabled() {
    أرجع g_ggml_sycl_memtrace > 0;
}

فراغ ggml_sycl_memtrace_جوڙيو(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بائيٽون) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    GGML_ASSERT(ptr != خالي_اشارو);
    GGML_ASSERT(بائيٽون != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالو(t.mutex);

    خودڪار it = t.جيئرو_by_ptr.find(ptr);
    إذا (it != t.جيئرو_by_ptr.end()) {
        t.جيئرو[it->second.پهريون] -= it->second.second;
        t.ڪل_جيئرو -= it->second.second;
    }

    t.جيئرو_by_ptr[ptr] = { type, بائيٽون };
    t.جيئرو[type] += بائيٽون;
    t.ڪل_جيئرو += بائيٽون;

    إذا (t.جيئرو[type] > t.peak[type]) {
        t.peak[type] = t.جيئرو[type];
    }
    إذا (t.ڪل_جيئرو > t.ڪل_peak) {
        t.ڪل_peak = t.ڪل_جيئرو;
    }

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, بائيٽون);
    }

    ساكن ثابت size_t قدم = قدم_بائيٽون();
    إذا (t.ڪل_peak >= t.آخري_logged_peak + قدم) {
        t.آخري_logged_peak = t.ڪل_peak;
        حرف ٽيگ[96];
        std::snprintf(ٽيگ, حجم(ٽيگ), "peak grew (+%zu MiB from %s)", بائيٽون / MIB,
                      mem_type_name(type));
        report_locked(ٽيگ);
    }
}

فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr) {
    إذا (!ggml_sycl_memtrace_enabled() || ptr == خالي_اشارو) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالو(t.mutex);

    خودڪار it = t.جيئرو_by_ptr.find(ptr);
    إذا (it == t.جيئرو_by_ptr.end()) {
        أرجع;
    }
    ثابت ggml_sycl_mem_type type = it->second.پهريون;
    ثابت size_t بائيٽون = it->second.second;
    t.جيئرو[type] -= بائيٽون;
    t.ڪل_جيئرو -= بائيٽون;
    t.جيئرو_by_ptr.erase(it);

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, بائيٽون);
    }
}

فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائيٽون) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_ٽيگ " alloc FAILED: %9.3f MiB %s\n",
                   (مزدوج) بائيٽون / MIB, mem_type_name(type));
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالو(t.mutex);
    report_locked("at allocation failure");
}

فراغ ggml_sycl_memtrace_report(ثابت حرف * ٽيگ) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالو(t.mutex);
    report_locked(ٽيگ);
}

ساكن منطقي device_يادگيري_is_dedicated(صحيح device) {
    إذا (device < 0 || device >= ggml_sycl_info().device_ڳڻپ) {
        أرجع خطأ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    أرجع info.l0_device_type_valid && info.l0_discrete_gpu;
}

فراغ ggml_sycl_memtrace_report_device(ثابت حرف * ٽيگ, صحيح device, size_t dev_free, size_t dev_ڪل) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالو(t.mutex);

    ثابت size_t اندريون_use    = dev_ڪل > dev_free ? dev_ڪل - dev_free : 0;
    ثابت size_t ڪل     = dev_ڪل / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.ڪل_جيئرو / MIB;
    ثابت size_t buffers   = t.جيئرو[GGML_SYCL_MEM_بفر] / MIB;
    ثابت size_t peak      = t.ڪل_peak / MIB;

    إذا (اندريون_use >= t.ڪل_جيئرو && device_يادگيري_is_dedicated(device) && ڪل >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      ٽيگ, ڪل, freed, allocated, buffers, allocated - buffers,
                      ڪل - freed - allocated, peak);
    } وإلا {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      ٽيگ, ڪل, freed, اندريون_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
