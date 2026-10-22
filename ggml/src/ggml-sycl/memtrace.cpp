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
        حالة GGML_SYCL_MEM_مستقیم:   أرجع "direct";
        افتراضي:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحيح) type);
    }
}

بنية mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت فراغ *, std::pair<ggml_sycl_mem_type, size_t>> ژوندی_by_ptr;
    size_t ژوندی[GGML_SYCL_MEM_TYPE_شمېر] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_شمېر] = {};
    size_t ټول_ژوندی = 0;
    size_t ټول_peak = 0;
    size_t وروستی_logged_peak = 0;
};

ساكن mem_tracker & get_tracker() {
    ساكن mem_tracker t;
    أرجع t;
}

ساكن size_t ګام_بایټونه() {
    ثابت صحيح mib = g_ggml_sycl_memtrace_ګام > 0 ? g_ggml_sycl_memtrace_ګام : 64;
    أرجع (size_t) mib * MIB;
}

ساكن فراغ report_sites_locked() {
    mem_tracker & t = get_tracker();
    لكل (صحيح i = 0; i < GGML_SYCL_MEM_TYPE_شمېر; i++) {
        إذا (t.peak[i] == 0) {
            استمر;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_نښان "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.ژوندی[i] / MIB, t.peak[i] / MIB);
    }
}

ساكن فراغ report_locked(ثابت حرف * نښان) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.ټول_ژوندی / MIB;
    ثابت size_t buffers   = t.ژوندی[GGML_SYCL_MEM_بفر] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_نښان " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  نښان, allocated, buffers, allocated - buffers, t.ټول_peak / MIB);
    report_sites_locked();
}

ساكن فراغ log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایټونه) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_نښان " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().ټول_ژوندی / MIB, op, mem_type_name(type),
                  (مزدوج) بایټونه / MIB, ptr);
}

منطقي ggml_sycl_memtrace_enabled() {
    أرجع g_ggml_sycl_memtrace > 0;
}

فراغ ggml_sycl_memtrace_جمع_کړه(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایټونه) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    GGML_ASSERT(ptr != تش_اشاره);
    GGML_ASSERT(بایټونه != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قلف(t.mutex);

    خپلکاری it = t.ژوندی_by_ptr.find(ptr);
    إذا (it != t.ژوندی_by_ptr.end()) {
        t.ژوندی[it->second.لومړی] -= it->second.second;
        t.ټول_ژوندی -= it->second.second;
    }

    t.ژوندی_by_ptr[ptr] = { type, بایټونه };
    t.ژوندی[type] += بایټونه;
    t.ټول_ژوندی += بایټونه;

    إذا (t.ژوندی[type] > t.peak[type]) {
        t.peak[type] = t.ژوندی[type];
    }
    إذا (t.ټول_ژوندی > t.ټول_peak) {
        t.ټول_peak = t.ټول_ژوندی;
    }

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, بایټونه);
    }

    ساكن ثابت size_t ګام = ګام_بایټونه();
    إذا (t.ټول_peak >= t.وروستی_logged_peak + ګام) {
        t.وروستی_logged_peak = t.ټول_peak;
        حرف نښان[96];
        std::snprintf(نښان, حجم(نښان), "peak grew (+%zu MiB from %s)", بایټونه / MIB,
                      mem_type_name(type));
        report_locked(نښان);
    }
}

فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr) {
    إذا (!ggml_sycl_memtrace_enabled() || ptr == تش_اشاره) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قلف(t.mutex);

    خپلکاری it = t.ژوندی_by_ptr.find(ptr);
    إذا (it == t.ژوندی_by_ptr.end()) {
        أرجع;
    }
    ثابت ggml_sycl_mem_type type = it->second.لومړی;
    ثابت size_t بایټونه = it->second.second;
    t.ژوندی[type] -= بایټونه;
    t.ټول_ژوندی -= بایټونه;
    t.ژوندی_by_ptr.erase(it);

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, بایټونه);
    }
}

فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بایټونه) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_نښان " alloc FAILED: %9.3f MiB %s\n",
                   (مزدوج) بایټونه / MIB, mem_type_name(type));
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قلف(t.mutex);
    report_locked("at allocation failure");
}

فراغ ggml_sycl_memtrace_report(ثابت حرف * نښان) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قلف(t.mutex);
    report_locked(نښان);
}

ساكن منطقي device_حافظه_is_dedicated(صحيح device) {
    إذا (device < 0 || device >= ggml_sycl_info().device_شمېر) {
        أرجع خطأ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    أرجع info.l0_device_type_valid && info.l0_discrete_gpu;
}

فراغ ggml_sycl_memtrace_report_device(ثابت حرف * نښان, صحيح device, size_t dev_free, size_t dev_ټول) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قلف(t.mutex);

    ثابت size_t دننه_use    = dev_ټول > dev_free ? dev_ټول - dev_free : 0;
    ثابت size_t ټول     = dev_ټول / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.ټول_ژوندی / MIB;
    ثابت size_t buffers   = t.ژوندی[GGML_SYCL_MEM_بفر] / MIB;
    ثابت size_t peak      = t.ټول_peak / MIB;

    إذا (دننه_use >= t.ټول_ژوندی && device_حافظه_is_dedicated(device) && ټول >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_نښان " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      نښان, ټول, freed, allocated, buffers, allocated - buffers,
                      ټول - freed - allocated, peak);
    } وإلا {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_نښان " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      نښان, ټول, freed, دننه_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
