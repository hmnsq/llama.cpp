#ضمن "memtrace.hpp"

#ضمن "common.hpp"
#ضمن "ggml-impl.h"

#ضمن <cstdio>
#ضمن <mutex>
#ضمن <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ساكن ثابت حرف * mem_type_name(ggml_sycl_mem_type type) {
    بدل (type) {
        حالة GGML_SYCL_MEM_میانگیر:   أرجع "buffer";
        حالة GGML_SYCL_MEM_POOL_LEG: أرجع "pool_leg";
        حالة GGML_SYCL_MEM_POOL_VMM: أرجع "pool_vmm";
        حالة GGML_SYCL_MEM_ASYNC:    أرجع "async";
        حالة GGML_SYCL_MEM_مستقیم:   أرجع "direct";
        افتراضي:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحيح) type);
    }
}

بنية mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت فراغ *, std::pair<ggml_sycl_mem_type, size_t>> زنده_by_ptr;
    size_t زنده[GGML_SYCL_MEM_TYPE_شمار] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_شمار] = {};
    size_t مجموع_زنده = 0;
    size_t مجموع_peak = 0;
    size_t آخرین_logged_peak = 0;
};

ساكن mem_tracker & get_tracker() {
    ساكن mem_tracker t;
    أرجع t;
}

ساكن size_t گام_بایتها() {
    ثابت صحيح mib = g_ggml_sycl_memtrace_گام > 0 ? g_ggml_sycl_memtrace_گام : 64;
    أرجع (size_t) mib * MIB;
}

ساكن فراغ report_sites_locked() {
    mem_tracker & t = get_tracker();
    لكل (صحيح i = 0; i < GGML_SYCL_MEM_TYPE_شمار; i++) {
        إذا (t.peak[i] == 0) {
            استمر;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_برچسب "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.زنده[i] / MIB, t.peak[i] / MIB);
    }
}

ساكن فراغ report_locked(ثابت حرف * برچسب) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.مجموع_زنده / MIB;
    ثابت size_t buffers   = t.زنده[GGML_SYCL_MEM_میانگیر] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_برچسب " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  برچسب, allocated, buffers, allocated - buffers, t.مجموع_peak / MIB);
    report_sites_locked();
}

ساكن فراغ log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایتها) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_برچسب " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().مجموع_زنده / MIB, op, mem_type_name(type),
                  (مزدوج) بایتها / MIB, ptr);
}

منطقي ggml_sycl_memtrace_enabled() {
    أرجع g_ggml_sycl_memtrace > 0;
}

فراغ ggml_sycl_memtrace_جمع_کن(ggml_sycl_mem_type type, ثابت فراغ * ptr, size_t بایتها) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    GGML_ASSERT(ptr != اشاره_تهی);
    GGML_ASSERT(بایتها != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قفل(t.mutex);

    خودکار it = t.زنده_by_ptr.find(ptr);
    إذا (it != t.زنده_by_ptr.end()) {
        t.زنده[it->second.اولی] -= it->second.second;
        t.مجموع_زنده -= it->second.second;
    }

    t.زنده_by_ptr[ptr] = { type, بایتها };
    t.زنده[type] += بایتها;
    t.مجموع_زنده += بایتها;

    إذا (t.زنده[type] > t.peak[type]) {
        t.peak[type] = t.زنده[type];
    }
    إذا (t.مجموع_زنده > t.مجموع_peak) {
        t.مجموع_peak = t.مجموع_زنده;
    }

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, بایتها);
    }

    ساكن ثابت size_t گام = گام_بایتها();
    إذا (t.مجموع_peak >= t.آخرین_logged_peak + گام) {
        t.آخرین_logged_peak = t.مجموع_peak;
        حرف برچسب[96];
        std::snprintf(برچسب, حجم(برچسب), "peak grew (+%zu MiB from %s)", بایتها / MIB,
                      mem_type_name(type));
        report_locked(برچسب);
    }
}

فراغ ggml_sycl_memtrace_del(ثابت فراغ * ptr) {
    إذا (!ggml_sycl_memtrace_enabled() || ptr == اشاره_تهی) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قفل(t.mutex);

    خودکار it = t.زنده_by_ptr.find(ptr);
    إذا (it == t.زنده_by_ptr.end()) {
        أرجع;
    }
    ثابت ggml_sycl_mem_type type = it->second.اولی;
    ثابت size_t بایتها = it->second.second;
    t.زنده[type] -= بایتها;
    t.مجموع_زنده -= بایتها;
    t.زنده_by_ptr.erase(it);

    إذا (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, بایتها);
    }
}

فراغ ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بایتها) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_برچسب " alloc FAILED: %9.3f MiB %s\n",
                   (مزدوج) بایتها / MIB, mem_type_name(type));
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قفل(t.mutex);
    report_locked("at allocation failure");
}

فراغ ggml_sycl_memtrace_report(ثابت حرف * برچسب) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قفل(t.mutex);
    report_locked(برچسب);
}

ساكن منطقي device_حافظه_is_dedicated(صحيح device) {
    إذا (device < 0 || device >= ggml_sycl_info().device_شمار) {
        أرجع خطأ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    أرجع info.l0_device_type_valid && info.l0_discrete_gpu;
}

فراغ ggml_sycl_memtrace_report_device(ثابت حرف * برچسب, صحيح device, size_t dev_free, size_t dev_مجموع) {
    إذا (!ggml_sycl_memtrace_enabled()) {
        أرجع;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> قفل(t.mutex);

    ثابت size_t درون_use    = dev_مجموع > dev_free ? dev_مجموع - dev_free : 0;
    ثابت size_t مجموع     = dev_مجموع / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.مجموع_زنده / MIB;
    ثابت size_t buffers   = t.زنده[GGML_SYCL_MEM_میانگیر] / MIB;
    ثابت size_t peak      = t.مجموع_peak / MIB;

    إذا (درون_use >= t.مجموع_زنده && device_حافظه_is_dedicated(device) && مجموع >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_برچسب " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      برچسب, مجموع, freed, allocated, buffers, allocated - buffers,
                      مجموع - freed - allocated, peak);
    } وإلا {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_برچسب " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      برچسب, مجموع, freed, درون_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
