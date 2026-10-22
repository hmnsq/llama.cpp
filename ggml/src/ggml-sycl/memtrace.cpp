#شامل "memtrace.hpp"

#شامل "common.hpp"
#شامل "ggml-impl.h"

#شامل <cstdio>
#شامل <mutex>
#شامل <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ساکن ثابت حرف * mem_type_name(ggml_sycl_mem_type type) {
    انتخاب (type) {
        حالت GGML_SYCL_MEM_بفر:   واپس "buffer";
        حالت GGML_SYCL_MEM_POOL_LEG: واپس "pool_leg";
        حالت GGML_SYCL_MEM_POOL_VMM: واپس "pool_vmm";
        حالت GGML_SYCL_MEM_ASYNC:    واپس "async";
        حالت GGML_SYCL_MEM_براہ_راست:   واپس "direct";
        طےشدہ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحیح) type);
    }
}

ساخت mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت خالی *, std::pair<ggml_sycl_mem_type, size_t>> زندہ_by_ptr;
    size_t زندہ[GGML_SYCL_MEM_TYPE_گنتی] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_گنتی] = {};
    size_t کل_زندہ = 0;
    size_t کل_peak = 0;
    size_t آخری_logged_peak = 0;
};

ساکن mem_tracker & get_tracker() {
    ساکن mem_tracker t;
    واپس t;
}

ساکن size_t قدم_بائٹیں() {
    ثابت صحیح mib = g_ggml_sycl_memtrace_قدم > 0 ? g_ggml_sycl_memtrace_قدم : 64;
    واپس (size_t) mib * MIB;
}

ساکن خالی report_sites_locked() {
    mem_tracker & t = get_tracker();
    برائے (صحیح i = 0; i < GGML_SYCL_MEM_TYPE_گنتی; i++) {
        اگر (t.peak[i] == 0) {
            جاری;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.زندہ[i] / MIB, t.peak[i] / MIB);
    }
}

ساکن خالی report_locked(ثابت حرف * لیبل) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.کل_زندہ / MIB;
    ثابت size_t buffers   = t.زندہ[GGML_SYCL_MEM_بفر] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  لیبل, allocated, buffers, allocated - buffers, t.کل_peak / MIB);
    report_sites_locked();
}

ساکن خالی log_event_locked(ثابت حرف * op, ggml_sycl_mem_type type, ثابت خالی * ptr, size_t بائٹیں) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().کل_زندہ / MIB, op, mem_type_name(type),
                  (دوہرا) بائٹیں / MIB, ptr);
}

منطقی ggml_sycl_memtrace_enabled() {
    واپس g_ggml_sycl_memtrace > 0;
}

خالی ggml_sycl_memtrace_جوڑو(ggml_sycl_mem_type type, ثابت خالی * ptr, size_t بائٹیں) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        واپس;
    }
    GGML_ASSERT(ptr != خالی_اشاریہ);
    GGML_ASSERT(بائٹیں != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالا(t.mutex);

    خودکار it = t.زندہ_by_ptr.find(ptr);
    اگر (it != t.زندہ_by_ptr.end()) {
        t.زندہ[it->second.پہلا] -= it->second.second;
        t.کل_زندہ -= it->second.second;
    }

    t.زندہ_by_ptr[ptr] = { type, بائٹیں };
    t.زندہ[type] += بائٹیں;
    t.کل_زندہ += بائٹیں;

    اگر (t.زندہ[type] > t.peak[type]) {
        t.peak[type] = t.زندہ[type];
    }
    اگر (t.کل_زندہ > t.کل_peak) {
        t.کل_peak = t.کل_زندہ;
    }

    اگر (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, بائٹیں);
    }

    ساکن ثابت size_t قدم = قدم_بائٹیں();
    اگر (t.کل_peak >= t.آخری_logged_peak + قدم) {
        t.آخری_logged_peak = t.کل_peak;
        حرف لیبل[96];
        std::snprintf(لیبل, حجم(لیبل), "peak grew (+%zu MiB from %s)", بائٹیں / MIB,
                      mem_type_name(type));
        report_locked(لیبل);
    }
}

خالی ggml_sycl_memtrace_del(ثابت خالی * ptr) {
    اگر (!ggml_sycl_memtrace_enabled() || ptr == خالی_اشاریہ) {
        واپس;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالا(t.mutex);

    خودکار it = t.زندہ_by_ptr.find(ptr);
    اگر (it == t.زندہ_by_ptr.end()) {
        واپس;
    }
    ثابت ggml_sycl_mem_type type = it->second.پہلا;
    ثابت size_t بائٹیں = it->second.second;
    t.زندہ[type] -= بائٹیں;
    t.کل_زندہ -= بائٹیں;
    t.زندہ_by_ptr.erase(it);

    اگر (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, بائٹیں);
    }
}

خالی ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t بائٹیں) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_لیبل " alloc FAILED: %9.3f MiB %s\n",
                   (دوہرا) بائٹیں / MIB, mem_type_name(type));
    اگر (!ggml_sycl_memtrace_enabled()) {
        واپس;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالا(t.mutex);
    report_locked("at allocation failure");
}

خالی ggml_sycl_memtrace_report(ثابت حرف * لیبل) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        واپس;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالا(t.mutex);
    report_locked(لیبل);
}

ساکن منطقی device_حافظہ_is_dedicated(صحیح device) {
    اگر (device < 0 || device >= ggml_sycl_info().device_گنتی) {
        واپس جھوٹ;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    واپس info.l0_device_type_valid && info.l0_discrete_gpu;
}

خالی ggml_sycl_memtrace_report_device(ثابت حرف * لیبل, صحیح device, size_t dev_free, size_t dev_کل) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        واپس;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> تالا(t.mutex);

    ثابت size_t اندر_use    = dev_کل > dev_free ? dev_کل - dev_free : 0;
    ثابت size_t کل     = dev_کل / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.کل_زندہ / MIB;
    ثابت size_t buffers   = t.زندہ[GGML_SYCL_MEM_بفر] / MIB;
    ثابت size_t peak      = t.کل_peak / MIB;

    اگر (اندر_use >= t.کل_زندہ && device_حافظہ_is_dedicated(device) && کل >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      لیبل, کل, freed, allocated, buffers, allocated - buffers,
                      کل - freed - allocated, peak);
    } ورنہ {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      لیبل, کل, freed, اندر_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
