#شامل "memtrace.hpp"

#شامل "common.hpp"
#شامل "ggml-impl.h"

#شامل <cstdio>
#شامل <mutex>
#شامل <unordered_map>

constexpr size_t MIB = 1024 * 1024;

ایستا ثابت نویسه * mem_type_name(ggml_sycl_mem_type type) {
    انتخاب (type) {
        حالت GGML_SYCL_MEM_BUFFER:   بازگشت "buffer";
        حالت GGML_SYCL_MEM_POOL_LEG: بازگشت "pool_leg";
        حالت GGML_SYCL_MEM_POOL_VMM: بازگشت "pool_vmm";
        حالت GGML_SYCL_MEM_ASYNC:    بازگشت "async";
        حالت GGML_SYCL_MEM_DIRECT:   بازگشت "direct";
        پیشفرض:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (صحیح) type);
    }
}

ساختار mem_tracker {
    std::mutex mutex;
    std::unordered_map<ثابت تهی *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_شمار] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_شمار] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

ایستا mem_tracker & get_tracker() {
    ایستا mem_tracker t;
    بازگشت t;
}

ایستا size_t step_bytes() {
    ثابت صحیح mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    بازگشت (size_t) mib * MIB;
}

ایستا تهی report_sites_locked() {
    mem_tracker & t = get_tracker();
    برای (صحیح i = 0; i < GGML_SYCL_MEM_TYPE_شمار; i++) {
        اگر (t.peak[i] == 0) {
            ادامه;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

ایستا تهی report_locked(ثابت نویسه * tag) {
    mem_tracker & t = get_tracker();

    ثابت size_t allocated = t.total_live / MIB;
    ثابت size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

ایستا تهی log_event_locked(ثابت نویسه * op, ggml_sycl_mem_type type, ثابت تهی * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (دوگانه) bytes / MIB, ptr);
}

منطقی ggml_sycl_memtrace_enabled() {
    بازگشت g_ggml_sycl_memtrace > 0;
}

تهی ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ثابت تهی * ptr, size_t bytes) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        بازگشت;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    اگر (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    اگر (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    اگر (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    اگر (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    ایستا ثابت size_t step = step_bytes();
    اگر (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        نویسه tag[96];
        std::snprintf(tag, اندازه(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

تهی ggml_sycl_memtrace_del(ثابت تهی * ptr) {
    اگر (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        بازگشت;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    اگر (it == t.live_by_ptr.end()) {
        بازگشت;
    }
    ثابت ggml_sycl_mem_type type = it->second.first;
    ثابت size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    اگر (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

تهی ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (دوگانه) bytes / MIB, mem_type_name(type));
    اگر (!ggml_sycl_memtrace_enabled()) {
        بازگشت;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

تهی ggml_sycl_memtrace_report(ثابت نویسه * tag) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        بازگشت;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

ایستا منطقی device_memory_is_dedicated(صحیح device) {
    اگر (device < 0 || device >= ggml_sycl_info().device_شمار) {
        بازگشت نادرست;
    }
    ثابت sycl_device_info & info = ggml_sycl_info().devices[device];
    بازگشت info.l0_device_type_valid && info.l0_discrete_gpu;
}

تهی ggml_sycl_memtrace_report_device(ثابت نویسه * tag, صحیح device, size_t dev_free, size_t dev_total) {
    اگر (!ggml_sycl_memtrace_enabled()) {
        بازگشت;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ثابت size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ثابت size_t total     = dev_total / MIB;
    ثابت size_t freed     = dev_free / MIB;
    ثابت size_t allocated = t.total_live / MIB;
    ثابت size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ثابت size_t peak      = t.total_peak / MIB;

    اگر (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } وگرنه {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
