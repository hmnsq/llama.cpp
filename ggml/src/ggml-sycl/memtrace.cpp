#অন্তর্ভাব "memtrace.hpp"

#অন্তর্ভাব "common.hpp"
#অন্তর্ভাব "ggml-impl.h"

#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <mutex>
#অন্তর্ভাব <unordered_map>

constexpr size_t MIB = 1024 * 1024;

স্থির ধ্রুবক অক্ষর * mem_type_name(ggml_sycl_mem_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_SYCL_MEM_বাফার:   ফেরত "buffer";
        ক্ষেত্র GGML_SYCL_MEM_POOL_LEG: ফেরত "pool_leg";
        ক্ষেত্র GGML_SYCL_MEM_POOL_VMM: ফেরত "pool_vmm";
        ক্ষেত্র GGML_SYCL_MEM_ASYNC:    ফেরত "async";
        ক্ষেত্র GGML_SYCL_MEM_DIRECT:   ফেরত "direct";
        পূর্বনির্ধারিত:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (পূর্ণসংখ্যা) type);
    }
}

কাঠামো mem_tracker {
    std::mutex mutex;
    std::unordered_map<ধ্রুবক শূন্য *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_গুনতি] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_গুনতি] = {};
    size_t যোগ_live = 0;
    size_t যোগ_peak = 0;
    size_t last_logged_peak = 0;
};

স্থির mem_tracker & get_tracker() {
    স্থির mem_tracker t;
    ফেরত t;
}

স্থির size_t step_bytes() {
    ধ্রুবক পূর্ণসংখ্যা mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    ফেরত (size_t) mib * MIB;
}

স্থির শূন্য report_sites_locked() {
    mem_tracker & t = get_tracker();
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_SYCL_MEM_TYPE_গুনতি; i++) {
        যদি (t.peak[i] == 0) {
            চালু;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

স্থির শূন্য report_locked(ধ্রুবক অক্ষর * tag) {
    mem_tracker & t = get_tracker();

    ধ্রুবক size_t allocated = t.যোগ_live / MIB;
    ধ্রুবক size_t buffers   = t.live[GGML_SYCL_MEM_বাফার] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.যোগ_peak / MIB);
    report_sites_locked();
}

স্থির শূন্য log_event_locked(ধ্রুবক অক্ষর * op, ggml_sycl_mem_type type, ধ্রুবক শূন্য * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().যোগ_live / MIB, op, mem_type_name(type),
                  (দ্বিগুণ) bytes / MIB, ptr);
}

যৌক্তিক ggml_sycl_memtrace_enabled() {
    ফেরত g_ggml_sycl_memtrace > 0;
}

শূন্য ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ধ্রুবক শূন্য * ptr, size_t bytes) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ফেরত;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    যদি (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.যোগ_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.যোগ_live += bytes;

    যদি (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    যদি (t.যোগ_live > t.যোগ_peak) {
        t.যোগ_peak = t.যোগ_live;
    }

    যদি (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    স্থির ধ্রুবক size_t step = step_bytes();
    যদি (t.যোগ_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.যোগ_peak;
        অক্ষর tag[96];
        std::snprintf(tag, আকার(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

শূন্য ggml_sycl_memtrace_del(ধ্রুবক শূন্য * ptr) {
    যদি (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        ফেরত;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    যদি (it == t.live_by_ptr.end()) {
        ফেরত;
    }
    ধ্রুবক ggml_sycl_mem_type type = it->second.first;
    ধ্রুবক size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.যোগ_live -= bytes;
    t.live_by_ptr.erase(it);

    যদি (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

শূন্য ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (দ্বিগুণ) bytes / MIB, mem_type_name(type));
    যদি (!ggml_sycl_memtrace_enabled()) {
        ফেরত;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

শূন্য ggml_sycl_memtrace_report(ধ্রুবক অক্ষর * tag) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ফেরত;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

স্থির যৌক্তিক device_memory_is_dedicated(পূর্ণসংখ্যা device) {
    যদি (device < 0 || device >= ggml_sycl_info().device_গুনতি) {
        ফেরত মিথ্যা;
    }
    ধ্রুবক sycl_device_info & info = ggml_sycl_info().devices[device];
    ফেরত info.l0_device_type_valid && info.l0_discrete_gpu;
}

শূন্য ggml_sycl_memtrace_report_device(ধ্রুবক অক্ষর * tag, পূর্ণসংখ্যা device, size_t dev_free, size_t dev_যোগ) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ফেরত;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ধ্রুবক size_t in_use    = dev_যোগ > dev_free ? dev_যোগ - dev_free : 0;
    ধ্রুবক size_t যোগ     = dev_যোগ / MIB;
    ধ্রুবক size_t freed     = dev_free / MIB;
    ধ্রুবক size_t allocated = t.যোগ_live / MIB;
    ধ্রুবক size_t buffers   = t.live[GGML_SYCL_MEM_বাফার] / MIB;
    ধ্রুবক size_t peak      = t.যোগ_peak / MIB;

    যদি (in_use >= t.যোগ_live && device_memory_is_dedicated(device) && যোগ >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, যোগ, freed, allocated, buffers, allocated - buffers,
                      যোগ - freed - allocated, peak);
    } অন্যথা {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, যোগ, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
