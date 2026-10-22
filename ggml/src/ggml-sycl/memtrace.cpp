#অন্তর্ভাৱ "memtrace.hpp"

#অন্তর্ভাৱ "common.hpp"
#অন্তর্ভাৱ "ggml-impl.h"

#অন্তর্ভাৱ <cstdio>
#অন্তর্ভাৱ <mutex>
#অন্তর্ভাৱ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

স্থির ধ্রুৱ ৱর্ণ * mem_type_name(ggml_sycl_mem_type type) {
    ৱিকল্প (type) {
        পক্ষ GGML_SYCL_MEM_BUFFER:   ঘূৰাই "buffer";
        পক্ষ GGML_SYCL_MEM_POOL_LEG: ঘূৰাই "pool_leg";
        পক্ষ GGML_SYCL_MEM_POOL_VMM: ঘূৰাই "pool_vmm";
        পক্ষ GGML_SYCL_MEM_ASYNC:    ঘূৰাই "async";
        পক্ষ GGML_SYCL_MEM_DIRECT:   ঘূৰাই "direct";
        স্ৱভাৱ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (পূর্ণাঙ্ক) type);
    }
}

সংরচনা mem_tracker {
    std::mutex mutex;
    std::unordered_map<ধ্রুৱ শূন্য *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_গুনতি] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_গুনতি] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

স্থির mem_tracker & get_tracker() {
    স্থির mem_tracker t;
    ঘূৰাই t;
}

স্থির size_t step_bytes() {
    ধ্রুৱ পূর্ণাঙ্ক mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    ঘূৰাই (size_t) mib * MIB;
}

স্থির শূন্য report_sites_locked() {
    mem_tracker & t = get_tracker();
    বাবে (পূর্ণাঙ্ক i = 0; i < GGML_SYCL_MEM_TYPE_গুনতি; i++) {
        যদি (t.peak[i] == 0) {
            অনুৱর্ত;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

স্থির শূন্য report_locked(ধ্রুৱ ৱর্ণ * tag) {
    mem_tracker & t = get_tracker();

    ধ্রুৱ size_t allocated = t.total_live / MIB;
    ধ্রুৱ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

স্থির শূন্য log_event_locked(ধ্রুৱ ৱর্ণ * op, ggml_sycl_mem_type type, ধ্রুৱ শূন্য * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (দ্ৱিগুণ) bytes / MIB, ptr);
}

তার্কিক ggml_sycl_memtrace_enabled() {
    ঘূৰাই g_ggml_sycl_memtrace > 0;
}

শূন্য ggml_sycl_memtrace_add(ggml_sycl_mem_type type, ধ্রুৱ শূন্য * ptr, size_t bytes) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ঘূৰাই;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    যদি (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    যদি (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    যদি (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    যদি (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    স্থির ধ্রুৱ size_t step = step_bytes();
    যদি (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        ৱর্ণ tag[96];
        std::snprintf(tag, পরিমাণ(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

শূন্য ggml_sycl_memtrace_del(ধ্রুৱ শূন্য * ptr) {
    যদি (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        ঘূৰাই;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    যদি (it == t.live_by_ptr.end()) {
        ঘূৰাই;
    }
    ধ্রুৱ ggml_sycl_mem_type type = it->second.first;
    ধ্রুৱ size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    যদি (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

শূন্য ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (দ্ৱিগুণ) bytes / MIB, mem_type_name(type));
    যদি (!ggml_sycl_memtrace_enabled()) {
        ঘূৰাই;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

শূন্য ggml_sycl_memtrace_report(ধ্রুৱ ৱর্ণ * tag) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ঘূৰাই;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

স্থির তার্কিক device_memory_is_dedicated(পূর্ণাঙ্ক device) {
    যদি (device < 0 || device >= ggml_sycl_info().device_গুনতি) {
        ঘূৰাই মিছা;
    }
    ধ্রুৱ sycl_device_info & info = ggml_sycl_info().devices[device];
    ঘূৰাই info.l0_device_type_valid && info.l0_discrete_gpu;
}

শূন্য ggml_sycl_memtrace_report_device(ধ্রুৱ ৱর্ণ * tag, পূর্ণাঙ্ক device, size_t dev_free, size_t dev_total) {
    যদি (!ggml_sycl_memtrace_enabled()) {
        ঘূৰাই;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    ধ্রুৱ size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    ধ্রুৱ size_t total     = dev_total / MIB;
    ধ্রুৱ size_t freed     = dev_free / MIB;
    ধ্রুৱ size_t allocated = t.total_live / MIB;
    ধ্রুৱ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    ধ্রুৱ size_t peak      = t.total_peak / MIB;

    যদি (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } অন্যথা {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
