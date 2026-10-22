#અન્તર્ભાવ "memtrace.hpp"

#અન્તર્ભાવ "common.hpp"
#અન્તર્ભાવ "ggml-impl.h"

#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <mutex>
#અન્તર્ભાવ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

સ્થિર અચલ અક્ષર * mem_type_name(ggml_sycl_mem_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_SYCL_MEM_BUFFER:   પાછા "buffer";
        સ્થિતિ GGML_SYCL_MEM_POOL_LEG: પાછા "pool_leg";
        સ્થિતિ GGML_SYCL_MEM_POOL_VMM: પાછા "pool_vmm";
        સ્થિતિ GGML_SYCL_MEM_ASYNC:    પાછા "async";
        સ્થિતિ GGML_SYCL_MEM_DIRECT:   પાછા "direct";
        મૂળભૂત:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (પૂર્ણાંક) type);
    }
}

સંરચના mem_tracker {
    std::mutex mutex;
    std::unordered_map<અચલ શૂન્ય *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_ગણતરી] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_ગણતરી] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

સ્થિર mem_tracker & get_tracker() {
    સ્થિર mem_tracker t;
    પાછા t;
}

સ્થિર size_t step_bytes() {
    અચલ પૂર્ણાંક mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    પાછા (size_t) mib * MIB;
}

સ્થિર શૂન્ય report_sites_locked() {
    mem_tracker & t = get_tracker();
    માટે (પૂર્ણાંક i = 0; i < GGML_SYCL_MEM_TYPE_ગણતરી; i++) {
        જો (t.peak[i] == 0) {
            ચાલુ;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

સ્થિર શૂન્ય report_locked(અચલ અક્ષર * tag) {
    mem_tracker & t = get_tracker();

    અચલ size_t allocated = t.total_live / MIB;
    અચલ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

સ્થિર શૂન્ય log_event_locked(અચલ અક્ષર * op, ggml_sycl_mem_type type, અચલ શૂન્ય * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (દ્વિગુણ) bytes / MIB, ptr);
}

તાર્કિક ggml_sycl_memtrace_enabled() {
    પાછા g_ggml_sycl_memtrace > 0;
}

શૂન્ય ggml_sycl_memtrace_add(ggml_sycl_mem_type type, અચલ શૂન્ય * ptr, size_t bytes) {
    જો (!ggml_sycl_memtrace_enabled()) {
        પાછા;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    જો (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    જો (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    જો (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    જો (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    સ્થિર અચલ size_t step = step_bytes();
    જો (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        અક્ષર tag[96];
        std::snprintf(tag, કદ(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

શૂન્ય ggml_sycl_memtrace_del(અચલ શૂન્ય * ptr) {
    જો (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        પાછા;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    જો (it == t.live_by_ptr.end()) {
        પાછા;
    }
    અચલ ggml_sycl_mem_type type = it->second.first;
    અચલ size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    જો (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

શૂન્ય ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (દ્વિગુણ) bytes / MIB, mem_type_name(type));
    જો (!ggml_sycl_memtrace_enabled()) {
        પાછા;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

શૂન્ય ggml_sycl_memtrace_report(અચલ અક્ષર * tag) {
    જો (!ggml_sycl_memtrace_enabled()) {
        પાછા;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

સ્થિર તાર્કિક device_memory_is_dedicated(પૂર્ણાંક device) {
    જો (device < 0 || device >= ggml_sycl_info().device_ગણતરી) {
        પાછા ખોટું;
    }
    અચલ sycl_device_info & info = ggml_sycl_info().devices[device];
    પાછા info.l0_device_type_valid && info.l0_discrete_gpu;
}

શૂન્ય ggml_sycl_memtrace_report_device(અચલ અક્ષર * tag, પૂર્ણાંક device, size_t dev_free, size_t dev_total) {
    જો (!ggml_sycl_memtrace_enabled()) {
        પાછા;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    અચલ size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    અચલ size_t total     = dev_total / MIB;
    અચલ size_t freed     = dev_free / MIB;
    અચલ size_t allocated = t.total_live / MIB;
    અચલ size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    અચલ size_t peak      = t.total_peak / MIB;

    જો (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } નહીંતર {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
