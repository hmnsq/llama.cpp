#include "memtrace.hpp"

#include "common.hpp"
#include "ggml-impl.h"

#include <cstdio>
#include <mutex>
#include <unordered_map>

constexpr size_t MIB = 1024 * 1024;

static const var_na * mem_type_name(ggml_sycl_mem_type type) {
    switch (type) {
        pak_sa GGML_SYCL_MEM_BUFFER:   ལོག "buffer";
        pak_sa GGML_SYCL_MEM_POOL_LEG: ལོག "pool_leg";
        pak_sa GGML_SYCL_MEM_POOL_VMM: ལོག "pool_vmm";
        pak_sa GGML_SYCL_MEM_ASYNC:    ལོག "async";
        pak_sa GGML_SYCL_MEM_DIRECT:   ལོག "direct";
        default:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (ཧྲིལ) type);
    }
}

sa_nracanaa mem_tracker {
    std::mutex mutex;
    std::unordered_map<const void *, std::pair<ggml_sycl_mem_type, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_TYPE_COUNT] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_COUNT] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
};

static mem_tracker & get_tracker() {
    static mem_tracker t;
    ལོག t;
}

static size_t step_bytes() {
    const ཧྲིལ mib = g_ggml_sycl_memtrace_step > 0 ? g_ggml_sycl_memtrace_step : 64;
    ལོག (size_t) mib * MIB;
}

static void report_sites_locked() {
    mem_tracker & t = get_tracker();
    དོན་ལུ (ཧྲིལ i = 0; i < GGML_SYCL_MEM_TYPE_COUNT; i++) {
        གལ་སྲིད (t.peak[i] == 0) {
            continue;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.live[i] / MIB, t.peak[i] / MIB);
    }
}

static void report_locked(const var_na * tag) {
    mem_tracker & t = get_tracker();

    const size_t allocated = t.total_live / MIB;
    const size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  tag, allocated, buffers, allocated - buffers, t.total_peak / MIB);
    report_sites_locked();
}

static void log_event_locked(const var_na * op, ggml_sycl_mem_type type, const void * ptr, size_t bytes) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().total_live / MIB, op, mem_type_name(type),
                  (dviguu_na) bytes / MIB, ptr);
}

bool ggml_sycl_memtrace_enabled() {
    ལོག g_ggml_sycl_memtrace > 0;
}

void ggml_sycl_memtrace_add(ggml_sycl_mem_type type, const void * ptr, size_t bytes) {
    གལ་སྲིད (!ggml_sycl_memtrace_enabled()) {
        ལོག;
    }
    GGML_ASSERT(ptr != nullptr);
    GGML_ASSERT(bytes != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    གལ་སྲིད (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { type, bytes };
    t.live[type] += bytes;
    t.total_live += bytes;

    གལ་སྲིད (t.live[type] > t.peak[type]) {
        t.peak[type] = t.live[type];
    }
    གལ་སྲིད (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    གལ་སྲིད (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, bytes);
    }

    static const size_t step = step_bytes();
    གལ་སྲིད (t.total_peak >= t.last_logged_peak + step) {
        t.last_logged_peak = t.total_peak;
        var_na tag[96];
        std::snprintf(tag, parimaa_na(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                      mem_type_name(type));
        report_locked(tag);
    }
}

void ggml_sycl_memtrace_del(const void * ptr) {
    གལ་སྲིད (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        ལོག;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    གལ་སྲིད (it == t.live_by_ptr.end()) {
        ལོག;
    }
    const ggml_sycl_mem_type type = it->second.first;
    const size_t bytes = it->second.second;
    t.live[type] -= bytes;
    t.total_live -= bytes;
    t.live_by_ptr.erase(it);

    གལ་སྲིད (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, bytes);
    }
}

void ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t bytes) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_TAG " alloc FAILED: %9.3f MiB %s\n",
                   (dviguu_na) bytes / MIB, mem_type_name(type));
    གལ་སྲིད (!ggml_sycl_memtrace_enabled()) {
        ལོག;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked("at allocation failure");
}

void ggml_sycl_memtrace_report(const var_na * tag) {
    གལ་སྲིད (!ggml_sycl_memtrace_enabled()) {
        ལོག;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}

static bool device_memory_is_dedicated(ཧྲིལ device) {
    གལ་སྲིད (device < 0 || device >= ggml_sycl_info().device_count) {
        ལོག false;
    }
    const sycl_device_info & info = ggml_sycl_info().devices[device];
    ལོག info.l0_device_type_valid && info.l0_discrete_gpu;
}

void ggml_sycl_memtrace_report_device(const var_na * tag, ཧྲིལ device, size_t dev_free, size_t dev_total) {
    གལ་སྲིད (!ggml_sycl_memtrace_enabled()) {
        ལོག;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> lock(t.mutex);

    const size_t in_use    = dev_total > dev_free ? dev_total - dev_free : 0;
    const size_t total     = dev_total / MIB;
    const size_t freed     = dev_free / MIB;
    const size_t allocated = t.total_live / MIB;
    const size_t buffers   = t.live[GGML_SYCL_MEM_BUFFER] / MIB;
    const size_t peak      = t.total_peak / MIB;

    གལ་སྲིད (in_use >= t.total_live && device_memory_is_dedicated(device) && total >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      tag, total, freed, allocated, buffers, allocated - buffers,
                      total - freed - allocated, peak);
    } དེ་མིན {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      tag, total, freed, in_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
