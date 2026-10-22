#అన్తర్భావ "memtrace.hpp"

#అన్తర్భావ "common.hpp"
#అన్తర్భావ "ggml-impl.h"

#అన్తర్భావ <cstdio>
#అన్తర్భావ <mutex>
#అన్తర్భావ <unordered_map>

constexpr size_t MIB = 1024 * 1024;

స్థిర స్థిరరాశి అక్షరం * mem_type_name(ggml_sycl_mem_type type) {
    ఎంపిక (type) {
        స్థితి GGML_SYCL_MEM_బఫరు:   తిరిగి "buffer";
        స్థితి GGML_SYCL_MEM_POOL_LEG: తిరిగి "pool_leg";
        స్థితి GGML_SYCL_MEM_POOL_VMM: తిరిగి "pool_vmm";
        స్థితి GGML_SYCL_MEM_ASYNC:    తిరిగి "async";
        స్థితి GGML_SYCL_MEM_ప్రత్యక్ష:   తిరిగి "direct";
        అప్రమేయ:                     GGML_ABORT("[%s] The type value %d is not supported\n", __func__, (పూర్ణాంకం) type);
    }
}

నిర్మాణం mem_tracker {
    std::mutex mutex;
    std::unordered_map<స్థిరరాశి శూన్య *, std::pair<ggml_sycl_mem_type, size_t>> సజీవ_by_ptr;
    size_t సజీవ[GGML_SYCL_MEM_TYPE_లెక్క] = {};
    size_t peak[GGML_SYCL_MEM_TYPE_లెక్క] = {};
    size_t మొత్తం_సజీవ = 0;
    size_t మొత్తం_peak = 0;
    size_t చివరి_logged_peak = 0;
};

స్థిర mem_tracker & get_tracker() {
    స్థిర mem_tracker t;
    తిరిగి t;
}

స్థిర size_t అడుగు_బైట్లు() {
    స్థిరరాశి పూర్ణాంకం mib = g_ggml_sycl_memtrace_అడుగు > 0 ? g_ggml_sycl_memtrace_అడుగు : 64;
    తిరిగి (size_t) mib * MIB;
}

స్థిర శూన్య report_sites_locked() {
    mem_tracker & t = get_tracker();
    కొరకు (పూర్ణాంకం i = 0; i < GGML_SYCL_MEM_TYPE_లెక్క; i++) {
        ఒకవేళ (t.peak[i] == 0) {
            కొనసాగు;
        }
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ "   %-9s allocated %5zu MiB, peak %5zu MiB\n",
                      mem_type_name((ggml_sycl_mem_type) i), t.సజీవ[i] / MIB, t.peak[i] / MIB);
    }
}

స్థిర శూన్య report_locked(స్థిరరాశి అక్షరం * ట్యాగ్) {
    mem_tracker & t = get_tracker();

    స్థిరరాశి size_t allocated = t.మొత్తం_సజీవ / MIB;
    స్థిరరాశి size_t buffers   = t.సజీవ[GGML_SYCL_MEM_బఫరు] / MIB;

    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ " %s: allocated %5zu MiB (buffers %5zu + scratch %5zu),"
                  " peak %5zu MiB\n",
                  ట్యాగ్, allocated, buffers, allocated - buffers, t.మొత్తం_peak / MIB);
    report_sites_locked();
}

స్థిర శూన్య log_event_locked(స్థిరరాశి అక్షరం * op, ggml_sycl_mem_type type, స్థిరరాశి శూన్య * ptr, size_t బైట్లు) {
    GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ " allocated %5zu MiB  %-5s %-9s %9.3f MiB  ptr=%p\n",
                  get_tracker().మొత్తం_సజీవ / MIB, op, mem_type_name(type),
                  (ద్విగుణ) బైట్లు / MIB, ptr);
}

తార్కిక ggml_sycl_memtrace_enabled() {
    తిరిగి g_ggml_sycl_memtrace > 0;
}

శూన్య ggml_sycl_memtrace_కూడు(ggml_sycl_mem_type type, స్థిరరాశి శూన్య * ptr, size_t బైట్లు) {
    ఒకవేళ (!ggml_sycl_memtrace_enabled()) {
        తిరిగి;
    }
    GGML_ASSERT(ptr != శూన్యసూచిక);
    GGML_ASSERT(బైట్లు != 0);

    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> తాళం(t.mutex);

    స్వయంచాలక it = t.సజీవ_by_ptr.find(ptr);
    ఒకవేళ (it != t.సజీవ_by_ptr.end()) {
        t.సజీవ[it->second.మొదటి] -= it->second.second;
        t.మొత్తం_సజీవ -= it->second.second;
    }

    t.సజీవ_by_ptr[ptr] = { type, బైట్లు };
    t.సజీవ[type] += బైట్లు;
    t.మొత్తం_సజీవ += బైట్లు;

    ఒకవేళ (t.సజీవ[type] > t.peak[type]) {
        t.peak[type] = t.సజీవ[type];
    }
    ఒకవేళ (t.మొత్తం_సజీవ > t.మొత్తం_peak) {
        t.మొత్తం_peak = t.మొత్తం_సజీవ;
    }

    ఒకవేళ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("alloc", type, ptr, బైట్లు);
    }

    స్థిర స్థిరరాశి size_t అడుగు = అడుగు_బైట్లు();
    ఒకవేళ (t.మొత్తం_peak >= t.చివరి_logged_peak + అడుగు) {
        t.చివరి_logged_peak = t.మొత్తం_peak;
        అక్షరం ట్యాగ్[96];
        std::snprintf(ట్యాగ్, పరిమాణం(ట్యాగ్), "peak grew (+%zu MiB from %s)", బైట్లు / MIB,
                      mem_type_name(type));
        report_locked(ట్యాగ్);
    }
}

శూన్య ggml_sycl_memtrace_del(స్థిరరాశి శూన్య * ptr) {
    ఒకవేళ (!ggml_sycl_memtrace_enabled() || ptr == శూన్యసూచిక) {
        తిరిగి;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> తాళం(t.mutex);

    స్వయంచాలక it = t.సజీవ_by_ptr.find(ptr);
    ఒకవేళ (it == t.సజీవ_by_ptr.end()) {
        తిరిగి;
    }
    స్థిరరాశి ggml_sycl_mem_type type = it->second.మొదటి;
    స్థిరరాశి size_t బైట్లు = it->second.second;
    t.సజీవ[type] -= బైట్లు;
    t.మొత్తం_సజీవ -= బైట్లు;
    t.సజీవ_by_ptr.erase(it);

    ఒకవేళ (g_ggml_sycl_memtrace >= 2) {
        log_event_locked("free", type, ptr, బైట్లు);
    }
}

శూన్య ggml_sycl_memtrace_fail(ggml_sycl_mem_type type, size_t బైట్లు) {
    GGML_LOG_ERROR(GGML_SYCL_MEMTRACE_ట్యాగ్ " alloc FAILED: %9.3f MiB %s\n",
                   (ద్విగుణ) బైట్లు / MIB, mem_type_name(type));
    ఒకవేళ (!ggml_sycl_memtrace_enabled()) {
        తిరిగి;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> తాళం(t.mutex);
    report_locked("at allocation failure");
}

శూన్య ggml_sycl_memtrace_report(స్థిరరాశి అక్షరం * ట్యాగ్) {
    ఒకవేళ (!ggml_sycl_memtrace_enabled()) {
        తిరిగి;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> తాళం(t.mutex);
    report_locked(ట్యాగ్);
}

స్థిర తార్కిక device_జ్ఞాపకం_is_dedicated(పూర్ణాంకం device) {
    ఒకవేళ (device < 0 || device >= ggml_sycl_info().device_లెక్క) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి sycl_device_info & info = ggml_sycl_info().devices[device];
    తిరిగి info.l0_device_type_valid && info.l0_discrete_gpu;
}

శూన్య ggml_sycl_memtrace_report_device(స్థిరరాశి అక్షరం * ట్యాగ్, పూర్ణాంకం device, size_t dev_free, size_t dev_మొత్తం) {
    ఒకవేళ (!ggml_sycl_memtrace_enabled()) {
        తిరిగి;
    }
    mem_tracker & t = get_tracker();
    std::lock_guard<std::mutex> తాళం(t.mutex);

    స్థిరరాశి size_t లోపల_use    = dev_మొత్తం > dev_free ? dev_మొత్తం - dev_free : 0;
    స్థిరరాశి size_t మొత్తం     = dev_మొత్తం / MIB;
    స్థిరరాశి size_t freed     = dev_free / MIB;
    స్థిరరాశి size_t allocated = t.మొత్తం_సజీవ / MIB;
    స్థిరరాశి size_t buffers   = t.సజీవ[GGML_SYCL_MEM_బఫరు] / MIB;
    స్థిరరాశి size_t peak      = t.మొత్తం_peak / MIB;

    ఒకవేళ (లోపల_use >= t.మొత్తం_సజీవ && device_జ్ఞాపకం_is_dedicated(device) && మొత్తం >= freed + allocated) {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ " %s: total %5zu MiB = free %5zu + allocated %5zu"
                      " (buffers %5zu + scratch %5zu) + other %5zu, peak %5zu MiB\n",
                      ట్యాగ్, మొత్తం, freed, allocated, buffers, allocated - buffers,
                      మొత్తం - freed - allocated, peak);
    } లేకపోతే {
        GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ " %s: total %5zu MiB, free %5zu, in use %5zu;"
                      " allocated %5zu (buffers %5zu + scratch %5zu), peak %5zu MiB\n",
                      ట్యాగ్, మొత్తం, freed, లోపల_use / MIB, allocated, buffers,
                      allocated - buffers, peak);
    }
    report_sites_locked();
}
