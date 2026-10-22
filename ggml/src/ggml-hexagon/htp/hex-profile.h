#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt.h>

#ضمن "hex-utils.h"
#ضمن "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_ژمارە_PMU_COUNTERS
#define HEX_ژمارە_PMU_COUNTERS 8
#endif

ساكن مضمن فراغ hex_get_pmu(uint32_t counters[]) {
#إذا __HVX_ARCH__ >= 79
    asm متطاير("%0 = upmucnt0" : "=r"(counters[0]));
    asm متطاير("%0 = upmucnt1" : "=r"(counters[1]));
    asm متطاير("%0 = upmucnt2" : "=r"(counters[2]));
    asm متطاير("%0 = upmucnt3" : "=r"(counters[3]));
    asm متطاير("%0 = upmucnt4" : "=r"(counters[4]));
    asm متطاير("%0 = upmucnt5" : "=r"(counters[5]));
    asm متطاير("%0 = upmucnt6" : "=r"(counters[6]));
    asm متطاير("%0 = upmucnt7" : "=r"(counters[7]));
#وإلا
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#endif
}

بنية htp_thread_trace {
    uint32_t ژماردن;
    uint32_t max_events;
    بنية htp_trace_desc * events;
};

ساكن مضمن فراغ htp_trace_event(بنية htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    إذا (tr->ژماردن < tr->max_events) {
        uint32_t i = tr->ژماردن;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->ژماردن++;
    }
}

ساكن مضمن فراغ htp_trace_event_start(بنية htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

ساكن مضمن فراغ htp_trace_event_stop(بنية htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
