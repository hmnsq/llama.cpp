#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <qurt.h>

#അന്തര്ഭാവ "hex-utils.h"
#അന്തര്ഭാവ "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_അക്കം_PMU_COUNTERS
#define HEX_അക്കം_PMU_COUNTERS 8
#endif

സ്ഥിതം വരിയിൽ ശൂന്യം hex_get_pmu(uint32_t counters[]) {
#എങ്കിൽ __HVX_ARCH__ >= 79
    asm അസ്ഥിര("%0 = upmucnt0" : "=r"(counters[0]));
    asm അസ്ഥിര("%0 = upmucnt1" : "=r"(counters[1]));
    asm അസ്ഥിര("%0 = upmucnt2" : "=r"(counters[2]));
    asm അസ്ഥിര("%0 = upmucnt3" : "=r"(counters[3]));
    asm അസ്ഥിര("%0 = upmucnt4" : "=r"(counters[4]));
    asm അസ്ഥിര("%0 = upmucnt5" : "=r"(counters[5]));
    asm അസ്ഥിര("%0 = upmucnt6" : "=r"(counters[6]));
    asm അസ്ഥിര("%0 = upmucnt7" : "=r"(counters[7]));
#അല്ലെങ്കിൽ
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

ഘടന htp_thread_trace {
    uint32_t എണ്ണം;
    uint32_t max_events;
    ഘടന htp_trace_desc * events;
};

സ്ഥിതം വരിയിൽ ശൂന്യം htp_trace_event(ഘടന htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    എങ്കിൽ (tr->എണ്ണം < tr->max_events) {
        uint32_t i = tr->എണ്ണം;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->എണ്ണം++;
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_trace_event_start(ഘടന htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

സ്ഥിതം വരിയിൽ ശൂന്യം htp_trace_event_stop(ഘടന htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
