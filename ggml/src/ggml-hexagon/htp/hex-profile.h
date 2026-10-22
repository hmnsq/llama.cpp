#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <qurt.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_ಅಂಕೆ_PMU_COUNTERS
#define HEX_ಅಂಕೆ_PMU_COUNTERS 8
#endif

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_get_pmu(uint32_t counters[]) {
#ಯದಿ __HVX_ARCH__ >= 79
    asm ಚಞ್ಚಲ("%0 = upmucnt0" : "=r"(counters[0]));
    asm ಚಞ್ಚಲ("%0 = upmucnt1" : "=r"(counters[1]));
    asm ಚಞ್ಚಲ("%0 = upmucnt2" : "=r"(counters[2]));
    asm ಚಞ್ಚಲ("%0 = upmucnt3" : "=r"(counters[3]));
    asm ಚಞ್ಚಲ("%0 = upmucnt4" : "=r"(counters[4]));
    asm ಚಞ್ಚಲ("%0 = upmucnt5" : "=r"(counters[5]));
    asm ಚಞ್ಚಲ("%0 = upmucnt6" : "=r"(counters[6]));
    asm ಚಞ್ಚಲ("%0 = upmucnt7" : "=r"(counters[7]));
#ಅನ್ಯಥಾ
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

ಸಂರಚನಾ htp_thread_trace {
    uint32_t ಎಣಿಕೆ;
    uint32_t max_events;
    ಸಂರಚನಾ htp_trace_desc * events;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_trace_event(ಸಂರಚನಾ htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ಯದಿ (tr->ಎಣಿಕೆ < tr->max_events) {
        uint32_t i = tr->ಎಣಿಕೆ;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->ಎಣಿಕೆ++;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_trace_event_start(ಸಂರಚನಾ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ htp_trace_event_stop(ಸಂರಚನಾ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
