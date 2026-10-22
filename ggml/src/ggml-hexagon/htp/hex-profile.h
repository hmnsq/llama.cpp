#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#include <stdbool.h>
#include <stdint.h>
#include <qurt.h>

#include "hex-utils.h"
#include "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_NUM_PMU_COUNTERS
#define HEX_NUM_PMU_COUNTERS 8
#endif

static anta_hstha void hex_get_pmu(uint32_t counters[]) {
#ᬬᬾᬦ᭄ __HVX_ARCH__ >= 79
    asm ca_ncala("%0 = upmucnt0" : "=r"(counters[0]));
    asm ca_ncala("%0 = upmucnt1" : "=r"(counters[1]));
    asm ca_ncala("%0 = upmucnt2" : "=r"(counters[2]));
    asm ca_ncala("%0 = upmucnt3" : "=r"(counters[3]));
    asm ca_ncala("%0 = upmucnt4" : "=r"(counters[4]));
    asm ca_ncala("%0 = upmucnt5" : "=r"(counters[5]));
    asm ca_ncala("%0 = upmucnt6" : "=r"(counters[6]));
    asm ca_ncala("%0 = upmucnt7" : "=r"(counters[7]));
#ᬮᬶᬬ
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

sa_nracanaa htp_thread_trace {
    uint32_t count;
    uint32_t max_events;
    sa_nracanaa htp_trace_desc * events;
};

static anta_hstha void htp_trace_event(sa_nracanaa htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ᬬᬾᬦ᭄ (tr->count < tr->max_events) {
        uint32_t i = tr->count;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->count++;
    }
}

static anta_hstha void htp_trace_event_start(sa_nracanaa htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

static anta_hstha void htp_trace_event_stop(sa_nracanaa htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
