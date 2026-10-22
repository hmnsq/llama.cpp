#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <qurt.h>

#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_ଅଙ୍କ_PMU_COUNTERS
#define HEX_ଅଙ୍କ_PMU_COUNTERS 8
#endif

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hex_get_pmu(uint32_t counters[]) {
#ଯଦି __HVX_ARCH__ >= 79
    asm ଅସ୍ଥିର("%0 = upmucnt0" : "=r"(counters[0]));
    asm ଅସ୍ଥିର("%0 = upmucnt1" : "=r"(counters[1]));
    asm ଅସ୍ଥିର("%0 = upmucnt2" : "=r"(counters[2]));
    asm ଅସ୍ଥିର("%0 = upmucnt3" : "=r"(counters[3]));
    asm ଅସ୍ଥିର("%0 = upmucnt4" : "=r"(counters[4]));
    asm ଅସ୍ଥିର("%0 = upmucnt5" : "=r"(counters[5]));
    asm ଅସ୍ଥିର("%0 = upmucnt6" : "=r"(counters[6]));
    asm ଅସ୍ଥିର("%0 = upmucnt7" : "=r"(counters[7]));
#ଅନ୍ୟଥା
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

ସଂରଚନା htp_thread_trace {
    uint32_t ଗଣନା_2;
    uint32_t max_events;
    ସଂରଚନା htp_trace_desc * events;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_trace_event(ସଂରଚନା htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ଯଦି (tr->ଗଣନା_2 < tr->max_events) {
        uint32_t i = tr->ଗଣନା_2;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->ଗଣନା_2++;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_trace_event_start(ସଂରଚନା htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_trace_event_stop(ସଂରଚନା htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
