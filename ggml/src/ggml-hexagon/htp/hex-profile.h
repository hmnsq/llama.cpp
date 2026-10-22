#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <qurt.h>

#અન્તર્ભાવ "hex-utils.h"
#અન્તર્ભાવ "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_અંક_PMU_COUNTERS
#define HEX_અંક_PMU_COUNTERS 8
#endif

સ્થિર અંતઃરેખ શૂન્ય hex_get_pmu(uint32_t counters[]) {
#જો __HVX_ARCH__ >= 79
    asm અસ્થિર("%0 = upmucnt0" : "=r"(counters[0]));
    asm અસ્થિર("%0 = upmucnt1" : "=r"(counters[1]));
    asm અસ્થિર("%0 = upmucnt2" : "=r"(counters[2]));
    asm અસ્થિર("%0 = upmucnt3" : "=r"(counters[3]));
    asm અસ્થિર("%0 = upmucnt4" : "=r"(counters[4]));
    asm અસ્થિર("%0 = upmucnt5" : "=r"(counters[5]));
    asm અસ્થિર("%0 = upmucnt6" : "=r"(counters[6]));
    asm અસ્થિર("%0 = upmucnt7" : "=r"(counters[7]));
#નહીંતર
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

સંરચના htp_thread_trace {
    uint32_t ગણતરી;
    uint32_t max_events;
    સંરચના htp_trace_desc * events;
};

સ્થિર અંતઃરેખ શૂન્ય htp_trace_event(સંરચના htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    જો (tr->ગણતરી < tr->max_events) {
        uint32_t i = tr->ગણતરી;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->ગણતરી++;
    }
}

સ્થિર અંતઃરેખ શૂન્ય htp_trace_event_start(સંરચના htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

સ્થિર અંતઃરેખ શૂન્ય htp_trace_event_stop(સંરચના htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
