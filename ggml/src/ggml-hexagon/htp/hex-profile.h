#నిర్వచితం_కాకపోతే HEX_PROFILE_H
#నిర్వచించు HEX_PROFILE_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <qurt.h>

#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "htp-ops.h"

#నిర్వచించు HTP_TRACE_EVT_ప్రారంభం 0
#నిర్వచించు HTP_TRACE_EVT_STOP  1

#నిర్వచితం_కాకపోతే HEX_అంకె_PMU_COUNTERS
#నిర్వచించు HEX_అంకె_PMU_COUNTERS 8
#అయితే_ముగింపు

స్థిర అంతఃవరుస శూన్య hex_get_pmu(uint32_t counters[]) {
#ఒకవేళ __HVX_ARCH__ >= 79
    asm అస్థిర("%0 = upmucnt0" : "=r"(counters[0]));
    asm అస్థిర("%0 = upmucnt1" : "=r"(counters[1]));
    asm అస్థిర("%0 = upmucnt2" : "=r"(counters[2]));
    asm అస్థిర("%0 = upmucnt3" : "=r"(counters[3]));
    asm అస్థిర("%0 = upmucnt4" : "=r"(counters[4]));
    asm అస్థిర("%0 = upmucnt5" : "=r"(counters[5]));
    asm అస్థిర("%0 = upmucnt6" : "=r"(counters[6]));
    asm అస్థిర("%0 = upmucnt7" : "=r"(counters[7]));
#లేకపోతే
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#అయితే_ముగింపు
}

నిర్మాణం htp_thread_trace {
    uint32_t లెక్క;
    uint32_t max_events;
    నిర్మాణం htp_trace_desc * events;
};

స్థిర అంతఃవరుస శూన్య htp_trace_event(నిర్మాణం htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ఒకవేళ (tr->లెక్క < tr->max_events) {
        uint32_t i = tr->లెక్క;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].చక్రాలు = (uint32_t) hex_get_చక్రాలు();
        tr->లెక్క++;
    }
}

స్థిర అంతఃవరుస శూన్య htp_trace_event_ప్రారంభం(నిర్మాణం htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_ప్రారంభం);
}

స్థిర అంతఃవరుస శూన్య htp_trace_event_stop(నిర్మాణం htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#అయితే_ముగింపు /* HEX_PROFILE_H */
