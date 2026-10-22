#اگر_غیر_تعریف_شدہ HEX_PROFILE_H
#تعریف_کرو HEX_PROFILE_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <qurt.h>

#شامل "hex-utils.h"
#شامل "htp-ops.h"

#تعریف_کرو HTP_TRACE_EVT_آغاز 0
#تعریف_کرو HTP_TRACE_EVT_STOP  1

#اگر_غیر_تعریف_شدہ HEX_عدد_PMU_COUNTERS
#تعریف_کرو HEX_عدد_PMU_COUNTERS 8
#اگر_ختم

ساکن اندرونی خالی hex_get_pmu(uint32_t counters[]) {
#اگر __HVX_ARCH__ >= 79
    asm متلون("%0 = upmucnt0" : "=r"(counters[0]));
    asm متلون("%0 = upmucnt1" : "=r"(counters[1]));
    asm متلون("%0 = upmucnt2" : "=r"(counters[2]));
    asm متلون("%0 = upmucnt3" : "=r"(counters[3]));
    asm متلون("%0 = upmucnt4" : "=r"(counters[4]));
    asm متلون("%0 = upmucnt5" : "=r"(counters[5]));
    asm متلون("%0 = upmucnt6" : "=r"(counters[6]));
    asm متلون("%0 = upmucnt7" : "=r"(counters[7]));
#ورنہ
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#اگر_ختم
}

ساخت htp_thread_trace {
    uint32_t گنتی;
    uint32_t max_events;
    ساخت htp_trace_desc * events;
};

ساکن اندرونی خالی htp_trace_event(ساخت htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    اگر (tr->گنتی < tr->max_events) {
        uint32_t i = tr->گنتی;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].ادوار = (uint32_t) hex_get_ادوار();
        tr->گنتی++;
    }
}

ساکن اندرونی خالی htp_trace_event_آغاز(ساخت htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_آغاز);
}

ساکن اندرونی خالی htp_trace_event_stop(ساخت htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#اگر_ختم /* HEX_PROFILE_H */
