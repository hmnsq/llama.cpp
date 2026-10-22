#که_نه_تعریف_شوی HEX_PROFILE_H
#تعریف_کړه HEX_PROFILE_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt.h>

#ضمن "hex-utils.h"
#ضمن "htp-ops.h"

#تعریف_کړه HTP_TRACE_EVT_پیل 0
#تعریف_کړه HTP_TRACE_EVT_STOP  1

#که_نه_تعریف_شوی HEX_عدد_PMU_COUNTERS
#تعریف_کړه HEX_عدد_PMU_COUNTERS 8
#د_که_پای

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
#د_که_پای
}

بنية htp_thread_trace {
    uint32_t شمېر;
    uint32_t max_events;
    بنية htp_trace_desc * events;
};

ساكن مضمن فراغ htp_trace_event(بنية htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    إذا (tr->شمېر < tr->max_events) {
        uint32_t i = tr->شمېر;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].دورې = (uint32_t) hex_get_دورې();
        tr->شمېر++;
    }
}

ساكن مضمن فراغ htp_trace_event_پیل(بنية htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_پیل);
}

ساكن مضمن فراغ htp_trace_event_stop(بنية htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#د_که_پای /* HEX_PROFILE_H */
