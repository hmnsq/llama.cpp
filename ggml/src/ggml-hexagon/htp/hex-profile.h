#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_PROFILE_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_PROFILE_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TRACE_EVT_ਸ਼ੁਰੂ 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_TRACE_EVT_STOP  1

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_ਅੰਕ_PMU_COUNTERS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_ਅੰਕ_PMU_COUNTERS 8
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_get_pmu(uint32_t counters[]) {
#ਜੇ __HVX_ARCH__ >= 79
    asm ਅਸਥਿਰ("%0 = upmucnt0" : "=r"(counters[0]));
    asm ਅਸਥਿਰ("%0 = upmucnt1" : "=r"(counters[1]));
    asm ਅਸਥਿਰ("%0 = upmucnt2" : "=r"(counters[2]));
    asm ਅਸਥਿਰ("%0 = upmucnt3" : "=r"(counters[3]));
    asm ਅਸਥਿਰ("%0 = upmucnt4" : "=r"(counters[4]));
    asm ਅਸਥਿਰ("%0 = upmucnt5" : "=r"(counters[5]));
    asm ਅਸਥਿਰ("%0 = upmucnt6" : "=r"(counters[6]));
    asm ਅਸਥਿਰ("%0 = upmucnt7" : "=r"(counters[7]));
#ਨਹੀਂਤਾਂ
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#ਜੇ_ਅੰਤ
}

ਬਣਤਰ htp_thread_trace {
    uint32_t ਗਿਣਤੀ_2;
    uint32_t max_events;
    ਬਣਤਰ htp_trace_desc * events;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_trace_event(ਬਣਤਰ htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ਜੇ (tr->ਗਿਣਤੀ_2 < tr->max_events) {
        uint32_t i = tr->ਗਿਣਤੀ_2;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].ਚੱਕਰਾਂ = (uint32_t) hex_get_ਚੱਕਰਾਂ();
        tr->ਗਿਣਤੀ_2++;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_trace_event_ਸ਼ੁਰੂ(ਬਣਤਰ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_ਸ਼ੁਰੂ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_trace_event_stop(ਬਣਤਰ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#ਜੇ_ਅੰਤ /* HEX_PROFILE_H */
