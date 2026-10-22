#ifndef HEX_PROFILE_H
#define HEX_PROFILE_H

#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <qurt.h>

#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "htp-ops.h"

#define HTP_TRACE_EVT_START 0
#define HTP_TRACE_EVT_STOP  1

#ifndef HEX_இலக்கம்_PMU_COUNTERS
#define HEX_இலக்கம்_PMU_COUNTERS 8
#endif

நிலையான உள்வரி வெறுமை hex_get_pmu(uint32_t counters[]) {
#எனில் __HVX_ARCH__ >= 79
    asm நிலையற்ற("%0 = upmucnt0" : "=r"(counters[0]));
    asm நிலையற்ற("%0 = upmucnt1" : "=r"(counters[1]));
    asm நிலையற்ற("%0 = upmucnt2" : "=r"(counters[2]));
    asm நிலையற்ற("%0 = upmucnt3" : "=r"(counters[3]));
    asm நிலையற்ற("%0 = upmucnt4" : "=r"(counters[4]));
    asm நிலையற்ற("%0 = upmucnt5" : "=r"(counters[5]));
    asm நிலையற்ற("%0 = upmucnt6" : "=r"(counters[6]));
    asm நிலையற்ற("%0 = upmucnt7" : "=r"(counters[7]));
#இல்லையெனில்
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

கட்டமைப்பு htp_thread_trace {
    uint32_t மொத்தம்;
    uint32_t max_events;
    கட்டமைப்பு htp_trace_desc * events;
};

நிலையான உள்வரி வெறுமை htp_trace_event(கட்டமைப்பு htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    எனில் (tr->மொத்தம் < tr->max_events) {
        uint32_t i = tr->மொத்தம்;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->மொத்தம்++;
    }
}

நிலையான உள்வரி வெறுமை htp_trace_event_start(கட்டமைப்பு htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

நிலையான உள்வரி வெறுமை htp_trace_event_stop(கட்டமைப்பு htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#endif /* HEX_PROFILE_H */
