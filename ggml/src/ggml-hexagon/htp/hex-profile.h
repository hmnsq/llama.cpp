#यदि_अपरिभाषितम् HEX_PROFILE_H
#परिभाषय HEX_PROFILE_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <qurt.h>

#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "htp-ops.h"

#परिभाषय HTP_TRACE_EVT_START 0
#परिभाषय HTP_TRACE_EVT_STOP  1

#यदि_अपरिभाषितम् HEX_अङ्कः_PMU_COUNTERS
#परिभाषय HEX_अङ्कः_PMU_COUNTERS 8
#यदि_अन्तः

स्थिर अन्तःस्थ शून्य hex_get_pmu(uint32_t counters[]) {
#यदि __HVX_ARCH__ >= 79
    asm चञ्चल("%0 = upmucnt0" : "=r"(counters[0]));
    asm चञ्चल("%0 = upmucnt1" : "=r"(counters[1]));
    asm चञ्चल("%0 = upmucnt2" : "=r"(counters[2]));
    asm चञ्चल("%0 = upmucnt3" : "=r"(counters[3]));
    asm चञ्चल("%0 = upmucnt4" : "=r"(counters[4]));
    asm चञ्चल("%0 = upmucnt5" : "=r"(counters[5]));
    asm चञ्चल("%0 = upmucnt6" : "=r"(counters[6]));
    asm चञ्चल("%0 = upmucnt7" : "=r"(counters[7]));
#अन्यथा
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#यदि_अन्तः
}

संरचना htp_thread_trace {
    uint32_t गणनम्;
    uint32_t max_events;
    संरचना htp_trace_desc * events;
};

स्थिर अन्तःस्थ शून्य htp_trace_event(संरचना htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    यदि (tr->गणनम् < tr->max_events) {
        uint32_t i = tr->गणनम्;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].आवृत्तयः = (uint32_t) hex_get_आवृत्तयः();
        tr->गणनम्++;
    }
}

स्थिर अन्तःस्थ शून्य htp_trace_event_start(संरचना htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

स्थिर अन्तःस्थ शून्य htp_trace_event_stop(संरचना htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#यदि_अन्तः /* HEX_PROFILE_H */
