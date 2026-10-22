#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_PROFILE_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_PROFILE_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <qurt.h>

#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TRACE_EVT_ಆರಂಭ 0
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TRACE_EVT_STOP  1

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_ಅಂಕೆ_PMU_COUNTERS
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_ಅಂಕೆ_PMU_COUNTERS 8
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hex_get_pmu(uint32_t counters[]) {
#ಆದರೆ __HVX_ARCH__ >= 79
    asm ಅಸ್ಥಿರ("%0 = upmucnt0" : "=r"(counters[0]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt1" : "=r"(counters[1]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt2" : "=r"(counters[2]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt3" : "=r"(counters[3]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt4" : "=r"(counters[4]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt5" : "=r"(counters[5]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt6" : "=r"(counters[6]));
    asm ಅಸ್ಥಿರ("%0 = upmucnt7" : "=r"(counters[7]));
#ಇಲ್ಲದಿದ್ದರೆ
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#ಆದರೆ_ಅಂತ್ಯ
}

ರಚನೆ htp_thread_trace {
    uint32_t ಎಣಿಕೆ_2;
    uint32_t max_events;
    ರಚನೆ htp_trace_desc * events;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_trace_event(ರಚನೆ htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    ಆದರೆ (tr->ಎಣಿಕೆ_2 < tr->max_events) {
        uint32_t i = tr->ಎಣಿಕೆ_2;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].ಚಕ್ರಗಳು = (uint32_t) hex_get_ಚಕ್ರಗಳು();
        tr->ಎಣಿಕೆ_2++;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_trace_event_ಆರಂಭ(ರಚನೆ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_ಆರಂಭ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_trace_event_stop(ರಚನೆ htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#ಆದರೆ_ಅಂತ್ಯ /* HEX_PROFILE_H */
