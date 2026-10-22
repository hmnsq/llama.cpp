#अगर_अघोषित HEX_PROFILE_H
#घोषणा HEX_PROFILE_H

#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <qurt.h>

#समावेश "hex-utils.h"
#समावेश "htp-ops.h"

#घोषणा HTP_TRACE_EVT_START 0
#घोषणा HTP_TRACE_EVT_STOP  1

#अगर_अघोषित HEX_अंक_PMU_COUNTERS
#घोषणा HEX_अंक_PMU_COUNTERS 8
#पूर्ण_अगर

अटल अंतरभूत व्योम hex_get_pmu(uint32_t counters[]) {
#अगर __HVX_ARCH__ >= 79
    यंत्र अस्थिर("%0 = upmucnt0" : "=r"(counters[0]));
    यंत्र अस्थिर("%0 = upmucnt1" : "=r"(counters[1]));
    यंत्र अस्थिर("%0 = upmucnt2" : "=r"(counters[2]));
    यंत्र अस्थिर("%0 = upmucnt3" : "=r"(counters[3]));
    यंत्र अस्थिर("%0 = upmucnt4" : "=r"(counters[4]));
    यंत्र अस्थिर("%0 = upmucnt5" : "=r"(counters[5]));
    यंत्र अस्थिर("%0 = upmucnt6" : "=r"(counters[6]));
    यंत्र अस्थिर("%0 = upmucnt7" : "=r"(counters[7]));
#अन्यथा
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
#पूर्ण_अगर
}

काष्ठा htp_thread_trace {
    uint32_t गिनती;
    uint32_t max_events;
    काष्ठा htp_trace_desc * events;
};

अटल अंतरभूत व्योम htp_trace_event(काष्ठा htp_thread_trace * tr, uint16_t id, uint16_t info, uint32_t type) {
    अगर (tr->गिनती < tr->max_events) {
        uint32_t i = tr->गिनती;
        tr->events[i].id     = id;
        tr->events[i].info   = info | (type == HTP_TRACE_EVT_STOP ? 0x8000 : 0);
        tr->events[i].cycles = (uint32_t) hex_get_cycles();
        tr->गिनती++;
    }
}

अटल अंतरभूत व्योम htp_trace_event_start(काष्ठा htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_START);
}

अटल अंतरभूत व्योम htp_trace_event_stop(काष्ठा htp_thread_trace * tr, uint16_t id, uint16_t info) {
    htp_trace_event(tr, id, info, HTP_TRACE_EVT_STOP);
}

#पूर्ण_अगर /* HEX_PROFILE_H */
