#నిర్వచితం_కాకపోతే HEX_UTILS_H
#నిర్వచించు HEX_UTILS_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <qurt_memory.h>
#అన్తర్భావ <qurt.h>

#అన్తర్భావ "hexagon_types.h"
#అన్తర్భావ "hexagon_protos.h"

#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-dump.h"
#అన్తర్భావ "hex-common.h"

స్థిర అంతఃవరుస uint64_t hex_get_చక్రాలు() {
    uint64_t చక్రాలు = 0;
    asm అస్థిర(" %0 = c15:14\n" : "=r"(చక్రాలు));
    తిరిగి చక్రాలు;
}

స్థిర అంతఃవరుస uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm అస్థిర(" %0 = c19:18\n" : "=r"(pktcnt));
    తిరిగి pktcnt;
}

స్థిర అంతఃవరుస శూన్య hex_l2fetch(స్థిరరాశి శూన్య * p, uint32_t వెడల్పు, uint32_t stride, uint32_t ఎత్తు) {
    స్థిరరాశి uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(వెడల్పు, ఎత్తు));
    Q6_l2fetch_AP((శూన్య *) p, control);
}

స్థిర అంతఃవరుస శూన్య hex_l2fetch_ఖండం(స్థిరరాశి శూన్య * addr, size_t కొలత) {
    ఒకవేళ (కొలత == 0) తిరిగి;
    స్థిరరాశి uint32_t వెడల్పు = 16384; // 16KB rows
    స్థిరరాశి uint32_t ఎత్తు = (కొలత + వెడల్పు - 1) / వెడల్పు;
    hex_l2fetch(addr, వెడల్పు, వెడల్పు, ఎత్తు);
}

#నిర్వచించు HEX_L2_పంక్తి_కొలత           128
#నిర్వచించు HEX_L2_ఖండం_కొలత          (HEX_L2_పంక్తి_కొలత * 4) // flush granularity (chunks per thread)
#నిర్వచించు HEX_L2_FLUSH_WQ_పరిమితి  (4 * 1024)
#నిర్వచించు HEX_L2_FLUSH_ALL_పరిమితి (4 * 1024 * 1024)

స్థిర అంతఃవరుస శూన్య hex_l2flush(శూన్య * addr, size_t కొలత) {
    ఒకవేళ (కొలత == 0) తిరిగి;
    స్థిరరాశి uint32_t s = ((uint32_t) addr) & ~(HEX_L2_పంక్తి_కొలత - 1);
    స్థిరరాశి uint32_t e = (((uint32_t) addr) + కొలత + HEX_L2_పంక్తి_కొలత - 1) & ~(HEX_L2_పంక్తి_కొలత - 1);
    కొరకు (uint32_t i = s; i < e; i += HEX_L2_పంక్తి_కొలత) {
        Q6_dccleaninva_A((శూన్య *) i);
    }
}

స్థిర అంతఃవరుస శూన్య hex_pause() {
    asm అస్థిర(" pause(#255)\n");
}

#అయితే_ముగింపు /* HEX_UTILS_H */
