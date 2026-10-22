#اگر_غیر_تعریف_شدہ HEX_UTILS_H
#تعریف_کرو HEX_UTILS_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <qurt_memory.h>
#شامل <qurt.h>

#شامل "hexagon_types.h"
#شامل "hexagon_protos.h"

#شامل "hex-fastdiv.h"
#شامل "hex-dump.h"
#شامل "hex-common.h"

ساکن اندرونی uint64_t hex_get_ادوار() {
    uint64_t ادوار = 0;
    asm متلون(" %0 = c15:14\n" : "=r"(ادوار));
    واپس ادوار;
}

ساکن اندرونی uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm متلون(" %0 = c19:18\n" : "=r"(pktcnt));
    واپس pktcnt;
}

ساکن اندرونی خالی hex_l2fetch(ثابت خالی * p, uint32_t چوڑائی, uint32_t stride, uint32_t اونچائی) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(چوڑائی, اونچائی));
    Q6_l2fetch_AP((خالی *) p, control);
}

ساکن اندرونی خالی hex_l2fetch_بلاک(ثابت خالی * addr, size_t سائز) {
    اگر (سائز == 0) واپس;
    ثابت uint32_t چوڑائی = 16384; // 16KB rows
    ثابت uint32_t اونچائی = (سائز + چوڑائی - 1) / چوڑائی;
    hex_l2fetch(addr, چوڑائی, چوڑائی, اونچائی);
}

#تعریف_کرو HEX_L2_سطر_سائز           128
#تعریف_کرو HEX_L2_بلاک_سائز          (HEX_L2_سطر_سائز * 4) // flush granularity (chunks per thread)
#تعریف_کرو HEX_L2_FLUSH_WQ_حد  (4 * 1024)
#تعریف_کرو HEX_L2_FLUSH_ALL_حد (4 * 1024 * 1024)

ساکن اندرونی خالی hex_l2flush(خالی * addr, size_t سائز) {
    اگر (سائز == 0) واپس;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_سطر_سائز - 1);
    ثابت uint32_t e = (((uint32_t) addr) + سائز + HEX_L2_سطر_سائز - 1) & ~(HEX_L2_سطر_سائز - 1);
    برائے (uint32_t i = s; i < e; i += HEX_L2_سطر_سائز) {
        Q6_dccleaninva_A((خالی *) i);
    }
}

ساکن اندرونی خالی hex_pause() {
    asm متلون(" pause(#255)\n");
}

#اگر_ختم /* HEX_UTILS_H */
