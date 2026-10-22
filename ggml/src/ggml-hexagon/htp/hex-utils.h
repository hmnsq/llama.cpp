#جيڪڏهن_وصف_نه_ٿيل HEX_UTILS_H
#وصف_ڪريو HEX_UTILS_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt_memory.h>
#ضمن <qurt.h>

#ضمن "hexagon_types.h"
#ضمن "hexagon_protos.h"

#ضمن "hex-fastdiv.h"
#ضمن "hex-dump.h"
#ضمن "hex-common.h"

ساكن مضمن uint64_t hex_get_چڪرن() {
    uint64_t چڪرن = 0;
    asm متطاير(" %0 = c15:14\n" : "=r"(چڪرن));
    أرجع چڪرن;
}

ساكن مضمن uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm متطاير(" %0 = c19:18\n" : "=r"(pktcnt));
    أرجع pktcnt;
}

ساكن مضمن فراغ hex_l2fetch(ثابت فراغ * p, uint32_t ويڪر, uint32_t stride, uint32_t اوچائي) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(ويڪر, اوچائي));
    Q6_l2fetch_AP((فراغ *) p, control);
}

ساكن مضمن فراغ hex_l2fetch_بلاڪ(ثابت فراغ * addr, size_t سائيز) {
    إذا (سائيز == 0) أرجع;
    ثابت uint32_t ويڪر = 16384; // 16KB rows
    ثابت uint32_t اوچائي = (سائيز + ويڪر - 1) / ويڪر;
    hex_l2fetch(addr, ويڪر, ويڪر, اوچائي);
}

#وصف_ڪريو HEX_L2_سٽ_سائيز           128
#وصف_ڪريو HEX_L2_بلاڪ_سائيز          (HEX_L2_سٽ_سائيز * 4) // flush granularity (chunks per thread)
#وصف_ڪريو HEX_L2_FLUSH_WQ_حد  (4 * 1024)
#وصف_ڪريو HEX_L2_FLUSH_ALL_حد (4 * 1024 * 1024)

ساكن مضمن فراغ hex_l2flush(فراغ * addr, size_t سائيز) {
    إذا (سائيز == 0) أرجع;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_سٽ_سائيز - 1);
    ثابت uint32_t e = (((uint32_t) addr) + سائيز + HEX_L2_سٽ_سائيز - 1) & ~(HEX_L2_سٽ_سائيز - 1);
    لكل (uint32_t i = s; i < e; i += HEX_L2_سٽ_سائيز) {
        Q6_dccleaninva_A((فراغ *) i);
    }
}

ساكن مضمن فراغ hex_pause() {
    asm متطاير(" pause(#255)\n");
}

#جيڪڏهن_پورو /* HEX_UTILS_H */
