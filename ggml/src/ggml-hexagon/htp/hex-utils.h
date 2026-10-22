#اگر_تعریف_نشده HEX_UTILS_H
#تعریف_کن HEX_UTILS_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt_memory.h>
#ضمن <qurt.h>

#ضمن "hexagon_types.h"
#ضمن "hexagon_protos.h"

#ضمن "hex-fastdiv.h"
#ضمن "hex-dump.h"
#ضمن "hex-common.h"

ساكن مضمن uint64_t hex_get_چرخهها() {
    uint64_t چرخهها = 0;
    asm متطاير(" %0 = c15:14\n" : "=r"(چرخهها));
    أرجع چرخهها;
}

ساكن مضمن uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm متطاير(" %0 = c19:18\n" : "=r"(pktcnt));
    أرجع pktcnt;
}

ساكن مضمن فراغ hex_l2fetch(ثابت فراغ * p, uint32_t عرض, uint32_t stride, uint32_t ارتفاع) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(عرض, ارتفاع));
    Q6_l2fetch_AP((فراغ *) p, control);
}

ساكن مضمن فراغ hex_l2fetch_بلوک(ثابت فراغ * addr, size_t اندازه) {
    إذا (اندازه == 0) أرجع;
    ثابت uint32_t عرض = 16384; // 16KB rows
    ثابت uint32_t ارتفاع = (اندازه + عرض - 1) / عرض;
    hex_l2fetch(addr, عرض, عرض, ارتفاع);
}

#تعریف_کن HEX_L2_سطر_اندازه           128
#تعریف_کن HEX_L2_بلوک_اندازه          (HEX_L2_سطر_اندازه * 4) // flush granularity (chunks per thread)
#تعریف_کن HEX_L2_FLUSH_WQ_آستانه  (4 * 1024)
#تعریف_کن HEX_L2_FLUSH_ALL_آستانه (4 * 1024 * 1024)

ساكن مضمن فراغ hex_l2flush(فراغ * addr, size_t اندازه) {
    إذا (اندازه == 0) أرجع;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_سطر_اندازه - 1);
    ثابت uint32_t e = (((uint32_t) addr) + اندازه + HEX_L2_سطر_اندازه - 1) & ~(HEX_L2_سطر_اندازه - 1);
    لكل (uint32_t i = s; i < e; i += HEX_L2_سطر_اندازه) {
        Q6_dccleaninva_A((فراغ *) i);
    }
}

ساكن مضمن فراغ hex_pause() {
    asm متطاير(" pause(#255)\n");
}

#پایان_اگر /* HEX_UTILS_H */
