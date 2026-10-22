#جے_بنا_تعریف HEX_UTILS_H
#تعریف_کرو HEX_UTILS_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt_memory.h>
#ضمن <qurt.h>

#ضمن "hexagon_types.h"
#ضمن "hexagon_protos.h"

#ضمن "hex-fastdiv.h"
#ضمن "hex-dump.h"
#ضمن "hex-common.h"

ساكن مضمن uint64_t hex_get_چکراں() {
    uint64_t چکراں = 0;
    asm متطاير(" %0 = c15:14\n" : "=r"(چکراں));
    أرجع چکراں;
}

ساكن مضمن uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm متطاير(" %0 = c19:18\n" : "=r"(pktcnt));
    أرجع pktcnt;
}

ساكن مضمن فراغ hex_l2fetch(ثابت فراغ * p, uint32_t چوڑائی, uint32_t stride, uint32_t اچائی) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(چوڑائی, اچائی));
    Q6_l2fetch_AP((فراغ *) p, control);
}

ساكن مضمن فراغ hex_l2fetch_بلاک(ثابت فراغ * addr, size_t ناپ) {
    إذا (ناپ == 0) أرجع;
    ثابت uint32_t چوڑائی = 16384; // 16KB rows
    ثابت uint32_t اچائی = (ناپ + چوڑائی - 1) / چوڑائی;
    hex_l2fetch(addr, چوڑائی, چوڑائی, اچائی);
}

#تعریف_کرو HEX_L2_سطر_ناپ           128
#تعریف_کرو HEX_L2_بلاک_ناپ          (HEX_L2_سطر_ناپ * 4) // flush granularity (chunks per thread)
#تعریف_کرو HEX_L2_FLUSH_WQ_حد  (4 * 1024)
#تعریف_کرو HEX_L2_FLUSH_ALL_حد (4 * 1024 * 1024)

ساكن مضمن فراغ hex_l2flush(فراغ * addr, size_t ناپ) {
    إذا (ناپ == 0) أرجع;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_سطر_ناپ - 1);
    ثابت uint32_t e = (((uint32_t) addr) + ناپ + HEX_L2_سطر_ناپ - 1) & ~(HEX_L2_سطر_ناپ - 1);
    لكل (uint32_t i = s; i < e; i += HEX_L2_سطر_ناپ) {
        Q6_dccleaninva_A((فراغ *) i);
    }
}

ساكن مضمن فراغ hex_pause() {
    asm متطاير(" pause(#255)\n");
}

#جے_انت /* HEX_UTILS_H */
