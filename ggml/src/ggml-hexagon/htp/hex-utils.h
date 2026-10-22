#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_UTILS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_UTILS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt_memory.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <qurt.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hexagon_types.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hexagon_protos.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-dump.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t hex_get_ਚੱਕਰਾਂ() {
    uint64_t ਚੱਕਰਾਂ = 0;
    asm ਅਸਥਿਰ(" %0 = c15:14\n" : "=r"(ਚੱਕਰਾਂ));
    ਵਾਪਸ ਚੱਕਰਾਂ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm ਅਸਥਿਰ(" %0 = c19:18\n" : "=r"(pktcnt));
    ਵਾਪਸ pktcnt;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_l2fetch(ਅਚਲ ਖਾਲੀ * p, uint32_t ਚੌੜਾਈ, uint32_t stride, uint32_t ਉਚਾਈ) {
    ਅਚਲ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(ਚੌੜਾਈ, ਉਚਾਈ));
    Q6_l2fetch_AP((ਖਾਲੀ *) p, control);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_l2fetch_ਖੰਡ(ਅਚਲ ਖਾਲੀ * addr, size_t ਮਾਪ) {
    ਜੇ (ਮਾਪ == 0) ਵਾਪਸ;
    ਅਚਲ uint32_t ਚੌੜਾਈ = 16384; // 16KB rows
    ਅਚਲ uint32_t ਉਚਾਈ = (ਮਾਪ + ਚੌੜਾਈ - 1) / ਚੌੜਾਈ;
    hex_l2fetch(addr, ਚੌੜਾਈ, ਚੌੜਾਈ, ਉਚਾਈ);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_L2_ਸਤਰ_ਮਾਪ           128
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_L2_ਖੰਡ_ਮਾਪ          (HEX_L2_ਸਤਰ_ਮਾਪ * 4) // flush granularity (chunks per thread)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_L2_FLUSH_WQ_ਹੱਦ  (4 * 1024)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_L2_FLUSH_ALL_ਹੱਦ (4 * 1024 * 1024)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_l2flush(ਖਾਲੀ * addr, size_t ਮਾਪ) {
    ਜੇ (ਮਾਪ == 0) ਵਾਪਸ;
    ਅਚਲ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_ਸਤਰ_ਮਾਪ - 1);
    ਅਚਲ uint32_t e = (((uint32_t) addr) + ਮਾਪ + HEX_L2_ਸਤਰ_ਮਾਪ - 1) & ~(HEX_L2_ਸਤਰ_ਮਾਪ - 1);
    ਲਈ (uint32_t i = s; i < e; i += HEX_L2_ਸਤਰ_ਮਾਪ) {
        Q6_dccleaninva_A((ਖਾਲੀ *) i);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hex_pause() {
    asm ਅਸਥਿਰ(" pause(#255)\n");
}

#ਜੇ_ਅੰਤ /* HEX_UTILS_H */
