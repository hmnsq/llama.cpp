#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <qurt_memory.h>
#شامل <qurt.h>

#شامل "hexagon_types.h"
#شامل "hexagon_protos.h"

#شامل "hex-fastdiv.h"
#شامل "hex-dump.h"
#شامل "hex-common.h"

ایستا درونخطی uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm فرار(" %0 = c15:14\n" : "=r"(cycles));
    بازگشت cycles;
}

ایستا درونخطی uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm فرار(" %0 = c19:18\n" : "=r"(pktcnt));
    بازگشت pktcnt;
}

ایستا درونخطی تهی hex_l2fetch(ثابت تهی * p, uint32_t width, uint32_t stride, uint32_t height) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((تهی *) p, control);
}

ایستا درونخطی تهی hex_l2fetch_block(ثابت تهی * addr, size_t size) {
    اگر (size == 0) بازگشت;
    ثابت uint32_t width = 16384; // 16KB rows
    ثابت uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_سطر_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_سطر_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

ایستا درونخطی تهی hex_l2flush(تهی * addr, size_t size) {
    اگر (size == 0) بازگشت;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_سطر_SIZE - 1);
    ثابت uint32_t e = (((uint32_t) addr) + size + HEX_L2_سطر_SIZE - 1) & ~(HEX_L2_سطر_SIZE - 1);
    برای (uint32_t i = s; i < e; i += HEX_L2_سطر_SIZE) {
        Q6_dccleaninva_A((تهی *) i);
    }
}

ایستا درونخطی تهی hex_pause() {
    asm فرار(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
