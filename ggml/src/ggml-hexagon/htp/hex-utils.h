#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <qurt_memory.h>
#ضمن <qurt.h>

#ضمن "hexagon_types.h"
#ضمن "hexagon_protos.h"

#ضمن "hex-fastdiv.h"
#ضمن "hex-dump.h"
#ضمن "hex-common.h"

ساكن مضمن uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm متطاير(" %0 = c15:14\n" : "=r"(cycles));
    ܦܢܐ cycles;
}

ساكن مضمن uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm متطاير(" %0 = c19:18\n" : "=r"(pktcnt));
    ܦܢܐ pktcnt;
}

ساكن مضمن فراغ hex_l2fetch(ثابت فراغ * p, uint32_t width, uint32_t stride, uint32_t height) {
    ثابت uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((فراغ *) p, control);
}

ساكن مضمن فراغ hex_l2fetch_block(ثابت فراغ * addr, size_t size) {
    ܐܢ (size == 0) ܦܢܐ;
    ثابت uint32_t width = 16384; // 16KB rows
    ثابت uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_LINE_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_LINE_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

ساكن مضمن فراغ hex_l2flush(فراغ * addr, size_t size) {
    ܐܢ (size == 0) ܦܢܐ;
    ثابت uint32_t s = ((uint32_t) addr) & ~(HEX_L2_LINE_SIZE - 1);
    ثابت uint32_t e = (((uint32_t) addr) + size + HEX_L2_LINE_SIZE - 1) & ~(HEX_L2_LINE_SIZE - 1);
    لكل (uint32_t i = s; i < e; i += HEX_L2_LINE_SIZE) {
        Q6_dccleaninva_A((فراغ *) i);
    }
}

ساكن مضمن فراغ hex_pause() {
    asm متطاير(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
