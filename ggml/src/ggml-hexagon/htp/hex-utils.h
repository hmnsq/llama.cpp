#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <qurt_memory.h>
#ಅನ್ತರ್ಭಾವ <qurt.h>

#ಅನ್ತರ್ಭಾವ "hexagon_types.h"
#ಅನ್ತರ್ಭಾವ "hexagon_protos.h"

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-dump.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm ಚಞ್ಚಲ(" %0 = c15:14\n" : "=r"(cycles));
    ಪ್ರತ್ಯಾಗಮ cycles;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm ಚಞ್ಚಲ(" %0 = c19:18\n" : "=r"(pktcnt));
    ಪ್ರತ್ಯಾಗಮ pktcnt;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_l2fetch(ಧ್ರುವ ಶೂನ್ಯ * p, uint32_t width, uint32_t stride, uint32_t height) {
    ಧ್ರುವ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((ಶೂನ್ಯ *) p, control);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_l2fetch_block(ಧ್ರುವ ಶೂನ್ಯ * addr, size_t size) {
    ಯದಿ (size == 0) ಪ್ರತ್ಯಾಗಮ;
    ಧ್ರುವ uint32_t width = 16384; // 16KB rows
    ಧ್ರುವ uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_ಸಾಲು_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_ಸಾಲು_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_l2flush(ಶೂನ್ಯ * addr, size_t size) {
    ಯದಿ (size == 0) ಪ್ರತ್ಯಾಗಮ;
    ಧ್ರುವ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_ಸಾಲು_SIZE - 1);
    ಧ್ರುವ uint32_t e = (((uint32_t) addr) + size + HEX_L2_ಸಾಲು_SIZE - 1) & ~(HEX_L2_ಸಾಲು_SIZE - 1);
    ಪ್ರತಿ (uint32_t i = s; i < e; i += HEX_L2_ಸಾಲು_SIZE) {
        Q6_dccleaninva_A((ಶೂನ್ಯ *) i);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ hex_pause() {
    asm ಚಞ್ಚಲ(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
