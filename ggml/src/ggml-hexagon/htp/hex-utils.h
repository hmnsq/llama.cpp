#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#অন্তর্ভাৱ <stdbool.h>
#অন্তর্ভাৱ <stdint.h>
#অন্তর্ভাৱ <qurt_memory.h>
#অন্তর্ভাৱ <qurt.h>

#অন্তর্ভাৱ "hexagon_types.h"
#অন্তর্ভাৱ "hexagon_protos.h"

#অন্তর্ভাৱ "hex-fastdiv.h"
#অন্তর্ভাৱ "hex-dump.h"
#অন্তর্ভাৱ "hex-common.h"

স্থির অন্তঃস্থ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm চঞ্চল(" %0 = c15:14\n" : "=r"(cycles));
    ঘূৰাই cycles;
}

স্থির অন্তঃস্থ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm চঞ্চল(" %0 = c19:18\n" : "=r"(pktcnt));
    ঘূৰাই pktcnt;
}

স্থির অন্তঃস্থ শূন্য hex_l2fetch(ধ্রুৱ শূন্য * p, uint32_t width, uint32_t stride, uint32_t height) {
    ধ্রুৱ uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((শূন্য *) p, control);
}

স্থির অন্তঃস্থ শূন্য hex_l2fetch_block(ধ্রুৱ শূন্য * addr, size_t size) {
    যদি (size == 0) ঘূৰাই;
    ধ্রুৱ uint32_t width = 16384; // 16KB rows
    ধ্রুৱ uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_সাৰি_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_সাৰি_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

স্থির অন্তঃস্থ শূন্য hex_l2flush(শূন্য * addr, size_t size) {
    যদি (size == 0) ঘূৰাই;
    ধ্রুৱ uint32_t s = ((uint32_t) addr) & ~(HEX_L2_সাৰি_SIZE - 1);
    ধ্রুৱ uint32_t e = (((uint32_t) addr) + size + HEX_L2_সাৰি_SIZE - 1) & ~(HEX_L2_সাৰি_SIZE - 1);
    বাবে (uint32_t i = s; i < e; i += HEX_L2_সাৰি_SIZE) {
        Q6_dccleaninva_A((শূন্য *) i);
    }
}

স্থির অন্তঃস্থ শূন্য hex_pause() {
    asm চঞ্চল(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
