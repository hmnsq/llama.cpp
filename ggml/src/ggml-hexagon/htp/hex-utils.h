#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <qurt_memory.h>
#অন্তর্ভাব <qurt.h>

#অন্তর্ভাব "hexagon_types.h"
#অন্তর্ভাব "hexagon_protos.h"

#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "hex-dump.h"
#অন্তর্ভাব "hex-common.h"

স্থির অন্তঃরেখ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm অস্থির(" %0 = c15:14\n" : "=r"(cycles));
    ফেরত cycles;
}

স্থির অন্তঃরেখ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm অস্থির(" %0 = c19:18\n" : "=r"(pktcnt));
    ফেরত pktcnt;
}

স্থির অন্তঃরেখ শূন্য hex_l2fetch(ধ্রুবক শূন্য * p, uint32_t width, uint32_t stride, uint32_t height) {
    ধ্রুবক uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((শূন্য *) p, control);
}

স্থির অন্তঃরেখ শূন্য hex_l2fetch_block(ধ্রুবক শূন্য * addr, size_t আকার_2) {
    যদি (আকার_2 == 0) ফেরত;
    ধ্রুবক uint32_t width = 16384; // 16KB rows
    ধ্রুবক uint32_t height = (আকার_2 + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_সারি_আকার           128
#define HEX_L2_BLOCK_আকার          (HEX_L2_সারি_আকার * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

স্থির অন্তঃরেখ শূন্য hex_l2flush(শূন্য * addr, size_t আকার_2) {
    যদি (আকার_2 == 0) ফেরত;
    ধ্রুবক uint32_t s = ((uint32_t) addr) & ~(HEX_L2_সারি_আকার - 1);
    ধ্রুবক uint32_t e = (((uint32_t) addr) + আকার_2 + HEX_L2_সারি_আকার - 1) & ~(HEX_L2_সারি_আকার - 1);
    জন্য (uint32_t i = s; i < e; i += HEX_L2_সারি_আকার) {
        Q6_dccleaninva_A((শূন্য *) i);
    }
}

স্থির অন্তঃরেখ শূন্য hex_pause() {
    asm অস্থির(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
