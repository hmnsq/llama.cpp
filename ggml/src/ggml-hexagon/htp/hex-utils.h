#ifndef HEX_UTILS_H
#define HEX_UTILS_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <qurt_memory.h>
#अन्तर्भाव <qurt.h>

#अन्तर्भाव "hexagon_types.h"
#अन्तर्भाव "hexagon_protos.h"

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-dump.h"
#अन्तर्भाव "hex-common.h"

स्थिर अन्तःस्थ uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    यंत्र चञ्चल(" %0 = c15:14\n" : "=r"(cycles));
    प्रत्यागम cycles;
}

स्थिर अन्तःस्थ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    यंत्र चञ्चल(" %0 = c19:18\n" : "=r"(pktcnt));
    प्रत्यागम pktcnt;
}

स्थिर अन्तःस्थ शून्य hex_l2fetch(ध्रुव शून्य * p, uint32_t width, uint32_t stride, uint32_t height) {
    ध्रुव uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((शून्य *) p, control);
}

स्थिर अन्तःस्थ शून्य hex_l2fetch_block(ध्रुव शून्य * addr, माप_प्रकार size) {
    यदि (size == 0) प्रत्यागम;
    ध्रुव uint32_t width = 16384; // 16KB rows
    ध्रुव uint32_t height = (size + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#define HEX_L2_पंक्ति_SIZE           128
#define HEX_L2_BLOCK_SIZE          (HEX_L2_पंक्ति_SIZE * 4) // flush granularity (chunks per thread)
#define HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#define HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

स्थिर अन्तःस्थ शून्य hex_l2flush(शून्य * addr, माप_प्रकार size) {
    यदि (size == 0) प्रत्यागम;
    ध्रुव uint32_t s = ((uint32_t) addr) & ~(HEX_L2_पंक्ति_SIZE - 1);
    ध्रुव uint32_t e = (((uint32_t) addr) + size + HEX_L2_पंक्ति_SIZE - 1) & ~(HEX_L2_पंक्ति_SIZE - 1);
    प्रति (uint32_t i = s; i < e; i += HEX_L2_पंक्ति_SIZE) {
        Q6_dccleaninva_A((शून्य *) i);
    }
}

स्थिर अन्तःस्थ शून्य hex_pause() {
    यंत्र चञ्चल(" pause(#255)\n");
}

#endif /* HEX_UTILS_H */
