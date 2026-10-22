#यदि_अपरिभाषितम् HEX_UTILS_H
#परिभाषय HEX_UTILS_H

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <qurt_memory.h>
#अन्तर्भाव <qurt.h>

#अन्तर्भाव "hexagon_types.h"
#अन्तर्भाव "hexagon_protos.h"

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-dump.h"
#अन्तर्भाव "hex-common.h"

स्थिर अन्तःस्थ uint64_t hex_get_आवृत्तयः() {
    uint64_t आवृत्तयः = 0;
    asm चञ्चल(" %0 = c15:14\n" : "=r"(आवृत्तयः));
    प्रत्यागम आवृत्तयः;
}

स्थिर अन्तःस्थ uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm चञ्चल(" %0 = c19:18\n" : "=r"(pktcnt));
    प्रत्यागम pktcnt;
}

स्थिर अन्तःस्थ शून्य hex_l2fetch(ध्रुव शून्य * p, uint32_t विस्तारः, uint32_t stride, uint32_t height) {
    ध्रुव uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(विस्तारः, height));
    Q6_l2fetch_AP((शून्य *) p, control);
}

स्थिर अन्तःस्थ शून्य hex_l2fetch_खण्डम्(ध्रुव शून्य * addr, size_t आकार) {
    यदि (आकार == 0) प्रत्यागम;
    ध्रुव uint32_t विस्तारः = 16384; // 16KB rows
    ध्रुव uint32_t height = (आकार + विस्तारः - 1) / विस्तारः;
    hex_l2fetch(addr, विस्तारः, विस्तारः, height);
}

#परिभाषय HEX_L2_पङ्क्तिः_आकार           128
#परिभाषय HEX_L2_खण्डम्_आकार          (HEX_L2_पङ्क्तिः_आकार * 4) // flush granularity (chunks per thread)
#परिभाषय HEX_L2_FLUSH_WQ_देहली  (4 * 1024)
#परिभाषय HEX_L2_FLUSH_ALL_देहली (4 * 1024 * 1024)

स्थिर अन्तःस्थ शून्य hex_l2flush(शून्य * addr, size_t आकार) {
    यदि (आकार == 0) प्रत्यागम;
    ध्रुव uint32_t s = ((uint32_t) addr) & ~(HEX_L2_पङ्क्तिः_आकार - 1);
    ध्रुव uint32_t e = (((uint32_t) addr) + आकार + HEX_L2_पङ्क्तिः_आकार - 1) & ~(HEX_L2_पङ्क्तिः_आकार - 1);
    प्रति (uint32_t i = s; i < e; i += HEX_L2_पङ्क्तिः_आकार) {
        Q6_dccleaninva_A((शून्य *) i);
    }
}

स्थिर अन्तःस्थ शून्य hex_pause() {
    asm चञ्चल(" pause(#255)\n");
}

#यदि_अन्तः /* HEX_UTILS_H */
