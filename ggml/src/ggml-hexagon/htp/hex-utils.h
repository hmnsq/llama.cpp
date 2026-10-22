#अगर_अघोषित HEX_UTILS_H
#घोषणा HEX_UTILS_H

#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <qurt_memory.h>
#समावेश <qurt.h>

#समावेश "hexagon_types.h"
#समावेश "hexagon_protos.h"

#समावेश "hex-fastdiv.h"
#समावेश "hex-dump.h"
#समावेश "hex-common.h"

अटल अंतरभूत uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    यंत्र अस्थिर(" %0 = c15:14\n" : "=r"(cycles));
    वापस cycles;
}

अटल अंतरभूत uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    यंत्र अस्थिर(" %0 = c19:18\n" : "=r"(pktcnt));
    वापस pktcnt;
}

अटल अंतरभूत व्योम hex_l2fetch(स्थिर व्योम * p, uint32_t width, uint32_t stride, uint32_t height) {
    स्थिर uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((व्योम *) p, control);
}

अटल अंतरभूत व्योम hex_l2fetch_block(स्थिर व्योम * addr, माप_प्रकार आकार) {
    अगर (आकार == 0) वापस;
    स्थिर uint32_t width = 16384; // 16KB rows
    स्थिर uint32_t height = (आकार + width - 1) / width;
    hex_l2fetch(addr, width, width, height);
}

#घोषणा HEX_L2_पंक्ति_आकार           128
#घोषणा HEX_L2_BLOCK_आकार          (HEX_L2_पंक्ति_आकार * 4) // flush granularity (chunks per thread)
#घोषणा HEX_L2_FLUSH_WQ_THRESHOLD  (4 * 1024)
#घोषणा HEX_L2_FLUSH_ALL_THRESHOLD (4 * 1024 * 1024)

अटल अंतरभूत व्योम hex_l2flush(व्योम * addr, माप_प्रकार आकार) {
    अगर (आकार == 0) वापस;
    स्थिर uint32_t s = ((uint32_t) addr) & ~(HEX_L2_पंक्ति_आकार - 1);
    स्थिर uint32_t e = (((uint32_t) addr) + आकार + HEX_L2_पंक्ति_आकार - 1) & ~(HEX_L2_पंक्ति_आकार - 1);
    क्रम (uint32_t i = s; i < e; i += HEX_L2_पंक्ति_आकार) {
        Q6_dccleaninva_A((व्योम *) i);
    }
}

अटल अंतरभूत व्योम hex_pause() {
    यंत्र अस्थिर(" pause(#255)\n");
}

#पूर्ण_अगर /* HEX_UTILS_H */
