#నిర్వచితం_కాకపోతే HVX_TYPES_H
#నిర్వచించు HVX_TYPES_H

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ <hexagon_types.h>

#నిర్వచించు SIZEOF_FP32 (4)
#నిర్వచించు SIZEOF_FP16 (2)
#నిర్వచించు VLEN        (128)
#నిర్వచించు VLEN_FP32   (VLEN / SIZEOF_FP32)
#నిర్వచించు VLEN_FP16   (VLEN / SIZEOF_FP16)

రకనిర్వచనం సంఘం {
    HVX_సదిశ v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    దశాంశం      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_సదిశ_Alias;

రకనిర్వచనం నిర్మాణం {
    HVX_సదిశ v[2];
} HVX_సదిశ_x2;

రకనిర్వచనం నిర్మాణం {
    HVX_సదిశ v[4];
} HVX_సదిశ_x4;

రకనిర్వచనం నిర్మాణం {
    HVX_సదిశ v[8];
} HVX_సదిశ_x8;

#అయితే_ముగింపు /* HVX_TYPES_H */
