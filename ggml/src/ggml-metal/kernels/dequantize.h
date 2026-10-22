#pragma once

#अन्तर्भाव "common.h"

#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#यदि defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#अन्यथा
#अन्तर्भाव "ggml-common.h"
#endif

#define QK_NL 16 // shared by mul_mm and get_rows_q instantiations

// NOTE: this is not dequantizing - we are simply fitting the template
ढाँचा <typename type4x4>
शून्य dequantize_f32(device ध्रुव float4x4 * src, लघु il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ढाँचा <typename type4>
शून्य dequantize_f32_t4(device ध्रुव float4 * src, लघु il, thread type4 & reg) {
    reg = (type4)(*src);
}

ढाँचा <typename type4x4>
शून्य dequantize_f16(device ध्रुव half4x4 * src, लघु il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ढाँचा <typename type4>
शून्य dequantize_f16_t4(device ध्रुव half4 * src, लघु il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#यदि defined(GGML_METAL_HAS_BF16)
ढाँचा <typename type4x4>
शून्य dequantize_bf16(device ध्रुव bfloat4x4 * src, लघु il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ढाँचा <typename type4>
शून्य dequantize_bf16_t4(device ध्रुव bfloat4 * src, लघु il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#endif

ढाँचा <typename type4x4>
शून्य dequantize_q1_0(device ध्रुव block_q1_0 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint8_t * qs = xb->qs;
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न neg_d = -d;

    ध्रुव पूर्णाङ्क byte_offset = il * 2;  // il*16 bits = il*2 bytes
    ध्रुव uint8_t b0 = qs[byte_offset];
    ध्रुव uint8_t b1 = qs[byte_offset + 1];

    float4x4 reg_f;

    reg_f[0][0] = select(neg_d, d, bool(b0 & 0x01));
    reg_f[0][1] = select(neg_d, d, bool(b0 & 0x02));
    reg_f[0][2] = select(neg_d, d, bool(b0 & 0x04));
    reg_f[0][3] = select(neg_d, d, bool(b0 & 0x08));
    reg_f[1][0] = select(neg_d, d, bool(b0 & 0x10));
    reg_f[1][1] = select(neg_d, d, bool(b0 & 0x20));
    reg_f[1][2] = select(neg_d, d, bool(b0 & 0x40));
    reg_f[1][3] = select(neg_d, d, bool(b0 & 0x80));

    reg_f[2][0] = select(neg_d, d, bool(b1 & 0x01));
    reg_f[2][1] = select(neg_d, d, bool(b1 & 0x02));
    reg_f[2][2] = select(neg_d, d, bool(b1 & 0x04));
    reg_f[2][3] = select(neg_d, d, bool(b1 & 0x08));
    reg_f[3][0] = select(neg_d, d, bool(b1 & 0x10));
    reg_f[3][1] = select(neg_d, d, bool(b1 & 0x20));
    reg_f[3][2] = select(neg_d, d, bool(b1 & 0x40));
    reg_f[3][3] = select(neg_d, d, bool(b1 & 0x80));

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q1_0_t4(device ध्रुव block_q1_0 * xb, लघु il, thread type4 & reg) {
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न neg_d = -d;
    ध्रुव पूर्णाङ्क base = il * 4;
    ध्रुव uint8_t byte = xb->qs[base / 8];
    ध्रुव पूर्णाङ्क s = base % 8;

    float4 reg_f;
    reg_f[0] = select(neg_d, d, bool((byte >> (s    )) & 1));
    reg_f[1] = select(neg_d, d, bool((byte >> (s + 1)) & 1));
    reg_f[2] = select(neg_d, d, bool((byte >> (s + 2)) & 1));
    reg_f[3] = select(neg_d, d, bool((byte >> (s + 3)) & 1));

    reg = (type4) reg_f;
}

ढाँचा <typename type4x4>
शून्य dequantize_q2_0(device ध्रुव block_q2_0 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint8_t * qs = xb->qs;
    ध्रुव भिन्न d = xb->d;

    ध्रुव पूर्णाङ्क byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        ध्रुव uint8_t b = qs[byte_offset + i];
        reg_f[i][0] = ((भिन्न)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((भिन्न)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((भिन्न)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((भिन्न)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q2_0_t4(device ध्रुव block_q2_0 * xb, लघु il, thread type4 & reg) {
    ध्रुव भिन्न d = xb->d;
    ध्रुव uint8_t b = xb->qs[il];

    float4 reg_f;
    reg_f[0] = ((भिन्न)((b >> 0) & 3) - 1.0f) * d;
    reg_f[1] = ((भिन्न)((b >> 2) & 3) - 1.0f) * d;
    reg_f[2] = ((भिन्न)((b >> 4) & 3) - 1.0f) * d;
    reg_f[3] = ((भिन्न)((b >> 6) & 3) - 1.0f) * d;

    reg = (type4) reg_f;
}

ढाँचा <typename type4x4>
शून्य dequantize_q4_0(device ध्रुव block_q4_0 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 1);
    ध्रुव भिन्न d1 = il ? (xb->d / 16.h) : xb->d;
    ध्रुव भिन्न d2 = d1 / 256.f;
    ध्रुव भिन्न md = -8.h * xb->d;
    ध्रुव ushort mask0 = il ? 0x00F0 : 0x000F;
    ध्रुव ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q4_0_t4(device ध्रुव block_q4_0 * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 1);
    ध्रुव भिन्न d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    ध्रुव भिन्न d2 = d1 / 256.f;
    ध्रुव भिन्न md = -8.h * xb->d;
    ध्रुव ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    ध्रुव ushort mask1 = mask0 << 8;

    प्रति (पूर्णाङ्क i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}



ढाँचा <typename type4x4>
शून्य dequantize_q4_1(device ध्रुव block_q4_1 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 2);
    ध्रुव भिन्न d1 = il ? (xb->d / 16.h) : xb->d;
    ध्रुव भिन्न d2 = d1 / 256.f;
    ध्रुव भिन्न  m = xb->m;
    ध्रुव ushort mask0 = il ? 0x00F0 : 0x000F;
    ध्रुव ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q4_1_t4(device ध्रुव block_q4_1 * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 2);
    ध्रुव भिन्न d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    ध्रुव भिन्न d2 = d1 / 256.f;
    ध्रुव भिन्न  m = xb->m;
    ध्रुव ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    ध्रुव ushort mask1 = mask0 << 8;

    प्रति (पूर्णाङ्क i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q5_0(device ध्रुव block_q5_0 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 3);
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न md = -16.h * xb->d;
    ध्रुव ushort mask = il ? 0x00F0 : 0x000F;

    ध्रुव uint32_t qh = *((device ध्रुव uint32_t *)xb->qh);

    ध्रुव पूर्णाङ्क x_mv = il ? 4 : 0;

    ध्रुव पूर्णाङ्क gh_mv = il ? 12 : 0;
    ध्रुव पूर्णाङ्क gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        ध्रुव uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ध्रुव uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ध्रुव int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        ध्रुव int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q5_0_t4(device ध्रुव block_q5_0 * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 3);
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न md = -16.h * xb->d;
    ध्रुव ushort mask = (il/4) ? 0x00F0 : 0x000F;

    ध्रुव uint32_t qh = *((device ध्रुव uint32_t *)xb->qh);

    ध्रुव पूर्णाङ्क x_mv = (il/4) ? 4 : 0;

    ध्रुव पूर्णाङ्क gh_mv = (il/4) ? 12 : 0;
    ध्रुव पूर्णाङ्क gh_bk = (il/4) ?  0 : 4;

    प्रति (पूर्णाङ्क ii = 0; ii < 2; ii++) {
        पूर्णाङ्क i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        ध्रुव uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ध्रुव uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ध्रुव int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        ध्रुव int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q5_1(device ध्रुव block_q5_1 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 4);
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न m = xb->m;
    ध्रुव ushort mask = il ? 0x00F0 : 0x000F;

    ध्रुव uint32_t qh = *((device ध्रुव uint32_t *)xb->qh);

    ध्रुव पूर्णाङ्क x_mv = il ? 4 : 0;

    ध्रुव पूर्णाङ्क gh_mv = il ? 12 : 0;
    ध्रुव पूर्णाङ्क gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        ध्रुव uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ध्रुव uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ध्रुव int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        ध्रुव int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q5_1_t4(device ध्रुव block_q5_1 * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint16_t * qs = ((device ध्रुव uint16_t *)xb + 4);
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न m = xb->m;
    ध्रुव ushort mask = (il/4) ? 0x00F0 : 0x000F;

    ध्रुव uint32_t qh = *((device ध्रुव uint32_t *)xb->qh);

    ध्रुव पूर्णाङ्क x_mv = (il/4) ? 4 : 0;

    ध्रुव पूर्णाङ्क gh_mv = (il/4) ? 12 : 0;
    ध्रुव पूर्णाङ्क gh_bk = (il/4) ?  0 : 4;

    प्रति (पूर्णाङ्क ii = 0; ii < 2; ii++) {
        पूर्णाङ्क i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        ध्रुव uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ध्रुव uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ध्रुव int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        ध्रुव int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q8_0(device ध्रुव block_q8_0 *xb, लघु il, thread type4x4 & reg) {
    device ध्रुव packed_char4 * qs = (device ध्रुव packed_char4 *) xb->qs;
    ध्रुव भिन्न d = xb->d;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_q8_0_t4(device ध्रुव block_q8_0 *xb, लघु il, thread type4 & reg) {
    device ध्रुव packed_char4 * qs = (device ध्रुव packed_char4 *) xb->qs;
    ध्रुव भिन्न d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

ढाँचा <typename type4x4>
शून्य dequantize_mxfp4(device ध्रुव block_mxfp4 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint8_t * q2 = (device ध्रुव uint8_t *)xb->qs;

    ध्रुव भिन्न d = e8m0_to_fp32(xb->e);
    ध्रुव uint8_t shr = il >= 1 ? 4 : 0;

    float4x4 reg_f;

    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

ढाँचा <typename type4>
शून्य dequantize_mxfp4_t4(device ध्रुव block_mxfp4 * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint8_t * q2 = (device ध्रुव uint8_t *)xb->qs;

    ध्रुव भिन्न d = e8m0_to_fp32(xb->e);
    ध्रुव लघु il4 = il%4;

    ध्रुव uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

ढाँचा <typename type4x4>
शून्य dequantize_q2_K(device ध्रुव block_q2_K *xb, लघु il, thread type4x4 & reg) {
    ध्रुव भिन्न d = xb->d;
    ध्रुव भिन्न min = xb->dmin;
    device ध्रुव uint8_t * q = (device ध्रुव uint8_t *)xb->qs;
    भिन्न dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q3_K(device ध्रुव block_q3_K *xb, लघु il, thread type4x4 & reg) {
    ध्रुव half d_all = xb->d;
    device ध्रुव uint8_t * q = (device ध्रुव uint8_t *)xb->qs;
    device ध्रुव uint8_t * h = (device ध्रुव uint8_t *)xb->hmask;
    device ध्रुव int8_t * scales = (device ध्रुव int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    भिन्न dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    ध्रुव भिन्न ml = 4.f * dl;

    il = (il/2) & 3;
    ध्रुव half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    ध्रुव uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

स्थिर अन्तःस्थ uchar2 get_scale_min_k4_just2(पूर्णाङ्क j, पूर्णाङ्क k, device ध्रुव uchar * q) {
    प्रत्यागम j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

ढाँचा <typename type4x4>
शून्य dequantize_q4_K(device ध्रुव block_q4_K * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uchar * q = xb->qs;

    लघु is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    ध्रुव uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    ध्रुव भिन्न d   = il < 2 ? xb->d : xb->d / 16.h;
    ध्रुव भिन्न min = xb->dmin;
    ध्रुव भिन्न dl = d * sc[0];
    ध्रुव भिन्न ml = min * sc[1];

    ध्रुव ushort mask = il < 2 ? 0x0F : 0xF0;
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q5_K(device ध्रुव block_q5_K *xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint8_t * q  = xb->qs;
    device ध्रुव uint8_t * qh = xb->qh;

    लघु is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    ध्रुव uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    ध्रुव भिन्न d = il < 2 ? xb->d : xb->d / 16.f;
    ध्रुव भिन्न min = xb->dmin;
    ध्रुव भिन्न dl = d * sc[0];
    ध्रुव भिन्न ml = min * sc[1];

    ध्रुव ushort mask  = il<2 ? 0x0F : 0xF0;
    ध्रुव भिन्न qh_val = il<2 ? 16.f : 256.f;
    प्रति (पूर्णाङ्क i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_q6_K(device ध्रुव block_q6_K *xb, लघु il, thread type4x4 & reg) {
    ध्रुव half d_all = xb->d;
    device ध्रुव uint16_t * ql = (device ध्रुव uint16_t *)xb->ql;
    device ध्रुव uint16_t * qh = (device ध्रुव uint16_t *)xb->qh;
    device ध्रुव int8_t * scales = (device ध्रुव int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    भिन्न sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    ध्रुव uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    ध्रुव uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    ध्रुव भिन्न ml = d_all * sc * 32.f;
    ध्रुव भिन्न dl0 = d_all * sc;
    ध्रुव भिन्न dl1 = dl0 / 256.f;
    ध्रुव भिन्न dl2 = dl0 / (256.f * 256.f);
    ध्रुव भिन्न dl3 = dl0 / (256.f * 256.f * 256.f);
    ध्रुव uint8_t shr_h = il>2 ? 2 : 0;
    ध्रुव uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    ध्रुव uint8_t shr_l = il>1 ? 4 : 0;
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        ध्रुव uint32_t  निचला = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        ध्रुव uint32_t ऊपरी = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        ध्रुव uint32_t q = ((ऊपरी << shl_h) >> shr_h) | (निचला >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((भिन्न)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((भिन्न)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((भिन्न)(q & 0xFF000000)) - ml;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq2_xxs(device ध्रुव block_iq2_xxs * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव भिन्न d = xb->d;
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device ध्रुव uint16_t * q2 = xb->qs + 4*ib32;
    ध्रुव uint32_t aux32_g = q2[0] | (q2[1] << 16);
    ध्रुव uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread ध्रुव uint8_t * aux8 = (thread ध्रुव uint8_t *)&aux32_g;
    ध्रुव भिन्न dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq2_xs(device ध्रुव block_iq2_xs * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव भिन्न d = xb->d;
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ध्रुव uint16_t * q2 = xb->qs + 4*ib32;
    ध्रुव भिन्न dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq3_xxs(device ध्रुव block_iq3_xxs * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव भिन्न d = xb->d;
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ध्रुव uint8_t * q3 = xb->qs + 8*ib32;
    device ध्रुव uint16_t * gas = (device ध्रुव uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    ध्रुव uint32_t aux32 = gas[0] | (gas[1] << 16);
    ध्रुव भिन्न dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq3_s(device ध्रुव block_iq3_s * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव भिन्न d = xb->d;
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ध्रुव uint8_t * qs = xb->qs + 8*ib32;
    device ध्रुव uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    ध्रुव uint8_t qh = xb->qh[ib32] >> 4*il;
    ध्रुव भिन्न dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq2_s(device ध्रुव block_iq2_s * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव भिन्न d = xb->d;
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ध्रुव uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device ध्रुव uint8_t * signs = qs + QK_K/8;
    ध्रुव uint8_t qh = xb->qh[ib32] >> 4*il;
    ध्रुव भिन्न dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    प्रति (पूर्णाङ्क i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq1_s(device ध्रुव block_iq1_s * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    ध्रुव भिन्न d = xb->d;
    device ध्रुव uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device ध्रुव uint16_t * qh = xb->qh;
    ध्रुव भिन्न dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    ध्रुव भिन्न ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    ध्रुव uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq1_m(device ध्रुव block_iq1_m * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    device ध्रुव uint16_t * sc = (device ध्रुव uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ध्रुव भिन्न d = scale.f16;

    device ध्रुव uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device ध्रुव uint8_t * qh = xb->qh + 2*ib32 + il;

    ध्रुव भिन्न dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    ध्रुव भिन्न ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    ध्रुव भिन्न ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_iq4_nl(device ध्रुव block_iq4_nl * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint16_t * q4 = (device ध्रुव uint16_t *)xb->qs;
    ध्रुव भिन्न d = xb->d;
    uint32_t aux32;
    thread ध्रुव uint8_t * q8 = (thread ध्रुव uint8_t *)&aux32;
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

ढाँचा <typename type4>
शून्य dequantize_iq4_nl_t4(device ध्रुव block_iq4_nl * xb, लघु il, thread type4 & reg) {
    device ध्रुव uint16_t * q4 = (device ध्रुव uint16_t *)xb->qs;
    ध्रुव भिन्न d = xb->d;
    uint32_t aux32;
    thread ध्रुव uint8_t * q8 = (thread ध्रुव uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

ढाँचा <typename type4x4>
शून्य dequantize_iq4_xs(device ध्रुव block_iq4_xs * xb, लघु il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ध्रुव पूर्णाङ्क ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ध्रुव uint32_t * q4 = (device ध्रुव uint32_t *)xb->qs + 4*ib32;
    ध्रुव पूर्णाङ्क ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    ध्रुव भिन्न d = (भिन्न)xb->d * (ls - 32);
    uint32_t aux32;
    thread ध्रुव uint8_t * q8 = (thread ध्रुव uint8_t *)&aux32;
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

ढाँचा <typename type4x4>
शून्य dequantize_tq2_0(device ध्रुव block_tq2_0 * xb, लघु il, thread type4x4 & reg) {
    device ध्रुव uint8_t * qs = xb->qs;
    ध्रुव भिन्न d = xb->d;

    float4x4 reg_f;

    // 2 bits per element, 4 elements per byte, 128 elements per 32-byte group
    ध्रुव लघु base = il * 16;
    प्रति (पूर्णाङ्क k = 0; k < 16; k++) {
        ध्रुव पूर्णाङ्क i = base + k;
        ध्रुव पूर्णाङ्क byte = ((i >> 7) & 1) * 32 + (i & 31);
        ध्रुव पूर्णाङ्क l = (i >> 5) & 3;
        reg_f[k/4][k%4] = d * (भिन्न)(((qs[byte] >> (2*l)) & 3) - 1);
    }

    reg = (type4x4) reg_f;
}
