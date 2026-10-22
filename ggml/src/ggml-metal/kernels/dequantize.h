#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_METAL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_METAL
#ਜੇ defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#ਨਹੀਂਤਾਂ
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ QK_NL 16 // shared by mul_mm and get_rows_q instantiations

// NOTE: this is not dequantizing - we are simply fitting the template
ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_f32(device ਅਚਲ float4x4 * src, ਛੋਟਾ il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_f32_t4(device ਅਚਲ float4 * src, ਛੋਟਾ il, thread type4 & reg) {
    reg = (type4)(*src);
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_f16(device ਅਚਲ half4x4 * src, ਛੋਟਾ il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_f16_t4(device ਅਚਲ half4 * src, ਛੋਟਾ il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#ਜੇ defined(GGML_METAL_HAS_BF16)
ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_bf16(device ਅਚਲ bfloat4x4 * src, ਛੋਟਾ il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_bf16_t4(device ਅਚਲ bfloat4 * src, ਛੋਟਾ il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q1_0(device ਅਚਲ ਖੰਡ_q1_0 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint8_t * qs = xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ neg_d = -d;

    ਅਚਲ ਪੂਰਨ byte_ਵਿੱਥ = il * 2;  // il*16 bits = il*2 bytes
    ਅਚਲ uint8_t b0 = qs[byte_ਵਿੱਥ];
    ਅਚਲ uint8_t b1 = qs[byte_ਵਿੱਥ + 1];

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

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q1_0_t4(device ਅਚਲ ਖੰਡ_q1_0 * xb, ਛੋਟਾ il, thread type4 & reg) {
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ neg_d = -d;
    ਅਚਲ ਪੂਰਨ ਅਧਾਰ = il * 4;
    ਅਚਲ uint8_t byte = xb->qs[ਅਧਾਰ / 8];
    ਅਚਲ ਪੂਰਨ s = ਅਧਾਰ % 8;

    float4 reg_f;
    reg_f[0] = select(neg_d, d, bool((byte >> (s    )) & 1));
    reg_f[1] = select(neg_d, d, bool((byte >> (s + 1)) & 1));
    reg_f[2] = select(neg_d, d, bool((byte >> (s + 2)) & 1));
    reg_f[3] = select(neg_d, d, bool((byte >> (s + 3)) & 1));

    reg = (type4) reg_f;
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q2_0(device ਅਚਲ ਖੰਡ_q2_0 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint8_t * qs = xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;

    ਅਚਲ ਪੂਰਨ byte_ਵਿੱਥ = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        ਅਚਲ uint8_t b = qs[byte_ਵਿੱਥ + i];
        reg_f[i][0] = ((ਦਸ਼ਮਲਵ)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((ਦਸ਼ਮਲਵ)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((ਦਸ਼ਮਲਵ)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((ਦਸ਼ਮਲਵ)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q2_0_t4(device ਅਚਲ ਖੰਡ_q2_0 * xb, ਛੋਟਾ il, thread type4 & reg) {
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ uint8_t b = xb->qs[il];

    float4 reg_f;
    reg_f[0] = ((ਦਸ਼ਮਲਵ)((b >> 0) & 3) - 1.0f) * d;
    reg_f[1] = ((ਦਸ਼ਮਲਵ)((b >> 2) & 3) - 1.0f) * d;
    reg_f[2] = ((ਦਸ਼ਮਲਵ)((b >> 4) & 3) - 1.0f) * d;
    reg_f[3] = ((ਦਸ਼ਮਲਵ)((b >> 6) & 3) - 1.0f) * d;

    reg = (type4) reg_f;
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q4_0(device ਅਚਲ ਖੰਡ_q4_0 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = il ? (xb->d / 16.h) : xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = d1 / 256.f;
    ਅਚਲ ਦਸ਼ਮਲਵ md = -8.h * xb->d;
    ਅਚਲ ushort mask0 = il ? 0x00F0 : 0x000F;
    ਅਚਲ ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q4_0_t4(device ਅਚਲ ਖੰਡ_q4_0 * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = d1 / 256.f;
    ਅਚਲ ਦਸ਼ਮਲਵ md = -8.h * xb->d;
    ਅਚਲ ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    ਅਚਲ ushort mask1 = mask0 << 8;

    ਲਈ (ਪੂਰਨ i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}



ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q4_1(device ਅਚਲ ਖੰਡ_q4_1 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 2);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = il ? (xb->d / 16.h) : xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = d1 / 256.f;
    ਅਚਲ ਦਸ਼ਮਲਵ  m = xb->m;
    ਅਚਲ ushort mask0 = il ? 0x00F0 : 0x000F;
    ਅਚਲ ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = ((qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qs[i] & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q4_1_t4(device ਅਚਲ ਖੰਡ_q4_1 * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 2);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = d1 / 256.f;
    ਅਚਲ ਦਸ਼ਮਲਵ  m = xb->m;
    ਅਚਲ ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    ਅਚਲ ushort mask1 = mask0 << 8;

    ਲਈ (ਪੂਰਨ i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q5_0(device ਅਚਲ ਖੰਡ_q5_0 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 3);
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ md = -16.h * xb->d;
    ਅਚਲ ushort ਨਕਾਬ = il ? 0x00F0 : 0x000F;

    ਅਚਲ uint32_t qh = *((device ਅਚਲ uint32_t *)xb->qh);

    ਅਚਲ ਪੂਰਨ x_mv = il ? 4 : 0;

    ਅਚਲ ਪੂਰਨ gh_mv = il ? 12 : 0;
    ਅਚਲ ਪੂਰਨ gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        ਅਚਲ uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ਅਚਲ uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ਅਚਲ int32_t x0 = ((((qs[i]     ) & ਨਕਾਬ) >> x_mv) | xh_0);
        ਅਚਲ int32_t x1 = ((((qs[i] >> 8) & ਨਕਾਬ) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q5_0_t4(device ਅਚਲ ਖੰਡ_q5_0 * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 3);
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ md = -16.h * xb->d;
    ਅਚਲ ushort ਨਕਾਬ = (il/4) ? 0x00F0 : 0x000F;

    ਅਚਲ uint32_t qh = *((device ਅਚਲ uint32_t *)xb->qh);

    ਅਚਲ ਪੂਰਨ x_mv = (il/4) ? 4 : 0;

    ਅਚਲ ਪੂਰਨ gh_mv = (il/4) ? 12 : 0;
    ਅਚਲ ਪੂਰਨ gh_bk = (il/4) ?  0 : 4;

    ਲਈ (ਪੂਰਨ ii = 0; ii < 2; ii++) {
        ਪੂਰਨ i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        ਅਚਲ uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ਅਚਲ uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ਅਚਲ int32_t x0 = ((((qs[i]     ) & ਨਕਾਬ) >> x_mv) | xh_0);
        ਅਚਲ int32_t x1 = ((((qs[i] >> 8) & ਨਕਾਬ) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q5_1(device ਅਚਲ ਖੰਡ_q5_1 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 4);
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ m = xb->m;
    ਅਚਲ ushort ਨਕਾਬ = il ? 0x00F0 : 0x000F;

    ਅਚਲ uint32_t qh = *((device ਅਚਲ uint32_t *)xb->qh);

    ਅਚਲ ਪੂਰਨ x_mv = il ? 4 : 0;

    ਅਚਲ ਪੂਰਨ gh_mv = il ? 12 : 0;
    ਅਚਲ ਪੂਰਨ gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        ਅਚਲ uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ਅਚਲ uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ਅਚਲ int32_t x0 = ((((qs[i]     ) & ਨਕਾਬ) >> x_mv) | xh_0);
        ਅਚਲ int32_t x1 = ((((qs[i] >> 8) & ਨਕਾਬ) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q5_1_t4(device ਅਚਲ ਖੰਡ_q5_1 * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint16_t * qs = ((device ਅਚਲ uint16_t *)xb + 4);
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ m = xb->m;
    ਅਚਲ ushort ਨਕਾਬ = (il/4) ? 0x00F0 : 0x000F;

    ਅਚਲ uint32_t qh = *((device ਅਚਲ uint32_t *)xb->qh);

    ਅਚਲ ਪੂਰਨ x_mv = (il/4) ? 4 : 0;

    ਅਚਲ ਪੂਰਨ gh_mv = (il/4) ? 12 : 0;
    ਅਚਲ ਪੂਰਨ gh_bk = (il/4) ?  0 : 4;

    ਲਈ (ਪੂਰਨ ii = 0; ii < 2; ii++) {
        ਪੂਰਨ i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        ਅਚਲ uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        ਅਚਲ uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        ਅਚਲ int32_t x0 = ((((qs[i]     ) & ਨਕਾਬ) >> x_mv) | xh_0);
        ਅਚਲ int32_t x1 = ((((qs[i] >> 8) & ਨਕਾਬ) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q8_0(device ਅਚਲ ਖੰਡ_q8_0 *xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ packed_char4 * qs = (device ਅਚਲ packed_char4 *) xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_q8_0_t4(device ਅਚਲ ਖੰਡ_q8_0 *xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ packed_char4 * qs = (device ਅਚਲ packed_char4 *) xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_mxfp4(device ਅਚਲ ਖੰਡ_mxfp4 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint8_t * q2 = (device ਅਚਲ uint8_t *)xb->qs;

    ਅਚਲ ਦਸ਼ਮਲਵ d = e8m0_ਮੰਜ਼ਿਲ_fp32(xb->e);
    ਅਚਲ uint8_t shr = il >= 1 ? 4 : 0;

    float4x4 reg_f;

    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_mxfp4_t4(device ਅਚਲ ਖੰਡ_mxfp4 * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint8_t * q2 = (device ਅਚਲ uint8_t *)xb->qs;

    ਅਚਲ ਦਸ਼ਮਲਵ d = e8m0_ਮੰਜ਼ਿਲ_fp32(xb->e);
    ਅਚਲ ਛੋਟਾ il4 = il%4;

    ਅਚਲ uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q2_K(device ਅਚਲ ਖੰਡ_q2_K *xb, ਛੋਟਾ il, thread type4x4 & reg) {
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਦਸ਼ਮਲਵ min = xb->dmin;
    device ਅਚਲ uint8_t * q = (device ਅਚਲ uint8_t *)xb->qs;
    ਦਸ਼ਮਲਵ dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar ਨਕਾਬ = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & ਨਕਾਬ) - ml;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q3_K(device ਅਚਲ ਖੰਡ_q3_K *xb, ਛੋਟਾ il, thread type4x4 & reg) {
    ਅਚਲ half d_all = xb->d;
    device ਅਚਲ uint8_t * q = (device ਅਚਲ uint8_t *)xb->qs;
    device ਅਚਲ uint8_t * h = (device ਅਚਲ uint8_t *)xb->hmask;
    device ਅਚਲ int8_t * scales = (device ਅਚਲ int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    ਦਸ਼ਮਲਵ dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    ਅਚਲ ਦਸ਼ਮਲਵ ml = 4.f * dl;

    il = (il/2) & 3;
    ਅਚਲ half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    ਅਚਲ uint8_t ਨਕਾਬ = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & ਨਕਾਬ) - (h[i] & m ? 0 : ml);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uchar2 get_scale_min_k4_just2(ਪੂਰਨ j, ਪੂਰਨ k, device ਅਚਲ uchar * q) {
    ਵਾਪਸ j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q4_K(device ਅਚਲ ਖੰਡ_q4_K * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uchar * q = xb->qs;

    ਛੋਟਾ is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    ਅਚਲ uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    ਅਚਲ ਦਸ਼ਮਲਵ d   = il < 2 ? xb->d : xb->d / 16.h;
    ਅਚਲ ਦਸ਼ਮਲਵ min = xb->dmin;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * sc[0];
    ਅਚਲ ਦਸ਼ਮਲਵ ml = min * sc[1];

    ਅਚਲ ushort ਨਕਾਬ = il < 2 ? 0x0F : 0xF0;
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & ਨਕਾਬ) - ml;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q5_K(device ਅਚਲ ਖੰਡ_q5_K *xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint8_t * q  = xb->qs;
    device ਅਚਲ uint8_t * qh = xb->qh;

    ਛੋਟਾ is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    ਅਚਲ uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    ਅਚਲ ਦਸ਼ਮਲਵ d = il < 2 ? xb->d : xb->d / 16.f;
    ਅਚਲ ਦਸ਼ਮਲਵ min = xb->dmin;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * sc[0];
    ਅਚਲ ਦਸ਼ਮਲਵ ml = min * sc[1];

    ਅਚਲ ushort ਨਕਾਬ  = il<2 ? 0x0F : 0xF0;
    ਅਚਲ ਦਸ਼ਮਲਵ qh_val = il<2 ? 16.f : 256.f;
    ਲਈ (ਪੂਰਨ i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & ਨਕਾਬ) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_q6_K(device ਅਚਲ ਖੰਡ_q6_K *xb, ਛੋਟਾ il, thread type4x4 & reg) {
    ਅਚਲ half d_all = xb->d;
    device ਅਚਲ uint16_t * ql = (device ਅਚਲ uint16_t *)xb->ql;
    device ਅਚਲ uint16_t * qh = (device ਅਚਲ uint16_t *)xb->qh;
    device ਅਚਲ int8_t * scales = (device ਅਚਲ int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    ਦਸ਼ਮਲਵ sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    ਅਚਲ uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    ਅਚਲ uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    ਅਚਲ ਦਸ਼ਮਲਵ ml = d_all * sc * 32.f;
    ਅਚਲ ਦਸ਼ਮਲਵ dl0 = d_all * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ dl1 = dl0 / 256.f;
    ਅਚਲ ਦਸ਼ਮਲਵ dl2 = dl0 / (256.f * 256.f);
    ਅਚਲ ਦਸ਼ਮਲਵ dl3 = dl0 / (256.f * 256.f * 256.f);
    ਅਚਲ uint8_t shr_h = il>2 ? 2 : 0;
    ਅਚਲ uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    ਅਚਲ uint8_t shr_l = il>1 ? 4 : 0;
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        ਅਚਲ uint32_t  ਨੀਵਾਂ = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        ਅਚਲ uint32_t ਉੱਚਾ = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        ਅਚਲ uint32_t q = ((ਉੱਚਾ << shl_h) >> shr_h) | (ਨੀਵਾਂ >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((ਦਸ਼ਮਲਵ)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((ਦਸ਼ਮਲਵ)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((ਦਸ਼ਮਲਵ)(q & 0xFF000000)) - ml;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq2_xxs(device ਅਚਲ ਖੰਡ_iq2_xxs * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device ਅਚਲ uint16_t * q2 = xb->qs + 4*ib32;
    ਅਚਲ uint32_t aux32_g = q2[0] | (q2[1] << 16);
    ਅਚਲ uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread ਅਚਲ uint8_t * aux8 = (thread ਅਚਲ uint8_t *)&aux32_g;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq2_xs(device ਅਚਲ ਖੰਡ_iq2_xs * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ਅਚਲ uint16_t * q2 = xb->qs + 4*ib32;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq3_xxs(device ਅਚਲ ਖੰਡ_iq3_xxs * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ਅਚਲ uint8_t * q3 = xb->qs + 8*ib32;
    device ਅਚਲ uint16_t * gas = (device ਅਚਲ uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    ਅਚਲ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq3_s(device ਅਚਲ ਖੰਡ_iq3_s * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ਅਚਲ uint8_t * qs = xb->qs + 8*ib32;
    device ਅਚਲ uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    ਅਚਲ uint8_t qh = xb->qh[ib32] >> 4*il;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq2_s(device ਅਚਲ ਖੰਡ_iq2_s * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ਅਚਲ uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device ਅਚਲ uint8_t * signs = qs + QK_K/8;
    ਅਚਲ uint8_t qh = xb->qh[ib32] >> 4*il;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    ਲਈ (ਪੂਰਨ i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq1_s(device ਅਚਲ ਖੰਡ_iq1_s * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    device ਅਚਲ uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device ਅਚਲ uint16_t * qh = xb->qh;
    ਅਚਲ ਦਸ਼ਮਲਵ dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    ਅਚਲ uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq1_m(device ਅਚਲ ਖੰਡ_iq1_m * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    device ਅਚਲ uint16_t * sc = (device ਅਚਲ uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ਅਚਲ ਦਸ਼ਮਲਵ d = scale.f16;

    device ਅਚਲ uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device ਅਚਲ uint8_t * qh = xb->qh + 2*ib32 + il;

    ਅਚਲ ਦਸ਼ਮਲਵ dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    ਅਚਲ ਦਸ਼ਮਲਵ ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq4_nl(device ਅਚਲ ਖੰਡ_iq4_nl * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint16_t * q4 = (device ਅਚਲ uint16_t *)xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    uint32_t aux32;
    thread ਅਚਲ uint8_t * q8 = (thread ਅਚਲ uint8_t *)&aux32;
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

ਸਾਂਚਾ <typename type4>
ਖਾਲੀ dequantize_iq4_nl_t4(device ਅਚਲ ਖੰਡ_iq4_nl * xb, ਛੋਟਾ il, thread type4 & reg) {
    device ਅਚਲ uint16_t * q4 = (device ਅਚਲ uint16_t *)xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;
    uint32_t aux32;
    thread ਅਚਲ uint8_t * q8 = (thread ਅਚਲ uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_iq4_xs(device ਅਚਲ ਖੰਡ_iq4_xs * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    ਅਚਲ ਪੂਰਨ ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device ਅਚਲ uint32_t * q4 = (device ਅਚਲ uint32_t *)xb->qs + 4*ib32;
    ਅਚਲ ਪੂਰਨ ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)xb->d * (ls - 32);
    uint32_t aux32;
    thread ਅਚਲ uint8_t * q8 = (thread ਅਚਲ uint8_t *)&aux32;
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

ਸਾਂਚਾ <typename type4x4>
ਖਾਲੀ dequantize_tq2_0(device ਅਚਲ ਖੰਡ_tq2_0 * xb, ਛੋਟਾ il, thread type4x4 & reg) {
    device ਅਚਲ uint8_t * qs = xb->qs;
    ਅਚਲ ਦਸ਼ਮਲਵ d = xb->d;

    float4x4 reg_f;

    // 2 bits per element, 4 elements per byte, 128 elements per 32-byte group
    ਅਚਲ ਛੋਟਾ ਅਧਾਰ = il * 16;
    ਲਈ (ਪੂਰਨ k = 0; k < 16; k++) {
        ਅਚਲ ਪੂਰਨ i = ਅਧਾਰ + k;
        ਅਚਲ ਪੂਰਨ byte = ((i >> 7) & 1) * 32 + (i & 31);
        ਅਚਲ ਪੂਰਨ l = (i >> 5) & 3;
        reg_f[k/4][k%4] = d * (ਦਸ਼ਮਲਵ)(((qs[byte] >> (2*l)) & 3) - 1);
    }

    reg = (type4x4) reg_f;
}
