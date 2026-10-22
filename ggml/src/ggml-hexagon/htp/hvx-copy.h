#నిర్వచితం_కాకపోతే HVX_COPY_H
#నిర్వచించు HVX_COPY_H

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hvx-base.h"

#నిర్వచించు hvx_splat_pragma(x) _Pragma(#x)
#నిర్వచించు hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    చేయి {                                                         \
        dst_type * పరిమిత vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_కొలత);                   \
        uint32_t nloe = n % (128 / elem_కొలత);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        కొరకు (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        ఒకవేళ (nloe) {                                              \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, src); \
        }                                                        \
    } అయితే(0)

స్థిర అంతఃవరుస శూన్య hvx_splat_a(శూన్య * పరిమిత dst, HVX_సదిశ src, uint32_t n, uint32_t elem_కొలత) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_splat_loop_body(HVX_సదిశ, hvx_vec_store_a, 4);
}

స్థిర అంతఃవరుస శూన్య hvx_splat_u(శూన్య * పరిమిత dst, HVX_సదిశ src, uint32_t n, uint32_t elem_కొలత) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

స్థిర అంతఃవరుస శూన్య hvx_splat_f32_a(శూన్య * పరిమిత dst, దశాంశం v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, పరిమాణం(దశాంశం));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_f32_u(శూన్య * పరిమిత dst, దశాంశం v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, పరిమాణం(దశాంశం));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_f16_a(శూన్య * పరిమిత dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, పరిమాణం(__fp16));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_f16_u(శూన్య * పరిమిత dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, పరిమాణం(__fp16));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_u16_a(శూన్య * పరిమిత dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, పరిమాణం(uint16_t));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_u16_u(శూన్య * పరిమిత dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, పరిమాణం(uint16_t));
}

స్థిర అంతఃవరుస శూన్య hvx_splat_u8_a(శూన్య * పరిమిత dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

స్థిర అంతఃవరుస శూన్య hvx_splat_u8_u(శూన్య * పరిమిత dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#నిర్వచించు hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    చేయి {                                                             \
        dst_type * పరిమిత vdst = (dst_type *) dst;                 \
        src_type * పరిమిత vsrc = (src_type *) src;                 \
                                                                     \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                       \
        స్థిరరాశి uint32_t nvec = n / epv;                               \
        స్థిరరాశి uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        కొరకు (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        ఒకవేళ (nloe) {                                                  \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, vsrc[i]); \
        }                                                            \
    } అయితే(0)

// Generic copy routines
స్థిర అంతఃవరుస శూన్య hvx_copy_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n, uint32_t elem_కొలత) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_copy_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n, uint32_t elem_కొలత) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_copy_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

స్థిర అంతఃవరుస శూన్య hvx_copy_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n, uint32_t elem_కొలత) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

స్థిర అంతఃవరుస శూన్య hvx_copy_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n, uint32_t elem_కొలత) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_aa(dst, src, n, పరిమాణం(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_au(dst, src, n, పరిమాణం(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_ua(dst, src, n, పరిమాణం(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_uu(dst, src, n, పరిమాణం(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_aa(dst, src, n, పరిమాణం(దశాంశం));
}

// copy n fp32 elements : destination is aligned, source is unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_au(dst, src, n, పరిమాణం(దశాంశం));
}

// copy n fp32 elements : destination is unaligned, source is aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_ua(dst, src, n, పరిమాణం(దశాంశం));
}

// copy n fp32 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_uu(dst, src, n, పరిమాణం(దశాంశం));
}

//// fp32 -> fp16

#నిర్వచించు hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    చేయి {                                                                            \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                \
        src_type * పరిమిత vsrc = (src_type *) src;                                \
                                                                                    \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(__fp16);                                  \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                                      \
        స్థిరరాశి uint32_t nvec = n / epv;                                              \
        స్థిరరాశి uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        కొరకు (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_గమ్యం_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        ఒకవేళ (nloe) {                                                                 \
            HVX_సదిశ v = hvx_vec_f32_గమ్యం_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);                      \
        }                                                                           \
    } అయితే(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_f32_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_f32_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f16_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#నిర్వచించు hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    చేయి {                                                                            \
        dst_type * పరిమిత vdst = (dst_type *) dst;                                \
        src_type * పరిమిత vsrc = (src_type *) src;                                \
                                                                                    \
        స్థిరరాశి HVX_సదిశ one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(__fp16);                                  \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                                      \
        స్థిరరాశి uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        కొరకు (i = 0; i < nvec; ++i) {                                                \
            HVX_సదిశ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        ఒకవేళ (nloe) {                                                                 \
            HVX_సదిశ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_సదిశ vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            ఒకవేళ (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            ఒకవేళ (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * పరిమాణం(దశాంశం), vd);                \
            }                                                                       \
        }                                                                           \
    } అయితే(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_f16_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_f16_au(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_సదిశ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_f16_ua(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_సదిశ, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_f16_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#నిర్వచించు hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    చేయి {                                                          \
        dst_type * పరిమిత vdst = (dst_type *) dst;              \
        src_type * పరిమిత vsrc = (src_type *) src;              \
                                                                  \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(int32_t);               \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                    \
        స్థిరరాశి uint32_t nvec = n / epv;                            \
        స్థిరరాశి uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        కొరకు (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        ఒకవేళ (nloe) {                                               \
            HVX_సదిశ v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);    \
        }                                                         \
    } అయితే(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_i32_f32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_i32_f32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#నిర్వచించు hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    చేయి {                                                          \
        dst_type * పరిమిత vdst = (dst_type *) dst;              \
        src_type * పరిమిత vsrc = (src_type *) src;              \
                                                                  \
        స్థిరరాశి uint32_t elem_కొలత = పరిమాణం(దశాంశం);                 \
        స్థిరరాశి uint32_t epv  = 128 / elem_కొలత;                    \
        స్థిరరాశి uint32_t nvec = n / epv;                            \
        స్థిరరాశి uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        కొరకు (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        ఒకవేళ (nloe) {                                               \
            HVX_సదిశ v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((శూన్య *) &vdst[i], nloe * elem_కొలత, v);    \
        }                                                         \
    } అయితే(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_i32_aa(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) dst % 128 == 0);
    ప్రతిజ్ఞా((గుర్తులేని దీర్ఘ) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_సదిశ, HVX_సదిశ, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
స్థిర అంతఃవరుస శూన్య hvx_copy_f32_i32_uu(uint8_t * పరిమిత dst, స్థిరరాశి uint8_t * పరిమిత src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#అయితే_ముగింపు // HVX_COPY_H
