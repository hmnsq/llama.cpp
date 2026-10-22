#ifndef HVX_COPY_H
#define HVX_COPY_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hvx-base.h"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    कुरु {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        प्रति (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        यदि (nloe) {                                              \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } यावत्(0)

स्थिर अन्तःस्थ शून्य hvx_splat_a(शून्य * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

स्थिर अन्तःस्थ शून्य hvx_splat_u(शून्य * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

स्थिर अन्तःस्थ शून्य hvx_splat_f32_a(शून्य * restrict dst, भिन्न v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, परिमाण(भिन्न));
}

स्थिर अन्तःस्थ शून्य hvx_splat_f32_u(शून्य * restrict dst, भिन्न v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, परिमाण(भिन्न));
}

स्थिर अन्तःस्थ शून्य hvx_splat_f16_a(शून्य * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, परिमाण(__fp16));
}

स्थिर अन्तःस्थ शून्य hvx_splat_f16_u(शून्य * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, परिमाण(__fp16));
}

स्थिर अन्तःस्थ शून्य hvx_splat_u16_a(शून्य * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, परिमाण(uint16_t));
}

स्थिर अन्तःस्थ शून्य hvx_splat_u16_u(शून्य * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, परिमाण(uint16_t));
}

स्थिर अन्तःस्थ शून्य hvx_splat_u8_a(शून्य * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

स्थिर अन्तःस्थ शून्य hvx_splat_u8_u(शून्य * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    कुरु {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        ध्रुव uint32_t epv  = 128 / elem_size;                       \
        ध्रुव uint32_t nvec = n / epv;                               \
        ध्रुव uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        प्रति (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        यदि (nloe) {                                                  \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } यावत्(0)

// Generic copy routines
स्थिर अन्तःस्थ शून्य hvx_copy_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_copy_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

स्थिर अन्तःस्थ शून्य hvx_copy_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

स्थिर अन्तःस्थ शून्य hvx_copy_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
स्थिर अन्तःस्थ शून्य hvx_copy_f16_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, परिमाण(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, परिमाण(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, परिमाण(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, परिमाण(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
स्थिर अन्तःस्थ शून्य hvx_copy_f32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, परिमाण(भिन्न));
}

// copy n fp32 elements : destination is aligned, source is unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, परिमाण(भिन्न));
}

// copy n fp32 elements : destination is unaligned, source is aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, परिमाण(भिन्न));
}

// copy n fp32 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, परिमाण(भिन्न));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    कुरु {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        ध्रुव uint32_t elem_size = परिमाण(__fp16);                                  \
        ध्रुव uint32_t epv  = 128 / elem_size;                                      \
        ध्रुव uint32_t nvec = n / epv;                                              \
        ध्रुव uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        प्रति (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        यदि (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } यावत्(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_f32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_f32_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_f32_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f16_f32_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    कुरु {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        ध्रुव HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ध्रुव uint32_t elem_size = परिमाण(__fp16);                                  \
        ध्रुव uint32_t epv  = 128 / elem_size;                                      \
        ध्रुव uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        प्रति (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        यदि (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            यदि (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            यदि (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * परिमाण(भिन्न), vd);                \
            }                                                                       \
        }                                                                           \
    } यावत्(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_f16_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_f16_au(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_f16_ua(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_f16_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    कुरु {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        ध्रुव uint32_t elem_size = परिमाण(int32_t);               \
        ध्रुव uint32_t epv  = 128 / elem_size;                    \
        ध्रुव uint32_t nvec = n / epv;                            \
        ध्रुव uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        प्रति (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        यदि (nloe) {                                               \
            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } यावत्(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
स्थिर अन्तःस्थ शून्य hvx_copy_i32_f32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_i32_f32_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    कुरु {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        ध्रुव uint32_t elem_size = परिमाण(भिन्न);                 \
        ध्रुव uint32_t epv  = 128 / elem_size;                    \
        ध्रुव uint32_t nvec = n / epv;                            \
        ध्रुव uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        प्रति (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        यदि (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((शून्य *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } यावत्(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_i32_aa(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    प्रतिज्ञा((unsigned long) dst % 128 == 0);
    प्रतिज्ञा((unsigned long) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
स्थिर अन्तःस्थ शून्य hvx_copy_f32_i32_uu(uint8_t * restrict dst, ध्रुव uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#endif // HVX_COPY_H
