#اگر_غیر_تعریف_شدہ HVX_COPY_H
#تعریف_کرو HVX_COPY_H

#شامل <assert.h>
#شامل <stddef.h>
#شامل <stdint.h>

#شامل "hvx-base.h"

#تعریف_کرو hvx_splat_pragma(x) _Pragma(#x)
#تعریف_کرو hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    کرو {                                                         \
        dst_type * محدود vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_سائز);                   \
        uint32_t nloe = n % (128 / elem_سائز);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        برائے (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        اگر (nloe) {                                              \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, src); \
        }                                                        \
    } جب(0)

ساکن اندرونی خالی hvx_splat_a(خالی * محدود dst, HVX_سمتیہ src, uint32_t n, uint32_t elem_سائز) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_splat_loop_body(HVX_سمتیہ, hvx_vec_store_a, 4);
}

ساکن اندرونی خالی hvx_splat_u(خالی * محدود dst, HVX_سمتیہ src, uint32_t n, uint32_t elem_سائز) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

ساکن اندرونی خالی hvx_splat_f32_a(خالی * محدود dst, اعشاری v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, حجم(اعشاری));
}

ساکن اندرونی خالی hvx_splat_f32_u(خالی * محدود dst, اعشاری v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, حجم(اعشاری));
}

ساکن اندرونی خالی hvx_splat_f16_a(خالی * محدود dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, حجم(__fp16));
}

ساکن اندرونی خالی hvx_splat_f16_u(خالی * محدود dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, حجم(__fp16));
}

ساکن اندرونی خالی hvx_splat_u16_a(خالی * محدود dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, حجم(uint16_t));
}

ساکن اندرونی خالی hvx_splat_u16_u(خالی * محدود dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, حجم(uint16_t));
}

ساکن اندرونی خالی hvx_splat_u8_a(خالی * محدود dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

ساکن اندرونی خالی hvx_splat_u8_u(خالی * محدود dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#تعریف_کرو hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    کرو {                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                 \
        src_type * محدود vsrc = (src_type *) src;                 \
                                                                     \
        ثابت uint32_t epv  = 128 / elem_سائز;                       \
        ثابت uint32_t nvec = n / epv;                               \
        ثابت uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        برائے (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        اگر (nloe) {                                                  \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, vsrc[i]); \
        }                                                            \
    } جب(0)

// Generic copy routines
ساکن اندرونی خالی hvx_copy_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_سائز) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_copy_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_سائز) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_copy_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساکن اندرونی خالی hvx_copy_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_سائز) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساکن اندرونی خالی hvx_copy_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_سائز) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
ساکن اندرونی خالی hvx_copy_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_aa(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
ساکن اندرونی خالی hvx_copy_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_au(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
ساکن اندرونی خالی hvx_copy_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_ua(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_uu(dst, src, n, حجم(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
ساکن اندرونی خالی hvx_copy_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_aa(dst, src, n, حجم(اعشاری));
}

// copy n fp32 elements : destination is aligned, source is unaligned
ساکن اندرونی خالی hvx_copy_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_au(dst, src, n, حجم(اعشاری));
}

// copy n fp32 elements : destination is unaligned, source is aligned
ساکن اندرونی خالی hvx_copy_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_ua(dst, src, n, حجم(اعشاری));
}

// copy n fp32 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_uu(dst, src, n, حجم(اعشاری));
}

//// fp32 -> fp16

#تعریف_کرو hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    کرو {                                                                            \
        dst_type * محدود vdst = (dst_type *) dst;                                \
        src_type * محدود vsrc = (src_type *) src;                                \
                                                                                    \
        ثابت uint32_t elem_سائز = حجم(__fp16);                                  \
        ثابت uint32_t epv  = 128 / elem_سائز;                                      \
        ثابت uint32_t nvec = n / epv;                                              \
        ثابت uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        برائے (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_مقصد_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        اگر (nloe) {                                                                 \
            HVX_سمتیہ v = hvx_vec_f32_مقصد_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, v);                      \
        }                                                                           \
    } جب(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
ساکن اندرونی خالی hvx_copy_f16_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
ساکن اندرونی خالی hvx_copy_f16_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
ساکن اندرونی خالی hvx_copy_f16_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_f16_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#تعریف_کرو hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    کرو {                                                                            \
        dst_type * محدود vdst = (dst_type *) dst;                                \
        src_type * محدود vsrc = (src_type *) src;                                \
                                                                                    \
        ثابت HVX_سمتیہ one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ثابت uint32_t elem_سائز = حجم(__fp16);                                  \
        ثابت uint32_t epv  = 128 / elem_سائز;                                      \
        ثابت uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        برائے (i = 0; i < nvec; ++i) {                                                \
            HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        اگر (nloe) {                                                                 \
            HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_سمتیہ vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            اگر (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            اگر (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * حجم(اعشاری), vd);                \
            }                                                                       \
        }                                                                           \
    } جب(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
ساکن اندرونی خالی hvx_copy_f32_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
ساکن اندرونی خالی hvx_copy_f32_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
ساکن اندرونی خالی hvx_copy_f32_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_f32_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#تعریف_کرو hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    کرو {                                                          \
        dst_type * محدود vdst = (dst_type *) dst;              \
        src_type * محدود vsrc = (src_type *) src;              \
                                                                  \
        ثابت uint32_t elem_سائز = حجم(int32_t);               \
        ثابت uint32_t epv  = 128 / elem_سائز;                    \
        ثابت uint32_t nvec = n / epv;                            \
        ثابت uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        برائے (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        اگر (nloe) {                                               \
            HVX_سمتیہ v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, v);    \
        }                                                         \
    } جب(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
ساکن اندرونی خالی hvx_copy_i32_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_i32_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#تعریف_کرو hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    کرو {                                                          \
        dst_type * محدود vdst = (dst_type *) dst;              \
        src_type * محدود vsrc = (src_type *) src;              \
                                                                  \
        ثابت uint32_t elem_سائز = حجم(اعشاری);                 \
        ثابت uint32_t epv  = 128 / elem_سائز;                    \
        ثابت uint32_t nvec = n / epv;                            \
        ثابت uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        برائے (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        اگر (nloe) {                                               \
            HVX_سمتیہ v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((خالی *) &vdst[i], nloe * elem_سائز, v);    \
        }                                                         \
    } جب(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
ساکن اندرونی خالی hvx_copy_f32_i32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((غیرعلامتی طویل) dst % 128 == 0);
    أكد((غیرعلامتی طویل) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
ساکن اندرونی خالی hvx_copy_f32_i32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#اگر_ختم // HVX_COPY_H
