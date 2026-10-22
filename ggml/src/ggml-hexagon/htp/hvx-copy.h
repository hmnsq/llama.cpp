#جے_بنا_تعریف HVX_COPY_H
#تعریف_کرو HVX_COPY_H

#ضمن <assert.h>
#ضمن <stddef.h>
#ضمن <stdint.h>

#ضمن "hvx-base.h"

#تعریف_کرو hvx_splat_pragma(x) _Pragma(#x)
#تعریف_کرو hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    افعل {                                                         \
        dst_type * محدود vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_ناپ);                   \
        uint32_t nloe = n % (128 / elem_ناپ);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        لكل (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        إذا (nloe) {                                              \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, src); \
        }                                                        \
    } طالما(0)

ساكن مضمن فراغ hvx_splat_a(فراغ * محدود dst, HVX_سمتیہ src, uint32_t n, uint32_t elem_ناپ) {
    أكد((بنانشانی لما) dst % 128 == 0);
    hvx_splat_loop_body(HVX_سمتیہ, hvx_vec_store_a, 4);
}

ساكن مضمن فراغ hvx_splat_u(فراغ * محدود dst, HVX_سمتیہ src, uint32_t n, uint32_t elem_ناپ) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

ساكن مضمن فراغ hvx_splat_f32_a(فراغ * محدود dst, عائم v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, حجم(عائم));
}

ساكن مضمن فراغ hvx_splat_f32_u(فراغ * محدود dst, عائم v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, حجم(عائم));
}

ساكن مضمن فراغ hvx_splat_f16_a(فراغ * محدود dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, حجم(__fp16));
}

ساكن مضمن فراغ hvx_splat_f16_u(فراغ * محدود dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, حجم(__fp16));
}

ساكن مضمن فراغ hvx_splat_u16_a(فراغ * محدود dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, حجم(uint16_t));
}

ساكن مضمن فراغ hvx_splat_u16_u(فراغ * محدود dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, حجم(uint16_t));
}

ساكن مضمن فراغ hvx_splat_u8_a(فراغ * محدود dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

ساكن مضمن فراغ hvx_splat_u8_u(فراغ * محدود dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#تعریف_کرو hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    افعل {                                                             \
        dst_type * محدود vdst = (dst_type *) dst;                 \
        src_type * محدود vsrc = (src_type *) src;                 \
                                                                     \
        ثابت uint32_t epv  = 128 / elem_ناپ;                       \
        ثابت uint32_t nvec = n / epv;                               \
        ثابت uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        لكل (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        إذا (nloe) {                                                  \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, vsrc[i]); \
        }                                                            \
    } طالما(0)

// Generic copy routines
ساكن مضمن فراغ hvx_copy_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_ناپ) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_copy_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_ناپ) {
    أكد((بنانشانی لما) dst % 128 == 0);
    hvx_copy_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

ساكن مضمن فراغ hvx_copy_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_ناپ) {
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

ساكن مضمن فراغ hvx_copy_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n, uint32_t elem_ناپ) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
ساكن مضمن فراغ hvx_copy_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_aa(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
ساكن مضمن فراغ hvx_copy_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_au(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
ساكن مضمن فراغ hvx_copy_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_ua(dst, src, n, حجم(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_uu(dst, src, n, حجم(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
ساكن مضمن فراغ hvx_copy_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_aa(dst, src, n, حجم(عائم));
}

// copy n fp32 elements : destination is aligned, source is unaligned
ساكن مضمن فراغ hvx_copy_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_au(dst, src, n, حجم(عائم));
}

// copy n fp32 elements : destination is unaligned, source is aligned
ساكن مضمن فراغ hvx_copy_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_ua(dst, src, n, حجم(عائم));
}

// copy n fp32 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_uu(dst, src, n, حجم(عائم));
}

//// fp32 -> fp16

#تعریف_کرو hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    افعل {                                                                            \
        dst_type * محدود vdst = (dst_type *) dst;                                \
        src_type * محدود vsrc = (src_type *) src;                                \
                                                                                    \
        ثابت uint32_t elem_ناپ = حجم(__fp16);                                  \
        ثابت uint32_t epv  = 128 / elem_ناپ;                                      \
        ثابت uint32_t nvec = n / epv;                                              \
        ثابت uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        لكل (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_منزل_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        إذا (nloe) {                                                                 \
            HVX_سمتیہ v = hvx_vec_f32_منزل_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, v);                      \
        }                                                                           \
    } طالما(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
ساكن مضمن فراغ hvx_copy_f16_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
ساكن مضمن فراغ hvx_copy_f16_f32_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
ساكن مضمن فراغ hvx_copy_f16_f32_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_f16_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#تعریف_کرو hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    افعل {                                                                            \
        dst_type * محدود vdst = (dst_type *) dst;                                \
        src_type * محدود vsrc = (src_type *) src;                                \
                                                                                    \
        ثابت HVX_سمتیہ one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ثابت uint32_t elem_ناپ = حجم(__fp16);                                  \
        ثابت uint32_t epv  = 128 / elem_ناپ;                                      \
        ثابت uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        لكل (i = 0; i < nvec; ++i) {                                                \
            HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        إذا (nloe) {                                                                 \
            HVX_سمتیہ_Pair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_سمتیہ vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            إذا (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            إذا (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * حجم(عائم), vd);                \
            }                                                                       \
        }                                                                           \
    } طالما(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
ساكن مضمن فراغ hvx_copy_f32_f16_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
ساكن مضمن فراغ hvx_copy_f32_f16_au(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_سمتیہ, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
ساكن مضمن فراغ hvx_copy_f32_f16_ua(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_سمتیہ, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_f32_f16_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#تعریف_کرو hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    افعل {                                                          \
        dst_type * محدود vdst = (dst_type *) dst;              \
        src_type * محدود vsrc = (src_type *) src;              \
                                                                  \
        ثابت uint32_t elem_ناپ = حجم(int32_t);               \
        ثابت uint32_t epv  = 128 / elem_ناپ;                    \
        ثابت uint32_t nvec = n / epv;                            \
        ثابت uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        لكل (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        إذا (nloe) {                                               \
            HVX_سمتیہ v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, v);    \
        }                                                         \
    } طالما(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
ساكن مضمن فراغ hvx_copy_i32_f32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_i32_f32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#تعریف_کرو hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    افعل {                                                          \
        dst_type * محدود vdst = (dst_type *) dst;              \
        src_type * محدود vsrc = (src_type *) src;              \
                                                                  \
        ثابت uint32_t elem_ناپ = حجم(عائم);                 \
        ثابت uint32_t epv  = 128 / elem_ناپ;                    \
        ثابت uint32_t nvec = n / epv;                            \
        ثابت uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        لكل (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        إذا (nloe) {                                               \
            HVX_سمتیہ v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((فراغ *) &vdst[i], nloe * elem_ناپ, v);    \
        }                                                         \
    } طالما(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
ساكن مضمن فراغ hvx_copy_f32_i32_aa(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    أكد((بنانشانی لما) dst % 128 == 0);
    أكد((بنانشانی لما) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_سمتیہ, HVX_سمتیہ, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
ساكن مضمن فراغ hvx_copy_f32_i32_uu(uint8_t * محدود dst, ثابت uint8_t * محدود src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#جے_انت // HVX_COPY_H
