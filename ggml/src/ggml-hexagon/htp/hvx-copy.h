#ifndef HVX_COPY_H
#define HVX_COPY_H

#অন্তর্ভাৱ <assert.h>
#অন্তর্ভাৱ <stddef.h>
#অন্তর্ভাৱ <stdint.h>

#অন্তর্ভাৱ "hvx-base.h"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    কৰা {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        বাবে (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        যদি (nloe) {                                              \
            vec_store((শূন্য *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } যেতিয়ালৈ(0)

স্থির অন্তঃস্থ শূন্য hvx_splat_a(শূন্য * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

স্থির অন্তঃস্থ শূন্য hvx_splat_u(শূন্য * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

স্থির অন্তঃস্থ শূন্য hvx_splat_f32_a(শূন্য * restrict dst, ভিন্ন v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, পরিমাণ(ভিন্ন));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_f32_u(শূন্য * restrict dst, ভিন্ন v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, পরিমাণ(ভিন্ন));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_f16_a(শূন্য * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, পরিমাণ(__fp16));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_f16_u(শূন্য * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, পরিমাণ(__fp16));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_u16_a(শূন্য * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, পরিমাণ(uint16_t));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_u16_u(শূন্য * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, পরিমাণ(uint16_t));
}

স্থির অন্তঃস্থ শূন্য hvx_splat_u8_a(শূন্য * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

স্থির অন্তঃস্থ শূন্য hvx_splat_u8_u(শূন্য * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    কৰা {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        ধ্রুৱ uint32_t epv  = 128 / elem_size;                       \
        ধ্রুৱ uint32_t nvec = n / epv;                               \
        ধ্রুৱ uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        বাবে (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        যদি (nloe) {                                                  \
            vec_store((শূন্য *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } যেতিয়ালৈ(0)

// Generic copy routines
স্থির অন্তঃস্থ শূন্য hvx_copy_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

স্থির অন্তঃস্থ শূন্য hvx_copy_au(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

স্থির অন্তঃস্থ শূন্য hvx_copy_ua(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

স্থির অন্তঃস্থ শূন্য hvx_copy_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, পরিমাণ(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_au(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, পরিমাণ(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_ua(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, পরিমাণ(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, পরিমাণ(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, পরিমাণ(ভিন্ন));
}

// copy n fp32 elements : destination is aligned, source is unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_au(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, পরিমাণ(ভিন্ন));
}

// copy n fp32 elements : destination is unaligned, source is aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_ua(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, পরিমাণ(ভিন্ন));
}

// copy n fp32 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, পরিমাণ(ভিন্ন));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    কৰা {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        ধ্রুৱ uint32_t elem_size = পরিমাণ(__fp16);                                  \
        ধ্রুৱ uint32_t epv  = 128 / elem_size;                                      \
        ধ্রুৱ uint32_t nvec = n / epv;                                              \
        ধ্রুৱ uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        বাবে (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        যদি (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((শূন্য *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } যেতিয়ালৈ(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_f32_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_f32_au(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_f32_ua(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f16_f32_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    কৰা {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        ধ্রুৱ HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        ধ্রুৱ uint32_t elem_size = পরিমাণ(__fp16);                                  \
        ধ্রুৱ uint32_t epv  = 128 / elem_size;                                      \
        ধ্রুৱ uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        বাবে (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        যদি (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            যদি (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            যদি (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * পরিমাণ(ভিন্ন), vd);                \
            }                                                                       \
        }                                                                           \
    } যেতিয়ালৈ(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_f16_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_f16_au(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_f16_ua(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_f16_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    কৰা {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        ধ্রুৱ uint32_t elem_size = পরিমাণ(int32_t);               \
        ধ্রুৱ uint32_t epv  = 128 / elem_size;                    \
        ধ্রুৱ uint32_t nvec = n / epv;                            \
        ধ্রুৱ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        বাবে (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        যদি (nloe) {                                               \
            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((শূন্য *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } যেতিয়ালৈ(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_i32_f32_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_i32_f32_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    কৰা {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        ধ্রুৱ uint32_t elem_size = পরিমাণ(ভিন্ন);                 \
        ধ্রুৱ uint32_t epv  = 128 / elem_size;                    \
        ধ্রুৱ uint32_t nvec = n / epv;                            \
        ধ্রুৱ uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        বাবে (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        যদি (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((শূন্য *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } যেতিয়ালৈ(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_i32_aa(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    প্রতিজ্ঞা((unsigned long) dst % 128 == 0);
    প্রতিজ্ঞা((unsigned long) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
স্থির অন্তঃস্থ শূন্য hvx_copy_f32_i32_uu(uint8_t * restrict dst, ধ্রুৱ uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#endif // HVX_COPY_H
