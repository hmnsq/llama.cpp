#अगर_अघोषित HVX_COPY_H
#घोषणा HVX_COPY_H

#समावेश <assert.h>
#समावेश <stddef.h>
#समावेश <stdint.h>

#समावेश "hvx-base.h"

#घोषणा hvx_splat_pragma(x) _Pragma(#x)
#घोषणा hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    करो {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_आकार);                   \
        uint32_t nloe = n % (128 / elem_आकार);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        क्रम (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        अगर (nloe) {                                              \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, src); \
        }                                                        \
    } जबतक(0)

अटल अंतरभूत व्योम hvx_splat_a(व्योम * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_आकार) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

अटल अंतरभूत व्योम hvx_splat_u(व्योम * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_आकार) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

अटल अंतरभूत व्योम hvx_splat_f32_a(व्योम * restrict dst, भग्न v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, माप(भग्न));
}

अटल अंतरभूत व्योम hvx_splat_f32_u(व्योम * restrict dst, भग्न v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, माप(भग्न));
}

अटल अंतरभूत व्योम hvx_splat_f16_a(व्योम * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, माप(__fp16));
}

अटल अंतरभूत व्योम hvx_splat_f16_u(व्योम * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, माप(__fp16));
}

अटल अंतरभूत व्योम hvx_splat_u16_a(व्योम * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, माप(uint16_t));
}

अटल अंतरभूत व्योम hvx_splat_u16_u(व्योम * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, माप(uint16_t));
}

अटल अंतरभूत व्योम hvx_splat_u8_a(व्योम * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

अटल अंतरभूत व्योम hvx_splat_u8_u(व्योम * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#घोषणा hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    करो {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        स्थिर uint32_t epv  = 128 / elem_आकार;                       \
        स्थिर uint32_t nvec = n / epv;                               \
        स्थिर uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        क्रम (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        अगर (nloe) {                                                  \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, vsrc[i]); \
        }                                                            \
    } जबतक(0)

// Generic copy routines
अटल अंतरभूत व्योम hvx_copy_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n, uint32_t elem_आकार) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_copy_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n, uint32_t elem_आकार) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

अटल अंतरभूत व्योम hvx_copy_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n, uint32_t elem_आकार) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

अटल अंतरभूत व्योम hvx_copy_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n, uint32_t elem_आकार) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
अटल अंतरभूत व्योम hvx_copy_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, माप(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
अटल अंतरभूत व्योम hvx_copy_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, माप(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
अटल अंतरभूत व्योम hvx_copy_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, माप(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, माप(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
अटल अंतरभूत व्योम hvx_copy_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, माप(भग्न));
}

// copy n fp32 elements : destination is aligned, source is unaligned
अटल अंतरभूत व्योम hvx_copy_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, माप(भग्न));
}

// copy n fp32 elements : destination is unaligned, source is aligned
अटल अंतरभूत व्योम hvx_copy_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, माप(भग्न));
}

// copy n fp32 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, माप(भग्न));
}

//// fp32 -> fp16

#घोषणा hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    करो {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        स्थिर uint32_t elem_आकार = माप(__fp16);                                  \
        स्थिर uint32_t epv  = 128 / elem_आकार;                                      \
        स्थिर uint32_t nvec = n / epv;                                              \
        स्थिर uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        क्रम (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        अगर (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);                      \
        }                                                                           \
    } जबतक(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
अटल अंतरभूत व्योम hvx_copy_f16_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
अटल अंतरभूत व्योम hvx_copy_f16_f32_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
अटल अंतरभूत व्योम hvx_copy_f16_f32_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_f16_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#घोषणा hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    करो {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        स्थिर HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        स्थिर uint32_t elem_आकार = माप(__fp16);                                  \
        स्थिर uint32_t epv  = 128 / elem_आकार;                                      \
        स्थिर uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        क्रम (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        अगर (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            अगर (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            अगर (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * माप(भग्न), vd);                \
            }                                                                       \
        }                                                                           \
    } जबतक(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
अटल अंतरभूत व्योम hvx_copy_f32_f16_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
अटल अंतरभूत व्योम hvx_copy_f32_f16_au(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
अटल अंतरभूत व्योम hvx_copy_f32_f16_ua(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_f32_f16_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#घोषणा hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    करो {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        स्थिर uint32_t elem_आकार = माप(int32_t);               \
        स्थिर uint32_t epv  = 128 / elem_आकार;                    \
        स्थिर uint32_t nvec = n / epv;                            \
        स्थिर uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        क्रम (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        अगर (nloe) {                                               \
            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);    \
        }                                                         \
    } जबतक(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
अटल अंतरभूत व्योम hvx_copy_i32_f32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_i32_f32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#घोषणा hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    करो {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        स्थिर uint32_t elem_आकार = माप(भग्न);                 \
        स्थिर uint32_t epv  = 128 / elem_आकार;                    \
        स्थिर uint32_t nvec = n / epv;                            \
        स्थिर uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        क्रम (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        अगर (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((व्योम *) &vdst[i], nloe * elem_आकार, v);    \
        }                                                         \
    } जबतक(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
अटल अंतरभूत व्योम hvx_copy_f32_i32_aa(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    निश्चित((अचिन्हित दीर्घ) dst % 128 == 0);
    निश्चित((अचिन्हित दीर्घ) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
अटल अंतरभूत व्योम hvx_copy_f32_i32_uu(uint8_t * restrict dst, स्थिर uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#पूर्ण_अगर // HVX_COPY_H
