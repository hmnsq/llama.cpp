#ifndef HVX_COPY_H
#define HVX_COPY_H

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdint.h>

#അന്തര്ഭാവ "hvx-base.h"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    ചെയ്യുക {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        വേണ്ടി (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        എങ്കിൽ (nloe) {                                              \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } ആകുമ്പോൾ(0)

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_a(ശൂന്യം * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_u(ശൂന്യം * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_f32_a(ശൂന്യം * restrict dst, ദശാംശം v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, വലുപ്പം(ദശാംശം));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_f32_u(ശൂന്യം * restrict dst, ദശാംശം v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, വലുപ്പം(ദശാംശം));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_f16_a(ശൂന്യം * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f16(v), n, വലുപ്പം(__fp16));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_f16_u(ശൂന്യം * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, വലുപ്പം(__fp16));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_u16_a(ശൂന്യം * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, വലുപ്പം(uint16_t));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_u16_u(ശൂന്യം * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, വലുപ്പം(uint16_t));
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_u8_a(ശൂന്യം * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_splat_u8_u(ശൂന്യം * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    ചെയ്യുക {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                       \
        സ്ഥിരം uint32_t nvec = n / epv;                               \
        സ്ഥിരം uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        വേണ്ടി (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        എങ്കിൽ (nloe) {                                                  \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } ആകുമ്പോൾ(0)

// Generic copy routines
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : destination and source are aligned to HVX Vector (128)
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, വലുപ്പം(__fp16));
}

// copy n fp16 elements : destination is aligned, source is unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, വലുപ്പം(__fp16));
}

// copy n fp16 elements : destination is unaligned, source is aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, വലുപ്പം(__fp16));
}

// copy n fp16 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, വലുപ്പം(__fp16));
}

// copy n fp32 elements : destination and source are aligned to HVX Vector (128)
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, വലുപ്പം(ദശാംശം));
}

// copy n fp32 elements : destination is aligned, source is unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, വലുപ്പം(ദശാംശം));
}

// copy n fp32 elements : destination is unaligned, source is aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, വലുപ്പം(ദശാംശം));
}

// copy n fp32 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, വലുപ്പം(ദശാംശം));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    ചെയ്യുക {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(__fp16);                                  \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                                      \
        സ്ഥിരം uint32_t nvec = n / epv;                                              \
        സ്ഥിരം uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        വേണ്ടി (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        എങ്കിൽ (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } ആകുമ്പോൾ(0)

// copy/convert n fp32 elements into n fp16 elements : destination and source are aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is aligned, source is unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_f32_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : destination is unaligned, source is aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_f32_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f16_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    ചെയ്യുക {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        സ്ഥിരം HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(__fp16);                                  \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                                      \
        സ്ഥിരം uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        വേണ്ടി (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        എങ്കിൽ (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            എങ്കിൽ (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            എങ്കിൽ (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * വലുപ്പം(ദശാംശം), vd);                \
            }                                                                       \
        }                                                                           \
    } ആകുമ്പോൾ(0)

// copy/convert n fp16 elements into n fp32 elements : destination and source are aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_f16_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is aligned, source is unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_f16_au(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : destination is unaligned, source is aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_f16_ua(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_f16_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp32 -> int32

#define hvx_copy_i32_f32_loop_body(dst_type, src_type, vec_store) \
    ചെയ്യുക {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(int32_t);               \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                    \
        സ്ഥിരം uint32_t nvec = n / epv;                            \
        സ്ഥിരം uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        വേണ്ടി (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vw_equals_Vsf(vsrc[i]);                  \
        }                                                         \
        എങ്കിൽ (nloe) {                                               \
            HVX_Vector v = Q6_Vw_equals_Vsf(vsrc[i]);             \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } ആകുമ്പോൾ(0)

// copy/convert n fp32 elements into n int32 elements : destination and source are aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_i32_f32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_i32_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n int32 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_i32_f32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_i32_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// int32 -> fp32

#define hvx_copy_f32_i32_loop_body(dst_type, src_type, vec_store) \
    ചെയ്യുക {                                                          \
        dst_type * restrict vdst = (dst_type *) dst;              \
        src_type * restrict vsrc = (src_type *) src;              \
                                                                  \
        സ്ഥിരം uint32_t elem_size = വലുപ്പം(ദശാംശം);                 \
        സ്ഥിരം uint32_t epv  = 128 / elem_size;                    \
        സ്ഥിരം uint32_t nvec = n / epv;                            \
        സ്ഥിരം uint32_t nloe = n % epv;                            \
                                                                  \
        uint32_t i = 0;                                           \
        _Pragma("unroll(4)")                                      \
        വേണ്ടി (; i < nvec; i++) {                                   \
            vdst[i] = Q6_Vsf_equals_Vw(vsrc[i]);                  \
        }                                                         \
        എങ്കിൽ (nloe) {                                               \
            HVX_Vector v = Q6_Vsf_equals_Vw(vsrc[i]);             \
            vec_store((ശൂന്യം *) &vdst[i], nloe * elem_size, v);    \
        }                                                         \
    } ആകുമ്പോൾ(0)

// copy/convert n int32 elements into n fp32 elements : destination and source are aligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_i32_aa(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    പ്രതിജ്ഞാ((unsigned long) dst % 128 == 0);
    പ്രതിജ്ഞാ((unsigned long) src % 128 == 0);
    hvx_copy_f32_i32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n int32 elements into n fp32 elements : destination and source are unaligned
സ്ഥിതം വരിയിൽ ശൂന്യം hvx_copy_f32_i32_uu(uint8_t * restrict dst, സ്ഥിരം uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_i32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#endif // HVX_COPY_H
