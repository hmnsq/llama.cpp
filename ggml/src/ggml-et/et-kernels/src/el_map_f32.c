// Element-wise operations: dst[i] = src0[i] op src1[i]
#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#تعریف_کړه DEFINE_بلاک_OP(name, op_insn)                                                                             \
    ساكن مضمن فراغ name(عائم * dst_بلاک, ثابت عائم * src0_بلاک, ثابت عائم * src1_بلاک, صحيح elements) { \
        ثابت int32_t vec_پای_ټکی = (elements / 8) * 8;                                                                \
        ثابت int32_t لکۍ    = elements - vec_پای_ټکی;                                                                \
                                                                                                                   \
        بینښه اوږد لنډمهاله_ماسک;                                                                                   \
        __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));                                                         \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        لكل (int32_t i = 0; i < vec_پای_ټکی; i += 8) {                                                                 \
            __asm__ متطاير(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(عائم (*)[8]) & dst_بلاک[i])                                                         \
                : [s0] "m"(*(ثابت عائم (*)[8]) & src0_بلاک[i]), [s1] "m"(*(ثابت عائم (*)[8]) & src1_بلاک[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        إذا (لکۍ > 0) {                                                                                            \
            ثابت بینښه اوږد لکۍ_m0 = (1ul << لکۍ) - 1;                                                       \
            __asm__ متطاير(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_بلاک[vec_پای_ټکی]), [s1] "r"(&src1_بلاک[vec_پای_ټکی]), [d] "r"(&dst_بلاک[vec_پای_ټکی]),    \
                  [tm] "r"(لکۍ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));                                                          \
    }

DEFINE_بلاک_OP(بلاک_mul_cache_aligned, "fmul.ps")
DEFINE_بلاک_OP(بلاک_جمع_کړه_cache_aligned, "fadd.ps")
DEFINE_بلاک_OP(بلاک_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#تعریف_کړه DEFINE_بلاک_OP_خپرونه(name, op_insn)                                                                     \
    ساكن مضمن فراغ name(عائم * dst_بلاک, ثابت عائم * src0_بلاک, عائم scalar, صحيح elements) {               \
        ثابت int32_t vec_پای_ټکی = (elements / 8) * 8;                                                                  \
        ثابت int32_t لکۍ    = elements - vec_پای_ټکی;                                                                  \
                                                                                                                     \
        بینښه اوږد لنډمهاله_ماسک;                                                                                     \
        __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));                                                           \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        لكل (int32_t i = 0; i < vec_پای_ټکی; i += 8) {                                                                   \
            __asm__ متطاير(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(عائم (*)[8]) & dst_بلاک[i])                                                           \
                : [s0] "m"(*(ثابت عائم (*)[8]) & src0_بلاک[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        إذا (لکۍ > 0) {                                                                                              \
            ثابت بینښه اوږد لکۍ_m0 = (1ul << لکۍ) - 1;                                                         \
            __asm__ متطاير(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_بلاک[vec_پای_ټکی]), [ps] "r"(&scalar), [d] "r"(&dst_بلاک[vec_پای_ټکی]), [tm] "r"(لکۍ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));                                                            \
    }

DEFINE_بلاک_OP_خپرونه(بلاک_mul_خپرونه, "fmul.ps")
DEFINE_بلاک_OP_خپرونه(بلاک_جمع_کړه_خپرونه, "fadd.ps")
DEFINE_بلاک_OP_خپرونه(بلاک_sub_خپرونه, "fsub.ps")

ساكن مضمن عائم scalar_el_map(عائم src0, عائم src1, تعداد ggml_op operation) {
    بدل (operation) {
        حالة GGML_OP_MUL:
            أرجع src0 * src1;
        حالة GGML_OP_جمع_کړه:
            أرجع src0 + src1;
        حالة GGML_OP_SUB:
            أرجع src0 - src1;
        افتراضي:
            أرجع 0.0f;
    }
}

صحيح ثبت_ټکی(بنية ggml_et_دوګونی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    عائم * src0_معلومات = (عائم *) src0->معلومات;
    عائم * src1_معلومات = (عائم *) src1->معلومات;
    عائم * dst_معلومات  = (عائم *) dst->معلومات;

    إذا (!src0_معلومات || !src1_معلومات || !dst_معلومات) {
        أرجع -1;  // Null data pointer
    }

#که_تعریف_شوی ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ثابت size_t src0_بایټونه = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بایټونه = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_معلومات, src0_بایټونه);
    evict_region_past_l2(src1_معلومات, src1_بایټونه);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#د_که_پای

    تعداد ggml_op operation = dst->op;

    إذا (operation != GGML_OP_MUL && operation != GGML_OP_جمع_کړه && operation != GGML_OP_SUB) {
        أرجع -1;  // Unsupported operation
    }

    ثابت int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ثابت int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ثابت int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ثابت size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ثابت bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ثابت bool no_خپرونه = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ثابت bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    إذا (no_خپرونه && all_contiguous) {
        ثابت int64_t ټول_elements         = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes
        ثابت int64_t ټول_cachelines       = (ټول_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

        ثابت int64_t cl_فی_thread = (ټول_cachelines + عدد_تارونه - 1) / عدد_تارونه;
        ثابت int64_t cl_پیل      = thread_id * cl_فی_thread;
        int64_t       cl_پای_ټکی        = cl_پیل + cl_فی_thread;
        إذا (cl_پای_ټکی > ټول_cachelines) {
            cl_پای_ټکی = ټول_cachelines;
        }

        إذا (cl_پیل >= ټول_cachelines) {
            أرجع 0;
        }

        ثابت int64_t elem_پیل = cl_پیل * elements_فی_cacheline;
        int64_t       elem_پای_ټکی   = cl_پای_ټکی * elements_فی_cacheline;
        إذا (elem_پای_ټکی > ټول_elements) {
            elem_پای_ټکی = ټول_elements;
        }
        ثابت int32_t شمېر = (int32_t) (elem_پای_ټکی - elem_پیل);

        بدل (operation) {
            حالة GGML_OP_MUL:
                بلاک_mul_cache_aligned(dst_معلومات + elem_پیل, src0_معلومات + elem_پیل, src1_معلومات + elem_پیل, شمېر);
                اكسر;
            حالة GGML_OP_جمع_کړه:
                بلاک_جمع_کړه_cache_aligned(dst_معلومات + elem_پیل, src0_معلومات + elem_پیل, src1_معلومات + elem_پیل, شمېر);
                اكسر;
            حالة GGML_OP_SUB:
                بلاک_sub_cache_aligned(dst_معلومات + elem_پیل, src0_معلومات + elem_پیل, src1_معلومات + elem_پیل, شمېر);
                اكسر;
            افتراضي:
                أرجع 1;
        }
#که_تعریف_شوی ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_معلومات + elem_پیل, (size_t) شمېر * حجم(عائم));
        WAIT_CACHEOPS;
        FENCE;
#د_که_پای
        أرجع 0;
    }

    // Slow path: broadcasting or non-contiguous
    ثابت int64_t ټول_لیکې = ne1 * ne2 * ne3;

    int64_t پیل_لیکه;
    int64_t پای_ټکی_لیکه;

    إذا (cache_aligned) {
        ثابت int64_t لیکې_فی_thread = (ټول_لیکې + عدد_تارونه - 1) / عدد_تارونه;
        پیل_لیکه                     = thread_id * لیکې_فی_thread;
        پای_ټکی_لیکه = (پیل_لیکه + لیکې_فی_thread < ټول_لیکې) ? (پیل_لیکه + لیکې_فی_thread) : ټول_لیکې;
    } وإلا {
        ثابت int64_t لیکې_فی_ډله = et_لیکې_فی_cacheline_ډله(ne0, حجم(عائم));
        ثابت int64_t ټول_groups   = (ټول_لیکې + لیکې_فی_ډله - 1) / لیکې_فی_ډله;

        إذا (thread_id >= ټول_groups) {
            أرجع 0;
        }

        ثابت int64_t ډله_پیل = thread_id;
        لكل (int64_t grp = ډله_پیل; grp < ټول_groups; grp += عدد_تارونه) {
            ثابت int64_t ډله_لیکه_پیل = grp * لیکې_فی_ډله;
            int64_t       ډله_لیکه_پای_ټکی   = ډله_لیکه_پیل + لیکې_فی_ډله;
            إذا (ډله_لیکه_پای_ټکی > ټول_لیکې) {
                ډله_لیکه_پای_ټکی = ټول_لیکې;
            }

#که_تعریف_شوی ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ثابت int64_t لومړی_i03      = ډله_لیکه_پیل / (ne2 * ne1);
            ثابت int64_t لومړی_i02      = (ډله_لیکه_پیل - لومړی_i03 * ne2 * ne1) / ne1;
            ثابت int64_t لومړی_i01      = (ډله_لیکه_پیل - لومړی_i03 * ne2 * ne1 - لومړی_i02 * ne1);
            حرف *        ډله_dst_بنسټ = (حرف *) dst_معلومات + لومړی_i03 * nb3 + لومړی_i02 * nb2 + لومړی_i01 * nb1;
#د_که_پای

            لكل (int64_t ir = ډله_لیکه_پیل; ir < ډله_لیکه_پای_ټکی; ir++) {
                ثابت int64_t i03 = ir / (ne2 * ne1);
                ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ثابت int64_t i13 = i03 % ne13;
                ثابت int64_t i12 = i02 % ne12;
                ثابت int64_t i11 = i01 % ne11;

                عائم *       dst_ptr = (عائم *) ((حرف *) dst_معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ثابت عائم * src0_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ثابت عائم * src1_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i13 * nb13 + i12 * nb12 + i11 * nb11);

                إذا (ne10 == 1) {
                    ثابت عائم scalar = src1_ptr[0];
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } وإلا {
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#که_تعریف_شوی ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ثابت int64_t nrows = ډله_لیکه_پای_ټکی - ډله_لیکه_پیل;
            إذا (nrows > 0) {
                FENCE;
                evict_region_past_l2(ډله_dst_بنسټ, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#د_که_پای
        }

        أرجع 0;
    }

    إذا (پیل_لیکه >= ټول_لیکې) {
        أرجع 0;
    }

    لكل (int64_t ir = پیل_لیکه; ir < پای_ټکی_لیکه; ir++) {
        // Convert flat row index to 3D coordinates
        ثابت int64_t i03 = ir / (ne2 * ne1);
        ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ثابت int64_t i13 = i03 % ne13;
        ثابت int64_t i12 = i02 % ne12;
        ثابت int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        عائم *       dst_ptr  = (عائم *) ((حرف *) dst_معلومات + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ثابت عائم * src0_ptr = (ثابت عائم *) ((ثابت حرف *) src0_معلومات + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ثابت عائم * src1_ptr = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i13 * nb13 + i12 * nb12 + i11 * nb11);

        إذا (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            عائم scalar = src1_ptr[0];
            بدل (operation) {
                حالة GGML_OP_MUL:
                    بلاک_mul_خپرونه(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_جمع_کړه:
                    بلاک_جمع_کړه_خپرونه(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_SUB:
                    بلاک_sub_خپرونه(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                افتراضي:
                    أرجع 1;
            }
        } وإلا {
            // Broadcasting in dimension 0: src1 repeats across src0
            ثابت int64_t nr0 = ne0 / ne10;

            لكل (int64_t r = 0; r < nr0; r++) {
                ثابت عائم * src0_بلاک = src0_ptr + r * ne10;
                عائم *       dst_بلاک  = dst_ptr + r * ne10;

                بدل (operation) {
                    حالة GGML_OP_MUL:
                        بلاک_mul_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    حالة GGML_OP_جمع_کړه:
                        بلاک_جمع_کړه_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    حالة GGML_OP_SUB:
                        بلاک_sub_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    افتراضي:
                        أرجع 1;
                }
            }
        }
    }

#که_تعریف_شوی ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    إذا (پای_ټکی_لیکه > پیل_لیکه) {
        FENCE;
        evict_region_past_l2((حرف *) dst_معلومات + پیل_لیکه * nb1, (size_t) (پای_ټکی_لیکه - پیل_لیکه) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#د_که_پای
    أرجع 0;
}
