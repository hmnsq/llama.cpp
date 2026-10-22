// Element-wise operations: dst[i] = src0[i] op src1[i]
#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#تعریف_کن DEFINE_بلوک_OP(name, op_insn)                                                                             \
    ساكن مضمن فراغ name(عائم * dst_بلوک, ثابت عائم * src0_بلوک, ثابت عائم * src1_بلوک, صحيح elements) { \
        ثابت int32_t vec_انتها = (elements / 8) * 8;                                                                \
        ثابت int32_t دم    = elements - vec_انتها;                                                                \
                                                                                                                   \
        بیعلامت دراز موقت_ماسک;                                                                                   \
        __asm__ متطاير("mova.x.m %0" : "=r"(موقت_ماسک));                                                         \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        لكل (int32_t i = 0; i < vec_انتها; i += 8) {                                                                 \
            __asm__ متطاير(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(عائم (*)[8]) & dst_بلوک[i])                                                         \
                : [s0] "m"(*(ثابت عائم (*)[8]) & src0_بلوک[i]), [s1] "m"(*(ثابت عائم (*)[8]) & src1_بلوک[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        إذا (دم > 0) {                                                                                            \
            ثابت بیعلامت دراز دم_m0 = (1ul << دم) - 1;                                                       \
            __asm__ متطاير(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_بلوک[vec_انتها]), [s1] "r"(&src1_بلوک[vec_انتها]), [d] "r"(&dst_بلوک[vec_انتها]),    \
                  [tm] "r"(دم_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ متطاير("mova.m.x %0" ::"r"(موقت_ماسک));                                                          \
    }

DEFINE_بلوک_OP(بلوک_mul_cache_aligned, "fmul.ps")
DEFINE_بلوک_OP(بلوک_جمع_کن_cache_aligned, "fadd.ps")
DEFINE_بلوک_OP(بلوک_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#تعریف_کن DEFINE_بلوک_OP_پخش(name, op_insn)                                                                     \
    ساكن مضمن فراغ name(عائم * dst_بلوک, ثابت عائم * src0_بلوک, عائم scalar, صحيح elements) {               \
        ثابت int32_t vec_انتها = (elements / 8) * 8;                                                                  \
        ثابت int32_t دم    = elements - vec_انتها;                                                                  \
                                                                                                                     \
        بیعلامت دراز موقت_ماسک;                                                                                     \
        __asm__ متطاير("mova.x.m %0" : "=r"(موقت_ماسک));                                                           \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        لكل (int32_t i = 0; i < vec_انتها; i += 8) {                                                                   \
            __asm__ متطاير(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(عائم (*)[8]) & dst_بلوک[i])                                                           \
                : [s0] "m"(*(ثابت عائم (*)[8]) & src0_بلوک[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        إذا (دم > 0) {                                                                                              \
            ثابت بیعلامت دراز دم_m0 = (1ul << دم) - 1;                                                         \
            __asm__ متطاير(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_بلوک[vec_انتها]), [ps] "r"(&scalar), [d] "r"(&dst_بلوک[vec_انتها]), [tm] "r"(دم_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ متطاير("mova.m.x %0" ::"r"(موقت_ماسک));                                                            \
    }

DEFINE_بلوک_OP_پخش(بلوک_mul_پخش, "fmul.ps")
DEFINE_بلوک_OP_پخش(بلوک_جمع_کن_پخش, "fadd.ps")
DEFINE_بلوک_OP_پخش(بلوک_sub_پخش, "fsub.ps")

ساكن مضمن عائم scalar_el_map(عائم src0, عائم src1, تعداد ggml_op operation) {
    بدل (operation) {
        حالة GGML_OP_MUL:
            أرجع src0 * src1;
        حالة GGML_OP_جمع_کن:
            أرجع src0 + src1;
        حالة GGML_OP_SUB:
            أرجع src0 - src1;
        افتراضي:
            أرجع 0.0f;
    }
}

صحيح مدخل_نقطه(بنية ggml_et_دودویی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

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

    عائم * src0_داده = (عائم *) src0->داده;
    عائم * src1_داده = (عائم *) src1->داده;
    عائم * dst_داده  = (عائم *) dst->داده;

    إذا (!src0_داده || !src1_داده || !dst_داده) {
        أرجع -1;  // Null data pointer
    }

#اگر_تعریف_شده ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ثابت size_t src0_بایتها = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بایتها = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_داده, src0_بایتها);
    evict_region_past_l2(src1_داده, src1_بایتها);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#پایان_اگر

    تعداد ggml_op operation = dst->op;

    إذا (operation != GGML_OP_MUL && operation != GGML_OP_جمع_کن && operation != GGML_OP_SUB) {
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
    ثابت bool no_پخش = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ثابت bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    إذا (no_پخش && all_contiguous) {
        ثابت int64_t مجموع_elements         = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elements_در_هر_cacheline = 16;  // 64 bytes / 4 bytes
        ثابت int64_t مجموع_cachelines       = (مجموع_elements + elements_در_هر_cacheline - 1) / elements_در_هر_cacheline;

        ثابت int64_t cl_در_هر_thread = (مجموع_cachelines + رقم_نخها - 1) / رقم_نخها;
        ثابت int64_t cl_شروع      = thread_id * cl_در_هر_thread;
        int64_t       cl_انتها        = cl_شروع + cl_در_هر_thread;
        إذا (cl_انتها > مجموع_cachelines) {
            cl_انتها = مجموع_cachelines;
        }

        إذا (cl_شروع >= مجموع_cachelines) {
            أرجع 0;
        }

        ثابت int64_t elem_شروع = cl_شروع * elements_در_هر_cacheline;
        int64_t       elem_انتها   = cl_انتها * elements_در_هر_cacheline;
        إذا (elem_انتها > مجموع_elements) {
            elem_انتها = مجموع_elements;
        }
        ثابت int32_t شمار = (int32_t) (elem_انتها - elem_شروع);

        بدل (operation) {
            حالة GGML_OP_MUL:
                بلوک_mul_cache_aligned(dst_داده + elem_شروع, src0_داده + elem_شروع, src1_داده + elem_شروع, شمار);
                اكسر;
            حالة GGML_OP_جمع_کن:
                بلوک_جمع_کن_cache_aligned(dst_داده + elem_شروع, src0_داده + elem_شروع, src1_داده + elem_شروع, شمار);
                اكسر;
            حالة GGML_OP_SUB:
                بلوک_sub_cache_aligned(dst_داده + elem_شروع, src0_داده + elem_شروع, src1_داده + elem_شروع, شمار);
                اكسر;
            افتراضي:
                أرجع 1;
        }
#اگر_تعریف_شده ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_داده + elem_شروع, (size_t) شمار * حجم(عائم));
        WAIT_CACHEOPS;
        FENCE;
#پایان_اگر
        أرجع 0;
    }

    // Slow path: broadcasting or non-contiguous
    ثابت int64_t مجموع_ردیفها = ne1 * ne2 * ne3;

    int64_t شروع_ردیف;
    int64_t انتها_ردیف;

    إذا (cache_aligned) {
        ثابت int64_t ردیفها_در_هر_thread = (مجموع_ردیفها + رقم_نخها - 1) / رقم_نخها;
        شروع_ردیف                     = thread_id * ردیفها_در_هر_thread;
        انتها_ردیف = (شروع_ردیف + ردیفها_در_هر_thread < مجموع_ردیفها) ? (شروع_ردیف + ردیفها_در_هر_thread) : مجموع_ردیفها;
    } وإلا {
        ثابت int64_t ردیفها_در_هر_گروه = et_ردیفها_در_هر_cacheline_گروه(ne0, حجم(عائم));
        ثابت int64_t مجموع_groups   = (مجموع_ردیفها + ردیفها_در_هر_گروه - 1) / ردیفها_در_هر_گروه;

        إذا (thread_id >= مجموع_groups) {
            أرجع 0;
        }

        ثابت int64_t گروه_شروع = thread_id;
        لكل (int64_t grp = گروه_شروع; grp < مجموع_groups; grp += رقم_نخها) {
            ثابت int64_t گروه_ردیف_شروع = grp * ردیفها_در_هر_گروه;
            int64_t       گروه_ردیف_انتها   = گروه_ردیف_شروع + ردیفها_در_هر_گروه;
            إذا (گروه_ردیف_انتها > مجموع_ردیفها) {
                گروه_ردیف_انتها = مجموع_ردیفها;
            }

#اگر_تعریف_شده ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ثابت int64_t اولی_i03      = گروه_ردیف_شروع / (ne2 * ne1);
            ثابت int64_t اولی_i02      = (گروه_ردیف_شروع - اولی_i03 * ne2 * ne1) / ne1;
            ثابت int64_t اولی_i01      = (گروه_ردیف_شروع - اولی_i03 * ne2 * ne1 - اولی_i02 * ne1);
            حرف *        گروه_dst_پایه = (حرف *) dst_داده + اولی_i03 * nb3 + اولی_i02 * nb2 + اولی_i01 * nb1;
#پایان_اگر

            لكل (int64_t ir = گروه_ردیف_شروع; ir < گروه_ردیف_انتها; ir++) {
                ثابت int64_t i03 = ir / (ne2 * ne1);
                ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ثابت int64_t i13 = i03 % ne13;
                ثابت int64_t i12 = i02 % ne12;
                ثابت int64_t i11 = i01 % ne11;

                عائم *       dst_ptr = (عائم *) ((حرف *) dst_داده + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ثابت عائم * src0_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src0_داده + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ثابت عائم * src1_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src1_داده + i13 * nb13 + i12 * nb12 + i11 * nb11);

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

#اگر_تعریف_شده ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ثابت int64_t nrows = گروه_ردیف_انتها - گروه_ردیف_شروع;
            إذا (nrows > 0) {
                FENCE;
                evict_region_past_l2(گروه_dst_پایه, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#پایان_اگر
        }

        أرجع 0;
    }

    إذا (شروع_ردیف >= مجموع_ردیفها) {
        أرجع 0;
    }

    لكل (int64_t ir = شروع_ردیف; ir < انتها_ردیف; ir++) {
        // Convert flat row index to 3D coordinates
        ثابت int64_t i03 = ir / (ne2 * ne1);
        ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ثابت int64_t i13 = i03 % ne13;
        ثابت int64_t i12 = i02 % ne12;
        ثابت int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        عائم *       dst_ptr  = (عائم *) ((حرف *) dst_داده + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ثابت عائم * src0_ptr = (ثابت عائم *) ((ثابت حرف *) src0_داده + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ثابت عائم * src1_ptr = (ثابت عائم *) ((ثابت حرف *) src1_داده + i13 * nb13 + i12 * nb12 + i11 * nb11);

        إذا (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            عائم scalar = src1_ptr[0];
            بدل (operation) {
                حالة GGML_OP_MUL:
                    بلوک_mul_پخش(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_جمع_کن:
                    بلوک_جمع_کن_پخش(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_SUB:
                    بلوک_sub_پخش(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                افتراضي:
                    أرجع 1;
            }
        } وإلا {
            // Broadcasting in dimension 0: src1 repeats across src0
            ثابت int64_t nr0 = ne0 / ne10;

            لكل (int64_t r = 0; r < nr0; r++) {
                ثابت عائم * src0_بلوک = src0_ptr + r * ne10;
                عائم *       dst_بلوک  = dst_ptr + r * ne10;

                بدل (operation) {
                    حالة GGML_OP_MUL:
                        بلوک_mul_cache_aligned(dst_بلوک, src0_بلوک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    حالة GGML_OP_جمع_کن:
                        بلوک_جمع_کن_cache_aligned(dst_بلوک, src0_بلوک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    حالة GGML_OP_SUB:
                        بلوک_sub_cache_aligned(dst_بلوک, src0_بلوک, src1_ptr, (صحيح) ne10);
                        اكسر;
                    افتراضي:
                        أرجع 1;
                }
            }
        }
    }

#اگر_تعریف_شده ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    إذا (انتها_ردیف > شروع_ردیف) {
        FENCE;
        evict_region_past_l2((حرف *) dst_داده + شروع_ردیف * nb1, (size_t) (انتها_ردیف - شروع_ردیف) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#پایان_اگر
    أرجع 0;
}
