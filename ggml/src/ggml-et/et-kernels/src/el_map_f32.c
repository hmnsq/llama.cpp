// Element-wise operations: dst[i] = src0[i] op src1[i]
#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#تعریف_کرو DEFINE_بلاک_OP(name, op_insn)                                                                             \
    ساكن مضمن فراغ name(عائم * dst_بلاک, ثابت عائم * src0_بلاک, ثابت عائم * src1_بلاک, صحيح elements) { \
        ثابت int32_t vec_سرا = (elements / 8) * 8;                                                                \
        ثابت int32_t پوچھ    = elements - vec_سرا;                                                                \
                                                                                                                   \
        بنانشانی لما عارضی_نقاب;                                                                                   \
        __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));                                                         \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        لكل (int32_t i = 0; i < vec_سرا; i += 8) {                                                                 \
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
        إذا (پوچھ > 0) {                                                                                            \
            ثابت بنانشانی لما پوچھ_m0 = (1ul << پوچھ) - 1;                                                       \
            __asm__ متطاير(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_بلاک[vec_سرا]), [s1] "r"(&src1_بلاک[vec_سرا]), [d] "r"(&dst_بلاک[vec_سرا]),    \
                  [tm] "r"(پوچھ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));                                                          \
    }

DEFINE_بلاک_OP(بلاک_mul_cache_aligned, "fmul.ps")
DEFINE_بلاک_OP(بلاک_جوڑو_cache_aligned, "fadd.ps")
DEFINE_بلاک_OP(بلاک_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#تعریف_کرو DEFINE_بلاک_OP_نشر(name, op_insn)                                                                     \
    ساكن مضمن فراغ name(عائم * dst_بلاک, ثابت عائم * src0_بلاک, عائم scalar, صحيح elements) {               \
        ثابت int32_t vec_سرا = (elements / 8) * 8;                                                                  \
        ثابت int32_t پوچھ    = elements - vec_سرا;                                                                  \
                                                                                                                     \
        بنانشانی لما عارضی_نقاب;                                                                                     \
        __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));                                                           \
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        لكل (int32_t i = 0; i < vec_سرا; i += 8) {                                                                   \
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
        إذا (پوچھ > 0) {                                                                                              \
            ثابت بنانشانی لما پوچھ_m0 = (1ul << پوچھ) - 1;                                                         \
            __asm__ متطاير(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_بلاک[vec_سرا]), [ps] "r"(&scalar), [d] "r"(&dst_بلاک[vec_سرا]), [tm] "r"(پوچھ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));                                                            \
    }

DEFINE_بلاک_OP_نشر(بلاک_mul_نشر, "fmul.ps")
DEFINE_بلاک_OP_نشر(بلاک_جوڑو_نشر, "fadd.ps")
DEFINE_بلاک_OP_نشر(بلاک_sub_نشر, "fsub.ps")

ساكن مضمن عائم scalar_el_map(عائم src0, عائم src1, تعداد ggml_op operation) {
    بدل (operation) {
        حالة GGML_OP_MUL:
            أرجع src0 * src1;
        حالة GGML_OP_جوڑو:
            أرجع src0 + src1;
        حالة GGML_OP_SUB:
            أرجع src0 - src1;
        افتراضي:
            أرجع 0.0f;
    }
}

صحيح اندراج_نقطہ(بنية ggml_et_ثنائی_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

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

    عائم * src0_انکڑے = (عائم *) src0->انکڑے;
    عائم * src1_انکڑے = (عائم *) src1->انکڑے;
    عائم * dst_انکڑے  = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !src1_انکڑے || !dst_انکڑے) {
        أرجع -1;  // Null data pointer
    }

#جے_تعریف_شدہ ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ثابت size_t src0_بائٹاں = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بائٹاں = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_انکڑے, src0_بائٹاں);
    evict_region_past_l2(src1_انکڑے, src1_بائٹاں);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#جے_انت

    تعداد ggml_op operation = dst->op;

    إذا (operation != GGML_OP_MUL && operation != GGML_OP_جوڑو && operation != GGML_OP_SUB) {
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
    ثابت bool no_نشر = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ثابت bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    إذا (no_نشر && all_contiguous) {
        ثابت int64_t کل_elements         = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes
        ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

        ثابت int64_t cl_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t cl_شروع      = thread_id * cl_فی_thread;
        int64_t       cl_سرا        = cl_شروع + cl_فی_thread;
        إذا (cl_سرا > کل_cachelines) {
            cl_سرا = کل_cachelines;
        }

        إذا (cl_شروع >= کل_cachelines) {
            أرجع 0;
        }

        ثابت int64_t elem_شروع = cl_شروع * elements_فی_cacheline;
        int64_t       elem_سرا   = cl_سرا * elements_فی_cacheline;
        إذا (elem_سرا > کل_elements) {
            elem_سرا = کل_elements;
        }
        ثابت int32_t گنتی = (int32_t) (elem_سرا - elem_شروع);

        بدل (operation) {
            حالة GGML_OP_MUL:
                بلاک_mul_cache_aligned(dst_انکڑے + elem_شروع, src0_انکڑے + elem_شروع, src1_انکڑے + elem_شروع, گنتی);
                اكسر;
            حالة GGML_OP_جوڑو:
                بلاک_جوڑو_cache_aligned(dst_انکڑے + elem_شروع, src0_انکڑے + elem_شروع, src1_انکڑے + elem_شروع, گنتی);
                اكسر;
            حالة GGML_OP_SUB:
                بلاک_sub_cache_aligned(dst_انکڑے + elem_شروع, src0_انکڑے + elem_شروع, src1_انکڑے + elem_شروع, گنتی);
                اكسر;
            افتراضي:
                أرجع 1;
        }
#جے_تعریف_شدہ ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_انکڑے + elem_شروع, (size_t) گنتی * حجم(عائم));
        WAIT_CACHEOPS;
        FENCE;
#جے_انت
        أرجع 0;
    }

    // Slow path: broadcasting or non-contiguous
    ثابت int64_t کل_صفاں = ne1 * ne2 * ne3;

    int64_t شروع_صف;
    int64_t سرا_صف;

    إذا (cache_aligned) {
        ثابت int64_t صفاں_فی_thread = (کل_صفاں + عدد_دھاگے - 1) / عدد_دھاگے;
        شروع_صف                     = thread_id * صفاں_فی_thread;
        سرا_صف = (شروع_صف + صفاں_فی_thread < کل_صفاں) ? (شروع_صف + صفاں_فی_thread) : کل_صفاں;
    } وإلا {
        ثابت int64_t صفاں_فی_گروہ = et_صفاں_فی_cacheline_گروہ(ne0, حجم(عائم));
        ثابت int64_t کل_groups   = (کل_صفاں + صفاں_فی_گروہ - 1) / صفاں_فی_گروہ;

        إذا (thread_id >= کل_groups) {
            أرجع 0;
        }

        ثابت int64_t گروہ_شروع = thread_id;
        لكل (int64_t grp = گروہ_شروع; grp < کل_groups; grp += عدد_دھاگے) {
            ثابت int64_t گروہ_صف_شروع = grp * صفاں_فی_گروہ;
            int64_t       گروہ_صف_سرا   = گروہ_صف_شروع + صفاں_فی_گروہ;
            إذا (گروہ_صف_سرا > کل_صفاں) {
                گروہ_صف_سرا = کل_صفاں;
            }

#جے_تعریف_شدہ ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ثابت int64_t پہلا_i03      = گروہ_صف_شروع / (ne2 * ne1);
            ثابت int64_t پہلا_i02      = (گروہ_صف_شروع - پہلا_i03 * ne2 * ne1) / ne1;
            ثابت int64_t پہلا_i01      = (گروہ_صف_شروع - پہلا_i03 * ne2 * ne1 - پہلا_i02 * ne1);
            حرف *        گروہ_dst_ادھار = (حرف *) dst_انکڑے + پہلا_i03 * nb3 + پہلا_i02 * nb2 + پہلا_i01 * nb1;
#جے_انت

            لكل (int64_t ir = گروہ_صف_شروع; ir < گروہ_صف_سرا; ir++) {
                ثابت int64_t i03 = ir / (ne2 * ne1);
                ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ثابت int64_t i13 = i03 % ne13;
                ثابت int64_t i12 = i02 % ne12;
                ثابت int64_t i11 = i01 % ne11;

                عائم *       dst_ptr = (عائم *) ((حرف *) dst_انکڑے + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ثابت عائم * src0_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ثابت عائم * src1_ptr =
                    (ثابت عائم *) ((ثابت حرف *) src1_انکڑے + i13 * nb13 + i12 * nb12 + i11 * nb11);

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

#جے_تعریف_شدہ ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ثابت int64_t nrows = گروہ_صف_سرا - گروہ_صف_شروع;
            إذا (nrows > 0) {
                FENCE;
                evict_region_past_l2(گروہ_dst_ادھار, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#جے_انت
        }

        أرجع 0;
    }

    إذا (شروع_صف >= کل_صفاں) {
        أرجع 0;
    }

    لكل (int64_t ir = شروع_صف; ir < سرا_صف; ir++) {
        // Convert flat row index to 3D coordinates
        ثابت int64_t i03 = ir / (ne2 * ne1);
        ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ثابت int64_t i13 = i03 % ne13;
        ثابت int64_t i12 = i02 % ne12;
        ثابت int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        عائم *       dst_ptr  = (عائم *) ((حرف *) dst_انکڑے + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ثابت عائم * src0_ptr = (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ثابت عائم * src1_ptr = (ثابت عائم *) ((ثابت حرف *) src1_انکڑے + i13 * nb13 + i12 * nb12 + i11 * nb11);

        إذا (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            عائم scalar = src1_ptr[0];
            بدل (operation) {
                حالة GGML_OP_MUL:
                    بلاک_mul_نشر(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_جوڑو:
                    بلاک_جوڑو_نشر(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
                    اكسر;
                حالة GGML_OP_SUB:
                    بلاک_sub_نشر(dst_ptr, src0_ptr, scalar, (صحيح) ne0);
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
                    حالة GGML_OP_جوڑو:
                        بلاک_جوڑو_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحيح) ne10);
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

#جے_تعریف_شدہ ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    إذا (سرا_صف > شروع_صف) {
        FENCE;
        evict_region_past_l2((حرف *) dst_انکڑے + شروع_صف * nb1, (size_t) (سرا_صف - شروع_صف) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#جے_انت
    أرجع 0;
}
