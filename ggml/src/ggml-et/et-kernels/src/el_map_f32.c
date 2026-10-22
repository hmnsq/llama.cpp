// Element-wise operations: dst[i] = src0[i] op src1[i]
#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#تعریف_کرو DEFINE_بلاک_OP(name, op_insn)                                                                             \
    ساکن اندرونی خالی name(اعشاری * dst_بلاک, ثابت اعشاری * src0_بلاک, ثابت اعشاری * src1_بلاک, صحیح elements) { \
        ثابت int32_t vec_انتہا = (elements / 8) * 8;                                                                \
        ثابت int32_t دم    = elements - vec_انتہا;                                                                \
                                                                                                                   \
        غیرعلامتی طویل عارضی_نقاب;                                                                                   \
        __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));                                                         \
        __asm__ متلون("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        برائے (int32_t i = 0; i < vec_انتہا; i += 8) {                                                                 \
            __asm__ متلون(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])                                                         \
                : [s0] "m"(*(ثابت اعشاری (*)[8]) & src0_بلاک[i]), [s1] "m"(*(ثابت اعشاری (*)[8]) & src1_بلاک[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        اگر (دم > 0) {                                                                                            \
            ثابت غیرعلامتی طویل دم_m0 = (1ul << دم) - 1;                                                       \
            __asm__ متلون(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_بلاک[vec_انتہا]), [s1] "r"(&src1_بلاک[vec_انتہا]), [d] "r"(&dst_بلاک[vec_انتہا]),    \
                  [tm] "r"(دم_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));                                                          \
    }

DEFINE_بلاک_OP(بلاک_mul_cache_aligned, "fmul.ps")
DEFINE_بلاک_OP(بلاک_جوڑو_cache_aligned, "fadd.ps")
DEFINE_بلاک_OP(بلاک_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#تعریف_کرو DEFINE_بلاک_OP_نشر(name, op_insn)                                                                     \
    ساکن اندرونی خالی name(اعشاری * dst_بلاک, ثابت اعشاری * src0_بلاک, اعشاری scalar, صحیح elements) {               \
        ثابت int32_t vec_انتہا = (elements / 8) * 8;                                                                  \
        ثابت int32_t دم    = elements - vec_انتہا;                                                                  \
                                                                                                                     \
        غیرعلامتی طویل عارضی_نقاب;                                                                                     \
        __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));                                                           \
        __asm__ متلون("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        برائے (int32_t i = 0; i < vec_انتہا; i += 8) {                                                                   \
            __asm__ متلون(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])                                                           \
                : [s0] "m"(*(ثابت اعشاری (*)[8]) & src0_بلاک[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        اگر (دم > 0) {                                                                                              \
            ثابت غیرعلامتی طویل دم_m0 = (1ul << دم) - 1;                                                         \
            __asm__ متلون(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_بلاک[vec_انتہا]), [ps] "r"(&scalar), [d] "r"(&dst_بلاک[vec_انتہا]), [tm] "r"(دم_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));                                                            \
    }

DEFINE_بلاک_OP_نشر(بلاک_mul_نشر, "fmul.ps")
DEFINE_بلاک_OP_نشر(بلاک_جوڑو_نشر, "fadd.ps")
DEFINE_بلاک_OP_نشر(بلاک_sub_نشر, "fsub.ps")

ساکن اندرونی اعشاری scalar_el_map(اعشاری src0, اعشاری src1, شمار ggml_op operation) {
    انتخاب (operation) {
        حالت GGML_OP_MUL:
            واپس src0 * src1;
        حالت GGML_OP_جوڑو:
            واپس src0 + src1;
        حالت GGML_OP_SUB:
            واپس src0 - src1;
        طےشدہ:
            واپس 0.0f;
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_ثنائی_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;  // Invalid pointer
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * src1_ڈیٹا = (اعشاری *) src1->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

#اگر_تعریف_شدہ ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ثابت size_t src0_بائٹیں = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بائٹیں = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ڈیٹا, src0_بائٹیں);
    evict_region_past_l2(src1_ڈیٹا, src1_بائٹیں);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#اگر_ختم

    شمار ggml_op operation = dst->op;

    اگر (operation != GGML_OP_MUL && operation != GGML_OP_جوڑو && operation != GGML_OP_SUB) {
        واپس -1;  // Unsupported operation
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

    اگر (no_نشر && all_contiguous) {
        ثابت int64_t کل_elements         = ne0 * ne1 * ne2 * ne3;
        ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes
        ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

        ثابت int64_t cl_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
        ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
        int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
        اگر (cl_انتہا > کل_cachelines) {
            cl_انتہا = کل_cachelines;
        }

        اگر (cl_آغاز >= کل_cachelines) {
            واپس 0;
        }

        ثابت int64_t elem_آغاز = cl_آغاز * elements_فی_cacheline;
        int64_t       elem_انتہا   = cl_انتہا * elements_فی_cacheline;
        اگر (elem_انتہا > کل_elements) {
            elem_انتہا = کل_elements;
        }
        ثابت int32_t گنتی = (int32_t) (elem_انتہا - elem_آغاز);

        انتخاب (operation) {
            حالت GGML_OP_MUL:
                بلاک_mul_cache_aligned(dst_ڈیٹا + elem_آغاز, src0_ڈیٹا + elem_آغاز, src1_ڈیٹا + elem_آغاز, گنتی);
                توڑو;
            حالت GGML_OP_جوڑو:
                بلاک_جوڑو_cache_aligned(dst_ڈیٹا + elem_آغاز, src0_ڈیٹا + elem_آغاز, src1_ڈیٹا + elem_آغاز, گنتی);
                توڑو;
            حالت GGML_OP_SUB:
                بلاک_sub_cache_aligned(dst_ڈیٹا + elem_آغاز, src0_ڈیٹا + elem_آغاز, src1_ڈیٹا + elem_آغاز, گنتی);
                توڑو;
            طےشدہ:
                واپس 1;
        }
#اگر_تعریف_شدہ ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_ڈیٹا + elem_آغاز, (size_t) گنتی * حجم(اعشاری));
        WAIT_CACHEOPS;
        FENCE;
#اگر_ختم
        واپس 0;
    }

    // Slow path: broadcasting or non-contiguous
    ثابت int64_t کل_صفیں = ne1 * ne2 * ne3;

    int64_t آغاز_صف;
    int64_t انتہا_صف;

    اگر (cache_aligned) {
        ثابت int64_t صفیں_فی_thread = (کل_صفیں + عدد_دھاگے - 1) / عدد_دھاگے;
        آغاز_صف                     = thread_id * صفیں_فی_thread;
        انتہا_صف = (آغاز_صف + صفیں_فی_thread < کل_صفیں) ? (آغاز_صف + صفیں_فی_thread) : کل_صفیں;
    } ورنہ {
        ثابت int64_t صفیں_فی_گروہ = et_صفیں_فی_cacheline_گروہ(ne0, حجم(اعشاری));
        ثابت int64_t کل_groups   = (کل_صفیں + صفیں_فی_گروہ - 1) / صفیں_فی_گروہ;

        اگر (thread_id >= کل_groups) {
            واپس 0;
        }

        ثابت int64_t گروہ_آغاز = thread_id;
        برائے (int64_t grp = گروہ_آغاز; grp < کل_groups; grp += عدد_دھاگے) {
            ثابت int64_t گروہ_صف_آغاز = grp * صفیں_فی_گروہ;
            int64_t       گروہ_صف_انتہا   = گروہ_صف_آغاز + صفیں_فی_گروہ;
            اگر (گروہ_صف_انتہا > کل_صفیں) {
                گروہ_صف_انتہا = کل_صفیں;
            }

#اگر_تعریف_شدہ ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ثابت int64_t پہلا_i03      = گروہ_صف_آغاز / (ne2 * ne1);
            ثابت int64_t پہلا_i02      = (گروہ_صف_آغاز - پہلا_i03 * ne2 * ne1) / ne1;
            ثابت int64_t پہلا_i01      = (گروہ_صف_آغاز - پہلا_i03 * ne2 * ne1 - پہلا_i02 * ne1);
            حرف *        گروہ_dst_بنیاد = (حرف *) dst_ڈیٹا + پہلا_i03 * nb3 + پہلا_i02 * nb2 + پہلا_i01 * nb1;
#اگر_ختم

            برائے (int64_t ir = گروہ_صف_آغاز; ir < گروہ_صف_انتہا; ir++) {
                ثابت int64_t i03 = ir / (ne2 * ne1);
                ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ثابت int64_t i13 = i03 % ne13;
                ثابت int64_t i12 = i02 % ne12;
                ثابت int64_t i11 = i01 % ne11;

                اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ثابت اعشاری * src0_ptr =
                    (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ثابت اعشاری * src1_ptr =
                    (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i13 * nb13 + i12 * nb12 + i11 * nb11);

                اگر (ne10 == 1) {
                    ثابت اعشاری scalar = src1_ptr[0];
                    برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } ورنہ {
                    برائے (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#اگر_تعریف_شدہ ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ثابت int64_t nrows = گروہ_صف_انتہا - گروہ_صف_آغاز;
            اگر (nrows > 0) {
                FENCE;
                evict_region_past_l2(گروہ_dst_بنیاد, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#اگر_ختم
        }

        واپس 0;
    }

    اگر (آغاز_صف >= کل_صفیں) {
        واپس 0;
    }

    برائے (int64_t ir = آغاز_صف; ir < انتہا_صف; ir++) {
        // Convert flat row index to 3D coordinates
        ثابت int64_t i03 = ir / (ne2 * ne1);
        ثابت int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ثابت int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ثابت int64_t i13 = i03 % ne13;
        ثابت int64_t i12 = i02 % ne12;
        ثابت int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        اعشاری *       dst_ptr  = (اعشاری *) ((حرف *) dst_ڈیٹا + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ثابت اعشاری * src0_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ثابت اعشاری * src1_ptr = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i13 * nb13 + i12 * nb12 + i11 * nb11);

        اگر (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            اعشاری scalar = src1_ptr[0];
            انتخاب (operation) {
                حالت GGML_OP_MUL:
                    بلاک_mul_نشر(dst_ptr, src0_ptr, scalar, (صحیح) ne0);
                    توڑو;
                حالت GGML_OP_جوڑو:
                    بلاک_جوڑو_نشر(dst_ptr, src0_ptr, scalar, (صحیح) ne0);
                    توڑو;
                حالت GGML_OP_SUB:
                    بلاک_sub_نشر(dst_ptr, src0_ptr, scalar, (صحیح) ne0);
                    توڑو;
                طےشدہ:
                    واپس 1;
            }
        } ورنہ {
            // Broadcasting in dimension 0: src1 repeats across src0
            ثابت int64_t nr0 = ne0 / ne10;

            برائے (int64_t r = 0; r < nr0; r++) {
                ثابت اعشاری * src0_بلاک = src0_ptr + r * ne10;
                اعشاری *       dst_بلاک  = dst_ptr + r * ne10;

                انتخاب (operation) {
                    حالت GGML_OP_MUL:
                        بلاک_mul_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحیح) ne10);
                        توڑو;
                    حالت GGML_OP_جوڑو:
                        بلاک_جوڑو_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحیح) ne10);
                        توڑو;
                    حالت GGML_OP_SUB:
                        بلاک_sub_cache_aligned(dst_بلاک, src0_بلاک, src1_ptr, (صحیح) ne10);
                        توڑو;
                    طےشدہ:
                        واپس 1;
                }
            }
        }
    }

#اگر_تعریف_شدہ ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    اگر (انتہا_صف > آغاز_صف) {
        FENCE;
        evict_region_past_l2((حرف *) dst_ڈیٹا + آغاز_صف * nb1, (size_t) (انتہا_صف - آغاز_صف) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#اگر_ختم
    واپس 0;
}
