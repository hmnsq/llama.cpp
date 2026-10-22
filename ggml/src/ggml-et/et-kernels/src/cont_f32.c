//******************************************************************************
// Bare Metal CONT F32 Kernel
// Converts non-contiguous tensors to contiguous memory layout
//
// Fast path:    src contiguous: flat vectorized copy by cache lines
// Aligned path: nb00==4 and ne00 % 16 == 0: distribute rows, no coherency issue
// Unaligned:    nb00==4 and ne00 not aligned: distribute by cache lines,
//               reverse-compute src coords, handle partial rows at boundaries
// Fallback:     nb00 != 4: scalar per-element
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_cont_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor (non-contiguous)
    నిర్మాణం ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
స్థిర అంతఃవరుస శూన్య vec_copy_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int32_t n) {
    int32_t       i       = 0;
    స్థిరరాశి int32_t vec_చివర = (n / 8) * 8;
    కొరకు (; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(దశాంశం (*)[8]) & dst[i])
            : [s] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src[i])
            : "f10");
    }
    కొరకు (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
స్థిర అంతఃవరుస శూన్య scalar_copy_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int32_t n) {
    కొరకు (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_cont_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి int64_t nb00 = src0->nb[0];
    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];

    స్థిరరాశి int64_t మొత్తం_elements = ne00 * ne01 * ne02 * ne03;

    ఒకవేళ (మొత్తం_elements == 0) {
        తిరిగి 0;
    }

    స్థిరరాశి bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    ఒకవేళ (src_contiguous) {
        స్థిరరాశి int64_t elems_చొప్పున_cl = 16;
        స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

        స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
        int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
        ఒకవేళ (cl_చివర > మొత్తం_cl) {
            cl_చివర = మొత్తం_cl;
        }
        ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cl) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t es = cl_ప్రారంభం * elems_చొప్పున_cl;
        int64_t       ee = cl_చివర * elems_చొప్పున_cl;
        ఒకవేళ (ee > మొత్తం_elements) {
            ee = మొత్తం_elements;
        }

        vec_copy_f32(dst_దత్తాంశం + es, src0_దత్తాంశం + es, (int32_t) (ee - es));
        తిరిగి 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    ఒకవేళ (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        స్థిరరాశి int64_t elems_చొప్పున_cl = 16;
        స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

        స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
        int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
        ఒకవేళ (cl_చివర > మొత్తం_cl) {
            cl_చివర = మొత్తం_cl;
        }
        ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cl) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t es = cl_ప్రారంభం * elems_చొప్పున_cl;
        int64_t       ee = cl_చివర * elems_చొప్పున_cl;
        ఒకవేళ (ee > మొత్తం_elements) {
            ee = మొత్తం_elements;
        }

        కొరకు (int64_t idx = es; idx < ee; idx++) {
            స్థిరరాశి int64_t i00  = idx % ne00;
            స్థిరరాశి int64_t rem1 = idx / ne00;
            స్థిరరాశి int64_t i01  = rem1 % ne01;
            స్థిరరాశి int64_t rem2 = rem1 / ne01;
            స్థిరరాశి int64_t i02  = rem2 % ne02;
            స్థిరరాశి int64_t i03  = rem2 / ne02;

            స్థిరరాశి దశాంశం * sp =
                (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_దత్తాంశం[idx] = *sp;
        }
        తిరిగి 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    ఒకవేళ (ne00 % 16 == 0) {
        స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు      = ne01 * ne02 * ne03;
        స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_thread = (మొత్తం_అడ్డువరుసలు + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస       = thread_id * అడ్డువరుసలు_చొప్పున_thread;
        స్థిరరాశి int64_t చివర_అడ్డువరుస = (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread < మొత్తం_అడ్డువరుసలు) ? (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread) : మొత్తం_అడ్డువరుసలు;

        ఒకవేళ (ప్రారంభం_అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
            తిరిగి 0;
        }

        కొరకు (int64_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ir++) {
            స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
            స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            స్థిరరాశి int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);
            దశాంశం *       dst_అడ్డువరుస = dst_దత్తాంశం + ir * ne00;

            vec_copy_f32(dst_అడ్డువరుస, src_అడ్డువరుస, (int32_t) ne00);
        }
        తిరిగి 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        స్థిరరాశి int64_t elems_చొప్పున_cl = 16;
        స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

        స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
        int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
        ఒకవేళ (cl_చివర > మొత్తం_cl) {
            cl_చివర = మొత్తం_cl;
        }
        ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cl) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t es = cl_ప్రారంభం * elems_చొప్పున_cl;
        int64_t       ee = cl_చివర * elems_చొప్పున_cl;
        ఒకవేళ (ee > మొత్తం_elements) {
            ee = మొత్తం_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t అడ్డువరుస_idx = pos / ne00;
        int64_t నిలువు     = pos % ne00;

        అయితే (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            స్థిరరాశి int64_t i03 = అడ్డువరుస_idx / (ne02 * ne01);
            స్థిరరాశి int64_t i02 = (అడ్డువరుస_idx - i03 * ne02 * ne01) / ne01;
            స్థిరరాశి int64_t i01 = అడ్డువరుస_idx - i03 * ne02 * ne01 - i02 * ne01;

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t అడ్డువరుస_మిగిలిన   = ne00 - నిలువు;
            int64_t chunk_మిగిలిన = ee - pos;
            int32_t n               = (int32_t) (అడ్డువరుస_మిగిలిన < chunk_మిగిలిన ? అడ్డువరుస_మిగిలిన : chunk_మిగిలిన);

            vec_copy_f32(dst_దత్తాంశం + pos, src_అడ్డువరుస + నిలువు, n);

            pos += n;
            నిలువు = 0;  // subsequent rows start at column 0
            అడ్డువరుస_idx++;
        }
    }

    తిరిగి 0;
}
