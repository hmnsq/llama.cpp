#అన్తర్భావ "amx.h"
#అన్తర్భావ "common.h"
#అన్తర్భావ "mmq.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "traits.h"

#ఒకవేళ defined(__linux__)
#అన్తర్భావ <sys/syscall.h>
#అన్తర్భావ <unistd.h>
#అయితే_ముగింపు

#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <memory>

#ఒకవేళ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
నామాకాశ ggml::cpu::amx {
తరగతి tensor_traits : బహిరంగ ggml::cpu::tensor_traits {
    తార్కిక work_కొలత(పూర్ణాంకం /* n_threads */, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        తిరిగి నిజం;
    }

    తార్కిక compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    }
};

స్థిర ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_బఫరు_t, నిర్మాణం ggml_tensor *) {
    స్థిర tensor_traits traits;
    తిరిగి &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
స్థిర శూన్య ggml_backend_amx_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    విడుదల(బఫరు->context);
}

స్థిర శూన్య * ggml_backend_amx_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి (శూన్య *) (బఫరు->context);
}

స్థిర గణన ggml_స్థితి ggml_backend_amx_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor) {
    tensor->extra = (శూన్య *) ggml::cpu::amx::get_tensor_traits(బఫరు, tensor);

    GGML_వాడని(బఫరు);
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_amx_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor,
                                                  uint8_t value, size_t స్థానభ్రంశం, size_t size) {
    memset((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, value, size);

    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_amx_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor,
                                               స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    ఒకవేళ (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_బరువు(tensor, దత్తాంశం, స్థానభ్రంశం, size);
    } లేకపోతే {
        జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);
    }

    GGML_వాడని(బఫరు);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}
*/

స్థిర శూన్య ggml_backend_amx_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    memset(బఫరు->context, value, బఫరు->size);
}

స్థిర ggml_backend_బఫరు_i ggml_backend_amx_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_amx_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_amx_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_amx_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ శూన్యసూచిక,
    /* .set_tensor_2d   = */ శూన్యసూచిక,
    /* .get_tensor_2d   = */ శూన్యసూచిక,
    /* .cpy_tensor      = */ శూన్యసూచిక,
    /* .clear           = */ ggml_backend_amx_బఫరు_clear,
    /* .reset           = */ శూన్యసూచిక,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_amx_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "AMX";

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_amx_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    శూన్య * దత్తాంశం = ggml_aligned_malloc(size);
    ఒకవేళ (దత్తాంశం == ఖాళీ) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        తిరిగి ఖాళీ;
    }

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_amx_బఫరు_interface, దత్తాంశం, size);
}

స్థిర size_t ggml_backend_amx_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి TENSOR_ALIGNMENT;

    GGML_వాడని(buft);
}

నామాకాశ ggml::cpu::amx {
తరగతి extra_బఫరు_type : ggml::cpu::extra_బఫరు_type {
    తార్కిక supports_op(ggml_backend_dev_t, స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (op->op != GGML_OP_MUL_MAT) {
            తిరిగి అబద్ధం;
        }
        స్వయంచాలక * src0 = op->src[0];
        స్వయంచాలక * src1 = op->src[1];

        ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!src0->బఫరు || src0->బఫరు->buft != ggml_backend_amx_బఫరు_type()) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src1->బఫరు && !ggml_backend_buft_is_host(src1->బఫరు->buft)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (op->ne[0] % (TILE_N * 2)) {
            తిరిగి అబద్ధం;
        }
        పూర్ణాంకం alignment;
        ఎంపిక (src0->type) {
            స్థితి GGML_TYPE_Q4_0:
            స్థితి GGML_TYPE_Q4_1:
            స్థితి GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ఆపు;
            స్థితి GGML_TYPE_Q4_K:
            స్థితి GGML_TYPE_Q5_K:
            స్థితి GGML_TYPE_Q6_K:
            స్థితి GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ఆపు;
            స్థితి GGML_TYPE_F16:
                alignment = 16;
                ఆపు;
            అప్రమేయ:
                తిరిగి అబద్ధం;
        }
        ఒకవేళ (src0->ne[0] % alignment) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src1->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        తిరిగి నిజం;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (op->op == GGML_OP_MUL_MAT && op->src[0]->బఫరు &&
            op->src[0]->బఫరు->buft == ggml_backend_amx_బఫరు_type()) {
            తిరిగి (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        తిరిగి శూన్యసూచిక;
    }
};
}  // namespace ggml::cpu::amx

స్థిర size_t ggml_backend_amx_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    తిరిగి ggml_backend_amx_get_alloc_కొలత(tensor);

    GGML_వాడని(buft);
}

#నిర్వచించు ARCH_GET_XCOMP_PERM     0x1022
#నిర్వచించు ARCH_REQ_XCOMP_PERM     0x1023
#నిర్వచించు XFEATURE_XTILECFG       17
#నిర్వచించు XFEATURE_XTILEDATA      18

స్థిర తార్కిక ggml_amx_init() {
#ఒకవేళ defined(__linux__)
    ఒకవేళ (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
#లేదావేళ defined(_WIN32)
    తిరిగి నిజం;
#లేకపోతే
    తిరిగి అబద్ధం;
#అయితే_ముగింపు
}

ggml_backend_బఫరు_type_t ggml_backend_amx_బఫరు_type() {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_బఫరు_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_బఫరు_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_బఫరు_type_alloc_బఫరు,
                        /* .alloc_buffer_n      = */ శూన్యసూచిక,
                        /* .get_alignment       = */ ggml_backend_amx_బఫరు_type_get_alignment,
                        /* .get_max_size        = */ శూన్యసూచిక,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_బఫరు_type_get_alloc_కొలత,
                        /* .get_alloc_size_n    = */ ఖాళీ,
                        /* .is_host             = */ శూన్యసూచిక,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ కొత్త ggml::cpu::amx::extra_బఫరు_type(),
    };

    ఒకవేళ (!ggml_amx_init()) {
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి &ggml_backend_బఫరు_type_amx;
}

#అయితే_ముగింపు  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
