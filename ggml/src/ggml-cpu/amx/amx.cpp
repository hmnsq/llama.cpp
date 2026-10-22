#અન્તર્ભાવ "amx.h"
#અન્તર્ભાવ "common.h"
#અન્તર્ભાવ "mmq.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "traits.h"

#જો defined(__linux__)
#અન્તર્ભાવ <sys/syscall.h>
#અન્તર્ભાવ <unistd.h>
#endif

#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <memory>

#જો defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
નામાકાશ ggml::cpu::amx {
વર્ગ tensor_traits : જાહેર ggml::cpu::tensor_traits {
    તાર્કિક work_size(પૂર્ણાંક /* n_threads */, અચલ સંરચના ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        પાછા સાચું;
    }

    તાર્કિક compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * op) override {
        જો (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            પાછા સાચું;
        }
        પાછા ખોટું;
    }
};

સ્થિર ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, સંરચના ggml_tensor *) {
    સ્થિર tensor_traits traits;
    પાછા &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
સ્થિર શૂન્ય ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    મુક્ત(buffer->context);
}

સ્થિર શૂન્ય * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    પાછા (શૂન્ય *) (buffer->context);
}

સ્થિર ગણના ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor) {
    tensor->extra = (શૂન્ય *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((અક્ષર *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

સ્થિર શૂન્ય ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor,
                                               અચલ શૂન્ય * data, size_t offset, size_t size) {
    જો (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } નહીંતર {
        memcpy((અક્ષર *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
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

સ્થિર શૂન્ય ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

સ્થિર ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

સ્થિર અચલ અક્ષર * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા "AMX";

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    શૂન્ય * data = ggml_aligned_malloc(size);
    જો (data == ખાલી) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        પાછા ખાલી;
    }

    પાછા ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

સ્થિર size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

નામાકાશ ggml::cpu::amx {
વર્ગ extra_buffer_type : ggml::cpu::extra_buffer_type {
    તાર્કિક supports_op(ggml_backend_dev_t, અચલ સંરચના ggml_tensor * op) override {
        જો (op->op != GGML_OP_MUL_MAT) {
            પાછા ખોટું;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        જો (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            પાછા ખોટું;
        }
        જો (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            પાછા ખોટું;
        }
        જો (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            પાછા ખોટું;
        }
        જો (op->ne[0] % (TILE_N * 2)) {
            પાછા ખોટું;
        }
        પૂર્ણાંક alignment;
        પસંદગી (src0->type) {
            સ્થિતિ GGML_TYPE_Q4_0:
            સ્થિતિ GGML_TYPE_Q4_1:
            સ્થિતિ GGML_TYPE_Q8_0:
                alignment = TILE_K;
                તોડો;
            સ્થિતિ GGML_TYPE_Q4_K:
            સ્થિતિ GGML_TYPE_Q5_K:
            સ્થિતિ GGML_TYPE_Q6_K:
            સ્થિતિ GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                તોડો;
            સ્થિતિ GGML_TYPE_F16:
                alignment = 16;
                તોડો;
            મૂળભૂત:
                પાછા ખોટું;
        }
        જો (src0->ne[0] % alignment) {
            પાછા ખોટું;
        }
        જો (src1->type != GGML_TYPE_F32) {
            પાછા ખોટું;
        }
        પાછા સાચું;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(અચલ સંરચના ggml_tensor * op) override {
        જો (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            પાછા (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        પાછા nullptr;
    }
};
}  // namespace ggml::cpu::amx

સ્થિર size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    પાછા ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

સ્થિર તાર્કિક ggml_amx_init() {
#જો defined(__linux__)
    જો (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        પાછા ખોટું;
    }
    પાછા સાચું;
#અથવાજો defined(_WIN32)
    પાછા સાચું;
#નહીંતર
    પાછા ખોટું;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ ખાલી,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ નવું ggml::cpu::amx::extra_buffer_type(),
    };

    જો (!ggml_amx_init()) {
        પાછા nullptr;
    }

    પાછા &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
