#ಅನ್ತರ್ಭಾವ "amx.h"
#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "mmq.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "traits.h"

#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/syscall.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#endif

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>

#ಯದಿ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
ನಾಮಾಕಾಶ ggml::cpu::amx {
ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
    ತಾರ್ಕಿಕ work_size(ಪೂರ್ಣಾಙ್ಕ /* n_threads */, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ compute_forward(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * op) override {
        ಯದಿ (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
};

ಸ್ಥಿರ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, ಸಂರಚನಾ ggml_tensor *) {
    ಸ್ಥಿರ tensor_traits traits;
    ಪ್ರತ್ಯಾಗಮ &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ಮುಕ್ತ(buffer->context);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    ಪ್ರತ್ಯಾಗಮ (ಶೂನ್ಯ *) (buffer->context);
}

ಸ್ಥಿರ ಗಣನಾ ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor) {
    tensor->extra = (ಶೂನ್ಯ *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    ಪ್ರತ್ಯಾಗಮ GGML_STATUS_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((ವರ್ಣ *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, ಸಂರಚನಾ ggml_tensor * tensor,
                                               ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size) {
    ಯದಿ (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } ಅನ್ಯಥಾ {
        memcpy((ವರ್ಣ *) tensor->data + offset, data, size);
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

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

ಸ್ಥಿರ ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ "AMX";

    GGML_UNUSED(buft);
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ಶೂನ್ಯ * data = ggml_aligned_malloc(size);
    ಯದಿ (data == ರಿಕ್ತ) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

ಸ್ಥಿರ size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ಪ್ರತ್ಯಾಗಮ TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

ನಾಮಾಕಾಶ ggml::cpu::amx {
ವರ್ಗ extra_buffer_type : ggml::cpu::extra_buffer_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) override {
        ಯದಿ (op->op != GGML_OP_MUL_MAT) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        ಯದಿ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (op->ne[0] % (TILE_N * 2)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪೂರ್ಣಾಙ್ಕ alignment;
        ವಿಕಲ್ಪ (src0->type) {
            ಪಕ್ಷ GGML_TYPE_Q4_0:
            ಪಕ್ಷ GGML_TYPE_Q4_1:
            ಪಕ್ಷ GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_Q4_K:
            ಪಕ್ಷ GGML_TYPE_Q5_K:
            ಪಕ್ಷ GGML_TYPE_Q6_K:
            ಪಕ್ಷ GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ಭಙ್ಗ;
            ಪಕ್ಷ GGML_TYPE_F16:
                alignment = 16;
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src0->ne[0] % alignment) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (src1->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) override {
        ಯದಿ (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            ಪ್ರತ್ಯಾಗಮ (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
};
}  // namespace ggml::cpu::amx

ಸ್ಥಿರ size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ಧ್ರುವ ggml_tensor * tensor) {
    ಪ್ರತ್ಯಾಗಮ ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_amx_init() {
#ಯದಿ defined(__linux__)
    ಯದಿ (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#ನೋಚೇದ್ಯದಿ defined(_WIN32)
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    ಸ್ಥಿರ ಸಂರಚನಾ ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ ರಿಕ್ತ,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ನೂತನ ggml::cpu::amx::extra_buffer_type(),
    };

    ಯದಿ (!ggml_amx_init()) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
