#ಅನ್ತರ್ಭಾವ "amx.h"
#ಅನ್ತರ್ಭಾವ "common.h"
#ಅನ್ತರ್ಭಾವ "mmq.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "traits.h"

#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/syscall.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <memory>

#ಆದರೆ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
ನಾಮಾಕಾಶ ggml::cpu::amx {
ವರ್ಗ tensor_traits : ಸಾರ್ವಜನಿಕ ggml::cpu::tensor_traits {
    ತಾರ್ಕಿಕ work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ /* n_threads */, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ತಾರ್ಕಿಕ compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * op) override {
        ಆದರೆ (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
};

ಸ್ಥಿರ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_ಬಫರ್_t, ರಚನೆ ggml_tensor *) {
    ಸ್ಥಿರ tensor_traits traits;
    ಹಿಂತಿರುಗಿ &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಬಿಡುಗಡೆ(ಬಫರ್->context);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_amx_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) (ಬಫರ್->context);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_amx_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor) {
    tensor->extra = (ಶೂನ್ಯ *) ggml::cpu::amx::get_tensor_traits(ಬಫರ್, tensor);

    GGML_ಬಳಸದ(ಬಫರ್);
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor,
                                                  uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    memset((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, value, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor,
                                               ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    ಆದರೆ (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_ತೂಕ(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
    }

    GGML_ಬಳಸದ(ಬಫರ್);
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

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_amx_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    memset(ಬಫರ್->context, value, ಬಫರ್->size);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_i ggml_backend_amx_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_amx_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_amx_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_amx_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .set_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .get_tensor_2d   = */ ಶೂನ್ಯಸೂಚಕ,
    /* .cpy_tensor      = */ ಶೂನ್ಯಸೂಚಕ,
    /* .clear           = */ ggml_backend_amx_ಬಫರ್_clear,
    /* .reset           = */ ಶೂನ್ಯಸೂಚಕ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_amx_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "AMX";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_amx_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ = ggml_aligned_malloc(size);
    ಆದರೆ (ದತ್ತಾಂಶ == ಖಾಲಿ) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_amx_ಬಫರ್_interface, ದತ್ತಾಂಶ, size);
}

ಸ್ಥಿರ size_t ggml_backend_amx_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ TENSOR_ALIGNMENT;

    GGML_ಬಳಸದ(buft);
}

ನಾಮಾಕಾಶ ggml::cpu::amx {
ವರ್ಗ extra_ಬಫರ್_type : ggml::cpu::extra_ಬಫರ್_type {
    ತಾರ್ಕಿಕ supports_op(ggml_backend_dev_t, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) override {
        ಆದರೆ (op->op != GGML_OP_MUL_MAT) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ವಯಂಚಾಲಿತ * src0 = op->src[0];
        ಸ್ವಯಂಚಾಲಿತ * src1 = op->src[1];

        ಆದರೆ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!src0->ಬಫರ್ || src0->ಬಫರ್->buft != ggml_backend_amx_ಬಫರ್_type()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src1->ಬಫರ್ && !ggml_backend_buft_is_host(src1->ಬಫರ್->buft)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (op->ne[0] % (TILE_N * 2)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಪೂರ್ಣಾಂಕ alignment;
        ಆಯ್ಕೆ (src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TYPE_F16:
                alignment = 16;
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src0->ne[0] % alignment) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (src1->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) override {
        ಆದರೆ (op->op == GGML_OP_MUL_MAT && op->src[0]->ಬಫರ್ &&
            op->src[0]->ಬಫರ್->buft == ggml_backend_amx_ಬಫರ್_type()) {
            ಹಿಂತಿರುಗಿ (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
};
}  // namespace ggml::cpu::amx

ಸ್ಥಿರ size_t ggml_backend_amx_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_backend_amx_get_alloc_ಗಾತ್ರ(tensor);

    GGML_ಬಳಸದ(buft);
}

#ವ್ಯಾಖ್ಯಾನಿಸು ARCH_GET_XCOMP_PERM     0x1022
#ವ್ಯಾಖ್ಯಾನಿಸು ARCH_REQ_XCOMP_PERM     0x1023
#ವ್ಯಾಖ್ಯಾನಿಸು XFEATURE_XTILECFG       17
#ವ್ಯಾಖ್ಯಾನಿಸು XFEATURE_XTILEDATA      18

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_amx_init() {
#ಆದರೆ defined(__linux__)
    ಆದರೆ (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಅಥವಾಆದರೆ defined(_WIN32)
    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
}

ggml_backend_ಬಫರ್_type_t ggml_backend_amx_ಬಫರ್_type() {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_ಬಫರ್_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_ಬಫರ್_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_ಬಫರ್_type_alloc_ಬಫರ್,
                        /* .alloc_buffer_n      = */ ಶೂನ್ಯಸೂಚಕ,
                        /* .get_alignment       = */ ggml_backend_amx_ಬಫರ್_type_get_alignment,
                        /* .get_max_size        = */ ಶೂನ್ಯಸೂಚಕ,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
                        /* .get_alloc_size_n    = */ ಖಾಲಿ,
                        /* .is_host             = */ ಶೂನ್ಯಸೂಚಕ,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ಹೊಸ ggml::cpu::amx::extra_ಬಫರ್_type(),
    };

    ಆದರೆ (!ggml_amx_init()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ &ggml_backend_ಬಫರ್_type_amx;
}

#ಆದರೆ_ಅಂತ್ಯ  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
