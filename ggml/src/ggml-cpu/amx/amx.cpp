#ਅਨ੍ਤਰ੍ਭਾਵ "amx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "mmq.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"

#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/syscall.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>

#ਜੇ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
ਨਾਮਾਕਾਸ਼ ggml::cpu::amx {
ਵਰਗ tensor_traits : ਜਨਤਕ ggml::cpu::tensor_traits {
    ਤਾਰਕਿਕ work_ਮਾਪ(ਪੂਰਨ /* n_threads */, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ਵਾਪਸ ਸੱਚ;
    }

    ਤਾਰਕਿਕ compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ਵਾਪਸ ਸੱਚ;
        }
        ਵਾਪਸ ਝੂਠ;
    }
};

ਸਥਿਰ ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_ਬਫਰ_t, ਬਣਤਰ ggml_tensor *) {
    ਸਥਿਰ tensor_traits traits;
    ਵਾਪਸ &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
ਸਥਿਰ ਖਾਲੀ ggml_backend_amx_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਮੁਕਤ(ਬਫਰ->context);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_amx_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ (ਖਾਲੀ *) (ਬਫਰ->context);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_amx_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor) {
    tensor->extra = (ਖਾਲੀ *) ggml::cpu::amx::get_tensor_traits(ਬਫਰ, tensor);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_amx_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor,
                                                  uint8_t value, size_t ਵਿੱਥ, size_t size) {
    memset((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, value, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_amx_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor,
                                               ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    ਜੇ (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_ਭਾਰ(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    } ਨਹੀਂਤਾਂ {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);
    }

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
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

ਸਥਿਰ ਖਾਲੀ ggml_backend_amx_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    memset(ਬਫਰ->context, value, ਬਫਰ->size);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_i ggml_backend_amx_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_amx_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_amx_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_amx_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .set_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .get_tensor_2d   = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .cpy_tensor      = */ ਖਾਲੀ_ਸੂਚਕ,
    /* .clear           = */ ggml_backend_amx_ਬਫਰ_clear,
    /* .reset           = */ ਖਾਲੀ_ਸੂਚਕ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_amx_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "AMX";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_amx_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਖਾਲੀ * ਅੰਕੜੇ = ggml_aligned_malloc(size);
    ਜੇ (ਅੰਕੜੇ == ਨਿਰਾ) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_amx_ਬਫਰ_interface, ਅੰਕੜੇ, size);
}

ਸਥਿਰ size_t ggml_backend_amx_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ TENSOR_ALIGNMENT;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::amx {
ਵਰਗ extra_ਬਫਰ_type : ggml::cpu::extra_ਬਫਰ_type {
    ਤਾਰਕਿਕ supports_op(ggml_backend_dev_t, ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (op->op != GGML_OP_MUL_MAT) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਸਵੈਚਾਲਿਤ * src0 = op->src[0];
        ਸਵੈਚਾਲਿਤ * src1 = op->src[1];

        ਜੇ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!src0->ਬਫਰ || src0->ਬਫਰ->buft != ggml_backend_amx_ਬਫਰ_type()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src1->ਬਫਰ && !ggml_backend_buft_is_host(src1->ਬਫਰ->buft)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (op->ne[0] % (TILE_N * 2)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਪੂਰਨ alignment;
        ਚੋਣ (src0->type) {
            ਹਾਲਤ GGML_TYPE_Q4_0:
            ਹਾਲਤ GGML_TYPE_Q4_1:
            ਹਾਲਤ GGML_TYPE_Q8_0:
                alignment = TILE_K;
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_Q4_K:
            ਹਾਲਤ GGML_TYPE_Q5_K:
            ਹਾਲਤ GGML_TYPE_Q6_K:
            ਹਾਲਤ GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                ਰੋਕੋ;
            ਹਾਲਤ GGML_TYPE_F16:
                alignment = 16;
                ਰੋਕੋ;
            ਮੂਲ:
                ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src0->ne[0] % alignment) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (src1->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਸੱਚ;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(ਅਚਲ ਬਣਤਰ ggml_tensor * op) override {
        ਜੇ (op->op == GGML_OP_MUL_MAT && op->src[0]->ਬਫਰ &&
            op->src[0]->ਬਫਰ->buft == ggml_backend_amx_ਬਫਰ_type()) {
            ਵਾਪਸ (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
};
}  // namespace ggml::cpu::amx

ਸਥਿਰ size_t ggml_backend_amx_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_backend_amx_get_alloc_ਮਾਪ(tensor);

    GGML_ਅਣਵਰਤਿਆ(buft);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ARCH_GET_XCOMP_PERM     0x1022
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ARCH_REQ_XCOMP_PERM     0x1023
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ XFEATURE_XTILECFG       17
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ XFEATURE_XTILEDATA      18

ਸਥਿਰ ਤਾਰਕਿਕ ggml_amx_init() {
#ਜੇ defined(__linux__)
    ਜੇ (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;
#ਜਾਂਜੇ defined(_WIN32)
    ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
}

ggml_backend_ਬਫਰ_type_t ggml_backend_amx_ਬਫਰ_type() {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_ਬਫਰ_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_ਬਫਰ_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_ਬਫਰ_type_alloc_ਬਫਰ,
                        /* .alloc_buffer_n      = */ ਖਾਲੀ_ਸੂਚਕ,
                        /* .get_alignment       = */ ggml_backend_amx_ਬਫਰ_type_get_alignment,
                        /* .get_max_size        = */ ਖਾਲੀ_ਸੂਚਕ,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_ਬਫਰ_type_get_alloc_ਮਾਪ,
                        /* .get_alloc_size_n    = */ ਨਿਰਾ,
                        /* .is_host             = */ ਖਾਲੀ_ਸੂਚਕ,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ਨਵਾਂ ggml::cpu::amx::extra_ਬਫਰ_type(),
    };

    ਜੇ (!ggml_amx_init()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ &ggml_backend_ਬਫਰ_type_amx;
}

#ਜੇ_ਅੰਤ  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
