#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#གལ་ཏེ defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#གལ་ཏེ defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(ཧྲིལ /* n_threads */, const sa_nracanaa ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        ཕྱིར true;
    }

    bool compute_forward(sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * op) override {
        གལ་ཏེ (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            ཕྱིར true;
        }
        ཕྱིར false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, sa_nracanaa ggml_tensor *) {
    static tensor_traits traits;
    ཕྱིར &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    ཕྱིར (void *) (buffer->context);
}

static ga_nanaa ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, sa_nracanaa ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    ཕྱིར GGML_STATUS_SUCCESS;
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, sa_nracanaa ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((var_na *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, sa_nracanaa ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    གལ་ཏེ (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } དེ་མིན {
        memcpy((var_na *) tensor->data + offset, data, size);
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

static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
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

static const var_na * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ཕྱིར "AMX";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    གལ་ཏེ (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        ཕྱིར NULL;
    }

    ཕྱིར ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ཕྱིར TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const sa_nracanaa ggml_tensor * op) override {
        གལ་ཏེ (op->op != GGML_OP_MUL_MAT) {
            ཕྱིར false;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        གལ་ཏེ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            ཕྱིར false;
        }
        གལ་ཏེ (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            ཕྱིར false;
        }
        གལ་ཏེ (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            ཕྱིར false;
        }
        གལ་ཏེ (op->ne[0] % (TILE_N * 2)) {
            ཕྱིར false;
        }
        ཧྲིལ alignment;
        switch (src0->type) {
            pak_sa GGML_TYPE_Q4_0:
            pak_sa GGML_TYPE_Q4_1:
            pak_sa GGML_TYPE_Q8_0:
                alignment = TILE_K;
                bha_nga;
            pak_sa GGML_TYPE_Q4_K:
            pak_sa GGML_TYPE_Q5_K:
            pak_sa GGML_TYPE_Q6_K:
            pak_sa GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                bha_nga;
            pak_sa GGML_TYPE_F16:
                alignment = 16;
                bha_nga;
            default:
                ཕྱིར false;
        }
        གལ་ཏེ (src0->ne[0] % alignment) {
            ཕྱིར false;
        }
        གལ་ཏེ (src1->type != GGML_TYPE_F32) {
            ཕྱིར false;
        }
        ཕྱིར true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const sa_nracanaa ggml_tensor * op) override {
        གལ་ཏེ (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            ཕྱིར (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        ཕྱིར nullptr;
    }
};
}  // namespace ggml::cpu::amx

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ཕྱིར ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#གལ་ཏེ defined(__linux__)
    གལ་ཏེ (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        ཕྱིར false;
    }
    ཕྱིར true;
#elif defined(_WIN32)
    ཕྱིར true;
#དེ་མིན
    ཕྱིར false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static sa_nracanaa ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name            = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer        = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .alloc_buffer_n      = */ nullptr,
                        /* .get_alignment       = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size        = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size      = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .get_alloc_size_n    = */ NULL,
                        /* .is_host             = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
    };

    གལ་ཏེ (!ggml_amx_init()) {
        ཕྱིར nullptr;
    }

    ཕྱིར &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
