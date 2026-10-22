#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#如果 defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#如果 defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(整数 /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        返回 true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        如果 (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            返回 true;
        }
        返回 false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    返回 &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    返回 (void *) (buffer->context);
}

static enum ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    返回 GGML_STATUS_SUCCESS;
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    如果 (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } 否则 {
        memcpy((char *) tensor->data + offset, data, size);
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

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    返回 "AMX";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    如果 (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        返回 NULL;
    }

    返回 ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    返回 TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        如果 (op->op != GGML_OP_MUL_MAT) {
            返回 false;
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        如果 (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            返回 false;
        }
        如果 (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            返回 false;
        }
        如果 (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            返回 false;
        }
        如果 (op->ne[0] % (TILE_N * 2)) {
            返回 false;
        }
        整数 alignment;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
                alignment = TILE_K;
                break;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                break;
            case GGML_TYPE_F16:
                alignment = 16;
                break;
            default:
                返回 false;
        }
        如果 (src0->ne[0] % alignment) {
            返回 false;
        }
        如果 (src1->type != GGML_TYPE_F32) {
            返回 false;
        }
        返回 true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        如果 (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            返回 (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        返回 nullptr;
    }
};
}  // namespace ggml::cpu::amx

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    返回 ggml_backend_amx_get_alloc_size(tensor);

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool ggml_amx_init() {
#如果 defined(__linux__)
    如果 (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        返回 false;
    }
    返回 true;
#elif defined(_WIN32)
    返回 true;
#否则
    返回 false;
#endif
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
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

    如果 (!ggml_amx_init()) {
        返回 nullptr;
    }

    返回 &ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
