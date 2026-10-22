// Note: porting this file to C++ is a work in progress

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WIN32_LEAN_AND_ਔਸਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NOMINMAX
#   define NOMINMAX
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-alloc.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <unordered_map>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __APPLE__
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/sysctl.h>
#ਜੇ_ਅੰਤ


// backend buffer type

ਅਚਲ ਅੱਖਰ * ggml_backend_buft_name(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(buft);
    ਵਾਪਸ buft->iface.get_name(buft);
}

ggml_backend_ਬਫਰ_t ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    ਜੇ (size == 0) {
        // return a dummy buffer for zero-sized allocations
        ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, {}, ਨਿਰਾ, 0);
    }
    ਵਾਪਸ buft->iface.alloc_ਬਫਰ(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ਬਣਤਰ ggml_backend_buft_alloc_ਬਫਰ_n_plan_ਵਸਤੂ {
    size_t size;  // total bytes for this buffer
    ਪੂਰਨ    ਪਹਿਲਾ; // first tensor index (inclusive)
    ਪੂਰਨ    ਆਖਰੀ;  // last tensor index (exclusive)
};

ਵਰਤੋਂ ggml_backend_buft_alloc_ਬਫਰ_n_plan_t = std::vector<ggml_backend_buft_alloc_ਬਫਰ_n_plan_ਵਸਤੂ>;

ਸਥਿਰ ggml_backend_buft_alloc_ਬਫਰ_n_plan_t ggml_backend_buft_alloc_ਬਫਰ_n_plan(
        ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ggml_backend_buft_alloc_ਬਫਰ_n_plan_t plan;

    ਅਚਲ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ਅਚਲ size_t max_size  = ggml_backend_buft_get_max_ਮਾਪ(buft);

    size_t cur_buf_ਮਾਪ = 0;
    ਪੂਰਨ    ਪਹਿਲਾ        = 0;

    ਲਈ (ਪੂਰਨ i = 0; i < n_tensors; i++) {
        size_t this_ਮਾਪ = 0;
        ਬਣਤਰ ggml_tensor * t = tensors[i];
        ਜੇ (t->ਅੰਕੜੇ == ਨਿਰਾ && t->view_src == ਨਿਰਾ) {
            this_ਮਾਪ = GGML_PAD(ggml_backend_buft_get_alloc_ਮਾਪ(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        ਜੇ (cur_buf_ਮਾਪ > 0 && (cur_buf_ਮਾਪ + this_ਮਾਪ) > max_size) {
            plan.push_back({ cur_buf_ਮਾਪ, ਪਹਿਲਾ, i });
            cur_buf_ਮਾਪ = this_ਮਾਪ;
            ਪਹਿਲਾ        = i;
        } ਨਹੀਂਤਾਂ {
            cur_buf_ਮਾਪ += this_ਮਾਪ;
        }
    }

    ਜੇ (cur_buf_ਮਾਪ > 0) {
        plan.push_back({ cur_buf_ਮਾਪ, ਪਹਿਲਾ, n_tensors });
    }

    ਵਾਪਸ plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_buft_alloc_ਬਫਰ_n_default(ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ਅਚਲ ggml_backend_buft_alloc_ਬਫਰ_n_plan_t plan = ggml_backend_buft_alloc_ਬਫਰ_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_ਬਫਰ_t> buffers;
    buffers.reserve(plan.size());

    ਲਈ (ਅਚਲ ggml_backend_buft_alloc_ਬਫਰ_n_plan_ਵਸਤੂ & ਵਸਤੂ : plan) {
        ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_buft_alloc_ਬਫਰ(buft, ਵਸਤੂ.size);
        ਜੇ (ਬਫਰ == ਨਿਰਾ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), ਵਸਤੂ.size);
            ਲਈ (ggml_backend_ਬਫਰ_t b : buffers) {
                ggml_backend_ਬਫਰ_free(b);
            }
            ਵਾਪਸ ਨਿਰਾ;
        }

        ਬਣਤਰ ggml_tallocr tallocr = ggml_tallocr_new(ਬਫਰ);

        // allocate tensors in the current buffer
        ਬਣਤਰ ggml_tensor * t_failed = ਨਿਰਾ;
        ਲਈ (ਪੂਰਨ j = ਵਸਤੂ.ਪਹਿਲਾ; j < ਵਸਤੂ.ਆਖਰੀ; j++) {
            ਬਣਤਰ ggml_tensor * t = tensors[j];
            ਜੇ (t->ਅੰਕੜੇ == ਨਿਰਾ) {
                ਜੇ (t->view_src == ਨਿਰਾ) {
                    ਜੇ (ggml_tallocr_alloc(&tallocr, t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (t->ਬਫਰ == ਨਿਰਾ) {
                    ਜੇ (ggml_backend_view_init(t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (t->view_src != ਨਿਰਾ && t->ਬਫਰ == ਨਿਰਾ) {
                    // view of a pre-allocated tensor
                    ਜੇ (ggml_backend_view_init(t) != GGML_ਸਥਿਤੀ_SUCCESS) {
                        t_failed = t;
                        ਰੋਕੋ;
                    }
                }
            }
        }
        ਜੇ (t_failed != ਨਿਰਾ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ਲਈ (ggml_backend_ਬਫਰ_t b : buffers) {
                ggml_backend_ਬਫਰ_free(b);
            }
            ggml_backend_ਬਫਰ_free(ਬਫਰ);
            ਵਾਪਸ ਨਿਰਾ;
        }

        buffers.push_back(ਬਫਰ);
    }

    ਜੇ (buffers.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਜੇ (buffers.size() == 1) {
        ਵਾਪਸ buffers[0];
    }

    ਵਾਪਸ ggml_backend_multi_ਬਫਰ_alloc_ਬਫਰ(buffers.ਅੰਕੜੇ(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ਸਥਿਰ size_t ggml_backend_buft_get_alloc_ਮਾਪ_n_default(ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    ਅਚਲ ggml_backend_buft_alloc_ਬਫਰ_n_plan_t plan = ggml_backend_buft_alloc_ਬਫਰ_n_plan(buft, tensors, n_tensors);

    size_t ਕੁੱਲ = 0;
    ਲਈ (ਅਚਲ ggml_backend_buft_alloc_ਬਫਰ_n_plan_ਵਸਤੂ & ਵਸਤੂ : plan) {
        ਕੁੱਲ += ਵਸਤੂ.size;
    }
    ਵਾਪਸ ਕੁੱਲ;
}

ggml_backend_ਬਫਰ_t ggml_backend_buft_alloc_ਬਫਰ_n(ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    GGML_ASSERT(buft);
    ਜੇ (buft->iface.alloc_ਬਫਰ_n) {
        ਵਾਪਸ buft->iface.alloc_ਬਫਰ_n(buft, tensors, n_tensors);
    }
    ਵਾਪਸ ggml_backend_buft_alloc_ਬਫਰ_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(buft);
    ਵਾਪਸ buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    ਜੇ (buft->iface.get_max_ਮਾਪ) {
        ਵਾਪਸ buft->iface.get_max_ਮਾਪ(buft);
    }
    ਵਾਪਸ SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    ਜੇ (buft->iface.get_alloc_ਮਾਪ) {
        size_t size = buft->iface.get_alloc_ਮਾਪ(buft, tensor);
        ਪ੍ਰਤਿਜ੍ਞਾ(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_ਮਾਪ_may_expand(tensor->op));

        ਵਾਪਸ size;
    }
    ਵਾਪਸ ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_ਮਾਪ_n(ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors) {
    GGML_ASSERT(buft);
    ਜੇ (buft->iface.get_alloc_ਮਾਪ_n) {
        ਵਾਪਸ buft->iface.get_alloc_ਮਾਪ_n(buft, tensors, n_tensors);
    }
    ਵਾਪਸ ggml_backend_buft_get_alloc_ਮਾਪ_n_default(buft, tensors, n_tensors);
}

ਤਾਰਕਿਕ ggml_backend_buft_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(buft);
    ਜੇ (buft->iface.is_host) {
        ਵਾਪਸ buft->iface.is_host(buft);
    }
    ਵਾਪਸ ਝੂਠ;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(buft);
    ਵਾਪਸ buft->device;
}

// backend buffer

ggml_backend_ਬਫਰ_t ggml_backend_ਬਫਰ_init(
               ggml_backend_ਬਫਰ_type_t buft,
        ਬਣਤਰ ggml_backend_ਬਫਰ_i      iface,
               ਖਾਲੀ *                     context,
               size_t                     size) {
    ggml_backend_ਬਫਰ_t ਬਫਰ = ਨਵਾਂ ggml_backend_ਬਫਰ {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_ਬਫਰ_USAGE_ANY
    };

    ਵਾਪਸ ਬਫਰ;
}

ਅਚਲ ਅੱਖਰ * ggml_backend_ਬਫਰ_name(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ggml_backend_buft_name(ggml_backend_ਬਫਰ_get_type(ਬਫਰ));
}

ਖਾਲੀ ggml_backend_ਬਫਰ_free(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਜੇ (ਬਫਰ == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    ਜੇ (ਬਫਰ->iface.free_ਬਫਰ != ਨਿਰਾ) {
        ਬਫਰ->iface.free_ਬਫਰ(ਬਫਰ);
    }
    ਮਿਟਾਓ ਬਫਰ;
}

size_t ggml_backend_ਬਫਰ_get_ਮਾਪ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ਵਾਪਸ ਬਫਰ->size;
}

ਖਾਲੀ * ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    // get_base is optional if the buffer is zero-sized
    ਜੇ (!ggml_backend_ਬਫਰ_is_meta(ਬਫਰ) && ਬਫਰ->size == 0) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    ਜੇ (!ਬਫਰ->iface.get_ਅਧਾਰ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਖਾਲੀ * ਅਧਾਰ = ਬਫਰ->iface.get_ਅਧਾਰ(ਬਫਰ);

    GGML_ASSERT(ਅਧਾਰ != ਨਿਰਾ && "backend buffer base cannot be NULL");

    ਵਾਪਸ ਅਧਾਰ;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(ਬਫਰ);
    // init_tensor is optional
    ਜੇ (ਬਫਰ->iface.init_tensor) {
        ਵਾਪਸ ਬਫਰ->iface.init_tensor(ਬਫਰ, tensor);
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਖਾਲੀ ggml_backend_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    GGML_ASSERT(ਬਫਰ);
    // clear is optional if the buffer is zero-sized
    ਜੇ (ਬਫਰ->size == 0) {
        ਵਾਪਸ;
    }

    ਬਫਰ->iface.clear(ਬਫਰ, value);
}

size_t ggml_backend_ਬਫਰ_get_alignment(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ggml_backend_buft_get_alignment(ggml_backend_ਬਫਰ_get_type(ਬਫਰ));
}

size_t ggml_backend_ਬਫਰ_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ggml_backend_buft_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_get_type(ਬਫਰ));
}

size_t ggml_backend_ਬਫਰ_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਵਾਪਸ ggml_backend_buft_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_get_type(ਬਫਰ), tensor);
}

ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_host(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ggml_backend_buft_is_host(ggml_backend_ਬਫਰ_get_type(ਬਫਰ));
}

ਖਾਲੀ ggml_backend_ਬਫਰ_ਸੈੱਟ_usage(ggml_backend_ਬਫਰ_t ਬਫਰ, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage) {
    GGML_ASSERT(ਬਫਰ);
    ਬਫਰ->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    ਜੇ (ggml_backend_ਬਫਰ_is_multi_ਬਫਰ(ਬਫਰ)) {
        ggml_backend_multi_ਬਫਰ_ਸੈੱਟ_usage(ਬਫਰ, usage);
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_backend_ਬਫਰ_is_meta(ਬਫਰ)) {
        ggml_backend_meta_ਬਫਰ_ਸੈੱਟ_usage(ਬਫਰ, usage);
    }
}

ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage ggml_backend_ਬਫਰ_get_usage(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ਵਾਪਸ ਬਫਰ->usage;
}

ggml_backend_ਬਫਰ_type_t ggml_backend_ਬਫਰ_get_type(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ਵਾਪਸ ਬਫਰ->buft;
}

ਖਾਲੀ ggml_backend_ਬਫਰ_reset(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ਜੇ (ਬਫਰ->iface.reset) {
        ਬਫਰ->iface.reset(ਬਫਰ);
    }
}

ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_copy_tensor(ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst) {
    ggml_backend_ਬਫਰ_t dst_buf = dst->view_src ? dst->view_src->ਬਫਰ : dst->ਬਫਰ;
    ਜੇ (dst_buf->iface.cpy_tensor) {
        ਵਾਪਸ dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    ਵਾਪਸ ਝੂਠ;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    ਜੇ (backend == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ backend->guid;
}

ਅਚਲ ਅੱਖਰ * ggml_backend_name(ggml_backend_t backend) {
    ਜੇ (backend == ਨਿਰਾ) {
        ਵਾਪਸ "NULL";
    }
    ਵਾਪਸ backend->iface.get_name(backend);
}

ਖਾਲੀ ggml_backend_free(ggml_backend_t backend) {
    ਜੇ (backend == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    backend->iface.ਮੁਕਤ(backend);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_get_default_ਬਫਰ_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ਵਾਪਸ ggml_backend_dev_ਬਫਰ_type(backend->device);
}

ggml_backend_ਬਫਰ_t ggml_backend_alloc_ਬਫਰ(ggml_backend_t backend, size_t size) {
    ਵਾਪਸ ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_get_default_ਬਫਰ_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ਵਾਪਸ ggml_backend_buft_get_alignment(ggml_backend_get_default_ਬਫਰ_type(backend));
}

size_t ggml_backend_get_max_ਮਾਪ(ggml_backend_t backend) {
    ਵਾਪਸ ggml_backend_buft_get_max_ਮਾਪ(ggml_backend_get_default_ਬਫਰ_type(backend));
}

ਖਾਲੀ ggml_backend_tensor_ਸੈੱਟ_async(ggml_backend_t backend, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ਜੇ (backend->iface.ਸੈੱਟ_tensor_async == ਨਿਰਾ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_ਸੈੱਟ(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    } ਨਹੀਂਤਾਂ {
        backend->iface.ਸੈੱਟ_tensor_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    }
}

ਖਾਲੀ ggml_backend_tensor_get_async(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    ਜੇ (backend->iface.get_tensor_async == ਨਿਰਾ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    } ਨਹੀਂਤਾਂ {
        backend->iface.get_tensor_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    }
}

ਖਾਲੀ ggml_backend_tensor_ਸੈੱਟ_2d_async(ggml_backend_t backend, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");

    ਜੇ (n_copies <= 1 || backend->iface.ਸੈੱਟ_tensor_2d_async == ਨਿਰਾ) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_ਸੈੱਟ_async(backend, tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, ਵਿੱਥ + i*stride_tensor, size);
        }
        ਵਾਪਸ;
    }
    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.ਸੈੱਟ_tensor_2d_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਖਾਲੀ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");

    ਜੇ (n_copies <= 1 || backend->iface.get_tensor_2d_async == ਨਿਰਾ) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, ਵਿੱਥ + i*stride_tensor, size);
        }
        ਵਾਪਸ;
    }
    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਖਾਲੀ ggml_backend_tensor_ਸੈੱਟ(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
    GGML_ASSERT(buf != ਨਿਰਾ && "tensor buffer not set");

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.ਸੈੱਟ_tensor(buf, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਖਾਲੀ ggml_backend_tensor_get(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
    GGML_ASSERT(buf != ਨਿਰਾ && "tensor buffer not set");

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
}

ਖਾਲੀ ggml_backend_tensor_ਸੈੱਟ_2d(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    GGML_ASSERT(tensor);
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
    GGML_ASSERT(buf != ਨਿਰਾ && "tensor buffer not set");

    ਜੇ (n_copies <= 1 || buf->iface.ਸੈੱਟ_tensor_2d == ਨਿਰਾ) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_ਸੈੱਟ(tensor, (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, ਵਿੱਥ + i*stride_tensor, size);
        }
        ਵਾਪਸ;
    }
    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.ਸੈੱਟ_tensor_2d(buf, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਖਾਲੀ ggml_backend_tensor_get_2d(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ) {
    GGML_ASSERT(tensor);
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
    GGML_ASSERT(buf != ਨਿਰਾ && "tensor buffer not set");

    ਜੇ (n_copies <= 1 || buf->iface.get_tensor_2d == ਨਿਰਾ) {
        ਲਈ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (ਅੱਖਰ *) ਅੰਕੜੇ + i*stride_ਅੰਕੜੇ, ਵਿੱਥ + i*stride_tensor, size);
        }
        ਵਾਪਸ;
    }
    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, ਅੰਕੜੇ, ਵਿੱਥ, size, n_copies, stride_tensor, stride_ਅੰਕੜੇ);
}

ਖਾਲੀ ggml_backend_tensor_memset(ਬਣਤਰ ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;

    ਜੇ (size == 0) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(buf != ਨਿਰਾ && "tensor buffer not set");
    GGML_ASSERT(tensor->ਅੰਕੜੇ != ਨਿਰਾ && "tensor not allocated");
    GGML_ASSERT(ਵਿੱਥ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ਨਿਰਾ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, ਵਿੱਥ, size);
}

ਖਾਲੀ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ਜੇ (backend->iface.synchronize == ਨਿਰਾ) {
        ਵਾਪਸ;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_ਗ੍ਰਾਫ_plan_t ggml_backend_ਗ੍ਰਾਫ_plan_create(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ਗ੍ਰਾਫ_plan_create != ਨਿਰਾ);

    ਵਾਪਸ backend->iface.ਗ੍ਰਾਫ_plan_create(backend, cgraph);
}

ਖਾਲੀ ggml_backend_ਗ੍ਰਾਫ_plan_free(ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ਗ੍ਰਾਫ_plan_free != ਨਿਰਾ);

    backend->iface.ਗ੍ਰਾਫ_plan_free(backend, plan);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_ਗ੍ਰਾਫ_plan_compute(ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ਗ੍ਰਾਫ_plan_compute != ਨਿਰਾ);

    ਵਾਪਸ backend->iface.ਗ੍ਰਾਫ_plan_compute(backend, plan);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    ਗਿਣਤੀ ggml_ਸਥਿਤੀ err = ggml_backend_ਗ੍ਰਾਫ_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    ਵਾਪਸ err;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_ਗ੍ਰਾਫ_compute_async(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    ਵਾਪਸ backend->iface.ਗ੍ਰਾਫ_compute(backend, cgraph);
}

ਤਾਰਕਿਕ ggml_backend_supports_op(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ਵਾਪਸ ggml_backend_dev_supports_op(backend->device, op);
}

ਤਾਰਕਿਕ ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(backend);
    ਵਾਪਸ ggml_backend_dev_supports_buft(backend->device, buft);
}

ਤਾਰਕਿਕ ggml_backend_offload_op(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ਵਾਪਸ ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ਵਾਪਸ backend->device;
}

// backend copy

ਖਾਲੀ ggml_backend_tensor_copy(ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ਜੇ (src == dst) {
        ਵਾਪਸ;
    }

    ਜੇ (ggml_backend_ਬਫਰ_is_host(src->ਬਫਰ)) {
        ggml_backend_tensor_ਸੈੱਟ(dst, src->ਅੰਕੜੇ, 0, ggml_nbytes(src));
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_backend_ਬਫਰ_is_host(dst->ਬਫਰ)) {
        ggml_backend_tensor_get(src, dst->ਅੰਕੜੇ, 0, ggml_nbytes(src));
    } ਨਹੀਂਤਾਂ ਜੇ (!ggml_backend_ਬਫਰ_copy_tensor(src, dst)) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_ਬਫਰ_name(src->ਬਫਰ), ggml_backend_ਬਫਰ_name(dst->ਬਫਰ));
#ਜੇ_ਅੰਤ // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        ਖਾਲੀ * ਅੰਕੜੇ = ਸਮ੍ਰਿਤੀ_ਦਿਓ(nbytes);
        ggml_backend_tensor_get(src, ਅੰਕੜੇ, 0, nbytes);
        ggml_backend_tensor_ਸੈੱਟ(dst, ਅੰਕੜੇ, 0, nbytes);
        ਮੁਕਤ(ਅੰਕੜੇ);
    }
}

ਖਾਲੀ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ਜੇ (src == dst) {
        ਵਾਪਸ;
    }

    GGML_ASSERT(backend_dst);
    ਜੇ (backend_dst->iface.cpy_tensor_async != ਨਿਰਾ) {
        ਜੇ (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            ਵਾਪਸ;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    ਜੇ (device == ਨਿਰਾ || device->iface.event_new == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ device->iface.event_new(device);
}

ਖਾਲੀ ggml_backend_event_free(ggml_backend_event_t event) {
    ਜੇ (event == ਨਿਰਾ) {
        ਵਾਪਸ;
    }
    event->device->iface.event_free(event->device, event);
}

ਖਾਲੀ ggml_backend_event_ਰਿਕਾਰਡ(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_ਰਿਕਾਰਡ != ਨਿਰਾ);

    backend->iface.event_ਰਿਕਾਰਡ(backend, event);
}

ਖਾਲੀ ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

ਖਾਲੀ ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ਨਿਰਾ);

    backend->iface.event_wait(backend, event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_ਗ੍ਰਾਫ_optimize(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params * params) {
    GGML_ASSERT(backend);
    ਜੇ (backend->iface.ਗ੍ਰਾਫ_optimize != ਨਿਰਾ) {
        backend->iface.ਗ੍ਰਾਫ_optimize(backend, cgraph, params);
    }
}

// Backend device

ਅਚਲ ਅੱਖਰ * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.get_name(device);
}

ਅਚਲ ਅੱਖਰ * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.get_description(device);
}

ਖਾਲੀ ggml_backend_dev_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    GGML_ASSERT(device);
    device->iface.get_ਸਮ੍ਰਿਤੀ(device, ਮੁਕਤ, ਕੁੱਲ);
}

ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.get_type(device);
}

ਖਾਲੀ ggml_backend_dev_get_props(ggml_backend_dev_t device, ਬਣਤਰ ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, ਆਕਾਰ(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ਅਚਲ ਅੱਖਰ * params) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.init_backend(device, params);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_dev_ਬਫਰ_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.get_ਬਫਰ_type(device);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_dev_host_ਬਫਰ_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ਜੇ (device->iface.get_host_ਬਫਰ_type == ਨਿਰਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ device->iface.get_host_ਬਫਰ_type(device);
}

ggml_backend_ਬਫਰ_t ggml_backend_dev_ਬਫਰ_from_host_ptr(ggml_backend_dev_t device, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.ਬਫਰ_from_host_ptr(device, ptr, size, max_tensor_ਮਾਪ);
}

ਤਾਰਕਿਕ ggml_backend_dev_supports_op(ggml_backend_dev_t device, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.supports_op(device, op);
}

ਤਾਰਕਿਕ ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ASSERT(device);
    ਵਾਪਸ device->iface.supports_buft(device, buft);
}

ਤਾਰਕਿਕ ggml_backend_dev_offload_op(ggml_backend_dev_t device, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    GGML_ASSERT(device);
    ਜੇ (device->iface.offload_op != ਨਿਰਾ) {
        ਵਾਪਸ device->iface.offload_op(device, op);
    }

    ਵਾਪਸ ਝੂਠ;
}

// Backend (reg)

ਅਚਲ ਅੱਖਰ * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ਵਾਪਸ reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ਵਾਪਸ reg->iface.get_device_ਗਿਣਤੀ(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(reg);
    ਵਾਪਸ reg->iface.get_device(reg, ਸੂਚਕਾਂਕ);
}

ਖਾਲੀ * ggml_backend_reg_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    GGML_ASSERT(reg);
    ਜੇ (!reg->iface.get_proc_ਪਤਾ) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ reg->iface.get_proc_ਪਤਾ(reg, name);
}

// multi-buffer buffer

ਬਣਤਰ ggml_backend_multi_ਬਫਰ_context {
    ggml_backend_ਬਫਰ_t * buffers;
    size_t n_buffers;
};

ਸਥਿਰ ਖਾਲੀ ggml_backend_multi_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ggml_backend_multi_ਬਫਰ_context * ctx = (ggml_backend_multi_ਬਫਰ_context *) ਬਫਰ->context;
    ਲਈ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ਬਫਰ_free(ctx->buffers[i]);
    }

    ਮੁਕਤ(ctx->buffers);
    ਮੁਕਤ(ctx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_multi_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    GGML_ASSERT(ਬਫਰ);
    ggml_backend_multi_ਬਫਰ_context * ctx = (ggml_backend_multi_ਬਫਰ_context *) ਬਫਰ->context;
    ਲਈ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ਬਫਰ_clear(ctx->buffers[i], value);
    }
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_i ggml_backend_multi_ਬਫਰ_i = {
    /* .free_buffer     = */ ggml_backend_multi_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ਨਿਰਾ,
    /* .init_tensor     = */ ਨਿਰਾ,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ਨਿਰਾ,
    /* .get_tensor      = */ ਨਿਰਾ,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ਨਿਰਾ,
    /* .clear           = */ ggml_backend_multi_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

ggml_backend_ਬਫਰ_t ggml_backend_multi_ਬਫਰ_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_t * buffers, size_t n_buffers) {
    ggml_backend_multi_ਬਫਰ_context * ctx = (ggml_backend_multi_ਬਫਰ_context *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(ਬਣਤਰ ggml_backend_multi_ਬਫਰ_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_ਬਫਰ_t *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(n_buffers * ਆਕਾਰ(ggml_backend_ਬਫਰ_t));

    GGML_ASSERT(ctx->buffers != ਨਿਰਾ);

    size_t ਕੁੱਲ_ਮਾਪ = 0;
    ਲਈ (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        ਕੁੱਲ_ਮਾਪ += ggml_backend_ਬਫਰ_get_ਮਾਪ(buffers[i]);
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buffers[0]->buft, ggml_backend_multi_ਬਫਰ_i, ctx, ਕੁੱਲ_ਮਾਪ);
}

ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_multi_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ਵਾਪਸ ਬਫਰ->iface.free_ਬਫਰ == ggml_backend_multi_ਬਫਰ_free_ਬਫਰ;
}

ਖਾਲੀ ggml_backend_multi_ਬਫਰ_ਸੈੱਟ_usage(ggml_backend_ਬਫਰ_t ਬਫਰ, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage) {
    GGML_ASSERT(ਬਫਰ);
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_multi_ਬਫਰ(ਬਫਰ));
    ggml_backend_multi_ਬਫਰ_context * ctx = (ggml_backend_multi_ਬਫਰ_context *) ਬਫਰ->context;
    ਲਈ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ਬਫਰ_ਸੈੱਟ_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ਸਥਿਰ ਬਣਤਰ ggml_tensor * ggml_dup_tensor_layout(ਬਣਤਰ ggml_context * ctx, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor) {
    ਬਣਤਰ ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    ਵਾਪਸ dup;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_is_view_op(ਗਿਣਤੀ ggml_op op) {
    ਵਾਪਸ op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SCHED_MAX_BACKENDS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SCHED_MAX_BACKENDS 16
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SCHED_MAX_ਵੰਡੋ_INPUTS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SCHED_MAX_ਵੰਡੋ_INPUTS 30
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SCHED_MAX_COPIES
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SCHED_MAX_COPIES 4
#ਜੇ_ਅੰਤ

ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ {
    ਪੂਰਨ backend_id;
    ਪੂਰਨ i_ਸ਼ੁਰੂ;
    ਪੂਰਨ i_ਸਿਰਾ;
    ਬਣਤਰ ggml_tensor ** inputs;
    ਪੂਰਨ n_inputs;
    ਪੂਰਨ inputs_ਸਮਰੱਥਾ;
    // graph view of this split
    ਬਣਤਰ ggml_cgraph graph;
};

ਬਣਤਰ ggml_backend_sched {
    ਤਾਰਕਿਕ is_reset; // true if the scheduler has been reset since the last graph split
    ਤਾਰਕਿਕ is_alloc;

    ਪੂਰਨ n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_ਬਫਰ_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ਬਣਤਰ ggml_hash_ਸੈੱਟ  hash_ਸੈੱਟ;
    ਪੂਰਨ                 * hv_tensor_backend_ids; // [hash_set.size]
    ਬਣਤਰ ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    ਪੂਰਨ * ਗੰਢ_backend_ids; // [graph_size]
    ਪੂਰਨ * ਪੱਤੀ_backend_ids; // [graph_size]

    ਪੂਰਨ * prev_ਗੰਢ_backend_ids; // [graph_size]
    ਪੂਰਨ * prev_ਪੱਤੀ_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ਬਣਤਰ ggml_cgraph graph;

    // graph splits
    ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * splits;
    ਪੂਰਨ n_splits;
    ਪੂਰਨ splits_ਸਮਰੱਥਾ;

    // pipeline parallelism support
    ਪੂਰਨ n_copies;
    ਪੂਰਨ cur_copy;
    ਪੂਰਨ ਅਗਲਾ_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ਬਣਤਰ ggml_tensor ** ਗ੍ਰਾਫ_inputs;
    ਪੂਰਨ n_ਗ੍ਰਾਫ_inputs;
    ਪੂਰਨ ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ;

    ਬਣਤਰ ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    ਖਾਲੀ * callback_eval_user_ਅੰਕੜੇ;

    ggml_backend_sched_copy_callback callback_copy;
    ਖਾਲੀ * callback_copy_user_ਅੰਕੜੇ;

    ggml_backend_sched_realloc_callback callback_realloc;
    ਖਾਲੀ * callback_realloc_user_ਅੰਕੜੇ;

    ਅੱਖਰ * context_ਬਫਰ;
    size_t context_ਬਫਰ_ਮਾਪ;

    ਤਾਰਕਿਕ op_offload;

    ਪੂਰਨ debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    ਪੂਰਨ ਗ੍ਰਾਫ_ਮਾਪ;
    ਪੂਰਨ prev_ਗ੍ਰਾਫ_ਮਾਪ;

    ਪੂਰਨ debug_realloc;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ hash_id(tensor) ggml_hash_ਲੱਭੋ_or_ਪਾਓ(&sched->hash_ਸੈੱਟ, tensor)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ਸਥਿਰ ਖਾਲੀ ggml_backend_sched_ਵੰਡੋ_inputs_grow(ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ) {
    ਪੂਰਨ new_cap = GGML_SCHED_MAX_ਵੰਡੋ_INPUTS;
    ਜੇ (ਵੰਡੋ->inputs_ਸਮਰੱਥਾ > 0) {
        new_cap = 2*ਵੰਡੋ->inputs_ਸਮਰੱਥਾ;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, ਵੰਡੋ->inputs_ਸਮਰੱਥਾ, new_cap);
    }
    ਸਵੈਚਾਲਿਤ * pnew = (ਬਣਤਰ ggml_tensor **) realloc((ਖਾਲੀ *) ਵੰਡੋ->inputs, new_cap * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਜੇ (pnew == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    ਵੰਡੋ->inputs = pnew;
    ਵੰਡੋ->inputs_ਸਮਰੱਥਾ = new_cap;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sched_ਗ੍ਰਾਫ_inputs_grow(ggml_backend_sched_t sched) {
    ਪੂਰਨ new_cap = GGML_SCHED_MAX_ਵੰਡੋ_INPUTS;
    ਜੇ (sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ > 0) {
        new_cap = 2*sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ, new_cap);
    }
    ਸਵੈਚਾਲਿਤ * pnew = (ਬਣਤਰ ggml_tensor **) realloc((ਖਾਲੀ *) sched->ਗ੍ਰਾਫ_inputs, new_cap * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
    ਜੇ (pnew == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->ਗ੍ਰਾਫ_inputs = pnew;
    sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ਸਥਿਰ ਪੂਰਨ ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_backends; i++) {
        ਜੇ (sched->backends[i] == backend) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ -1;
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_sched_backend_from_ਬਫਰ(ggml_backend_sched_t sched, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਬਣਤਰ ggml_tensor * op) {
    ggml_backend_ਬਫਰ_t ਬਫਰ = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
    ਜੇ (ਬਫਰ == ਨਿਰਾ) {
        ਵਾਪਸ -1;
    }

    // find highest prio backend that supports the buffer type and the op
    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_backends; i++) {
        ਜੇ (ggml_backend_supports_buft(sched->backends[i], ਬਫਰ->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            ਵਾਪਸ i;
        }
    }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_ਬਫਰ_name(ਬਫਰ), tensor->name);
#ਜੇ_ਅੰਤ

    ਵਾਪਸ -1;
}

#ਜੇ 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SCHED_MAX_SPLITS_DEBUG 4096
ਸਥਿਰ ਅੱਖਰ causes[GGML_DEFAULT_ਗ੍ਰਾਫ_ਮਾਪ*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_ਵੰਡੋ_INPUTS][128]; // debug only
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸੈੱਟ_CAUSE(ਗੰਢ, ...) sprintf(causes[hash_id(ਗੰਢ)], __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GET_CAUSE(ਗੰਢ) causes[hash_id(ਗੰਢ)]
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸੈੱਟ_CAUSE(ਗੰਢ, ...)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GET_CAUSE(ਗੰਢ) ""
#ਜੇ_ਅੰਤ

// returns the backend that should be used for the node based on the current locations
ਸਥਿਰ ਪੂਰਨ ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ਬਣਤਰ ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    ਪੂਰਨ cur_backend_id = ggml_backend_sched_backend_from_ਬਫਰ(sched, tensor, tensor);
    ਜੇ (cur_backend_id != -1) {
        ਸੈੱਟ_CAUSE(tensor, "1.dst");
        ਵਾਪਸ cur_backend_id;
    }

    // view_src
    ਜੇ (tensor->view_src != ਨਿਰਾ) {
        cur_backend_id = ggml_backend_sched_backend_from_ਬਫਰ(sched, tensor->view_src, tensor);
        ਜੇ (cur_backend_id != -1) {
            ਸੈੱਟ_CAUSE(tensor, "1.vsrc");
            ਵਾਪਸ cur_backend_id;
        }
    }

    ਜੇ (tensor->ਬਫਰ || (tensor->view_src && tensor->view_src->ਬਫਰ)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_ਬਫਰ_t ਬਫਰ = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_ਬਫਰ_name(ਬਫਰ), ggml_op_name(tensor->op));
    }

    // graph input
    ਜੇ (tensor->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        ਸੈੱਟ_CAUSE(tensor, "1.inp");
        ਵਾਪਸ cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    ਤਾਰਕਿਕ allow = ਸੱਚ;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    ਜੇ (allow) {
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
            ਅਚਲ ਬਣਤਰ ggml_tensor * src = tensor->src[i];
            ਜੇ (src == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ਜੇ (src->ਬਫਰ != ਨਿਰਾ && src->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
                ਪੂਰਨ src_backend_id = ggml_backend_sched_backend_from_ਬਫਰ(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                ਜੇ (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_ਬਫਰ_is_host(src->ਬਫਰ)) {
                    ਲਈ (ਪੂਰਨ b = 0; b < src_backend_id; b++) {
                        ਜੇ (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            ਸੈੱਟ_CAUSE(tensor, "1.off");
                            ਵਾਪਸ b;
                        }
                    }
                }
                ਸੈੱਟ_CAUSE(tensor, "1.wgt%d", i);
                ਵਾਪਸ src_backend_id;
            }
        }
    }

    ਵਾਪਸ -1;
}

ਸਥਿਰ ਅੱਖਰ * fmt_ਮਾਪ(size_t size) {
    ਸਥਿਰ ਅੱਖਰ ਬਫਰ[128];
    ਜੇ (size >= 1024*1024) {
        snprintf(ਬਫਰ, ਆਕਾਰ(ਬਫਰ), "%zuM", size/1024/1024);
    } ਨਹੀਂਤਾਂ {
        snprintf(ਬਫਰ, ਆਕਾਰ(ਬਫਰ), "%zuK", size/1024);
    }
    ਵਾਪਸ ਬਫਰ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * graph) {
    ਪੂਰਨ cur_ਵੰਡੋ = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਜੇ (cur_ਵੰਡੋ < sched->n_splits && i == sched->splits[cur_ਵੰਡੋ].i_ਸ਼ੁਰੂ) {
            ggml_backend_t ਵੰਡੋ_backend = sched->backends[sched->splits[cur_ਵੰਡੋ].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_ਵੰਡੋ, ggml_backend_name(ਵੰਡੋ_backend),
                sched->splits[cur_ਵੰਡੋ].n_inputs);
            ਲਈ (ਪੂਰਨ j = 0; j < sched->splits[cur_ਵੰਡੋ].n_inputs; j++) {
                ਜੇ (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_ਵੰਡੋ].inputs[j]->name,
                    fmt_ਮਾਪ(ggml_nbytes(sched->splits[cur_ਵੰਡੋ].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_ਵੰਡੋ++;
        }
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
            ਜਾਰੀ;
        }
        ਜੇ (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, ਗੰਢ);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(ਗੰਢ), ਗੰਢ->name,
                fmt_ਮਾਪ(ggml_nbytes(ਗੰਢ)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(ਗੰਢ),
                graph->use_counts[ggml_hash_ਲੱਭੋ(&graph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ)], ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                ਜੇ (src == ਨਿਰਾ) {
                    ਜਾਰੀ;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_ਮਾਪ(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sched_ਬਫਰ_supported(ggml_backend_sched_t sched, ਬਣਤਰ ggml_tensor * t, ਪੂਰਨ backend_id) {
    ggml_backend_ਬਫਰ_t buf = t->view_src ? t->view_src->ਬਫਰ : t->ਬਫਰ;
    ggml_backend_ਬਫਰ_type_t buft = ਨਿਰਾ;

    ਜੇ (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } ਨਹੀਂਤਾਂ {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        ਪੂਰਨ tensor_backend_id = tensor_backend_id(t);
        ਜੇ (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        ਜੇ (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    ਵਾਪਸ buft != ਨਿਰਾ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sched_ਸੈੱਟ_if_supported(ggml_backend_sched_t sched, ਬਣਤਰ ggml_tensor * ਗੰਢ, ਪੂਰਨ cur_backend_id, ਪੂਰਨ * ਗੰਢ_backend_id) {
    ਜੇ (ggml_backend_supports_op(sched->backends[cur_backend_id], ਗੰਢ)) {
        *ਗੰਢ_backend_id = cur_backend_id;
        ਸੈੱਟ_CAUSE(ਗੰਢ, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
ਖਾਲੀ ggml_backend_sched_ਵੰਡੋ_ਗ੍ਰਾਫ(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_ਗ੍ਰਾਫ_inputs = 0;
    sched->is_reset = ਝੂਠ;

    ਬਣਤਰ ggml_init_params params = {
        /* .mem_size =   */ sched->context_ਬਫਰ_ਮਾਪ,
        /* .mem_buffer = */ sched->context_ਬਫਰ,
        /* .no_alloc =   */ ਸੱਚ
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    ਜੇ (sched->ctx == ਨਿਰਾ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_ਗ੍ਰਾਫ_ਅਗਲਾ_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = graph->leafs[i];
        ਪੂਰਨ * ਪੱਤੀ_backend_id = &tensor_backend_id(ਪੱਤੀ);
        // do not overwrite user assignments
        ਜੇ (*ਪੱਤੀ_backend_id == -1) {
            *ਪੱਤੀ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ਪੱਤੀ);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
        // do not overwrite user assignments
        ਜੇ (*ਗੰਢ_backend_id == -1) {
            *ਗੰਢ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ਗੰਢ);

#ਜੇ 0
            // src
            ਜੇ (ਗੰਢ->op == GGML_OP_NONE) {
                ਜਾਰੀ;
            }

            ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                ਜੇ (src == ਨਿਰਾ) {
                    ਜਾਰੀ;
                }
                ਪੂਰਨ * src_backend_id = &tensor_backend_id(src);
                ਜੇ (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#ਜੇ_ਅੰਤ
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        ਪੂਰਨ cur_backend_id = -1;
        ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
            ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
            ਜੇ (*ਗੰਢ_backend_id != -1) {
                ਜੇ (*ਗੰਢ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ਨਹੀਂਤਾਂ {
                    cur_backend_id = *ਗੰਢ_backend_id;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (cur_backend_id != -1) {
                ggml_backend_sched_ਸੈੱਟ_if_supported(sched, ਗੰਢ, cur_backend_id, ਗੰਢ_backend_id);
            }
        }
    }
    // expand gpu up
    {
        ਪੂਰਨ cur_backend_id = -1;
        ਲਈ (ਪੂਰਨ i = graph->n_nodes - 1; i >= 0; i--) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
            ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
            ਜੇ (*ਗੰਢ_backend_id != -1) {
                ਜੇ (*ਗੰਢ_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ਨਹੀਂਤਾਂ {
                    cur_backend_id = *ਗੰਢ_backend_id;
                }
            } ਨਹੀਂਤਾਂ ਜੇ (cur_backend_id != -1) {
                ggml_backend_sched_ਸੈੱਟ_if_supported(sched, ਗੰਢ, cur_backend_id, ਗੰਢ_backend_id);
            }
        }
    }
    // expand rest down
    {
        ਪੂਰਨ cur_backend_id = -1;
        ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
            ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
            ਜੇ (*ਗੰਢ_backend_id != -1) {
                cur_backend_id = *ਗੰਢ_backend_id;
            } ਨਹੀਂਤਾਂ ਜੇ (cur_backend_id != -1) {
                ggml_backend_sched_ਸੈੱਟ_if_supported(sched, ਗੰਢ, cur_backend_id, ਗੰਢ_backend_id);
            }
        }
    }
    // expand rest up
    {
        ਪੂਰਨ cur_backend_id = -1;
        ਲਈ (ਪੂਰਨ i = graph->n_nodes - 1; i >= 0; i--) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
            ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
            ਜੇ (*ਗੰਢ_backend_id != -1) {
                cur_backend_id = *ਗੰਢ_backend_id;
            } ਨਹੀਂਤਾਂ ਜੇ (cur_backend_id != -1) {
                ggml_backend_sched_ਸੈੱਟ_if_supported(sched, ਗੰਢ, cur_backend_id, ਗੰਢ_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
            ਜਾਰੀ;
        }
        ਪੂਰਨ * ਗੰਢ_backend_id = &tensor_backend_id(ਗੰਢ);
        ਜੇ (*ਗੰਢ_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            ਪੂਰਨ n_supported_ਉੱਤਮ = -1;
            ਲਈ (ਪੂਰਨ b = 0; b < sched->n_backends; b++) {
                ਜੇ (ggml_backend_supports_op(sched->backends[b], ਗੰਢ)) {
                    ਪੂਰਨ n_supported = 0;
                    ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                        ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                        ਜੇ (src == ਨਿਰਾ) {
                            ਜਾਰੀ;
                        }
                        ਜੇ ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_ਬਫਰ_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    ਜੇ (n_supported > n_supported_ਉੱਤਮ) {
                        n_supported_ਉੱਤਮ = n_supported;
                        *ਗੰਢ_backend_id = b;
                        ਸੈੱਟ_CAUSE(ਗੰਢ, "3.best");
                    }
                }
            }
        } ਨਹੀਂਤਾਂ {
            // assigned node: upgrade to higher prio backend if possible
            ਲਈ (ਪੂਰਨ b = 0; b < *ਗੰਢ_backend_id; b++) {
                ਜੇ (sched->bufts[b] == sched->bufts[*ਗੰਢ_backend_id] && ggml_backend_supports_op(sched->backends[b], ਗੰਢ)) {
                    ਤਾਰਕਿਕ supported = ਸੱਚ;
                    ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                        ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                        ਜੇ (src == ਨਿਰਾ) {
                            ਜਾਰੀ;
                        }
                        ਜੇ (!ggml_backend_sched_ਬਫਰ_supported(sched, src, b)) {
                            supported = ਝੂਠ;
                            ਰੋਕੋ;
                        }
                    }
                    ਜੇ (supported) {
                        *ਗੰਢ_backend_id = b;
                        ਸੈੱਟ_CAUSE(ਗੰਢ, "3.upg");
                        ਰੋਕੋ;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਪੂਰਨ * cur_backend_id = &tensor_backend_id(ਗੰਢ);
        ਜੇ (ਗੰਢ->view_src != ਨਿਰਾ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(ਗੰਢ->view_src);
            ਸੈੱਟ_CAUSE(ਗੰਢ, "4.vsrc");
        }
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
            ਜੇ (src == ਨਿਰਾ) {
                ਜਾਰੀ;
            }
            ਪੂਰਨ * src_backend_id = &tensor_backend_id(src);
            ਜੇ (*src_backend_id == -1) {
                ਜੇ (src->view_src != ਨਿਰਾ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    ਸੈੱਟ_CAUSE(src, "4.vsrc");
                } ਨਹੀਂਤਾਂ {
                    *src_backend_id = *cur_backend_id;
                    ਸੈੱਟ_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        ਲਈ (ਪੂਰਨ b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_ਸੈੱਟ_if_supported(sched, ਗੰਢ, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        ਪੂਰਨ i_ਵੰਡੋ = 0;
        ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        ਪੂਰਨ i = 0;
        ਲਈ (; i < graph->n_nodes; i++) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
            ਜੇ (!ggml_is_view_op(ਗੰਢ->op)) {
                ਵੰਡੋ->backend_id = tensor_backend_id(ਗੰਢ);
                ਰੋਕੋ;
            }
        }
        ਵੰਡੋ->i_ਸ਼ੁਰੂ = 0;
        ਵੰਡੋ->n_inputs = 0;
        ਪੂਰਨ cur_backend_id = ਵੰਡੋ->backend_id;
        ਲਈ (; i < graph->n_nodes; i++) {
            ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];

            ਜੇ (ggml_is_view_op(ਗੰਢ->op)) {
                ਜਾਰੀ;
            }

            ਅਚਲ ਪੂਰਨ ਗੰਢ_backend_id = tensor_backend_id(ਗੰਢ);

            GGML_ASSERT(ਗੰਢ_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            ਤਾਰਕਿਕ need_new_ਵੰਡੋ = ਝੂਠ;
            ਜੇ (ਗੰਢ_backend_id == cur_backend_id && ਵੰਡੋ->n_inputs > 0) {
                ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                    ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                    ਜੇ (src == ਨਿਰਾ) {
                        ਜਾਰੀ;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    ਜੇ (src->ਬਫਰ != ਨਿਰਾ && src->ਬਫਰ->usage == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS) {
                        ਪੂਰਨ src_backend_id = tensor_backend_id(src);
                        ਜੇ (src_backend_id != cur_backend_id && !ggml_backend_sched_ਬਫਰ_supported(sched, src, cur_backend_id)) {
                            need_new_ਵੰਡੋ = ਸੱਚ;
                            ਰੋਕੋ;
                        }
                    }
                }
            }

            ਜੇ (ਗੰਢ_backend_id != cur_backend_id || need_new_ਵੰਡੋ) {
                ਵੰਡੋ->i_ਸਿਰਾ = i;
                i_ਵੰਡੋ++;
                ਜੇ (i_ਵੰਡੋ >= sched->splits_ਸਮਰੱਥਾ) {
                    ਪੂਰਨ old_cap = sched->splits_ਸਮਰੱਥਾ;
                    sched->splits_ਸਮਰੱਥਾ *= 2;
                    sched->splits = (ggml_backend_sched_ਵੰਡੋ *)
                        realloc(sched->splits, sched->splits_ਸਮਰੱਥਾ * ਆਕਾਰ(ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ));
                    GGML_ASSERT(sched->splits != ਨਿਰਾ);
                    ਲਈ (ਪੂਰਨ k = old_cap; k < sched->splits_ਸਮਰੱਥਾ; k++) {
                        memset(&sched->splits[k], 0, ਆਕਾਰ(ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ));
                    }
                }
                ਵੰਡੋ = &sched->splits[i_ਵੰਡੋ];
                ਵੰਡੋ->backend_id = ਗੰਢ_backend_id;
                ਵੰਡੋ->i_ਸ਼ੁਰੂ = i;
                ਵੰਡੋ->n_inputs = 0;
                cur_backend_id = ਗੰਢ_backend_id;
            }

            // find inputs that are not on the same backend
            ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
                ਬਣਤਰ ggml_tensor * src = ਗੰਢ->src[j];
                ਜੇ (src == ਨਿਰਾ) {
                    ਜਾਰੀ;
                }

                size_t src_id = hash_id(src);
                ਅਚਲ ਪੂਰਨ src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                ਜੇ (src_backend_id != cur_backend_id && !ggml_backend_sched_ਬਫਰ_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    ਜੇ (tensor_id_copy(src_id, cur_backend_id, 0) == ਨਿਰਾ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
                            ਬਣਤਰ ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            ਜੇ (sched->n_copies > 1) {
                                ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(tensor_copy);
                                ggml_ਸੈੱਟ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            ਸੈੱਟ_CAUSE(tensor_copy, "4.cpy");
                        }
                        ਪੂਰਨ n_inputs = ਵੰਡੋ->n_inputs++;
                        ਜੇ (n_inputs >= ਵੰਡੋ->inputs_ਸਮਰੱਥਾ) {
                            ggml_backend_sched_ਵੰਡੋ_inputs_grow(ਵੰਡੋ);
                        }
                        ਵੰਡੋ->inputs[n_inputs] = src;
                    }
                    ਗੰਢ->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        ਵੰਡੋ->i_ਸਿਰਾ = graph->n_nodes;
        sched->n_splits = i_ਵੰਡੋ + 1;
    }

    ਜੇ (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    ਜੇ (sched->n_copies > 1) {
        ਲਈ (ਪੂਰਨ i = 0; i < graph->n_leafs; i++) {
            ਬਣਤਰ ggml_tensor * ਪੱਤੀ = graph->leafs[i];
            ਜੇ ((ਪੱਤੀ->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) == 0) {
                ਜਾਰੀ;
            }

            ਅਚਲ size_t ਪੱਤੀ_id = hash_id(ਪੱਤੀ);
            ਅਚਲ ਪੂਰਨ ਪੱਤੀ_backend_id = tensor_backend_id(ਪੱਤੀ);
            GGML_ASSERT(ਪੱਤੀ_backend_id != -1); // all leafs should be assigned by now

            ਜੇ (tensor_id_copy(ਪੱਤੀ_id, ਪੱਤੀ_backend_id, 0) == ਨਿਰਾ) {
                ggml_backend_t backend = sched->backends[ਪੱਤੀ_backend_id];
                ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
                    ਬਣਤਰ ggml_tensor * tensor_copy;
                    ਜੇ (c == sched->cur_copy) {
                        tensor_copy = ਪੱਤੀ; // use the original tensor as the current copy
                    } ਨਹੀਂਤਾਂ {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, ਪੱਤੀ);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), ਪੱਤੀ->name, c);
                    }
                    ggml_ਸੈੱਟ_ਨਿਵੇਸ਼(tensor_copy);
                    ggml_ਸੈੱਟ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(ਪੱਤੀ_id, ਪੱਤੀ_backend_id, c) = tensor_copy;
                    ਸੈੱਟ_CAUSE(tensor_copy, "6.cpy");
                }
            }

            ਪੂਰਨ n_ਗ੍ਰਾਫ_inputs = sched->n_ਗ੍ਰਾਫ_inputs++;
            ਜੇ (n_ਗ੍ਰਾਫ_inputs >= sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ) {
                ggml_backend_sched_ਗ੍ਰਾਫ_inputs_grow(sched);
            }
            sched->ਗ੍ਰਾਫ_inputs[n_ਗ੍ਰਾਫ_inputs] = ਪੱਤੀ;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        ਪੂਰਨ * tmp = sched->ਗੰਢ_backend_ids;
        sched->ਗੰਢ_backend_ids = sched->prev_ਗੰਢ_backend_ids;
        sched->prev_ਗੰਢ_backend_ids = tmp;

        tmp = sched->ਪੱਤੀ_backend_ids;
        sched->ਪੱਤੀ_backend_ids = sched->prev_ਪੱਤੀ_backend_ids;
        sched->prev_ਪੱਤੀ_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](ਖਾਲੀ * user_ਅੰਕੜੇ, ggml_tensor * tensor, ggml_tensor * until) {
            ਸਵੈਚਾਲਿਤ & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_ਅੰਕੜੇ;
            std::vector<ggml_tensor *> & keep = deps[until];
            ਜੇ (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_splits; i++) {
        ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ = &sched->splits[i];
        ਵੰਡੋ->graph = ggml_ਗ੍ਰਾਫ_view(graph, ਵੰਡੋ->i_ਸ਼ੁਰੂ, ਵੰਡੋ->i_ਸਿਰਾ);

        ggml_backend_ਗ੍ਰਾਫ_optimize(sched->backends[ਵੰਡੋ->backend_id], &ਵੰਡੋ->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    ਪੂਰਨ n_dep_nodes = 0;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    ਪੂਰਨ ਕੁੱਲ_inputs = sched->n_ਗ੍ਰਾਫ_inputs;
    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_splits; i++) {
        ਕੁੱਲ_inputs += sched->splits[i].n_inputs;
    }
    ਪੂਰਨ ਗ੍ਰਾਫ_ਮਾਪ = std::max(graph->n_nodes, graph->n_leafs) + ਕੁੱਲ_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_ਗ੍ਰਾਫ_ਮਾਪ = sched->ਗ੍ਰਾਫ_ਮਾਪ;
    sched->ਗ੍ਰਾਫ_ਮਾਪ = ਗ੍ਰਾਫ_ਮਾਪ;

    ਜੇ (sched->graph.size < ਗ੍ਰਾਫ_ਮਾਪ) {
        sched->graph.size = ਗ੍ਰਾਫ_ਮਾਪ;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, ਗ੍ਰਾਫ_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, ਗ੍ਰਾਫ_ਮਾਪ * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ਨਿਰਾ);
        GGML_ASSERT(sched->graph.leafs != ਨਿਰਾ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ_copy = &sched->graph;

    ਪੂਰਨ n_dep_nodes_added = 0;

    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_splits; i++) {
        ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        ਲਈ (ਪੂਰਨ j = 0; j < ਵੰਡੋ->n_inputs; j++) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > (ਗ੍ਰਾਫ_copy->n_nodes + 1));

            ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼ = ਵੰਡੋ->inputs[j];
            ਅਚਲ size_t ਨਿਵੇਸ਼_id = hash_id(ਨਿਵੇਸ਼);
            ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼_cpy = tensor_id_copy(ਨਿਵੇਸ਼_id, ਵੰਡੋ->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼_dep = ggml_view_tensor(sched->ctx, ਨਿਵੇਸ਼);
            ਨਿਵੇਸ਼_dep->src[0] = ਨਿਵੇਸ਼;
            sched->ਗੰਢ_backend_ids[ਗ੍ਰਾਫ_copy->n_nodes] = sched->hv_tensor_backend_ids[ਨਿਵੇਸ਼_id];
            ਗ੍ਰਾਫ_copy->nodes[ਗ੍ਰਾਫ_copy->n_nodes++] = ਨਿਵੇਸ਼_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->ਗੰਢ_backend_ids[ਗ੍ਰਾਫ_copy->n_nodes] = ਵੰਡੋ->backend_id;
            ਗ੍ਰਾਫ_copy->nodes[ਗ੍ਰਾਫ_copy->n_nodes++] = ਨਿਵੇਸ਼_cpy;
        }

        ਲਈ (ਪੂਰਨ j = ਵੰਡੋ->i_ਸ਼ੁਰੂ; j < ਵੰਡੋ->i_ਸਿਰਾ; j++) {
            ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > ਗ੍ਰਾਫ_copy->n_nodes);
            sched->ਗੰਢ_backend_ids[ਗ੍ਰਾਫ_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            ਗ੍ਰਾਫ_copy->nodes[ਗ੍ਰਾਫ_copy->n_nodes++] = graph->nodes[j];

            ਜੇ (alloc_deps.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਜਾਰੀ;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            ਸਵੈਚਾਲਿਤ it = alloc_deps.find(graph->nodes[j]);
            ਜੇ (it != alloc_deps.end()) {
                ਅਚਲ std::vector<ggml_tensor *> & keep = it->second;
                ਲਈ (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ਬਣਤਰ ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    ਲਈ (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > ਗ੍ਰਾਫ_copy->n_nodes);
                    sched->ਗੰਢ_backend_ids[ਗ੍ਰਾਫ_copy->n_nodes] = ਵੰਡੋ->backend_id;
                    ਗ੍ਰਾਫ_copy->nodes[ਗ੍ਰਾਫ_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    ਜੇ (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        ਲਈ (ਪੂਰਨ i = 0; i < sched->n_ਗ੍ਰਾਫ_inputs; i++) {
            ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼ = sched->ਗ੍ਰਾਫ_inputs[i];
            size_t id = hash_id(ਨਿਵੇਸ਼);
            ਪੂਰਨ backend_id = tensor_backend_id(ਨਿਵੇਸ਼);
            ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
                ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼_cpy = tensor_id_copy(id, backend_id, c);
                sched->ਪੱਤੀ_backend_ids[ਗ੍ਰਾਫ_copy->n_leafs] = backend_id;
                ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > ਗ੍ਰਾਫ_copy->n_leafs);
                ਗ੍ਰਾਫ_copy->leafs[ਗ੍ਰਾਫ_copy->n_leafs++] = ਨਿਵੇਸ਼_cpy;
            }
        }

        ਲਈ (ਪੂਰਨ i = 0; i < sched->n_splits; i++) {
            ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ = &sched->splits[i];
            ਪੂਰਨ backend_id = ਵੰਡੋ->backend_id;
            ਲਈ (ਪੂਰਨ j = 0; j < ਵੰਡੋ->n_inputs; j++) {
                ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼ = ਵੰਡੋ->inputs[j];
                size_t id = hash_id(ਨਿਵੇਸ਼);
                ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
                    ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼_cpy = tensor_id_copy(id, backend_id, c);
                    sched->ਪੱਤੀ_backend_ids[ਗ੍ਰਾਫ_copy->n_leafs] = backend_id;
                    ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > ਗ੍ਰਾਫ_copy->n_leafs);
                    ਗ੍ਰਾਫ_copy->leafs[ਗ੍ਰਾਫ_copy->n_leafs++] = ਨਿਵੇਸ਼_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_leafs; i++) {
        ਬਣਤਰ ggml_tensor * ਪੱਤੀ = graph->leafs[i];
        sched->ਪੱਤੀ_backend_ids[ਗ੍ਰਾਫ_copy->n_leafs] = tensor_backend_id(ਪੱਤੀ);
        ਪ੍ਰਤਿਜ੍ਞਾ(ਗ੍ਰਾਫ_copy->size > ਗ੍ਰਾਫ_copy->n_leafs);
        ਗ੍ਰਾਫ_copy->leafs[ਗ੍ਰਾਫ_copy->n_leafs++] = ਪੱਤੀ;
    }

    // set ids for all splits
    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_ਗ੍ਰਾਫ_ਅਗਲਾ_uid();
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    ਤਾਰਕਿਕ backend_ids_ਬਦਲਿਆ = ਝੂਠ;
    ਲਈ (ਪੂਰਨ i = 0; i < sched->graph.n_nodes; i++) {
        ਜੇ (sched->ਗੰਢ_backend_ids[i] != sched->prev_ਗੰਢ_backend_ids[i] &&
            sched->bufts[sched->ਗੰਢ_backend_ids[i]] != sched->bufts[sched->prev_ਗੰਢ_backend_ids[i]]) {
            backend_ids_ਬਦਲਿਆ = ਸੱਚ;
            ਰੋਕੋ;
        }
    }
    ਜੇ (!backend_ids_ਬਦਲਿਆ) {
        ਲਈ (ਪੂਰਨ i = 0; i < sched->graph.n_leafs; i++) {
            ਜੇ (sched->ਪੱਤੀ_backend_ids[i] != sched->prev_ਪੱਤੀ_backend_ids[i] &&
                sched->bufts[sched->ਪੱਤੀ_backend_ids[i]] != sched->bufts[sched->prev_ਪੱਤੀ_backend_ids[i]]) {
                backend_ids_ਬਦਲਿਆ = ਸੱਚ;
                ਰੋਕੋ;
            }
        }
    }

    // allocate graph
    ਜੇ (backend_ids_ਬਦਲਿਆ || !ggml_gallocr_alloc_ਗ੍ਰਾਫ(sched->galloc, &sched->graph)) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_ਬਦਲਿਆ);
#ਜੇ_ਅੰਤ

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ਅਚਲ ਤਾਰਕਿਕ ਗ੍ਰਾਫ_ਮਾਪ_ਬਦਲਿਆ = sched->prev_ਗ੍ਰਾਫ_ਮਾਪ != sched->ਗ੍ਰਾਫ_ਮਾਪ;

        ਜੇ (sched->callback_realloc != ਨਿਰਾ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_ਬਦਲਿਆ, ਗ੍ਰਾਫ_ਮਾਪ_ਬਦਲਿਆ, sched->callback_realloc_user_ਅੰਕੜੇ);
        }

        ਜੇ (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ਅਚਲ ਤਾਰਕਿਕ unexpected = !backend_ids_ਬਦਲਿਆ && !ਗ੍ਰਾਫ_ਮਾਪ_ਬਦਲਿਆ;

            ਜੇ (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->ਗ੍ਰਾਫ_ਮਾਪ, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        ਲਈ (ਪੂਰਨ i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ਜੇ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ਗੰਢ_backend_ids, sched->ਪੱਤੀ_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (!ggml_gallocr_alloc_ਗ੍ਰਾਫ(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sched_is_host_ਭਾਰ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ t->ਬਫਰ != ਨਿਰਾ &&
        ggml_backend_ਬਫਰ_get_usage(t->ਬਫਰ) == GGML_BACKEND_ਬਫਰ_USAGE_WEIGHTS &&
        ggml_backend_ਬਫਰ_is_host(t->ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sched_copy_ਨਿਵੇਸ਼(ggml_backend_sched_t sched, ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ, ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼) {
    ਅਚਲ ਪੂਰਨ ਵੰਡੋ_backend_id = ਵੰਡੋ->backend_id;
    ggml_backend_t ਵੰਡੋ_backend = sched->backends[ਵੰਡੋ_backend_id];
    ggml_backend_t ਨਿਵੇਸ਼_backend = ggml_backend_sched_get_tensor_backend(sched, ਨਿਵੇਸ਼);
    ਬਣਤਰ ggml_tensor * ਨਿਵੇਸ਼_cpy = tensor_copy(ਨਿਵੇਸ਼, ਵੰਡੋ_backend_id, sched->cur_copy);

    ਜੇ (ਨਿਵੇਸ਼->flags & GGML_TENSOR_FLAG_ਨਿਵੇਸ਼) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        ਜੇ (sched->events[ਵੰਡੋ_backend_id][sched->cur_copy] != ਨਿਰਾ) {
            ggml_backend_event_synchronize(sched->events[ਵੰਡੋ_backend_id][sched->cur_copy]);
        } ਨਹੀਂਤਾਂ {
            ggml_backend_synchronize(ਵੰਡੋ_backend);
        }
        ggml_backend_tensor_copy(ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_cpy);
        ਵਾਪਸ;
    }

    // wait for the split backend to finish using the input before overwriting it
    ਜੇ (sched->events[ਵੰਡੋ_backend_id][sched->cur_copy] != ਨਿਰਾ) {
        ggml_backend_event_wait(ਵੰਡੋ_backend, sched->events[ਵੰਡੋ_backend_id][sched->cur_copy]);
    } ਨਹੀਂਤਾਂ {
        ggml_backend_synchronize(ਵੰਡੋ_backend);
    }

    ਜੇ (sched->callback_copy != ਨਿਰਾ && ggml_backend_sched_is_host_ਭਾਰ(ਨਿਵੇਸ਼) &&
        sched->callback_copy(ਵੰਡੋ_backend, ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_cpy, &ਵੰਡੋ->graph, sched->callback_copy_user_ਅੰਕੜੇ)) {
        ਵਾਪਸ;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    ਜੇ (!ਵੰਡੋ_backend->iface.cpy_tensor_async || !ਵੰਡੋ_backend->iface.cpy_tensor_async(ਨਿਵੇਸ਼_backend, ਵੰਡੋ_backend, ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_cpy)) {
        ggml_backend_synchronize(ਨਿਵੇਸ਼_backend);
        ਜੇ (sched->events[ਵੰਡੋ_backend_id][sched->cur_copy] != ਨਿਰਾ) {
            ggml_backend_event_synchronize(sched->events[ਵੰਡੋ_backend_id][sched->cur_copy]);
        } ਨਹੀਂਤਾਂ {
            ggml_backend_synchronize(ਵੰਡੋ_backend);
        }
        ggml_backend_tensor_copy(ਨਿਵੇਸ਼, ਨਿਵੇਸ਼_cpy);
    }
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * splits = sched->splits;

    ਪੂਰਨ prev_backend_id = -1;

    ਲਈ (ਪੂਰਨ ਵੰਡੋ_id = 0; ਵੰਡੋ_id < sched->n_splits; ਵੰਡੋ_id++) {
        ਬਣਤਰ ggml_backend_sched_ਵੰਡੋ * ਵੰਡੋ = &splits[ਵੰਡੋ_id];
        ਪੂਰਨ ਵੰਡੋ_backend_id = ਵੰਡੋ->backend_id;
        ggml_backend_t ਵੰਡੋ_backend = sched->backends[ਵੰਡੋ_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        ਜੇ (ਵੰਡੋ->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != ਵੰਡੋ_backend_id) {
            ਜੇ (sched->events[prev_backend_id][sched->cur_copy] != ਨਿਰਾ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } ਨਹੀਂਤਾਂ {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        ਲਈ (ਪੂਰਨ ਨਿਵੇਸ਼_id = 0; ਨਿਵੇਸ਼_id < ਵੰਡੋ->n_inputs; ਨਿਵੇਸ਼_id++) {
            ਜੇ (!ggml_backend_sched_is_host_ਭਾਰ(ਵੰਡੋ->inputs[ਨਿਵੇਸ਼_id])) {
                ggml_backend_sched_copy_ਨਿਵੇਸ਼(sched, ਵੰਡੋ, ਵੰਡੋ->inputs[ਨਿਵੇਸ਼_id]);
            }
        }
        ਲਈ (ਪੂਰਨ ਨਿਵੇਸ਼_id = 0; ਨਿਵੇਸ਼_id < ਵੰਡੋ->n_inputs; ਨਿਵੇਸ਼_id++) {
            ਜੇ (ggml_backend_sched_is_host_ਭਾਰ(ਵੰਡੋ->inputs[ਨਿਵੇਸ਼_id])) {
                ggml_backend_sched_copy_ਨਿਵੇਸ਼(sched, ਵੰਡੋ, ਵੰਡੋ->inputs[ਨਿਵੇਸ਼_id]);
            }
        }

        ਜੇ (!sched->callback_eval) {
            ਗਿਣਤੀ ggml_ਸਥਿਤੀ ec = ggml_backend_ਗ੍ਰਾਫ_compute_async(ਵੰਡੋ_backend, &ਵੰਡੋ->graph);
            ਜੇ (ec != GGML_ਸਥਿਤੀ_SUCCESS) {
                ਵਾਪਸ ec;
            }
        } ਨਹੀਂਤਾਂ {
            // similar to ggml_backend_compare_graph_backend
            ਲਈ (ਪੂਰਨ j0 = 0; j0 < ਵੰਡੋ->graph.n_nodes; j0++) {
                ਬਣਤਰ ggml_tensor * t = ਵੰਡੋ->graph.nodes[j0];

                // check if the user needs data from this node
                ਤਾਰਕਿਕ need = sched->callback_eval(t, ਸੱਚ, sched->callback_eval_user_ਅੰਕੜੇ);

                ਪੂਰਨ j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                ਜਦੋਂ (!need && j1 < ਵੰਡੋ->graph.n_nodes - 1) {
                    t = ਵੰਡੋ->graph.nodes[++j1];
                    need = sched->callback_eval(t, ਸੱਚ, sched->callback_eval_user_ਅੰਕੜੇ);
                }

                ਬਣਤਰ ggml_cgraph gv = ggml_ਗ੍ਰਾਫ_view(&ਵੰਡੋ->graph, j0, j1 + 1);

                ਗਿਣਤੀ ggml_ਸਥਿਤੀ ec = ggml_backend_ਗ੍ਰਾਫ_compute_async(ਵੰਡੋ_backend, &gv);
                ਜੇ (ec != GGML_ਸਥਿਤੀ_SUCCESS) {
                    ਵਾਪਸ ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(ਵੰਡੋ_backend);

                ਜੇ (need && !sched->callback_eval(t, ਝੂਠ, sched->callback_eval_user_ਅੰਕੜੇ)) {
                    ਰੋਕੋ;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        ਜੇ (sched->events[ਵੰਡੋ_backend_id][sched->cur_copy] != ਨਿਰਾ) {
            ggml_backend_event_ਰਿਕਾਰਡ(sched->events[ਵੰਡੋ_backend_id][sched->cur_copy], ਵੰਡੋ_backend);
        }

        prev_backend_id = ਵੰਡੋ_backend_id;
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_ਬਫਰ_type_t * bufts,
        ਪੂਰਨ n_backends,
        size_t ਗ੍ਰਾਫ_ਮਾਪ,
        ਤਾਰਕਿਕ parallel,
        ਤਾਰਕਿਕ op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ਬਣਤਰ ggml_backend_sched * sched = (ggml_backend_sched *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(1, ਆਕਾਰ(ਬਣਤਰ ggml_backend_sched));

    ਅਚਲ ਅੱਖਰ * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#ਜੇ_ਅੰਤ
    ਅਚਲ ਅੱਖਰ * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_ਸੈੱਟ    = ggml_hash_ਸੈੱਟ_new(ਗ੍ਰਾਫ_ਮਾਪ);
    sched->hv_tensor_backend_ids = (ਪੂਰਨ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(sched->hash_ਸੈੱਟ.size * ਆਕਾਰ(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) ਸਮ੍ਰਿਤੀ_ਦਿਓ(sched->hash_ਸੈੱਟ.size * sched->n_backends * sched->n_copies * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));

    ਅਚਲ size_t ggml_sched_max_splits = ਗ੍ਰਾਫ_ਮਾਪ; // at most there is one split for each node in the graph
    ਅਚਲ size_t nodes_ਮਾਪ = ਗ੍ਰਾਫ_ਮਾਪ + ggml_sched_max_splits*GGML_SCHED_MAX_ਵੰਡੋ_INPUTS*2;
    sched->ਗੰਢ_backend_ids = (ਪੂਰਨ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(nodes_ਮਾਪ, ਆਕਾਰ(sched->ਗੰਢ_backend_ids[0]));
    sched->ਪੱਤੀ_backend_ids = (ਪੂਰਨ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(nodes_ਮਾਪ, ਆਕਾਰ(sched->ਪੱਤੀ_backend_ids[0]));
    sched->prev_ਗੰਢ_backend_ids = (ਪੂਰਨ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(nodes_ਮਾਪ, ਆਕਾਰ(sched->prev_ਗੰਢ_backend_ids[0]));
    sched->prev_ਪੱਤੀ_backend_ids = (ਪੂਰਨ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(nodes_ਮਾਪ, ਆਕਾਰ(sched->prev_ਪੱਤੀ_backend_ids[0]));

    sched->ਗ੍ਰਾਫ_ਮਾਪ      = 0;
    sched->prev_ਗ੍ਰਾਫ_ਮਾਪ = 0;

    sched->context_ਬਫਰ_ਮਾਪ = ggml_sched_max_splits*GGML_SCHED_MAX_ਵੰਡੋ_INPUTS*2*ਆਕਾਰ(ਬਣਤਰ ggml_tensor) + ggml_ਗ੍ਰਾਫ_overhead_custom(ਗ੍ਰਾਫ_ਮਾਪ, ਝੂਠ);
    sched->context_ਬਫਰ = (ਅੱਖਰ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(sched->context_ਬਫਰ_ਮਾਪ);

    ਅਚਲ ਪੂਰਨ initial_splits_ਸਮਰੱਥਾ = 16;
    sched->splits = (ggml_backend_sched_ਵੰਡੋ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(initial_splits_ਸਮਰੱਥਾ, ਆਕਾਰ(sched->splits[0]));
    sched->splits_ਸਮਰੱਥਾ = initial_splits_ਸਮਰੱਥਾ;

    sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ = GGML_SCHED_MAX_ਵੰਡੋ_INPUTS;
    sched->ਗ੍ਰਾਫ_inputs = (ਬਣਤਰ ggml_tensor **) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(sched->ਗ੍ਰਾਫ_inputs_ਸਮਰੱਥਾ, ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));

    ਲਈ (ਪੂਰਨ b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_ਬਫਰ_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        ਜੇ (sched->n_copies > 1) {
            ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    ਵਾਪਸ sched;
}

ਖਾਲੀ ggml_backend_sched_free(ggml_backend_sched_t sched) {
    ਜੇ (sched == ਨਿਰਾ) {
        ਵਾਪਸ;
    }
    ਲਈ (ਪੂਰਨ b = 0; b < sched->n_backends; b++) {
        ਲਈ (ਪੂਰਨ c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_ਸੈੱਟ_free(&sched->hash_ਸੈੱਟ);
    ਲਈ (ਪੂਰਨ i = 0; i < sched->splits_ਸਮਰੱਥਾ; i++) {
        ਮੁਕਤ(sched->splits[i].inputs);
    }
    ਮੁਕਤ(sched->splits);
    ਮੁਕਤ(sched->ਗ੍ਰਾਫ_inputs);
    ਮੁਕਤ(sched->hv_tensor_backend_ids);
    ਮੁਕਤ(sched->hv_tensor_copies);
    ਮੁਕਤ(sched->ਗੰਢ_backend_ids);
    ਮੁਕਤ(sched->ਪੱਤੀ_backend_ids);
    ਮੁਕਤ(sched->prev_ਗੰਢ_backend_ids);
    ਮੁਕਤ(sched->prev_ਪੱਤੀ_backend_ids);
    ਮੁਕਤ(sched->context_ਬਫਰ);
    ਮੁਕਤ(sched->graph.nodes);
    ਮੁਕਤ(sched->graph.leafs);
    ਮੁਕਤ(sched);
}

ਖਾਲੀ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    ਜੇ (!sched->is_reset) {
        ggml_hash_ਸੈੱਟ_reset(&sched->hash_ਸੈੱਟ);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_ਸੈੱਟ.size * ਆਕਾਰ(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_ਸੈੱਟ.size * sched->n_backends * sched->n_copies * ਆਕਾਰ(ਬਣਤਰ ggml_tensor *));
        sched->is_reset = ਸੱਚ;
    }
    sched->is_alloc = ਝੂਠ;
}

ਖਾਲੀ ggml_backend_sched_reserve_ਮਾਪ(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * measure_ਗ੍ਰਾਫ, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ਪੂਰਨ)sched->hash_ਸੈੱਟ.size >= measure_ਗ੍ਰਾਫ->n_nodes + measure_ਗ੍ਰਾਫ->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ਵੰਡੋ_ਗ੍ਰਾਫ(sched, measure_ਗ੍ਰਾਫ);

    ggml_gallocr_reserve_n_ਮਾਪ(sched->galloc, &sched->graph, sched->ਗੰਢ_backend_ids, sched->ਪੱਤੀ_backend_ids, sizes);
}

ਤਾਰਕਿਕ ggml_backend_sched_reserve(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * measure_ਗ੍ਰਾਫ) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ਪੂਰਨ)sched->hash_ਸੈੱਟ.size >= measure_ਗ੍ਰਾਫ->n_nodes + measure_ਗ੍ਰਾਫ->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ਵੰਡੋ_ਗ੍ਰਾਫ(sched, measure_ਗ੍ਰਾਫ);

    ਜੇ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ਗੰਢ_backend_ids, sched->ਪੱਤੀ_backend_ids)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_backend_sched_reset(sched);

    ਵਾਪਸ ਸੱਚ;
}

ਤਾਰਕਿਕ ggml_backend_sched_alloc_ਗ੍ਰਾਫ(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ਪੂਰਨ)sched->hash_ਸੈੱਟ.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->ਅਗਲਾ_copy;
    sched->ਅਗਲਾ_copy = (sched->ਅਗਲਾ_copy + 1) % sched->n_copies;

    ggml_backend_sched_ਵੰਡੋ_ਗ੍ਰਾਫ(sched, graph);

    ਜੇ (!ggml_backend_sched_alloc_splits(sched)) {
        ਵਾਪਸ ਝੂਠ;
    }

    sched->is_alloc = ਸੱਚ;

    ਵਾਪਸ ਸੱਚ;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_sched_ਗ੍ਰਾਫ_compute(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * graph) {
    ਗਿਣਤੀ ggml_ਸਥਿਤੀ err = ggml_backend_sched_ਗ੍ਰਾਫ_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    ਵਾਪਸ err;
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_sched_ਗ੍ਰਾਫ_compute_async(ggml_backend_sched_t sched, ਬਣਤਰ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    ਜੇ (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    ਜੇ (!sched->is_alloc) {
        ਜੇ (!ggml_backend_sched_alloc_ਗ੍ਰਾਫ(sched, graph)) {
            ਵਾਪਸ GGML_ਸਥਿਤੀ_ALLOC_FAILED;
        }
    }

    ਵਾਪਸ ggml_backend_sched_compute_splits(sched);
}

ਖਾਲੀ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ਲਈ (ਪੂਰਨ i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    ਜੇ (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->ਅਗਲਾ_copy = 0;
    }
}

ਖਾਲੀ ggml_backend_sched_ਸੈੱਟ_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_ਅੰਕੜੇ = user_ਅੰਕੜੇ;
}

ਖਾਲੀ ggml_backend_sched_ਸੈੱਟ_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_ਅੰਕੜੇ = user_ਅੰਕੜੇ;
}

ਖਾਲੀ ggml_backend_sched_ਸੈੱਟ_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, ਖਾਲੀ * user_ਅੰਕੜੇ) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_ਅੰਕੜੇ = user_ਅੰਕੜੇ;
}

ਪੂਰਨ ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ਵਾਪਸ sched->n_splits;
}

ਪੂਰਨ ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ਵਾਪਸ sched->n_copies;
}

ਪੂਰਨ ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ਵਾਪਸ sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, ਪੂਰਨ i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    ਵਾਪਸ sched->backends[i];
}

ggml_backend_ਬਫਰ_type_t ggml_backend_sched_get_ਬਫਰ_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ਪੂਰਨ backend_ਸੂਚਕਾਂਕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ਸੂਚਕਾਂਕ >= 0 && backend_ਸੂਚਕਾਂਕ < sched->n_backends);

    ਵਾਪਸ sched->bufts[backend_ਸੂਚਕਾਂਕ];
}

size_t ggml_backend_sched_get_ਬਫਰ_ਮਾਪ(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ਪੂਰਨ backend_ਸੂਚਕਾਂਕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ਸੂਚਕਾਂਕ >= 0 && backend_ਸੂਚਕਾਂਕ < sched->n_backends);

    ਵਾਪਸ ggml_gallocr_get_ਬਫਰ_ਮਾਪ(sched->galloc, backend_ਸੂਚਕਾਂਕ);
}

ਖਾਲੀ ggml_backend_sched_ਸੈੱਟ_tensor_backend(ggml_backend_sched_t sched, ਬਣਤਰ ggml_tensor * ਗੰਢ, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ਪੂਰਨ backend_ਸੂਚਕਾਂਕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ਸੂਚਕਾਂਕ >= 0 && backend_ਸੂਚਕਾਂਕ < sched->n_backends);
    tensor_backend_id(ਗੰਢ) = backend_ਸੂਚਕਾਂਕ;
    ਸੈੱਟ_CAUSE(ਗੰਢ, "usr");
    sched->is_reset = ਝੂਠ;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ਬਣਤਰ ggml_tensor * ਗੰਢ) {
    GGML_ASSERT(sched);
    ਪੂਰਨ backend_ਸੂਚਕਾਂਕ = tensor_backend_id(ਗੰਢ);
    ਜੇ (backend_ਸੂਚਕਾਂਕ == -1) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ sched->backends[backend_ਸੂਚਕਾਂਕ];
}

// utils

ਤਾਰਕਿਕ ggml_op_alloc_ਮਾਪ_may_expand(ਗਿਣਤੀ ggml_op op) {
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ਹਾਲਤ GGML_OP_CUMSUM:
        ਹਾਲਤ GGML_OP_ARGSORT:
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
        ਹਾਲਤ GGML_OP_ARGMAX:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_view_init(ਬਣਤਰ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਬਫਰ == ਨਿਰਾ);
    GGML_ASSERT(tensor->view_src != ਨਿਰਾ);
    GGML_ASSERT(tensor->view_src->ਬਫਰ != ਨਿਰਾ);
    GGML_ASSERT(tensor->view_src->ਅੰਕੜੇ != ਨਿਰਾ);

    tensor->ਬਫਰ = tensor->view_src->ਬਫਰ;
    tensor->ਅੰਕੜੇ = (ਅੱਖਰ *)tensor->view_src->ਅੰਕੜੇ + tensor->view_offs;
    ਵਾਪਸ ggml_backend_ਬਫਰ_init_tensor(tensor->ਬਫਰ, tensor);
}

ਗਿਣਤੀ ggml_ਸਥਿਤੀ ggml_backend_tensor_alloc(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ਬਫਰ == ਨਿਰਾ);
    GGML_ASSERT(tensor->ਅੰਕੜੇ == ਨਿਰਾ);
    GGML_ASSERT(tensor->view_src == ਨਿਰਾ);
    GGML_ASSERT(addr >= ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਬਫਰ));
    GGML_ASSERT(ggml_backend_ਬਫਰ_is_meta(ਬਫਰ) ||
        (ਅੱਖਰ *) addr + ggml_backend_ਬਫਰ_get_alloc_ਮਾਪ(ਬਫਰ, tensor) <=
        (ਅੱਖਰ *) ggml_backend_ਬਫਰ_get_ਅਧਾਰ(ਬਫਰ) + ggml_backend_ਬਫਰ_get_ਮਾਪ(ਬਫਰ));

    tensor->ਬਫਰ = ਬਫਰ;
    tensor->ਅੰਕੜੇ = addr;
    ਵਾਪਸ ggml_backend_ਬਫਰ_init_tensor(ਬਫਰ, tensor);
}

ਸਥਿਰ ਬਣਤਰ ggml_tensor * ਗ੍ਰਾਫ_copy_dup_tensor(ਬਣਤਰ ggml_hash_ਸੈੱਟ hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor ** ਗੰਢ_copies,
    ਬਣਤਰ ggml_context * ctx_allocated, ਬਣਤਰ ggml_context * ctx_unallocated, ਬਣਤਰ ggml_tensor * src) {

    GGML_ASSERT(src != ਨਿਰਾ);
    GGML_ASSERT(src->ਅੰਕੜੇ && "graph must be allocated");

    size_t id = ggml_hash_ਪਾਓ(&hash_ਸੈੱਟ, src);
    ਜੇ (id == GGML_HASHSET_ALREADY_EXISTS) {
        ਵਾਪਸ ਗੰਢ_copies[ggml_hash_ਲੱਭੋ(&hash_ਸੈੱਟ, src)];
    }

    ਬਣਤਰ ggml_tensor * dst = ggml_dup_tensor_layout(src->ਅੰਕੜੇ && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    ਜੇ (src->view_src != ਨਿਰਾ) {
        dst->view_src = ਗ੍ਰਾਫ_copy_dup_tensor(hash_ਸੈੱਟ, ਗੰਢ_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst->op_params, src->op_params, ਆਕਾਰ(dst->op_params));
    ggml_ਸੈੱਟ_name(dst, src->name);

    // copy src
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਬਣਤਰ ggml_tensor * s = src->src[i];
        ਜੇ (s == ਨਿਰਾ) {
            ਜਾਰੀ;
        }
        dst->src[i] = ਗ੍ਰਾਫ_copy_dup_tensor(hash_ਸੈੱਟ, ਗੰਢ_copies, ctx_allocated, ctx_unallocated, s);
    }

    ਗੰਢ_copies[id] = dst;
    ਵਾਪਸ dst;
}

ਸਥਿਰ ਖਾਲੀ ਗ੍ਰਾਫ_copy_init_tensor(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor ** ਗੰਢ_copies, ਤਾਰਕਿਕ * ਗੰਢ_init, ਬਣਤਰ ggml_tensor * src) {
    size_t id = ggml_hash_ਲੱਭੋ(hash_ਸੈੱਟ, src);
    ਜੇ (ਗੰਢ_init[id]) {
        ਵਾਪਸ;
    }
    ਗੰਢ_init[id] = ਸੱਚ;

    ਬਣਤਰ ggml_tensor * dst = ਗੰਢ_copies[id];
    ਜੇ (dst->view_src != ਨਿਰਾ) {
        ਗ੍ਰਾਫ_copy_init_tensor(hash_ਸੈੱਟ, ਗੰਢ_copies, ਗੰਢ_init, src->view_src);
        ਗਿਣਤੀ ggml_ਸਥਿਤੀ ਸਥਿਤੀ = ggml_backend_view_init(dst);
        GGML_ASSERT(ਸਥਿਤੀ == GGML_ਸਥਿਤੀ_SUCCESS);
    }
    ਨਹੀਂਤਾਂ {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC; i++) {
        ਬਣਤਰ ggml_tensor * s = src->src[i];
        ਜੇ (s == ਨਿਰਾ) {
            ਜਾਰੀ;
        }
        ਗ੍ਰਾਫ_copy_init_tensor(hash_ਸੈੱਟ, ਗੰਢ_copies, ਗੰਢ_init, s);
    }
}

ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_copy ggml_backend_ਗ੍ਰਾਫ_copy(ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ਬਣਤਰ ggml_hash_ਸੈੱਟ hash_ਸੈੱਟ = ggml_hash_ਸੈੱਟ_new(graph->ਦੇਖਿਆ_hash_ਸੈੱਟ.size);
    ਬਣਤਰ ggml_tensor ** ਗੰਢ_copies = (ggml_tensor **) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(hash_ਸੈੱਟ.size, ਆਕਾਰ(ਗੰਢ_copies[0])); // NOLINT
    ਤਾਰਕਿਕ * ਗੰਢ_init = (ਤਾਰਕਿਕ *) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(hash_ਸੈੱਟ.size, ਆਕਾਰ(ਗੰਢ_init[0]));

    ਬਣਤਰ ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_ਸੈੱਟ.size + ggml_ਗ੍ਰਾਫ_overhead_custom(graph->size, ਝੂਠ),
        /* .mem_buffer = */ ਨਿਰਾ,
        /* .no_alloc   = */ ਸੱਚ
    };

    ਬਣਤਰ ggml_context * ctx_allocated = ggml_init(params);
    ਬਣਤਰ ggml_context * ctx_unallocated = ggml_init(params);

    ਜੇ (ctx_allocated == ਨਿਰਾ || ctx_unallocated == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_ਸੈੱਟ_free(&hash_ਸੈੱਟ);
        ਮੁਕਤ(ਗੰਢ_copies);
        ਮੁਕਤ(ਗੰਢ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ਵਾਪਸ {
            /* .buffer           = */ ਨਿਰਾ,
            /* .ctx_allocated    = */ ਨਿਰਾ,
            /* .ctx_unallocated  = */ ਨਿਰਾ,
            /* .graph            = */ ਨਿਰਾ,
        };
    }

    // dup nodes
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਗ੍ਰਾਫ_copy_dup_tensor(hash_ਸੈੱਟ, ਗੰਢ_copies, ctx_allocated, ctx_unallocated, ਗੰਢ);
    }

    // allocate nodes
    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    ਜੇ (ਬਫਰ == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_ਸੈੱਟ_free(&hash_ਸੈੱਟ);
        ਮੁਕਤ(ਗੰਢ_copies);
        ਮੁਕਤ(ਗੰਢ_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ਵਾਪਸ {
            /* .buffer           = */ ਨਿਰਾ,
            /* .ctx_allocated    = */ ਨਿਰਾ,
            /* .ctx_unallocated  = */ ਨਿਰਾ,
            /* .graph            = */ ਨਿਰਾ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਗ੍ਰਾਫ_copy_init_tensor(&hash_ਸੈੱਟ, ਗੰਢ_copies, ਗੰਢ_init, ਗੰਢ);
    }

    // build graph copy
    ਬਣਤਰ ggml_cgraph * ਗ੍ਰਾਫ_copy = ggml_new_ਗ੍ਰਾਫ_custom(ctx_allocated, graph->size, ਝੂਠ);
    ਲਈ (ਪੂਰਨ i = 0; i < graph->n_nodes; i++) {
        ਬਣਤਰ ggml_tensor * ਗੰਢ = graph->nodes[i];
        ਬਣਤਰ ggml_tensor * ਗੰਢ_copy = ਗੰਢ_copies[ggml_hash_ਲੱਭੋ(&hash_ਸੈੱਟ, ਗੰਢ)];
        ਗ੍ਰਾਫ_copy->nodes[i] = ਗੰਢ_copy;
    }
    ਗ੍ਰਾਫ_copy->n_nodes = graph->n_nodes;

    ggml_hash_ਸੈੱਟ_free(&hash_ਸੈੱਟ);
    ਮੁਕਤ(ਗੰਢ_copies);
    ਮੁਕਤ(ਗੰਢ_init);

    ਵਾਪਸ {
        /* .buffer           = */ ਬਫਰ,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ ਗ੍ਰਾਫ_copy,
    };
}

ਖਾਲੀ ggml_backend_ਗ੍ਰਾਫ_copy_free(ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_copy copy) {
    ggml_backend_ਬਫਰ_free(copy.ਬਫਰ);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

ਤਾਰਕਿਕ ggml_backend_compare_ਗ੍ਰਾਫ_backend(ggml_backend_t backend1, ggml_backend_t backend2, ਬਣਤਰ ggml_cgraph * graph, ggml_backend_eval_callback callback, ਖਾਲੀ * user_ਅੰਕੜੇ, ਬਣਤਰ ggml_tensor ਅਚਲ * ਅਚਲ * test_nodes, size_t ਅੰਕ_test_nodes) {
    ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_copy copy = ggml_backend_ਗ੍ਰਾਫ_copy(backend2, graph);
    ਜੇ (copy.ਬਫਰ == ਨਿਰਾ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਬਣਤਰ ggml_cgraph * g1 = graph;
    ਬਣਤਰ ggml_cgraph * g2 = copy.graph;

    ਪ੍ਰਤਿਜ੍ਞਾ(g1->n_nodes == g2->n_nodes);

    ਜੇ (ਅੰਕ_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_ਗ੍ਰਾਫ_compute(backend1, g1);
        ggml_backend_ਗ੍ਰਾਫ_compute(backend2, g2);

        ਤਾਰਕਿਕ verified = ਝੂਠ;
        ਲਈ (ਪੂਰਨ i = 0; i < g1->n_nodes; i++) {
            ਲਈ (size_t j = 0; j < ਅੰਕ_test_nodes; ++j) {
                ਜੇ (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_ਅੰਕੜੇ);
                    verified = ਸੱਚ;
                }
            }
        }
        GGML_ASSERT(verified);
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਪੂਰਨ i = 0; i < g1->n_nodes; i++) {
            ਬਣਤਰ ggml_tensor * t1 = g1->nodes[i];
            ਬਣਤਰ ggml_tensor * t2 = g2->nodes[i];

            ਪ੍ਰਤਿਜ੍ਞਾ(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ਬਣਤਰ ggml_cgraph g1v = ggml_ਗ੍ਰਾਫ_view(g1, i, i + 1);
            ਬਣਤਰ ggml_cgraph g2v = ggml_ਗ੍ਰਾਫ_view(g2, i, i + 1);

            ggml_backend_ਗ੍ਰਾਫ_compute(backend1, &g1v);
            ggml_backend_ਗ੍ਰਾਫ_compute(backend2, &g2v);

            ਜੇ (ggml_is_view_op(t1->op)) {
                ਜਾਰੀ;
            }

            // compare results, calculate rms etc
            ਜੇ (!callback(i, t1, t2, user_ਅੰਕੜੇ)) {
                ਰੋਕੋ;
            }
        }
    }
    ggml_backend_ਗ੍ਰਾਫ_copy_free(copy);

    ਵਾਪਸ ਸੱਚ;
}

// CPU backend - buffer

ਸਥਿਰ ਖਾਲੀ * ggml_backend_cpu_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    uintptr_t ਅੰਕੜੇ = (uintptr_t)ਬਫਰ->context;

    // align the buffer
    ਜੇ (ਅੰਕੜੇ % TENSOR_ALIGNMENT != 0) {
        ਅੰਕੜੇ = GGML_PAD(ਅੰਕੜੇ, TENSOR_ALIGNMENT);
    }

    ਵਾਪਸ (ਖਾਲੀ *)ਅੰਕੜੇ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_ASSERT(ਬਫਰ);
    ggml_aligned_free(ਬਫਰ->context, ਬਫਰ->size);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor, uint8_t value, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    memset((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, value, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t size) {
    GGML_ASSERT(tensor);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cpu_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst) {
    GGML_ASSERT(src);
    ਜੇ (ggml_backend_ਬਫਰ_is_host(src->ਬਫਰ)) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst->ਅੰਕੜੇ, src->ਅੰਕੜੇ, ggml_nbytes(src));
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_cpu_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    GGML_ASSERT(ਬਫਰ);
    memset(ਬਫਰ->context, value, ਬਫਰ->size);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_i ggml_backend_cpu_ਬਫਰ_i = {
    /* .free_buffer     = */ ggml_backend_cpu_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_cpu_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ਨਿਰਾ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_cpu_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_i ggml_backend_cpu_ਬਫਰ_from_ptr_i = {
    /* .free_buffer     = */ ਨਿਰਾ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ਨਿਰਾ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_cpu_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CPU";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_cpu_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਖਾਲੀ * ਅੰਕੜੇ = ggml_aligned_malloc(size);

    ਜੇ (ਅੰਕੜੇ == ਨਿਰਾ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        ਵਾਪਸ ਨਿਰਾ;
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_cpu_ਬਫਰ_i, ਅੰਕੜੇ, size);
}

ਸਥਿਰ size_t ggml_backend_cpu_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ TENSOR_ALIGNMENT;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_cpu_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_ਬਫਰ_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_ਬਫਰ_type_alloc_ਬਫਰ,
            /* .alloc_buffer_n      = */ ਨਿਰਾ,
            /* .get_alignment       = */ ggml_backend_cpu_ਬਫਰ_type_get_alignment,
            /* .get_max_size        = */ ਨਿਰਾ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ਨਿਰਾ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ਨਿਰਾ,
            /* .is_host             = */ ggml_backend_cpu_ਬਫਰ_type_is_host,
        },
        /* .device  = */ ਨਿਰਾ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_cpu_ਬਫਰ_from_ptr_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ "CPU_Mapped";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_cpu_ਬਫਰ_from_ptr_type(ਖਾਲੀ) {
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_cpu_ਬਫਰ_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_ਬਫਰ_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_ਬਫਰ_type_alloc_ਬਫਰ,
            /* .alloc_buffer_n      = */ ਨਿਰਾ,
            /* .get_alignment       = */ ggml_backend_cpu_ਬਫਰ_type_get_alignment,
            /* .get_max_size        = */ ਨਿਰਾ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ਨਿਰਾ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ਨਿਰਾ,
            /* .is_host             = */ ggml_backend_cpu_ਬਫਰ_type_is_host,
        },
        /* .device  = */ ਨਿਰਾ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ਨਿਰਾ,
    };

    ਵਾਪਸ &ggml_backend_cpu_ਬਫਰ_type;
}

ggml_backend_ਬਫਰ_t ggml_backend_cpu_ਬਫਰ_from_ptr(ਖਾਲੀ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    ਵਾਪਸ ggml_backend_ਬਫਰ_init(ggml_backend_cpu_ਬਫਰ_from_ptr_type(), ggml_backend_cpu_ਬਫਰ_from_ptr_i, ptr, size);
}
