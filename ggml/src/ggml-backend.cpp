// Note: porting this file to C++ is a work in progress

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#ವ್ಯಾಖ್ಯಾನಿಸು WIN32_LEAN_AND_ಸರಾಸರಿ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NOMINMAX
#   define NOMINMAX
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <windows.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-alloc.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <limits.h>
#ಅನ್ತರ್ಭಾವ <stdarg.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <vector>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __APPLE__
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <sys/sysctl.h>
#ಆದರೆ_ಅಂತ್ಯ


// backend buffer type

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_buft_name(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(buft);
    ಹಿಂತಿರುಗಿ buft->iface.get_name(buft);
}

ggml_backend_ಬಫರ್_t ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    ಆದರೆ (size == 0) {
        // return a dummy buffer for zero-sized allocations
        ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, {}, ಖಾಲಿ, 0);
    }
    ಹಿಂತಿರುಗಿ buft->iface.alloc_ಬಫರ್(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
ರಚನೆ ggml_backend_buft_alloc_ಬಫರ್_n_plan_ಅಂಶ {
    size_t size;  // total bytes for this buffer
    ಪೂರ್ಣಾಂಕ    ಮೊದಲ; // first tensor index (inclusive)
    ಪೂರ್ಣಾಂಕ    ಕೊನೆಯ;  // last tensor index (exclusive)
};

ಬಳಸಿ ggml_backend_buft_alloc_ಬಫರ್_n_plan_t = std::vector<ggml_backend_buft_alloc_ಬಫರ್_n_plan_ಅಂಶ>;

ಸ್ಥಿರ ggml_backend_buft_alloc_ಬಫರ್_n_plan_t ggml_backend_buft_alloc_ಬಫರ್_n_plan(
        ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ggml_backend_buft_alloc_ಬಫರ್_n_plan_t plan;

    ಸ್ಥಿರಾಂಕ size_t alignment = ggml_backend_buft_get_alignment(buft);
    ಸ್ಥಿರಾಂಕ size_t max_size  = ggml_backend_buft_get_max_ಗಾತ್ರ(buft);

    size_t cur_buf_ಗಾತ್ರ = 0;
    ಪೂರ್ಣಾಂಕ    ಮೊದಲ        = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n_tensors; i++) {
        size_t this_ಗಾತ್ರ = 0;
        ರಚನೆ ggml_tensor * t = tensors[i];
        ಆದರೆ (t->ದತ್ತಾಂಶ == ಖಾಲಿ && t->view_src == ಖಾಲಿ) {
            this_ಗಾತ್ರ = GGML_PAD(ggml_backend_buft_get_alloc_ಗಾತ್ರ(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        ಆದರೆ (cur_buf_ಗಾತ್ರ > 0 && (cur_buf_ಗಾತ್ರ + this_ಗಾತ್ರ) > max_size) {
            plan.push_back({ cur_buf_ಗಾತ್ರ, ಮೊದಲ, i });
            cur_buf_ಗಾತ್ರ = this_ಗಾತ್ರ;
            ಮೊದಲ        = i;
        } ಇಲ್ಲದಿದ್ದರೆ {
            cur_buf_ಗಾತ್ರ += this_ಗಾತ್ರ;
        }
    }

    ಆದರೆ (cur_buf_ಗಾತ್ರ > 0) {
        plan.push_back({ cur_buf_ಗಾತ್ರ, ಮೊದಲ, n_tensors });
    }

    ಹಿಂತಿರುಗಿ plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_buft_alloc_ಬಫರ್_n_default(ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ಸ್ಥಿರಾಂಕ ggml_backend_buft_alloc_ಬಫರ್_n_plan_t plan = ggml_backend_buft_alloc_ಬಫರ್_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_ಬಫರ್_t> buffers;
    buffers.reserve(plan.size());

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_backend_buft_alloc_ಬಫರ್_n_plan_ಅಂಶ & ಅಂಶ : plan) {
        ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_buft_alloc_ಬಫರ್(buft, ಅಂಶ.size);
        ಆದರೆ (ಬಫರ್ == ಖಾಲಿ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), ಅಂಶ.size);
            ಗಾಗಿ (ggml_backend_ಬಫರ್_t b : buffers) {
                ggml_backend_ಬಫರ್_free(b);
            }
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }

        ರಚನೆ ggml_tallocr tallocr = ggml_tallocr_new(ಬಫರ್);

        // allocate tensors in the current buffer
        ರಚನೆ ggml_tensor * t_failed = ಖಾಲಿ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಅಂಶ.ಮೊದಲ; j < ಅಂಶ.ಕೊನೆಯ; j++) {
            ರಚನೆ ggml_tensor * t = tensors[j];
            ಆದರೆ (t->ದತ್ತಾಂಶ == ಖಾಲಿ) {
                ಆದರೆ (t->view_src == ಖಾಲಿ) {
                    ಆದರೆ (ggml_tallocr_alloc(&tallocr, t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (t->ಬಫರ್ == ಖಾಲಿ) {
                    ಆದರೆ (ggml_backend_view_init(t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (t->view_src != ಖಾಲಿ && t->ಬಫರ್ == ಖಾಲಿ) {
                    // view of a pre-allocated tensor
                    ಆದರೆ (ggml_backend_view_init(t) != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                        t_failed = t;
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
        ಆದರೆ (t_failed != ಖಾಲಿ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            ಗಾಗಿ (ggml_backend_ಬಫರ್_t b : buffers) {
                ggml_backend_ಬಫರ್_free(b);
            }
            ggml_backend_ಬಫರ್_free(ಬಫರ್);
            ಹಿಂತಿರುಗಿ ಖಾಲಿ;
        }

        buffers.push_back(ಬಫರ್);
    }

    ಆದರೆ (buffers.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಆದರೆ (buffers.size() == 1) {
        ಹಿಂತಿರುಗಿ buffers[0];
    }

    ಹಿಂತಿರುಗಿ ggml_backend_multi_ಬಫರ್_alloc_ಬಫರ್(buffers.ದತ್ತಾಂಶ(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
ಸ್ಥಿರ size_t ggml_backend_buft_get_alloc_ಗಾತ್ರ_n_default(ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    ಸ್ಥಿರಾಂಕ ggml_backend_buft_alloc_ಬಫರ್_n_plan_t plan = ggml_backend_buft_alloc_ಬಫರ್_n_plan(buft, tensors, n_tensors);

    size_t ಒಟ್ಟು = 0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ggml_backend_buft_alloc_ಬಫರ್_n_plan_ಅಂಶ & ಅಂಶ : plan) {
        ಒಟ್ಟು += ಅಂಶ.size;
    }
    ಹಿಂತಿರುಗಿ ಒಟ್ಟು;
}

ggml_backend_ಬಫರ್_t ggml_backend_buft_alloc_ಬಫರ್_n(ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    GGML_ASSERT(buft);
    ಆದರೆ (buft->iface.alloc_ಬಫರ್_n) {
        ಹಿಂತಿರುಗಿ buft->iface.alloc_ಬಫರ್_n(buft, tensors, n_tensors);
    }
    ಹಿಂತಿರುಗಿ ggml_backend_buft_alloc_ಬಫರ್_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(buft);
    ಹಿಂತಿರುಗಿ buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    ಆದರೆ (buft->iface.get_max_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ buft->iface.get_max_ಗಾತ್ರ(buft);
    }
    ಹಿಂತಿರುಗಿ SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    ಆದರೆ (buft->iface.get_alloc_ಗಾತ್ರ) {
        size_t size = buft->iface.get_alloc_ಗಾತ್ರ(buft, tensor);
        ಪ್ರತಿಜ್ಞಾ(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_ಖಾಲಿಗಳು(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_ಗಾತ್ರ_may_expand(tensor->op));

        ಹಿಂತಿರುಗಿ size;
    }
    ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_ಗಾತ್ರ_n(ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors) {
    GGML_ASSERT(buft);
    ಆದರೆ (buft->iface.get_alloc_ಗಾತ್ರ_n) {
        ಹಿಂತಿರುಗಿ buft->iface.get_alloc_ಗಾತ್ರ_n(buft, tensors, n_tensors);
    }
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_alloc_ಗಾತ್ರ_n_default(buft, tensors, n_tensors);
}

ತಾರ್ಕಿಕ ggml_backend_buft_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(buft);
    ಆದರೆ (buft->iface.is_host) {
        ಹಿಂತಿರುಗಿ buft->iface.is_host(buft);
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(buft);
    ಹಿಂತಿರುಗಿ buft->device;
}

// backend buffer

ggml_backend_ಬಫರ್_t ggml_backend_ಬಫರ್_init(
               ggml_backend_ಬಫರ್_type_t buft,
        ರಚನೆ ggml_backend_ಬಫರ್_i      iface,
               ಶೂನ್ಯ *                     context,
               size_t                     size) {
    ggml_backend_ಬಫರ್_t ಬಫರ್ = ಹೊಸ ggml_backend_ಬಫರ್ {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_ಬಫರ್_USAGE_ANY
    };

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_ಬಫರ್_name(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_name(ggml_backend_ಬಫರ್_get_type(ಬಫರ್));
}

ಶೂನ್ಯ ggml_backend_ಬಫರ್_free(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಆದರೆ (ಬಫರ್ == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ಬಫರ್->iface.free_ಬಫರ್ != ಖಾಲಿ) {
        ಬಫರ್->iface.free_ಬಫರ್(ಬಫರ್);
    }
    ಅಳಿಸು ಬಫರ್;
}

size_t ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ಹಿಂತಿರುಗಿ ಬಫರ್->size;
}

ಶೂನ್ಯ * ggml_backend_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    // get_base is optional if the buffer is zero-sized
    ಆದರೆ (!ggml_backend_ಬಫರ್_is_meta(ಬಫರ್) && ಬಫರ್->size == 0) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    ಆದರೆ (!ಬಫರ್->iface.get_ತಳ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಶೂನ್ಯ * ತಳ = ಬಫರ್->iface.get_ತಳ(ಬಫರ್);

    GGML_ASSERT(ತಳ != ಖಾಲಿ && "backend buffer base cannot be NULL");

    ಹಿಂತಿರುಗಿ ತಳ;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(ಬಫರ್);
    // init_tensor is optional
    ಆದರೆ (ಬಫರ್->iface.init_tensor) {
        ಹಿಂತಿರುಗಿ ಬಫರ್->iface.init_tensor(ಬಫರ್, tensor);
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಶೂನ್ಯ ggml_backend_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    GGML_ASSERT(ಬಫರ್);
    // clear is optional if the buffer is zero-sized
    ಆದರೆ (ಬಫರ್->size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಬಫರ್->iface.clear(ಬಫರ್, value);
}

size_t ggml_backend_ಬಫರ್_get_alignment(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_alignment(ggml_backend_ಬಫರ್_get_type(ಬಫರ್));
}

size_t ggml_backend_ಬಫರ್_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_get_type(ಬಫರ್));
}

size_t ggml_backend_ಬಫರ್_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_get_type(ಬಫರ್), tensor);
}

ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_host(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_is_host(ggml_backend_ಬಫರ್_get_type(ಬಫರ್));
}

ಶೂನ್ಯ ggml_backend_ಬಫರ್_ಗಣ_usage(ggml_backend_ಬಫರ್_t ಬಫರ್, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage) {
    GGML_ASSERT(ಬಫರ್);
    ಬಫರ್->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    ಆದರೆ (ggml_backend_ಬಫರ್_is_multi_ಬಫರ್(ಬಫರ್)) {
        ggml_backend_multi_ಬಫರ್_ಗಣ_usage(ಬಫರ್, usage);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_backend_ಬಫರ್_is_meta(ಬಫರ್)) {
        ggml_backend_meta_ಬಫರ್_ಗಣ_usage(ಬಫರ್, usage);
    }
}

ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage ggml_backend_ಬಫರ್_get_usage(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ಹಿಂತಿರುಗಿ ಬಫರ್->usage;
}

ggml_backend_ಬಫರ್_type_t ggml_backend_ಬಫರ್_get_type(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ಹಿಂತಿರುಗಿ ಬಫರ್->buft;
}

ಶೂನ್ಯ ggml_backend_ಬಫರ್_reset(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ಆದರೆ (ಬಫರ್->iface.reset) {
        ಬಫರ್->iface.reset(ಬಫರ್);
    }
}

ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_copy_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst) {
    ggml_backend_ಬಫರ್_t dst_buf = dst->view_src ? dst->view_src->ಬಫರ್ : dst->ಬಫರ್;
    ಆದರೆ (dst_buf->iface.cpy_tensor) {
        ಹಿಂತಿರುಗಿ dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    ಆದರೆ (backend == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ backend->guid;
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_name(ggml_backend_t backend) {
    ಆದರೆ (backend == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ "NULL";
    }
    ಹಿಂತಿರುಗಿ backend->iface.get_name(backend);
}

ಶೂನ್ಯ ggml_backend_free(ggml_backend_t backend) {
    ಆದರೆ (backend == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    backend->iface.ಬಿಡುಗಡೆ(backend);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_get_default_ಬಫರ್_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ ggml_backend_dev_ಬಫರ್_type(backend->device);
}

ggml_backend_ಬಫರ್_t ggml_backend_alloc_ಬಫರ್(ggml_backend_t backend, size_t size) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_get_default_ಬಫರ್_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_alignment(ggml_backend_get_default_ಬಫರ್_type(backend));
}

size_t ggml_backend_get_max_ಗಾತ್ರ(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ ggml_backend_buft_get_max_ಗಾತ್ರ(ggml_backend_get_default_ಬಫರ್_type(backend));
}

ಶೂನ್ಯ ggml_backend_tensor_ಗಣ_async(ggml_backend_t backend, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ಆದರೆ (backend->iface.ಗಣ_tensor_async == ಖಾಲಿ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_ಗಣ(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        backend->iface.ಗಣ_tensor_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_get_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    ಆದರೆ (backend->iface.get_tensor_async == ಖಾಲಿ) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        backend->iface.get_tensor_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_ಗಣ_2d_async(ggml_backend_t backend, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");

    ಆದರೆ (n_copies <= 1 || backend->iface.ಗಣ_tensor_2d_async == ಖಾಲಿ) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_ಗಣ_async(backend, tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ + i*stride_tensor, size);
        }
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.ಗಣ_tensor_2d_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಶೂನ್ಯ ggml_backend_tensor_get_2d_async(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");

    ಆದರೆ (n_copies <= 1 || backend->iface.get_tensor_2d_async == ಖಾಲಿ) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ + i*stride_tensor, size);
        }
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಶೂನ್ಯ ggml_backend_tensor_ಗಣ(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
    GGML_ASSERT(buf != ಖಾಲಿ && "tensor buffer not set");

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.ಗಣ_tensor(buf, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಶೂನ್ಯ ggml_backend_tensor_get(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
    GGML_ASSERT(buf != ಖಾಲಿ && "tensor buffer not set");

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
}

ಶೂನ್ಯ ggml_backend_tensor_ಗಣ_2d(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    GGML_ASSERT(tensor);
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
    GGML_ASSERT(buf != ಖಾಲಿ && "tensor buffer not set");

    ಆದರೆ (n_copies <= 1 || buf->iface.ಗಣ_tensor_2d == ಖಾಲಿ) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_ಗಣ(tensor, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ + i*stride_tensor, size);
        }
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.ಗಣ_tensor_2d(buf, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಶೂನ್ಯ ggml_backend_tensor_get_2d(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ) {
    GGML_ASSERT(tensor);
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
    GGML_ASSERT(buf != ಖಾಲಿ && "tensor buffer not set");

    ಆದರೆ (n_copies <= 1 || buf->iface.get_tensor_2d == ಖಾಲಿ) {
        ಗಾಗಿ (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (ಅಕ್ಷರ *) ದತ್ತಾಂಶ + i*stride_ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ + i*stride_tensor, size);
        }
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size, n_copies, stride_tensor, stride_ದತ್ತಾಂಶ);
}

ಶೂನ್ಯ ggml_backend_tensor_memset(ರಚನೆ ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;

    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(buf != ಖಾಲಿ && "tensor buffer not set");
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ != ಖಾಲಿ && "tensor not allocated");
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != ಖಾಲಿ && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, ಸ್ಥಾನಾಂತರ, size);
}

ಶೂನ್ಯ ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಆದರೆ (backend->iface.synchronize == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_ನಕ್ಷೆ_plan_t ggml_backend_ನಕ್ಷೆ_plan_create(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ನಕ್ಷೆ_plan_create != ಖಾಲಿ);

    ಹಿಂತಿರುಗಿ backend->iface.ನಕ್ಷೆ_plan_create(backend, cgraph);
}

ಶೂನ್ಯ ggml_backend_ನಕ್ಷೆ_plan_free(ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ನಕ್ಷೆ_plan_free != ಖಾಲಿ);

    backend->iface.ನಕ್ಷೆ_plan_free(backend, plan);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_ನಕ್ಷೆ_plan_compute(ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.ನಕ್ಷೆ_plan_compute != ಖಾಲಿ);

    ಹಿಂತಿರುಗಿ backend->iface.ನಕ್ಷೆ_plan_compute(backend, plan);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_ನಕ್ಷೆ_compute(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ err = ggml_backend_ನಕ್ಷೆ_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    ಹಿಂತಿರುಗಿ err;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_ನಕ್ಷೆ_compute_async(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ backend->iface.ನಕ್ಷೆ_compute(backend, cgraph);
}

ತಾರ್ಕಿಕ ggml_backend_supports_op(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ ggml_backend_dev_supports_op(backend->device, op);
}

ತಾರ್ಕಿಕ ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ ggml_backend_dev_supports_buft(backend->device, buft);
}

ತಾರ್ಕಿಕ ggml_backend_offload_op(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    ಹಿಂತಿರುಗಿ backend->device;
}

// backend copy

ಶೂನ್ಯ ggml_backend_tensor_copy(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ಆದರೆ (src == dst) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (ggml_backend_ಬಫರ್_is_host(src->ಬಫರ್)) {
        ggml_backend_tensor_ಗಣ(dst, src->ದತ್ತಾಂಶ, 0, ggml_nbytes(src));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_backend_ಬಫರ್_is_host(dst->ಬಫರ್)) {
        ggml_backend_tensor_get(src, dst->ದತ್ತಾಂಶ, 0, ggml_nbytes(src));
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ggml_backend_ಬಫರ್_copy_tensor(src, dst)) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_ಬಫರ್_name(src->ಬಫರ್), ggml_backend_ಬಫರ್_name(dst->ಬಫರ್));
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        ಶೂನ್ಯ * ದತ್ತಾಂಶ = ಸ್ಮರಣೆ_ಕೊಡು(nbytes);
        ggml_backend_tensor_get(src, ದತ್ತಾಂಶ, 0, nbytes);
        ggml_backend_tensor_ಗಣ(dst, ದತ್ತಾಂಶ, 0, nbytes);
        ಬಿಡುಗಡೆ(ದತ್ತಾಂಶ);
    }
}

ಶೂನ್ಯ ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    ಆದರೆ (src == dst) {
        ಹಿಂತಿರುಗಿ;
    }

    GGML_ASSERT(backend_dst);
    ಆದರೆ (backend_dst->iface.cpy_tensor_async != ಖಾಲಿ) {
        ಆದರೆ (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            ಹಿಂತಿರುಗಿ;
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
    ಆದರೆ (device == ಖಾಲಿ || device->iface.event_new == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ device->iface.event_new(device);
}

ಶೂನ್ಯ ggml_backend_event_free(ggml_backend_event_t event) {
    ಆದರೆ (event == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }
    event->device->iface.event_free(event->device, event);
}

ಶೂನ್ಯ ggml_backend_event_ದಾಖಲೆ(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_ದಾಖಲೆ != ಖಾಲಿ);

    backend->iface.event_ದಾಖಲೆ(backend, event);
}

ಶೂನ್ಯ ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

ಶೂನ್ಯ ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != ಖಾಲಿ);

    backend->iface.event_wait(backend, event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_ನಕ್ಷೆ_optimize(ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params * params) {
    GGML_ASSERT(backend);
    ಆದರೆ (backend->iface.ನಕ್ಷೆ_optimize != ಖಾಲಿ) {
        backend->iface.ನಕ್ಷೆ_optimize(backend, cgraph, params);
    }
}

// Backend device

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.get_name(device);
}

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.get_description(device);
}

ಶೂನ್ಯ ggml_backend_dev_ಸ್ಮರಣೆ(ggml_backend_dev_t device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    GGML_ASSERT(device);
    device->iface.get_ಸ್ಮರಣೆ(device, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.get_type(device);
}

ಶೂನ್ಯ ggml_backend_dev_get_props(ggml_backend_dev_t device, ರಚನೆ ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, ಗಾತ್ರ(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.init_backend(device, params);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_dev_ಬಫರ್_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.get_ಬಫರ್_type(device);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_dev_host_ಬಫರ್_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    ಆದರೆ (device->iface.get_host_ಬಫರ್_type == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ device->iface.get_host_ಬಫರ್_type(device);
}

ggml_backend_ಬಫರ್_t ggml_backend_dev_ಬಫರ್_from_host_ptr(ggml_backend_dev_t device, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.ಬಫರ್_from_host_ptr(device, ptr, size, max_tensor_ಗಾತ್ರ);
}

ತಾರ್ಕಿಕ ggml_backend_dev_supports_op(ggml_backend_dev_t device, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.supports_op(device, op);
}

ತಾರ್ಕಿಕ ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ASSERT(device);
    ಹಿಂತಿರುಗಿ device->iface.supports_buft(device, buft);
}

ತಾರ್ಕಿಕ ggml_backend_dev_offload_op(ggml_backend_dev_t device, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    GGML_ASSERT(device);
    ಆದರೆ (device->iface.offload_op != ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ device->iface.offload_op(device, op);
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

// Backend (reg)

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ಹಿಂತಿರುಗಿ reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    ಹಿಂತಿರುಗಿ reg->iface.get_device_ಎಣಿಕೆ(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(reg);
    ಹಿಂತಿರುಗಿ reg->iface.get_device(reg, ಸೂಚ್ಯಂಕ);
}

ಶೂನ್ಯ * ggml_backend_reg_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    GGML_ASSERT(reg);
    ಆದರೆ (!reg->iface.get_proc_ವಿಳಾಸ) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ reg->iface.get_proc_ವಿಳಾಸ(reg, name);
}

// multi-buffer buffer

ರಚನೆ ggml_backend_multi_ಬಫರ್_context {
    ggml_backend_ಬಫರ್_t * buffers;
    size_t n_buffers;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_multi_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ggml_backend_multi_ಬಫರ್_context * ctx = (ggml_backend_multi_ಬಫರ್_context *) ಬಫರ್->context;
    ಗಾಗಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ಬಫರ್_free(ctx->buffers[i]);
    }

    ಬಿಡುಗಡೆ(ctx->buffers);
    ಬಿಡುಗಡೆ(ctx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_multi_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    GGML_ASSERT(ಬಫರ್);
    ggml_backend_multi_ಬಫರ್_context * ctx = (ggml_backend_multi_ಬಫರ್_context *) ಬಫರ್->context;
    ಗಾಗಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ಬಫರ್_clear(ctx->buffers[i], value);
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_i ggml_backend_multi_ಬಫರ್_i = {
    /* .free_buffer     = */ ggml_backend_multi_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ಖಾಲಿ,
    /* .init_tensor     = */ ಖಾಲಿ,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ಖಾಲಿ,
    /* .get_tensor      = */ ಖಾಲಿ,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ಖಾಲಿ,
    /* .clear           = */ ggml_backend_multi_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

ggml_backend_ಬಫರ್_t ggml_backend_multi_ಬಫರ್_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_t * buffers, size_t n_buffers) {
    ggml_backend_multi_ಬಫರ್_context * ctx = (ggml_backend_multi_ಬಫರ್_context *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(ರಚನೆ ggml_backend_multi_ಬಫರ್_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_ಬಫರ್_t *) ಸ್ಮರಣೆ_ಕೊಡು(n_buffers * ಗಾತ್ರ(ggml_backend_ಬಫರ್_t));

    GGML_ASSERT(ctx->buffers != ಖಾಲಿ);

    size_t ಒಟ್ಟು_ಗಾತ್ರ = 0;
    ಗಾಗಿ (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        ಒಟ್ಟು_ಗಾತ್ರ += ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(buffers[i]);
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buffers[0]->buft, ggml_backend_multi_ಬಫರ್_i, ctx, ಒಟ್ಟು_ಗಾತ್ರ);
}

ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_multi_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ಹಿಂತಿರುಗಿ ಬಫರ್->iface.free_ಬಫರ್ == ggml_backend_multi_ಬಫರ್_free_ಬಫರ್;
}

ಶೂನ್ಯ ggml_backend_multi_ಬಫರ್_ಗಣ_usage(ggml_backend_ಬಫರ್_t ಬಫರ್, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage) {
    GGML_ASSERT(ಬಫರ್);
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_multi_ಬಫರ್(ಬಫರ್));
    ggml_backend_multi_ಬಫರ್_context * ctx = (ggml_backend_multi_ಬಫರ್_context *) ಬಫರ್->context;
    ಗಾಗಿ (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_ಬಫರ್_ಗಣ_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
ಸ್ಥಿರ ರಚನೆ ggml_tensor * ggml_dup_tensor_layout(ರಚನೆ ggml_context * ctx, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor) {
    ರಚನೆ ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    ಹಿಂತಿರುಗಿ dup;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_is_view_op(ಎಣಿಕೆ ggml_op op) {
    ಹಿಂತಿರುಗಿ op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SCHED_MAX_BACKENDS
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SCHED_MAX_BACKENDS 16
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS 30
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SCHED_MAX_COPIES
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SCHED_MAX_COPIES 4
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು {
    ಪೂರ್ಣಾಂಕ backend_id;
    ಪೂರ್ಣಾಂಕ i_ಆರಂಭ;
    ಪೂರ್ಣಾಂಕ i_ಕೊನೆ;
    ರಚನೆ ggml_tensor ** inputs;
    ಪೂರ್ಣಾಂಕ n_inputs;
    ಪೂರ್ಣಾಂಕ inputs_ಸಾಮರ್ಥ್ಯ;
    // graph view of this split
    ರಚನೆ ggml_cgraph graph;
};

ರಚನೆ ggml_backend_sched {
    ತಾರ್ಕಿಕ is_reset; // true if the scheduler has been reset since the last graph split
    ತಾರ್ಕಿಕ is_alloc;

    ಪೂರ್ಣಾಂಕ n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_ಬಫರ್_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    ರಚನೆ ggml_hash_ಗಣ  hash_ಗಣ;
    ಪೂರ್ಣಾಂಕ                 * hv_tensor_backend_ids; // [hash_set.size]
    ರಚನೆ ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_ids; // [graph_size]
    ಪೂರ್ಣಾಂಕ * ಎಲೆ_backend_ids; // [graph_size]

    ಪೂರ್ಣಾಂಕ * prev_ಗಂಟು_backend_ids; // [graph_size]
    ಪೂರ್ಣಾಂಕ * prev_ಎಲೆ_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    ರಚನೆ ggml_cgraph graph;

    // graph splits
    ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * splits;
    ಪೂರ್ಣಾಂಕ n_splits;
    ಪೂರ್ಣಾಂಕ splits_ಸಾಮರ್ಥ್ಯ;

    // pipeline parallelism support
    ಪೂರ್ಣಾಂಕ n_copies;
    ಪೂರ್ಣಾಂಕ cur_copy;
    ಪೂರ್ಣಾಂಕ ಮುಂದಿನ_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    ರಚನೆ ggml_tensor ** ನಕ್ಷೆ_inputs;
    ಪೂರ್ಣಾಂಕ n_ನಕ್ಷೆ_inputs;
    ಪೂರ್ಣಾಂಕ ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ;

    ರಚನೆ ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    ಶೂನ್ಯ * callback_eval_user_ದತ್ತಾಂಶ;

    ggml_backend_sched_copy_callback callback_copy;
    ಶೂನ್ಯ * callback_copy_user_ದತ್ತಾಂಶ;

    ggml_backend_sched_realloc_callback callback_realloc;
    ಶೂನ್ಯ * callback_realloc_user_ದತ್ತಾಂಶ;

    ಅಕ್ಷರ * context_ಬಫರ್;
    size_t context_ಬಫರ್_ಗಾತ್ರ;

    ತಾರ್ಕಿಕ op_offload;

    ಪೂರ್ಣಾಂಕ debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    ಪೂರ್ಣಾಂಕ ನಕ್ಷೆ_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ prev_ನಕ್ಷೆ_ಗಾತ್ರ;

    ಪೂರ್ಣಾಂಕ debug_realloc;
};

#ವ್ಯಾಖ್ಯಾನಿಸು hash_id(tensor) ggml_hash_ಕಂಡುಹಿಡಿ_or_ಸೇರಿಸು(&sched->hash_ಗಣ, tensor)
#ವ್ಯಾಖ್ಯಾನಿಸು tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#ವ್ಯಾಖ್ಯಾನಿಸು tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#ವ್ಯಾಖ್ಯಾನಿಸು tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_ವಿಭಜಿಸು_inputs_grow(ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು) {
    ಪೂರ್ಣಾಂಕ new_cap = GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS;
    ಆದರೆ (ವಿಭಜಿಸು->inputs_ಸಾಮರ್ಥ್ಯ > 0) {
        new_cap = 2*ವಿಭಜಿಸು->inputs_ಸಾಮರ್ಥ್ಯ;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, ವಿಭಜಿಸು->inputs_ಸಾಮರ್ಥ್ಯ, new_cap);
    }
    ಸ್ವಯಂಚಾಲಿತ * pnew = (ರಚನೆ ggml_tensor **) realloc((ಶೂನ್ಯ *) ವಿಭಜಿಸು->inputs, new_cap * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ಆದರೆ (pnew == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    ವಿಭಜಿಸು->inputs = pnew;
    ವಿಭಜಿಸು->inputs_ಸಾಮರ್ಥ್ಯ = new_cap;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_ನಕ್ಷೆ_inputs_grow(ggml_backend_sched_t sched) {
    ಪೂರ್ಣಾಂಕ new_cap = GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS;
    ಆದರೆ (sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ > 0) {
        new_cap = 2*sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ, new_cap);
    }
    ಸ್ವಯಂಚಾಲಿತ * pnew = (ರಚನೆ ggml_tensor **) realloc((ಶೂನ್ಯ *) sched->ನಕ್ಷೆ_inputs, new_cap * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
    ಆದರೆ (pnew == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->ನಕ್ಷೆ_inputs = pnew;
    sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ = new_cap;
}

// returns the priority of the backend, lower id is higher priority
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_backends; i++) {
        ಆದರೆ (sched->backends[i] == backend) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ -1;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_sched_backend_from_ಬಫರ್(ggml_backend_sched_t sched, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) {
    ggml_backend_ಬಫರ್_t ಬಫರ್ = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
    ಆದರೆ (ಬಫರ್ == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // find highest prio backend that supports the buffer type and the op
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_backends; i++) {
        ಆದರೆ (ggml_backend_supports_buft(sched->backends[i], ಬಫರ್->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            ಹಿಂತಿರುಗಿ i;
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_ಬಫರ್_name(ಬಫರ್), tensor->name);
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ -1;
}

#ಆದರೆ 0
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SCHED_MAX_SPLITS_DEBUG 4096
ಸ್ಥಿರ ಅಕ್ಷರ causes[GGML_DEFAULT_ನಕ್ಷೆ_ಗಾತ್ರ*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS][128]; // debug only
#ವ್ಯಾಖ್ಯಾನಿಸು ಗಣ_CAUSE(ಗಂಟು, ...) sprintf(causes[hash_id(ಗಂಟು)], __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GET_CAUSE(ಗಂಟು) causes[hash_id(ಗಂಟು)]
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು ಗಣ_CAUSE(ಗಂಟು, ...)
#ವ್ಯಾಖ್ಯಾನಿಸು GET_CAUSE(ಗಂಟು) ""
#ಆದರೆ_ಅಂತ್ಯ

// returns the backend that should be used for the node based on the current locations
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, ರಚನೆ ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    ಪೂರ್ಣಾಂಕ cur_backend_id = ggml_backend_sched_backend_from_ಬಫರ್(sched, tensor, tensor);
    ಆದರೆ (cur_backend_id != -1) {
        ಗಣ_CAUSE(tensor, "1.dst");
        ಹಿಂತಿರುಗಿ cur_backend_id;
    }

    // view_src
    ಆದರೆ (tensor->view_src != ಖಾಲಿ) {
        cur_backend_id = ggml_backend_sched_backend_from_ಬಫರ್(sched, tensor->view_src, tensor);
        ಆದರೆ (cur_backend_id != -1) {
            ಗಣ_CAUSE(tensor, "1.vsrc");
            ಹಿಂತಿರುಗಿ cur_backend_id;
        }
    }

    ಆದರೆ (tensor->ಬಫರ್ || (tensor->view_src && tensor->view_src->ಬಫರ್)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_ಬಫರ್_t ಬಫರ್ = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_ಬಫರ್_name(ಬಫರ್), ggml_op_name(tensor->op));
    }

    // graph input
    ಆದರೆ (tensor->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        ಗಣ_CAUSE(tensor, "1.inp");
        ಹಿಂತಿರುಗಿ cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    ತಾರ್ಕಿಕ allow = ನಿಜ;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    ಆದರೆ (allow) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
            ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src = tensor->src[i];
            ಆದರೆ (src == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (src->ಬಫರ್ != ಖಾಲಿ && src->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
                ಪೂರ್ಣಾಂಕ src_backend_id = ggml_backend_sched_backend_from_ಬಫರ್(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                ಆದರೆ (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_ಬಫರ್_is_host(src->ಬಫರ್)) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < src_backend_id; b++) {
                        ಆದರೆ (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            ಗಣ_CAUSE(tensor, "1.off");
                            ಹಿಂತಿರುಗಿ b;
                        }
                    }
                }
                ಗಣ_CAUSE(tensor, "1.wgt%d", i);
                ಹಿಂತಿರುಗಿ src_backend_id;
            }
        }
    }

    ಹಿಂತಿರುಗಿ -1;
}

ಸ್ಥಿರ ಅಕ್ಷರ * fmt_ಗಾತ್ರ(size_t size) {
    ಸ್ಥಿರ ಅಕ್ಷರ ಬಫರ್[128];
    ಆದರೆ (size >= 1024*1024) {
        snprintf(ಬಫರ್, ಗಾತ್ರ(ಬಫರ್), "%zuM", size/1024/1024);
    } ಇಲ್ಲದಿದ್ದರೆ {
        snprintf(ಬಫರ್, ಗಾತ್ರ(ಬಫರ್), "%zuK", size/1024);
    }
    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * graph) {
    ಪೂರ್ಣಾಂಕ cur_ವಿಭಜಿಸು = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ಆದರೆ (cur_ವಿಭಜಿಸು < sched->n_splits && i == sched->splits[cur_ವಿಭಜಿಸು].i_ಆರಂಭ) {
            ggml_backend_t ವಿಭಜಿಸು_backend = sched->backends[sched->splits[cur_ವಿಭಜಿಸು].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_ವಿಭಜಿಸು, ggml_backend_name(ವಿಭಜಿಸು_backend),
                sched->splits[cur_ವಿಭಜಿಸು].n_inputs);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < sched->splits[cur_ವಿಭಜಿಸು].n_inputs; j++) {
                ಆದರೆ (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_ವಿಭಜಿಸು].inputs[j]->name,
                    fmt_ಗಾತ್ರ(ggml_nbytes(sched->splits[cur_ವಿಭಜಿಸು].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_ವಿಭಜಿಸು++;
        }
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, ಗಂಟು);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(ಗಂಟು), ಗಂಟು->name,
                fmt_ಗಾತ್ರ(ggml_nbytes(ಗಂಟು)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(ಗಂಟು),
                graph->use_counts[ggml_hash_ಕಂಡುಹಿಡಿ(&graph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು)], ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                ಆದರೆ (src == ಖಾಲಿ) {
                    ಮುಂದುವರಿ;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_ಗಾತ್ರ(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_ಬಫರ್_supported(ggml_backend_sched_t sched, ರಚನೆ ggml_tensor * t, ಪೂರ್ಣಾಂಕ backend_id) {
    ggml_backend_ಬಫರ್_t buf = t->view_src ? t->view_src->ಬಫರ್ : t->ಬಫರ್;
    ggml_backend_ಬಫರ್_type_t buft = ಖಾಲಿ;

    ಆದರೆ (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        ಪೂರ್ಣಾಂಕ tensor_backend_id = tensor_backend_id(t);
        ಆದರೆ (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        ಆದರೆ (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    ಹಿಂತಿರುಗಿ buft != ಖಾಲಿ && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_ಗಣ_if_supported(ggml_backend_sched_t sched, ರಚನೆ ggml_tensor * ಗಂಟು, ಪೂರ್ಣಾಂಕ cur_backend_id, ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id) {
    ಆದರೆ (ggml_backend_supports_op(sched->backends[cur_backend_id], ಗಂಟು)) {
        *ಗಂಟು_backend_id = cur_backend_id;
        ಗಣ_CAUSE(ಗಂಟು, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
ಶೂನ್ಯ ggml_backend_sched_ವಿಭಜಿಸು_ನಕ್ಷೆ(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_ನಕ್ಷೆ_inputs = 0;
    sched->is_reset = ಸುಳ್ಳು;

    ರಚನೆ ggml_init_params params = {
        /* .mem_size =   */ sched->context_ಬಫರ್_ಗಾತ್ರ,
        /* .mem_buffer = */ sched->context_ಬಫರ್,
        /* .no_alloc =   */ ನಿಜ
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    ಆದರೆ (sched->ctx == ಖಾಲಿ) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_ನಕ್ಷೆ_ಮುಂದಿನ_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = graph->leafs[i];
        ಪೂರ್ಣಾಂಕ * ಎಲೆ_backend_id = &tensor_backend_id(ಎಲೆ);
        // do not overwrite user assignments
        ಆದರೆ (*ಎಲೆ_backend_id == -1) {
            *ಎಲೆ_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ಎಲೆ);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
        // do not overwrite user assignments
        ಆದರೆ (*ಗಂಟು_backend_id == -1) {
            *ಗಂಟು_backend_id = ggml_backend_sched_backend_id_from_cur(sched, ಗಂಟು);

#ಆದರೆ 0
            // src
            ಆದರೆ (ಗಂಟು->op == GGML_OP_NONE) {
                ಮುಂದುವರಿ;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                ಆದರೆ (src == ಖಾಲಿ) {
                    ಮುಂದುವರಿ;
                }
                ಪೂರ್ಣಾಂಕ * src_backend_id = &tensor_backend_id(src);
                ಆದರೆ (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#ಆದರೆ_ಅಂತ್ಯ
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        ಪೂರ್ಣಾಂಕ cur_backend_id = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಆದರೆ (*ಗಂಟು_backend_id != -1) {
                ಆದರೆ (*ಗಂಟು_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    cur_backend_id = *ಗಂಟು_backend_id;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cur_backend_id != -1) {
                ggml_backend_sched_ಗಣ_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand gpu up
    {
        ಪೂರ್ಣಾಂಕ cur_backend_id = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = graph->n_nodes - 1; i >= 0; i--) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಆದರೆ (*ಗಂಟು_backend_id != -1) {
                ಆದರೆ (*ಗಂಟು_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    cur_backend_id = *ಗಂಟು_backend_id;
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cur_backend_id != -1) {
                ggml_backend_sched_ಗಣ_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand rest down
    {
        ಪೂರ್ಣಾಂಕ cur_backend_id = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಆದರೆ (*ಗಂಟು_backend_id != -1) {
                cur_backend_id = *ಗಂಟು_backend_id;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cur_backend_id != -1) {
                ggml_backend_sched_ಗಣ_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
            }
        }
    }
    // expand rest up
    {
        ಪೂರ್ಣಾಂಕ cur_backend_id = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = graph->n_nodes - 1; i >= 0; i--) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
            ಆದರೆ (*ಗಂಟು_backend_id != -1) {
                cur_backend_id = *ಗಂಟು_backend_id;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cur_backend_id != -1) {
                ggml_backend_sched_ಗಣ_if_supported(sched, ಗಂಟು, cur_backend_id, ಗಂಟು_backend_id);
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
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
            ಮುಂದುವರಿ;
        }
        ಪೂರ್ಣಾಂಕ * ಗಂಟು_backend_id = &tensor_backend_id(ಗಂಟು);
        ಆದರೆ (*ಗಂಟು_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            ಪೂರ್ಣಾಂಕ n_supported_ಉತ್ತಮ = -1;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < sched->n_backends; b++) {
                ಆದರೆ (ggml_backend_supports_op(sched->backends[b], ಗಂಟು)) {
                    ಪೂರ್ಣಾಂಕ n_supported = 0;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                        ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                        ಆದರೆ (src == ಖಾಲಿ) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_ಬಫರ್_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    ಆದರೆ (n_supported > n_supported_ಉತ್ತಮ) {
                        n_supported_ಉತ್ತಮ = n_supported;
                        *ಗಂಟು_backend_id = b;
                        ಗಣ_CAUSE(ಗಂಟು, "3.best");
                    }
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // assigned node: upgrade to higher prio backend if possible
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < *ಗಂಟು_backend_id; b++) {
                ಆದರೆ (sched->bufts[b] == sched->bufts[*ಗಂಟು_backend_id] && ggml_backend_supports_op(sched->backends[b], ಗಂಟು)) {
                    ತಾರ್ಕಿಕ supported = ನಿಜ;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                        ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                        ಆದರೆ (src == ಖಾಲಿ) {
                            ಮುಂದುವರಿ;
                        }
                        ಆದರೆ (!ggml_backend_sched_ಬಫರ್_supported(sched, src, b)) {
                            supported = ಸುಳ್ಳು;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                    ಆದರೆ (supported) {
                        *ಗಂಟು_backend_id = b;
                        ಗಣ_CAUSE(ಗಂಟು, "3.upg");
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ಪೂರ್ಣಾಂಕ * cur_backend_id = &tensor_backend_id(ಗಂಟು);
        ಆದರೆ (ಗಂಟು->view_src != ಖಾಲಿ && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(ಗಂಟು->view_src);
            ಗಣ_CAUSE(ಗಂಟು, "4.vsrc");
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
            ಆದರೆ (src == ಖಾಲಿ) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ * src_backend_id = &tensor_backend_id(src);
            ಆದರೆ (*src_backend_id == -1) {
                ಆದರೆ (src->view_src != ಖಾಲಿ) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    ಗಣ_CAUSE(src, "4.vsrc");
                } ಇಲ್ಲದಿದ್ದರೆ {
                    *src_backend_id = *cur_backend_id;
                    ಗಣ_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_ಗಣ_if_supported(sched, ಗಂಟು, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        ಪೂರ್ಣಾಂಕ i_ವಿಭಜಿಸು = 0;
        ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        ಪೂರ್ಣಾಂಕ i = 0;
        ಗಾಗಿ (; i < graph->n_nodes; i++) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
            ಆದರೆ (!ggml_is_view_op(ಗಂಟು->op)) {
                ವಿಭಜಿಸು->backend_id = tensor_backend_id(ಗಂಟು);
                ನಿಲ್ಲಿಸು;
            }
        }
        ವಿಭಜಿಸು->i_ಆರಂಭ = 0;
        ವಿಭಜಿಸು->n_inputs = 0;
        ಪೂರ್ಣಾಂಕ cur_backend_id = ವಿಭಜಿಸು->backend_id;
        ಗಾಗಿ (; i < graph->n_nodes; i++) {
            ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];

            ಆದರೆ (ggml_is_view_op(ಗಂಟು->op)) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_backend_id = tensor_backend_id(ಗಂಟು);

            GGML_ASSERT(ಗಂಟು_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            ತಾರ್ಕಿಕ need_new_ವಿಭಜಿಸು = ಸುಳ್ಳು;
            ಆದರೆ (ಗಂಟು_backend_id == cur_backend_id && ವಿಭಜಿಸು->n_inputs > 0) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                    ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                    ಆದರೆ (src == ಖಾಲಿ) {
                        ಮುಂದುವರಿ;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    ಆದರೆ (src->ಬಫರ್ != ಖಾಲಿ && src->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS) {
                        ಪೂರ್ಣಾಂಕ src_backend_id = tensor_backend_id(src);
                        ಆದರೆ (src_backend_id != cur_backend_id && !ggml_backend_sched_ಬಫರ್_supported(sched, src, cur_backend_id)) {
                            need_new_ವಿಭಜಿಸು = ನಿಜ;
                            ನಿಲ್ಲಿಸು;
                        }
                    }
                }
            }

            ಆದರೆ (ಗಂಟು_backend_id != cur_backend_id || need_new_ವಿಭಜಿಸು) {
                ವಿಭಜಿಸು->i_ಕೊನೆ = i;
                i_ವಿಭಜಿಸು++;
                ಆದರೆ (i_ವಿಭಜಿಸು >= sched->splits_ಸಾಮರ್ಥ್ಯ) {
                    ಪೂರ್ಣಾಂಕ old_cap = sched->splits_ಸಾಮರ್ಥ್ಯ;
                    sched->splits_ಸಾಮರ್ಥ್ಯ *= 2;
                    sched->splits = (ggml_backend_sched_ವಿಭಜಿಸು *)
                        realloc(sched->splits, sched->splits_ಸಾಮರ್ಥ್ಯ * ಗಾತ್ರ(ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು));
                    GGML_ASSERT(sched->splits != ಖಾಲಿ);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = old_cap; k < sched->splits_ಸಾಮರ್ಥ್ಯ; k++) {
                        memset(&sched->splits[k], 0, ಗಾತ್ರ(ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು));
                    }
                }
                ವಿಭಜಿಸು = &sched->splits[i_ವಿಭಜಿಸು];
                ವಿಭಜಿಸು->backend_id = ಗಂಟು_backend_id;
                ವಿಭಜಿಸು->i_ಆರಂಭ = i;
                ವಿಭಜಿಸು->n_inputs = 0;
                cur_backend_id = ಗಂಟು_backend_id;
            }

            // find inputs that are not on the same backend
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
                ರಚನೆ ggml_tensor * src = ಗಂಟು->src[j];
                ಆದರೆ (src == ಖಾಲಿ) {
                    ಮುಂದುವರಿ;
                }

                size_t src_id = hash_id(src);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                ಆದರೆ (src_backend_id != cur_backend_id && !ggml_backend_sched_ಬಫರ್_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    ಆದರೆ (tensor_id_copy(src_id, cur_backend_id, 0) == ಖಾಲಿ) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
                            ರಚನೆ ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            ಆದರೆ (sched->n_copies > 1) {
                                ggml_ಗಣ_ಒಳಹರಿವು(tensor_copy);
                                ggml_ಗಣ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            ಗಣ_CAUSE(tensor_copy, "4.cpy");
                        }
                        ಪೂರ್ಣಾಂಕ n_inputs = ವಿಭಜಿಸು->n_inputs++;
                        ಆದರೆ (n_inputs >= ವಿಭಜಿಸು->inputs_ಸಾಮರ್ಥ್ಯ) {
                            ggml_backend_sched_ವಿಭಜಿಸು_inputs_grow(ವಿಭಜಿಸು);
                        }
                        ವಿಭಜಿಸು->inputs[n_inputs] = src;
                    }
                    ಗಂಟು->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        ವಿಭಜಿಸು->i_ಕೊನೆ = graph->n_nodes;
        sched->n_splits = i_ವಿಭಜಿಸು + 1;
    }

    ಆದರೆ (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    ಆದರೆ (sched->n_copies > 1) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_leafs; i++) {
            ರಚನೆ ggml_tensor * ಎಲೆ = graph->leafs[i];
            ಆದರೆ ((ಎಲೆ->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) == 0) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ size_t ಎಲೆ_id = hash_id(ಎಲೆ);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಎಲೆ_backend_id = tensor_backend_id(ಎಲೆ);
            GGML_ASSERT(ಎಲೆ_backend_id != -1); // all leafs should be assigned by now

            ಆದರೆ (tensor_id_copy(ಎಲೆ_id, ಎಲೆ_backend_id, 0) == ಖಾಲಿ) {
                ggml_backend_t backend = sched->backends[ಎಲೆ_backend_id];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
                    ರಚನೆ ggml_tensor * tensor_copy;
                    ಆದರೆ (c == sched->cur_copy) {
                        tensor_copy = ಎಲೆ; // use the original tensor as the current copy
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, ಎಲೆ);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), ಎಲೆ->name, c);
                    }
                    ggml_ಗಣ_ಒಳಹರಿವು(tensor_copy);
                    ggml_ಗಣ_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(ಎಲೆ_id, ಎಲೆ_backend_id, c) = tensor_copy;
                    ಗಣ_CAUSE(tensor_copy, "6.cpy");
                }
            }

            ಪೂರ್ಣಾಂಕ n_ನಕ್ಷೆ_inputs = sched->n_ನಕ್ಷೆ_inputs++;
            ಆದರೆ (n_ನಕ್ಷೆ_inputs >= sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ) {
                ggml_backend_sched_ನಕ್ಷೆ_inputs_grow(sched);
            }
            sched->ನಕ್ಷೆ_inputs[n_ನಕ್ಷೆ_inputs] = ಎಲೆ;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        ಪೂರ್ಣಾಂಕ * tmp = sched->ಗಂಟು_backend_ids;
        sched->ಗಂಟು_backend_ids = sched->prev_ಗಂಟು_backend_ids;
        sched->prev_ಗಂಟು_backend_ids = tmp;

        tmp = sched->ಎಲೆ_backend_ids;
        sched->ಎಲೆ_backend_ids = sched->prev_ಎಲೆ_backend_ids;
        sched->prev_ಎಲೆ_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](ಶೂನ್ಯ * user_ದತ್ತಾಂಶ, ggml_tensor * tensor, ggml_tensor * until) {
            ಸ್ವಯಂಚಾಲಿತ & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_ದತ್ತಾಂಶ;
            std::vector<ggml_tensor *> & keep = deps[until];
            ಆದರೆ (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_splits; i++) {
        ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು = &sched->splits[i];
        ವಿಭಜಿಸು->graph = ggml_ನಕ್ಷೆ_view(graph, ವಿಭಜಿಸು->i_ಆರಂಭ, ವಿಭಜಿಸು->i_ಕೊನೆ);

        ggml_backend_ನಕ್ಷೆ_optimize(sched->backends[ವಿಭಜಿಸು->backend_id], &ವಿಭಜಿಸು->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    ಪೂರ್ಣಾಂಕ n_dep_nodes = 0;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_inputs = sched->n_ನಕ್ಷೆ_inputs;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_splits; i++) {
        ಒಟ್ಟು_inputs += sched->splits[i].n_inputs;
    }
    ಪೂರ್ಣಾಂಕ ನಕ್ಷೆ_ಗಾತ್ರ = std::max(graph->n_nodes, graph->n_leafs) + ಒಟ್ಟು_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_ನಕ್ಷೆ_ಗಾತ್ರ = sched->ನಕ್ಷೆ_ಗಾತ್ರ;
    sched->ನಕ್ಷೆ_ಗಾತ್ರ = ನಕ್ಷೆ_ಗಾತ್ರ;

    ಆದರೆ (sched->graph.size < ನಕ್ಷೆ_ಗಾತ್ರ) {
        sched->graph.size = ನಕ್ಷೆ_ಗಾತ್ರ;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, ನಕ್ಷೆ_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, ನಕ್ಷೆ_ಗಾತ್ರ * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != ಖಾಲಿ);
        GGML_ASSERT(sched->graph.leafs != ಖಾಲಿ);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    ರಚನೆ ggml_cgraph * ನಕ್ಷೆ_copy = &sched->graph;

    ಪೂರ್ಣಾಂಕ n_dep_nodes_added = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_splits; i++) {
        ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ವಿಭಜಿಸು->n_inputs; j++) {
            ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > (ನಕ್ಷೆ_copy->n_nodes + 1));

            ರಚನೆ ggml_tensor * ಒಳಹರಿವು = ವಿಭಜಿಸು->inputs[j];
            ಸ್ಥಿರಾಂಕ size_t ಒಳಹರಿವು_id = hash_id(ಒಳಹರಿವು);
            ರಚನೆ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(ಒಳಹರಿವು_id, ವಿಭಜಿಸು->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            ರಚನೆ ggml_tensor * ಒಳಹರಿವು_dep = ggml_view_tensor(sched->ctx, ಒಳಹರಿವು);
            ಒಳಹರಿವು_dep->src[0] = ಒಳಹರಿವು;
            sched->ಗಂಟು_backend_ids[ನಕ್ಷೆ_copy->n_nodes] = sched->hv_tensor_backend_ids[ಒಳಹರಿವು_id];
            ನಕ್ಷೆ_copy->nodes[ನಕ್ಷೆ_copy->n_nodes++] = ಒಳಹರಿವು_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->ಗಂಟು_backend_ids[ನಕ್ಷೆ_copy->n_nodes] = ವಿಭಜಿಸು->backend_id;
            ನಕ್ಷೆ_copy->nodes[ನಕ್ಷೆ_copy->n_nodes++] = ಒಳಹರಿವು_cpy;
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ವಿಭಜಿಸು->i_ಆರಂಭ; j < ವಿಭಜಿಸು->i_ಕೊನೆ; j++) {
            ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > ನಕ್ಷೆ_copy->n_nodes);
            sched->ಗಂಟು_backend_ids[ನಕ್ಷೆ_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            ನಕ್ಷೆ_copy->nodes[ನಕ್ಷೆ_copy->n_nodes++] = graph->nodes[j];

            ಆದರೆ (alloc_deps.ಖಾಲಿಗಳು()) {
                ಮುಂದುವರಿ;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            ಸ್ವಯಂಚಾಲಿತ it = alloc_deps.find(graph->nodes[j]);
            ಆದರೆ (it != alloc_deps.end()) {
                ಸ್ಥಿರಾಂಕ std::vector<ggml_tensor *> & keep = it->second;
                ಗಾಗಿ (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    ರಚನೆ ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    ಗಾಗಿ (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > ನಕ್ಷೆ_copy->n_nodes);
                    sched->ಗಂಟು_backend_ids[ನಕ್ಷೆ_copy->n_nodes] = ವಿಭಜಿಸು->backend_id;
                    ನಕ್ಷೆ_copy->nodes[ನಕ್ಷೆ_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    ಆದರೆ (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_ನಕ್ಷೆ_inputs; i++) {
            ರಚನೆ ggml_tensor * ಒಳಹರಿವು = sched->ನಕ್ಷೆ_inputs[i];
            size_t id = hash_id(ಒಳಹರಿವು);
            ಪೂರ್ಣಾಂಕ backend_id = tensor_backend_id(ಒಳಹರಿವು);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
                ರಚನೆ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(id, backend_id, c);
                sched->ಎಲೆ_backend_ids[ನಕ್ಷೆ_copy->n_leafs] = backend_id;
                ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > ನಕ್ಷೆ_copy->n_leafs);
                ನಕ್ಷೆ_copy->leafs[ನಕ್ಷೆ_copy->n_leafs++] = ಒಳಹರಿವು_cpy;
            }
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_splits; i++) {
            ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು = &sched->splits[i];
            ಪೂರ್ಣಾಂಕ backend_id = ವಿಭಜಿಸು->backend_id;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ವಿಭಜಿಸು->n_inputs; j++) {
                ರಚನೆ ggml_tensor * ಒಳಹರಿವು = ವಿಭಜಿಸು->inputs[j];
                size_t id = hash_id(ಒಳಹರಿವು);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
                    ರಚನೆ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_id_copy(id, backend_id, c);
                    sched->ಎಲೆ_backend_ids[ನಕ್ಷೆ_copy->n_leafs] = backend_id;
                    ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > ನಕ್ಷೆ_copy->n_leafs);
                    ನಕ್ಷೆ_copy->leafs[ನಕ್ಷೆ_copy->n_leafs++] = ಒಳಹರಿವು_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_leafs; i++) {
        ರಚನೆ ggml_tensor * ಎಲೆ = graph->leafs[i];
        sched->ಎಲೆ_backend_ids[ನಕ್ಷೆ_copy->n_leafs] = tensor_backend_id(ಎಲೆ);
        ಪ್ರತಿಜ್ಞಾ(ನಕ್ಷೆ_copy->size > ನಕ್ಷೆ_copy->n_leafs);
        ನಕ್ಷೆ_copy->leafs[ನಕ್ಷೆ_copy->n_leafs++] = ಎಲೆ;
    }

    // set ids for all splits
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_ನಕ್ಷೆ_ಮುಂದಿನ_uid();
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    ತಾರ್ಕಿಕ backend_ids_ಬದಲಾಯಿತು = ಸುಳ್ಳು;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->graph.n_nodes; i++) {
        ಆದರೆ (sched->ಗಂಟು_backend_ids[i] != sched->prev_ಗಂಟು_backend_ids[i] &&
            sched->bufts[sched->ಗಂಟು_backend_ids[i]] != sched->bufts[sched->prev_ಗಂಟು_backend_ids[i]]) {
            backend_ids_ಬದಲಾಯಿತು = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        }
    }
    ಆದರೆ (!backend_ids_ಬದಲಾಯಿತು) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->graph.n_leafs; i++) {
            ಆದರೆ (sched->ಎಲೆ_backend_ids[i] != sched->prev_ಎಲೆ_backend_ids[i] &&
                sched->bufts[sched->ಎಲೆ_backend_ids[i]] != sched->bufts[sched->prev_ಎಲೆ_backend_ids[i]]) {
                backend_ids_ಬದಲಾಯಿತು = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    // allocate graph
    ಆದರೆ (backend_ids_ಬದಲಾಯಿತು || !ggml_gallocr_alloc_ನಕ್ಷೆ(sched->galloc, &sched->graph)) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_ಬದಲಾಯಿತು);
#ಆದರೆ_ಅಂತ್ಯ

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ನಕ್ಷೆ_ಗಾತ್ರ_ಬದಲಾಯಿತು = sched->prev_ನಕ್ಷೆ_ಗಾತ್ರ != sched->ನಕ್ಷೆ_ಗಾತ್ರ;

        ಆದರೆ (sched->callback_realloc != ಖಾಲಿ) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_ಬದಲಾಯಿತು, ನಕ್ಷೆ_ಗಾತ್ರ_ಬದಲಾಯಿತು, sched->callback_realloc_user_ದತ್ತಾಂಶ);
        }

        ಆದರೆ (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ unexpected = !backend_ids_ಬದಲಾಯಿತು && !ನಕ್ಷೆ_ಗಾತ್ರ_ಬದಲಾಯಿತು;

            ಆದರೆ (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->ನಕ್ಷೆ_ಗಾತ್ರ, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ಆದರೆ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->ಎಲೆ_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (!ggml_gallocr_alloc_ನಕ್ಷೆ(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sched_is_host_ತೂಕ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ t->ಬಫರ್ != ಖಾಲಿ &&
        ggml_backend_ಬಫರ್_get_usage(t->ಬಫರ್) == GGML_BACKEND_ಬಫರ್_USAGE_WEIGHTS &&
        ggml_backend_ಬಫರ್_is_host(t->ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sched_copy_ಒಳಹರಿವು(ggml_backend_sched_t sched, ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು, ರಚನೆ ggml_tensor * ಒಳಹರಿವು) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_backend_id = ವಿಭಜಿಸು->backend_id;
    ggml_backend_t ವಿಭಜಿಸು_backend = sched->backends[ವಿಭಜಿಸು_backend_id];
    ggml_backend_t ಒಳಹರಿವು_backend = ggml_backend_sched_get_tensor_backend(sched, ಒಳಹರಿವು);
    ರಚನೆ ggml_tensor * ಒಳಹರಿವು_cpy = tensor_copy(ಒಳಹರಿವು, ವಿಭಜಿಸು_backend_id, sched->cur_copy);

    ಆದರೆ (ಒಳಹರಿವು->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        ಆದರೆ (sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy] != ಖಾಲಿ) {
            ggml_backend_event_synchronize(sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy]);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_backend_synchronize(ವಿಭಜಿಸು_backend);
        }
        ggml_backend_tensor_copy(ಒಳಹರಿವು, ಒಳಹರಿವು_cpy);
        ಹಿಂತಿರುಗಿ;
    }

    // wait for the split backend to finish using the input before overwriting it
    ಆದರೆ (sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy] != ಖಾಲಿ) {
        ggml_backend_event_wait(ವಿಭಜಿಸು_backend, sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy]);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_backend_synchronize(ವಿಭಜಿಸು_backend);
    }

    ಆದರೆ (sched->callback_copy != ಖಾಲಿ && ggml_backend_sched_is_host_ತೂಕ(ಒಳಹರಿವು) &&
        sched->callback_copy(ವಿಭಜಿಸು_backend, ಒಳಹರಿವು, ಒಳಹರಿವು_cpy, &ವಿಭಜಿಸು->graph, sched->callback_copy_user_ದತ್ತಾಂಶ)) {
        ಹಿಂತಿರುಗಿ;
    }

    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    ಆದರೆ (!ವಿಭಜಿಸು_backend->iface.cpy_tensor_async || !ವಿಭಜಿಸು_backend->iface.cpy_tensor_async(ಒಳಹರಿವು_backend, ವಿಭಜಿಸು_backend, ಒಳಹರಿವು, ಒಳಹರಿವು_cpy)) {
        ggml_backend_synchronize(ಒಳಹರಿವು_backend);
        ಆದರೆ (sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy] != ಖಾಲಿ) {
            ggml_backend_event_synchronize(sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy]);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_backend_synchronize(ವಿಭಜಿಸು_backend);
        }
        ggml_backend_tensor_copy(ಒಳಹರಿವು, ಒಳಹರಿವು_cpy);
    }
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * splits = sched->splits;

    ಪೂರ್ಣಾಂಕ prev_backend_id = -1;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_id = 0; ವಿಭಜಿಸು_id < sched->n_splits; ವಿಭಜಿಸು_id++) {
        ರಚನೆ ggml_backend_sched_ವಿಭಜಿಸು * ವಿಭಜಿಸು = &splits[ವಿಭಜಿಸು_id];
        ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_backend_id = ವಿಭಜಿಸು->backend_id;
        ggml_backend_t ವಿಭಜಿಸು_backend = sched->backends[ವಿಭಜಿಸು_backend_id];

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        ಆದರೆ (ವಿಭಜಿಸು->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != ವಿಭಜಿಸು_backend_id) {
            ಆದರೆ (sched->events[prev_backend_id][sched->cur_copy] != ಖಾಲಿ) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_id = 0; ಒಳಹರಿವು_id < ವಿಭಜಿಸು->n_inputs; ಒಳಹರಿವು_id++) {
            ಆದರೆ (!ggml_backend_sched_is_host_ತೂಕ(ವಿಭಜಿಸು->inputs[ಒಳಹರಿವು_id])) {
                ggml_backend_sched_copy_ಒಳಹರಿವು(sched, ವಿಭಜಿಸು, ವಿಭಜಿಸು->inputs[ಒಳಹರಿವು_id]);
            }
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_id = 0; ಒಳಹರಿವು_id < ವಿಭಜಿಸು->n_inputs; ಒಳಹರಿವು_id++) {
            ಆದರೆ (ggml_backend_sched_is_host_ತೂಕ(ವಿಭಜಿಸು->inputs[ಒಳಹರಿವು_id])) {
                ggml_backend_sched_copy_ಒಳಹರಿವು(sched, ವಿಭಜಿಸು, ವಿಭಜಿಸು->inputs[ಒಳಹರಿವು_id]);
            }
        }

        ಆದರೆ (!sched->callback_eval) {
            ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ec = ggml_backend_ನಕ್ಷೆ_compute_async(ವಿಭಜಿಸು_backend, &ವಿಭಜಿಸು->graph);
            ಆದರೆ (ec != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                ಹಿಂತಿರುಗಿ ec;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // similar to ggml_backend_compare_graph_backend
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j0 = 0; j0 < ವಿಭಜಿಸು->graph.n_nodes; j0++) {
                ರಚನೆ ggml_tensor * t = ವಿಭಜಿಸು->graph.nodes[j0];

                // check if the user needs data from this node
                ತಾರ್ಕಿಕ need = sched->callback_eval(t, ನಿಜ, sched->callback_eval_user_ದತ್ತಾಂಶ);

                ಪೂರ್ಣಾಂಕ j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                ಆಗ (!need && j1 < ವಿಭಜಿಸು->graph.n_nodes - 1) {
                    t = ವಿಭಜಿಸು->graph.nodes[++j1];
                    need = sched->callback_eval(t, ನಿಜ, sched->callback_eval_user_ದತ್ತಾಂಶ);
                }

                ರಚನೆ ggml_cgraph gv = ggml_ನಕ್ಷೆ_view(&ವಿಭಜಿಸು->graph, j0, j1 + 1);

                ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ec = ggml_backend_ನಕ್ಷೆ_compute_async(ವಿಭಜಿಸು_backend, &gv);
                ಆದರೆ (ec != GGML_ಸ್ಥಿತಿ_SUCCESS) {
                    ಹಿಂತಿರುಗಿ ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(ವಿಭಜಿಸು_backend);

                ಆದರೆ (need && !sched->callback_eval(t, ಸುಳ್ಳು, sched->callback_eval_user_ದತ್ತಾಂಶ)) {
                    ನಿಲ್ಲಿಸು;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        ಆದರೆ (sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy] != ಖಾಲಿ) {
            ggml_backend_event_ದಾಖಲೆ(sched->events[ವಿಭಜಿಸು_backend_id][sched->cur_copy], ವಿಭಜಿಸು_backend);
        }

        prev_backend_id = ವಿಭಜಿಸು_backend_id;
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_ಬಫರ್_type_t * bufts,
        ಪೂರ್ಣಾಂಕ n_backends,
        size_t ನಕ್ಷೆ_ಗಾತ್ರ,
        ತಾರ್ಕಿಕ parallel,
        ತಾರ್ಕಿಕ op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    ರಚನೆ ggml_backend_sched * sched = (ggml_backend_sched *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(1, ಗಾತ್ರ(ರಚನೆ ggml_backend_sched));

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_ಗಣ    = ggml_hash_ಗಣ_new(ನಕ್ಷೆ_ಗಾತ್ರ);
    sched->hv_tensor_backend_ids = (ಪೂರ್ಣಾಂಕ *) ಸ್ಮರಣೆ_ಕೊಡು(sched->hash_ಗಣ.size * ಗಾತ್ರ(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) ಸ್ಮರಣೆ_ಕೊಡು(sched->hash_ಗಣ.size * sched->n_backends * sched->n_copies * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));

    ಸ್ಥಿರಾಂಕ size_t ggml_sched_max_splits = ನಕ್ಷೆ_ಗಾತ್ರ; // at most there is one split for each node in the graph
    ಸ್ಥಿರಾಂಕ size_t nodes_ಗಾತ್ರ = ನಕ್ಷೆ_ಗಾತ್ರ + ggml_sched_max_splits*GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS*2;
    sched->ಗಂಟು_backend_ids = (ಪೂರ್ಣಾಂಕ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(nodes_ಗಾತ್ರ, ಗಾತ್ರ(sched->ಗಂಟು_backend_ids[0]));
    sched->ಎಲೆ_backend_ids = (ಪೂರ್ಣಾಂಕ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(nodes_ಗಾತ್ರ, ಗಾತ್ರ(sched->ಎಲೆ_backend_ids[0]));
    sched->prev_ಗಂಟು_backend_ids = (ಪೂರ್ಣಾಂಕ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(nodes_ಗಾತ್ರ, ಗಾತ್ರ(sched->prev_ಗಂಟು_backend_ids[0]));
    sched->prev_ಎಲೆ_backend_ids = (ಪೂರ್ಣಾಂಕ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(nodes_ಗಾತ್ರ, ಗಾತ್ರ(sched->prev_ಎಲೆ_backend_ids[0]));

    sched->ನಕ್ಷೆ_ಗಾತ್ರ      = 0;
    sched->prev_ನಕ್ಷೆ_ಗಾತ್ರ = 0;

    sched->context_ಬಫರ್_ಗಾತ್ರ = ggml_sched_max_splits*GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS*2*ಗಾತ್ರ(ರಚನೆ ggml_tensor) + ggml_ನಕ್ಷೆ_overhead_custom(ನಕ್ಷೆ_ಗಾತ್ರ, ಸುಳ್ಳು);
    sched->context_ಬಫರ್ = (ಅಕ್ಷರ *) ಸ್ಮರಣೆ_ಕೊಡು(sched->context_ಬಫರ್_ಗಾತ್ರ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ initial_splits_ಸಾಮರ್ಥ್ಯ = 16;
    sched->splits = (ggml_backend_sched_ವಿಭಜಿಸು *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(initial_splits_ಸಾಮರ್ಥ್ಯ, ಗಾತ್ರ(sched->splits[0]));
    sched->splits_ಸಾಮರ್ಥ್ಯ = initial_splits_ಸಾಮರ್ಥ್ಯ;

    sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ = GGML_SCHED_MAX_ವಿಭಜಿಸು_INPUTS;
    sched->ನಕ್ಷೆ_inputs = (ರಚನೆ ggml_tensor **) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(sched->ನಕ್ಷೆ_inputs_ಸಾಮರ್ಥ್ಯ, ಗಾತ್ರ(ರಚನೆ ggml_tensor *));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_ಬಫರ್_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        ಆದರೆ (sched->n_copies > 1) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    ಹಿಂತಿರುಗಿ sched;
}

ಶೂನ್ಯ ggml_backend_sched_free(ggml_backend_sched_t sched) {
    ಆದರೆ (sched == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < sched->n_backends; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_ಗಣ_free(&sched->hash_ಗಣ);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->splits_ಸಾಮರ್ಥ್ಯ; i++) {
        ಬಿಡುಗಡೆ(sched->splits[i].inputs);
    }
    ಬಿಡುಗಡೆ(sched->splits);
    ಬಿಡುಗಡೆ(sched->ನಕ್ಷೆ_inputs);
    ಬಿಡುಗಡೆ(sched->hv_tensor_backend_ids);
    ಬಿಡುಗಡೆ(sched->hv_tensor_copies);
    ಬಿಡುಗಡೆ(sched->ಗಂಟು_backend_ids);
    ಬಿಡುಗಡೆ(sched->ಎಲೆ_backend_ids);
    ಬಿಡುಗಡೆ(sched->prev_ಗಂಟು_backend_ids);
    ಬಿಡುಗಡೆ(sched->prev_ಎಲೆ_backend_ids);
    ಬಿಡುಗಡೆ(sched->context_ಬಫರ್);
    ಬಿಡುಗಡೆ(sched->graph.nodes);
    ಬಿಡುಗಡೆ(sched->graph.leafs);
    ಬಿಡುಗಡೆ(sched);
}

ಶೂನ್ಯ ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    ಆದರೆ (!sched->is_reset) {
        ggml_hash_ಗಣ_reset(&sched->hash_ಗಣ);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_ಗಣ.size * ಗಾತ್ರ(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_ಗಣ.size * sched->n_backends * sched->n_copies * ಗಾತ್ರ(ರಚನೆ ggml_tensor *));
        sched->is_reset = ನಿಜ;
    }
    sched->is_alloc = ಸುಳ್ಳು;
}

ಶೂನ್ಯ ggml_backend_sched_reserve_ಗಾತ್ರ(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * measure_ನಕ್ಷೆ, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಂಕ)sched->hash_ಗಣ.size >= measure_ನಕ್ಷೆ->n_nodes + measure_ನಕ್ಷೆ->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ವಿಭಜಿಸು_ನಕ್ಷೆ(sched, measure_ನಕ್ಷೆ);

    ggml_gallocr_reserve_n_ಗಾತ್ರ(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->ಎಲೆ_backend_ids, sizes);
}

ತಾರ್ಕಿಕ ggml_backend_sched_reserve(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * measure_ನಕ್ಷೆ) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಂಕ)sched->hash_ಗಣ.size >= measure_ನಕ್ಷೆ->n_nodes + measure_ನಕ್ಷೆ->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_ವಿಭಜಿಸು_ನಕ್ಷೆ(sched, measure_ನಕ್ಷೆ);

    ಆದರೆ (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->ಗಂಟು_backend_ids, sched->ಎಲೆ_backend_ids)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_backend_sched_reset(sched);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ತಾರ್ಕಿಕ ggml_backend_sched_alloc_ನಕ್ಷೆ(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((ಪೂರ್ಣಾಂಕ)sched->hash_ಗಣ.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->ಮುಂದಿನ_copy;
    sched->ಮುಂದಿನ_copy = (sched->ಮುಂದಿನ_copy + 1) % sched->n_copies;

    ggml_backend_sched_ವಿಭಜಿಸು_ನಕ್ಷೆ(sched, graph);

    ಆದರೆ (!ggml_backend_sched_alloc_splits(sched)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    sched->is_alloc = ನಿಜ;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_sched_ನಕ್ಷೆ_compute(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * graph) {
    ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ err = ggml_backend_sched_ನಕ್ಷೆ_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    ಹಿಂತಿರುಗಿ err;
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_sched_ನಕ್ಷೆ_compute_async(ggml_backend_sched_t sched, ರಚನೆ ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    ಆದರೆ (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    ಆದರೆ (!sched->is_alloc) {
        ಆದರೆ (!ggml_backend_sched_alloc_ನಕ್ಷೆ(sched, graph)) {
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_ALLOC_FAILED;
        }
    }

    ಹಿಂತಿರುಗಿ ggml_backend_sched_compute_splits(sched);
}

ಶೂನ್ಯ ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    ಆದರೆ (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->ಮುಂದಿನ_copy = 0;
    }
}

ಶೂನ್ಯ ggml_backend_sched_ಗಣ_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_ದತ್ತಾಂಶ = user_ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ ggml_backend_sched_ಗಣ_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_ದತ್ತಾಂಶ = user_ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ ggml_backend_sched_ಗಣ_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_ದತ್ತಾಂಶ = user_ದತ್ತಾಂಶ;
}

ಪೂರ್ಣಾಂಕ ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಹಿಂತಿರುಗಿ sched->n_splits;
}

ಪೂರ್ಣಾಂಕ ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಹಿಂತಿರುಗಿ sched->n_copies;
}

ಪೂರ್ಣಾಂಕ ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ಹಿಂತಿರುಗಿ sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, ಪೂರ್ಣಾಂಕ i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    ಹಿಂತಿರುಗಿ sched->backends[i];
}

ggml_backend_ಬಫರ್_type_t ggml_backend_sched_get_ಬಫರ್_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಂಕ backend_ಸೂಚ್ಯಂಕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ಸೂಚ್ಯಂಕ >= 0 && backend_ಸೂಚ್ಯಂಕ < sched->n_backends);

    ಹಿಂತಿರುಗಿ sched->bufts[backend_ಸೂಚ್ಯಂಕ];
}

size_t ggml_backend_sched_get_ಬಫರ್_ಗಾತ್ರ(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಂಕ backend_ಸೂಚ್ಯಂಕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ಸೂಚ್ಯಂಕ >= 0 && backend_ಸೂಚ್ಯಂಕ < sched->n_backends);

    ಹಿಂತಿರುಗಿ ggml_gallocr_get_ಬಫರ್_ಗಾತ್ರ(sched->galloc, backend_ಸೂಚ್ಯಂಕ);
}

ಶೂನ್ಯ ggml_backend_sched_ಗಣ_tensor_backend(ggml_backend_sched_t sched, ರಚನೆ ggml_tensor * ಗಂಟು, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಂಕ backend_ಸೂಚ್ಯಂಕ = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_ಸೂಚ್ಯಂಕ >= 0 && backend_ಸೂಚ್ಯಂಕ < sched->n_backends);
    tensor_backend_id(ಗಂಟು) = backend_ಸೂಚ್ಯಂಕ;
    ಗಣ_CAUSE(ಗಂಟು, "usr");
    sched->is_reset = ಸುಳ್ಳು;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ರಚನೆ ggml_tensor * ಗಂಟು) {
    GGML_ASSERT(sched);
    ಪೂರ್ಣಾಂಕ backend_ಸೂಚ್ಯಂಕ = tensor_backend_id(ಗಂಟು);
    ಆದರೆ (backend_ಸೂಚ್ಯಂಕ == -1) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ sched->backends[backend_ಸೂಚ್ಯಂಕ];
}

// utils

ತಾರ್ಕಿಕ ggml_op_alloc_ಗಾತ್ರ_may_expand(ಎಣಿಕೆ ggml_op op) {
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_view_init(ರಚನೆ ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ಬಫರ್ == ಖಾಲಿ);
    GGML_ASSERT(tensor->view_src != ಖಾಲಿ);
    GGML_ASSERT(tensor->view_src->ಬಫರ್ != ಖಾಲಿ);
    GGML_ASSERT(tensor->view_src->ದತ್ತಾಂಶ != ಖಾಲಿ);

    tensor->ಬಫರ್ = tensor->view_src->ಬಫರ್;
    tensor->ದತ್ತಾಂಶ = (ಅಕ್ಷರ *)tensor->view_src->ದತ್ತಾಂಶ + tensor->view_offs;
    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init_tensor(tensor->ಬಫರ್, tensor);
}

ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ggml_backend_tensor_alloc(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->ಬಫರ್ == ಖಾಲಿ);
    GGML_ASSERT(tensor->ದತ್ತಾಂಶ == ಖಾಲಿ);
    GGML_ASSERT(tensor->view_src == ಖಾಲಿ);
    GGML_ASSERT(addr >= ggml_backend_ಬಫರ್_get_ತಳ(ಬಫರ್));
    GGML_ASSERT(ggml_backend_ಬಫರ್_is_meta(ಬಫರ್) ||
        (ಅಕ್ಷರ *) addr + ggml_backend_ಬಫರ್_get_alloc_ಗಾತ್ರ(ಬಫರ್, tensor) <=
        (ಅಕ್ಷರ *) ggml_backend_ಬಫರ್_get_ತಳ(ಬಫರ್) + ggml_backend_ಬಫರ್_get_ಗಾತ್ರ(ಬಫರ್));

    tensor->ಬಫರ್ = ಬಫರ್;
    tensor->ದತ್ತಾಂಶ = addr;
    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init_tensor(ಬಫರ್, tensor);
}

ಸ್ಥಿರ ರಚನೆ ggml_tensor * ನಕ್ಷೆ_copy_dup_tensor(ರಚನೆ ggml_hash_ಗಣ hash_ಗಣ, ರಚನೆ ggml_tensor ** ಗಂಟು_copies,
    ರಚನೆ ggml_context * ctx_allocated, ರಚನೆ ggml_context * ctx_unallocated, ರಚನೆ ggml_tensor * src) {

    GGML_ASSERT(src != ಖಾಲಿ);
    GGML_ASSERT(src->ದತ್ತಾಂಶ && "graph must be allocated");

    size_t id = ggml_hash_ಸೇರಿಸು(&hash_ಗಣ, src);
    ಆದರೆ (id == GGML_HASHSET_ALREADY_EXISTS) {
        ಹಿಂತಿರುಗಿ ಗಂಟು_copies[ggml_hash_ಕಂಡುಹಿಡಿ(&hash_ಗಣ, src)];
    }

    ರಚನೆ ggml_tensor * dst = ggml_dup_tensor_layout(src->ದತ್ತಾಂಶ && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    ಆದರೆ (src->view_src != ಖಾಲಿ) {
        dst->view_src = ನಕ್ಷೆ_copy_dup_tensor(hash_ಗಣ, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    ಸ್ಮರಣೆ_ನಕಲು(dst->op_params, src->op_params, ಗಾತ್ರ(dst->op_params));
    ggml_ಗಣ_name(dst, src->name);

    // copy src
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ರಚನೆ ggml_tensor * s = src->src[i];
        ಆದರೆ (s == ಖಾಲಿ) {
            ಮುಂದುವರಿ;
        }
        dst->src[i] = ನಕ್ಷೆ_copy_dup_tensor(hash_ಗಣ, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, s);
    }

    ಗಂಟು_copies[id] = dst;
    ಹಿಂತಿರುಗಿ dst;
}

ಸ್ಥಿರ ಶೂನ್ಯ ನಕ್ಷೆ_copy_init_tensor(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor ** ಗಂಟು_copies, ತಾರ್ಕಿಕ * ಗಂಟು_init, ರಚನೆ ggml_tensor * src) {
    size_t id = ggml_hash_ಕಂಡುಹಿಡಿ(hash_ಗಣ, src);
    ಆದರೆ (ಗಂಟು_init[id]) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಂಟು_init[id] = ನಿಜ;

    ರಚನೆ ggml_tensor * dst = ಗಂಟು_copies[id];
    ಆದರೆ (dst->view_src != ಖಾಲಿ) {
        ನಕ್ಷೆ_copy_init_tensor(hash_ಗಣ, ಗಂಟು_copies, ಗಂಟು_init, src->view_src);
        ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = ggml_backend_view_init(dst);
        GGML_ASSERT(ಸ್ಥಿತಿ_2 == GGML_ಸ್ಥಿತಿ_SUCCESS);
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC; i++) {
        ರಚನೆ ggml_tensor * s = src->src[i];
        ಆದರೆ (s == ಖಾಲಿ) {
            ಮುಂದುವರಿ;
        }
        ನಕ್ಷೆ_copy_init_tensor(hash_ಗಣ, ಗಂಟು_copies, ಗಂಟು_init, s);
    }
}

ರಚನೆ ggml_backend_ನಕ್ಷೆ_copy ggml_backend_ನಕ್ಷೆ_copy(ggml_backend_t backend, ರಚನೆ ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    ರಚನೆ ggml_hash_ಗಣ hash_ಗಣ = ggml_hash_ಗಣ_new(graph->ಭೇಟಿಯಾದ_hash_ಗಣ.size);
    ರಚನೆ ggml_tensor ** ಗಂಟು_copies = (ggml_tensor **) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(hash_ಗಣ.size, ಗಾತ್ರ(ಗಂಟು_copies[0])); // NOLINT
    ತಾರ್ಕಿಕ * ಗಂಟು_init = (ತಾರ್ಕಿಕ *) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(hash_ಗಣ.size, ಗಾತ್ರ(ಗಂಟು_init[0]));

    ರಚನೆ ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_ಗಣ.size + ggml_ನಕ್ಷೆ_overhead_custom(graph->size, ಸುಳ್ಳು),
        /* .mem_buffer = */ ಖಾಲಿ,
        /* .no_alloc   = */ ನಿಜ
    };

    ರಚನೆ ggml_context * ctx_allocated = ggml_init(params);
    ರಚನೆ ggml_context * ctx_unallocated = ggml_init(params);

    ಆದರೆ (ctx_allocated == ಖಾಲಿ || ctx_unallocated == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_ಗಣ_free(&hash_ಗಣ);
        ಬಿಡುಗಡೆ(ಗಂಟು_copies);
        ಬಿಡುಗಡೆ(ಗಂಟು_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ಹಿಂತಿರುಗಿ {
            /* .buffer           = */ ಖಾಲಿ,
            /* .ctx_allocated    = */ ಖಾಲಿ,
            /* .ctx_unallocated  = */ ಖಾಲಿ,
            /* .graph            = */ ಖಾಲಿ,
        };
    }

    // dup nodes
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ನಕ್ಷೆ_copy_dup_tensor(hash_ಗಣ, ಗಂಟು_copies, ctx_allocated, ctx_unallocated, ಗಂಟು);
    }

    // allocate nodes
    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    ಆದರೆ (ಬಫರ್ == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_ಗಣ_free(&hash_ಗಣ);
        ಬಿಡುಗಡೆ(ಗಂಟು_copies);
        ಬಿಡುಗಡೆ(ಗಂಟು_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        ಹಿಂತಿರುಗಿ {
            /* .buffer           = */ ಖಾಲಿ,
            /* .ctx_allocated    = */ ಖಾಲಿ,
            /* .ctx_unallocated  = */ ಖಾಲಿ,
            /* .graph            = */ ಖಾಲಿ,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ನಕ್ಷೆ_copy_init_tensor(&hash_ಗಣ, ಗಂಟು_copies, ಗಂಟು_init, ಗಂಟು);
    }

    // build graph copy
    ರಚನೆ ggml_cgraph * ನಕ್ಷೆ_copy = ggml_new_ನಕ್ಷೆ_custom(ctx_allocated, graph->size, ಸುಳ್ಳು);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < graph->n_nodes; i++) {
        ರಚನೆ ggml_tensor * ಗಂಟು = graph->nodes[i];
        ರಚನೆ ggml_tensor * ಗಂಟು_copy = ಗಂಟು_copies[ggml_hash_ಕಂಡುಹಿಡಿ(&hash_ಗಣ, ಗಂಟು)];
        ನಕ್ಷೆ_copy->nodes[i] = ಗಂಟು_copy;
    }
    ನಕ್ಷೆ_copy->n_nodes = graph->n_nodes;

    ggml_hash_ಗಣ_free(&hash_ಗಣ);
    ಬಿಡುಗಡೆ(ಗಂಟು_copies);
    ಬಿಡುಗಡೆ(ಗಂಟು_init);

    ಹಿಂತಿರುಗಿ {
        /* .buffer           = */ ಬಫರ್,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ ನಕ್ಷೆ_copy,
    };
}

ಶೂನ್ಯ ggml_backend_ನಕ್ಷೆ_copy_free(ರಚನೆ ggml_backend_ನಕ್ಷೆ_copy copy) {
    ggml_backend_ಬಫರ್_free(copy.ಬಫರ್);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

ತಾರ್ಕಿಕ ggml_backend_compare_ನಕ್ಷೆ_backend(ggml_backend_t backend1, ggml_backend_t backend2, ರಚನೆ ggml_cgraph * graph, ggml_backend_eval_callback callback, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ, ರಚನೆ ggml_tensor ಸ್ಥಿರಾಂಕ * ಸ್ಥಿರಾಂಕ * test_nodes, size_t ಅಂಕೆ_test_nodes) {
    ರಚನೆ ggml_backend_ನಕ್ಷೆ_copy copy = ggml_backend_ನಕ್ಷೆ_copy(backend2, graph);
    ಆದರೆ (copy.ಬಫರ್ == ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ರಚನೆ ggml_cgraph * g1 = graph;
    ರಚನೆ ggml_cgraph * g2 = copy.graph;

    ಪ್ರತಿಜ್ಞಾ(g1->n_nodes == g2->n_nodes);

    ಆದರೆ (ಅಂಕೆ_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_ನಕ್ಷೆ_compute(backend1, g1);
        ggml_backend_ನಕ್ಷೆ_compute(backend2, g2);

        ತಾರ್ಕಿಕ verified = ಸುಳ್ಳು;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g1->n_nodes; i++) {
            ಗಾಗಿ (size_t j = 0; j < ಅಂಕೆ_test_nodes; ++j) {
                ಆದರೆ (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_ದತ್ತಾಂಶ);
                    verified = ನಿಜ;
                }
            }
        }
        GGML_ASSERT(verified);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < g1->n_nodes; i++) {
            ರಚನೆ ggml_tensor * t1 = g1->nodes[i];
            ರಚನೆ ggml_tensor * t2 = g2->nodes[i];

            ಪ್ರತಿಜ್ಞಾ(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            ರಚನೆ ggml_cgraph g1v = ggml_ನಕ್ಷೆ_view(g1, i, i + 1);
            ರಚನೆ ggml_cgraph g2v = ggml_ನಕ್ಷೆ_view(g2, i, i + 1);

            ggml_backend_ನಕ್ಷೆ_compute(backend1, &g1v);
            ggml_backend_ನಕ್ಷೆ_compute(backend2, &g2v);

            ಆದರೆ (ggml_is_view_op(t1->op)) {
                ಮುಂದುವರಿ;
            }

            // compare results, calculate rms etc
            ಆದರೆ (!callback(i, t1, t2, user_ದತ್ತಾಂಶ)) {
                ನಿಲ್ಲಿಸು;
            }
        }
    }
    ggml_backend_ನಕ್ಷೆ_copy_free(copy);

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// CPU backend - buffer

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_cpu_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    uintptr_t ದತ್ತಾಂಶ = (uintptr_t)ಬಫರ್->context;

    // align the buffer
    ಆದರೆ (ದತ್ತಾಂಶ % TENSOR_ALIGNMENT != 0) {
        ದತ್ತಾಂಶ = GGML_PAD(ದತ್ತಾಂಶ, TENSOR_ALIGNMENT);
    }

    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ದತ್ತಾಂಶ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_ASSERT(ಬಫರ್);
    ggml_aligned_free(ಬಫರ್->context, ಬಫರ್->size);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor, uint8_t value, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    memset((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, value, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_ASSERT(tensor);
    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst) {
    GGML_ASSERT(src);
    ಆದರೆ (ggml_backend_ಬಫರ್_is_host(src->ಬಫರ್)) {
        ಸ್ಮರಣೆ_ನಕಲು(dst->ದತ್ತಾಂಶ, src->ದತ್ತಾಂಶ, ggml_nbytes(src));
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_cpu_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    GGML_ASSERT(ಬಫರ್);
    memset(ಬಫರ್->context, value, ಬಫರ್->size);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_i ggml_backend_cpu_ಬಫರ್_i = {
    /* .free_buffer     = */ ggml_backend_cpu_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_cpu_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ಖಾಲಿ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_cpu_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_i ggml_backend_cpu_ಬಫರ್_from_ptr_i = {
    /* .free_buffer     = */ ಖಾಲಿ, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ಖಾಲಿ, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_cpu_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "CPU";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_cpu_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಶೂನ್ಯ * ದತ್ತಾಂಶ = ggml_aligned_malloc(size);

    ಆದರೆ (ದತ್ತಾಂಶ == ಖಾಲಿ) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_cpu_ಬಫರ್_i, ದತ್ತಾಂಶ, size);
}

ಸ್ಥಿರ size_t ggml_backend_cpu_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ TENSOR_ALIGNMENT;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_cpu_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(buft);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_cpu_ಬಫರ್_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_ಬಫರ್_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_ಬಫರ್_type_alloc_ಬಫರ್,
            /* .alloc_buffer_n      = */ ಖಾಲಿ,
            /* .get_alignment       = */ ggml_backend_cpu_ಬಫರ್_type_get_alignment,
            /* .get_max_size        = */ ಖಾಲಿ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ಖಾಲಿ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ಖಾಲಿ,
            /* .is_host             = */ ggml_backend_cpu_ಬಫರ್_type_is_host,
        },
        /* .device  = */ ಖಾಲಿ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_ಬಫರ್_type;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_cpu_ಬಫರ್_from_ptr_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ "CPU_Mapped";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_cpu_ಬಫರ್_from_ptr_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_cpu_ಬಫರ್_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_ಬಫರ್_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_ಬಫರ್_type_alloc_ಬಫರ್,
            /* .alloc_buffer_n      = */ ಖಾಲಿ,
            /* .get_alignment       = */ ggml_backend_cpu_ಬಫರ್_type_get_alignment,
            /* .get_max_size        = */ ಖಾಲಿ, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ಖಾಲಿ, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ ಖಾಲಿ,
            /* .is_host             = */ ggml_backend_cpu_ಬಫರ್_type_is_host,
        },
        /* .device  = */ ಖಾಲಿ, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ ಖಾಲಿ,
    };

    ಹಿಂತಿರುಗಿ &ggml_backend_cpu_ಬಫರ್_type;
}

ggml_backend_ಬಫರ್_t ggml_backend_cpu_ಬಫರ್_from_ptr(ಶೂನ್ಯ * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(ggml_backend_cpu_ಬಫರ್_from_ptr_type(), ggml_backend_cpu_ಬಫರ್_from_ptr_i, ptr, size);
}
