#ಸೂಚನೆ once

// ggml-backend internal header

#ಅನ್ತರ್ಭಾವ "ggml-backend.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ  __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_BACKEND_API_ಆವೃತ್ತಿ 3

    //
    // Backend buffer type
    //

    ರಚನೆ ggml_backend_ಬಫರ್_type_i {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *          (*get_name)        (ggml_backend_ಬಫರ್_type_t buft);
        // allocate a buffer of this type
        ggml_backend_ಬಫರ್_t (*alloc_ಬಫರ್)    (ggml_backend_ಬಫರ್_type_t buft, size_t ಗಾತ್ರ_2);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_ಬಫರ್_t (*alloc_ಬಫರ್_n)  (ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_ಬಫರ್_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_ಗಾತ್ರ)    (ggml_backend_ಬಫರ್_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_ಗಾತ್ರ)  (ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_ಗಾತ್ರ_n)(ggml_backend_ಬಫರ್_type_t buft, ರಚನೆ ggml_tensor ** tensors, ಪೂರ್ಣಾಂಕ n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_ಬಫರ್_type_t buft);
    };

    ರಚನೆ ggml_backend_ಬಫರ್_type {
        ರಚನೆ ggml_backend_ಬಫರ್_type_i  iface;
        ggml_backend_dev_t device;
        ಶೂನ್ಯ * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_ಗಾತ್ರ_may_expand(ಎಣಿಕೆ ggml_op op);

    //
    // Backend buffer
    //

    ರಚನೆ ggml_backend_ಬಫರ್_i {
        // (optional) free the buffer
        ಶೂನ್ಯ         (*free_ಬಫರ್)  (ggml_backend_ಬಫರ್_t ಬಫರ್);
        // base address of the buffer
        ಶೂನ್ಯ *       (*get_ತಳ)     (ggml_backend_ಬಫರ್_t ಬಫರ್);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ (*init_tensor)(ggml_backend_ಬಫರ್_t ಬಫರ್, ರಚನೆ ggml_tensor * tensor);
        // tensor data access
        ಶೂನ್ಯ         (*memset_tensor)(ggml_backend_ಬಫರ್_t ಬಫರ್,       ರಚನೆ ggml_tensor * tensor,     uint8_t ಮೌಲ್ಯ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
        ಶೂನ್ಯ         (*ಗಣ_tensor)   (ggml_backend_ಬಫರ್_t ಬಫರ್,       ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
        ಶೂನ್ಯ         (*get_tensor)   (ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor,       ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
        // (optional) 2d data copies
        ಶೂನ್ಯ         (*ಗಣ_tensor_2d)(ggml_backend_ಬಫರ್_t ಬಫರ್,       ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);
        ಶೂನ್ಯ         (*get_tensor_2d)(ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor,       ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_ಬಫರ್_t ಬಫರ್, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst);
        // clear the entire buffer
        ಶೂನ್ಯ         (*clear)        (ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t ಮೌಲ್ಯ);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        ಶೂನ್ಯ         (*reset)        (ggml_backend_ಬಫರ್_t ಬಫರ್);
    };

    ರಚನೆ ggml_backend_ಬಫರ್ {
        ರಚನೆ ggml_backend_ಬಫರ್_i  iface;
        ggml_backend_ಬಫರ್_type_t    buft;
        ಶೂನ್ಯ * context;
        size_t ಗಾತ್ರ_2;
        ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage;
    };

    GGML_API ggml_backend_ಬಫರ್_t ggml_backend_ಬಫರ್_init(
                   ggml_backend_ಬಫರ್_type_t buft,
            ರಚನೆ ggml_backend_ಬಫರ್_i      iface,
                   ಶೂನ್ಯ *                     context,
                   size_t                     ಗಾತ್ರ_2);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_ಬಫರ್_copy_tensor(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_ಬಫರ್_t ggml_backend_multi_ಬಫರ್_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_ಬಫರ್_is_multi_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್);
    GGML_API ಶೂನ್ಯ                  ggml_backend_multi_ಬಫರ್_ಗಣ_usage(ggml_backend_ಬಫರ್_t ಬಫರ್, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage);
    GGML_API ಶೂನ್ಯ                  ggml_backend_meta_ಬಫರ್_ಗಣ_usage (ggml_backend_ಬಫರ್_t ಬಫರ್, ಎಣಿಕೆ ggml_backend_ಬಫರ್_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_ಬಫರ್_is_meta(ggml_backend_ಬಫರ್_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_ಬಫರ್_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t ಸೂಚ್ಯಂಕ);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        ಶೂನ್ಯ (*ಕೂಡಿಸು_alloc_dep)(ಶೂನ್ಯ * user_ದತ್ತಾಂಶ, ರಚನೆ ggml_tensor * tensor, ರಚನೆ ggml_tensor * until);
        ಶೂನ್ಯ * user_ದತ್ತಾಂಶ;
    };

    ರಚನೆ ggml_backend_i {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*get_name)(ggml_backend_t backend);

        ಶೂನ್ಯ (*ಬಿಡುಗಡೆ)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        ಶೂನ್ಯ (*ಗಣ_tensor_async)   (ggml_backend_t backend,       ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
        ಶೂನ್ಯ (*get_tensor_async)   (ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor,       ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);
        ಶೂನ್ಯ (*ಗಣ_tensor_2d_async)(ggml_backend_t backend,       ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);
        ಶೂನ್ಯ (*get_tensor_2d_async)(ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor,       ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2, size_t n_copies, size_t stride_tensor, size_t stride_ದತ್ತಾಂಶ);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src, ರಚನೆ ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        ಶೂನ್ಯ (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_ನಕ್ಷೆ_plan_t (*ನಕ್ಷೆ_plan_create) (ggml_backend_t backend, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph);
        ಶೂನ್ಯ                      (*ನಕ್ಷೆ_plan_free)   (ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        ಶೂನ್ಯ                      (*ನಕ್ಷೆ_plan_update) (ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph);
        // compute the graph with the plan
        ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ          (*ನಕ್ಷೆ_plan_compute)(ggml_backend_t backend, ggml_backend_ನಕ್ಷೆ_plan_t plan);

        // compute graph (always async if supported by the backend)
        ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ          (*ನಕ್ಷೆ_compute)     (ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        ಶೂನ್ಯ (*event_ದಾಖಲೆ)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        ಶೂನ್ಯ (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        ಶೂನ್ಯ                      (*ನಕ್ಷೆ_optimize)    (ggml_backend_t backend, ರಚನೆ ggml_cgraph * cgraph, ರಚನೆ ggml_backend_ನಕ್ಷೆ_optimize_params * params);
    };

    ರಚನೆ ggml_backend {
        ggml_guid_t guid;
        ರಚನೆ ggml_backend_i iface;
        ggml_backend_dev_t device;
        ಶೂನ್ಯ * context;
    };

    ರಚನೆ ggml_backend_event {
        ರಚನೆ ggml_backend_device * device;
        ಶೂನ್ಯ * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    ರಚನೆ ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        ಶೂನ್ಯ         (*get_ಸ್ಮರಣೆ)(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು);

        // device type
        ಎಣಿಕೆ ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        ಶೂನ್ಯ (*get_props)(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params);

        // preferred buffer type
        ggml_backend_ಬಫರ್_type_t (*get_ಬಫರ್_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_ಬಫರ್_type_t (*get_host_ಬಫರ್_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_ಬಫರ್_t (*ಬಫರ್_from_host_ptr)(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2, size_t max_tensor_ಗಾತ್ರ);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        ಶೂನ್ಯ                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        ಶೂನ್ಯ                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    ರಚನೆ ggml_backend_device {
        ರಚನೆ ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        ಶೂನ್ಯ * context;
    };

    //
    // Backend (reg)
    //

    ರಚನೆ ggml_backend_reg_i {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_ಎಣಿಕೆ)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        ಶೂನ್ಯ * (*get_proc_ವಿಳಾಸ)(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name);
    };

    ರಚನೆ ggml_backend_reg {
        ಪೂರ್ಣಾಂಕ api_ಆವೃತ್ತಿ; // initialize to GGML_BACKEND_API_VERSION
        ರಚನೆ ggml_backend_reg_i iface;
        ಶೂನ್ಯ * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ggml_backend_reg_t (*ggml_backend_init_t)(ಶೂನ್ಯ);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಪೂರ್ಣಾಂಕ                (*ggml_backend_score_t)(ಶೂನ್ಯ);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            ಬಾಹ್ಯ "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(ಶೂನ್ಯ); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(ಶೂನ್ಯ) {                 \
                ಹಿಂತಿರುಗಿ reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            ಬಾಹ್ಯ "C" {                                   \
            GGML_BACKEND_API ಪೂರ್ಣಾಂಕ ggml_backend_score(ಶೂನ್ಯ); \
            }                                              \
            ಪೂರ್ಣಾಂಕ ggml_backend_score(ಶೂನ್ಯ) {                 \
                ಹಿಂತಿರುಗಿ score_fn();                         \
            }
#    ಇಲ್ಲದಿದ್ದರೆ
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(ಶೂನ್ಯ);  \
            ggml_backend_reg_t                  ggml_backend_init(ಶೂನ್ಯ) { \
                ಹಿಂತಿರುಗಿ reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API ಪೂರ್ಣಾಂಕ ggml_backend_score(ಶೂನ್ಯ);  \
            ಪೂರ್ಣಾಂಕ                  ggml_backend_score(ಶೂನ್ಯ) { \
                ಹಿಂತಿರುಗಿ score_fn();                          \
            }
#    endif
#ಇಲ್ಲದಿದ್ದರೆ
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ  __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ
