#ਹਦਾਇਤ once

// ggml-backend internal header

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ  __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BACKEND_API_ਸੰਸਕਰਣ 3

    //
    // Backend buffer type
    //

    ਬਣਤਰ ggml_backend_ਬਫਰ_type_i {
        ਅਚਲ ਅੱਖਰ *          (*get_name)        (ggml_backend_ਬਫਰ_type_t buft);
        // allocate a buffer of this type
        ggml_backend_ਬਫਰ_t (*alloc_ਬਫਰ)    (ggml_backend_ਬਫਰ_type_t buft, size_t ਮਾਪ);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_ਬਫਰ_t (*alloc_ਬਫਰ_n)  (ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_ਬਫਰ_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_ਮਾਪ)    (ggml_backend_ਬਫਰ_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_ਮਾਪ)  (ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_ਮਾਪ_n)(ggml_backend_ਬਫਰ_type_t buft, ਬਣਤਰ ggml_tensor ** tensors, ਪੂਰਨ n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_ਬਫਰ_type_t buft);
    };

    ਬਣਤਰ ggml_backend_ਬਫਰ_type {
        ਬਣਤਰ ggml_backend_ਬਫਰ_type_i  iface;
        ggml_backend_dev_t device;
        ਖਾਲੀ * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_ਮਾਪ_may_expand(ਗਿਣਤੀ ggml_op op);

    //
    // Backend buffer
    //

    ਬਣਤਰ ggml_backend_ਬਫਰ_i {
        // (optional) free the buffer
        ਖਾਲੀ         (*free_ਬਫਰ)  (ggml_backend_ਬਫਰ_t ਬਫਰ);
        // base address of the buffer
        ਖਾਲੀ *       (*get_ਅਧਾਰ)     (ggml_backend_ਬਫਰ_t ਬਫਰ);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        ਗਿਣਤੀ ggml_ਸਥਿਤੀ (*init_tensor)(ggml_backend_ਬਫਰ_t ਬਫਰ, ਬਣਤਰ ggml_tensor * tensor);
        // tensor data access
        ਖਾਲੀ         (*memset_tensor)(ggml_backend_ਬਫਰ_t ਬਫਰ,       ਬਣਤਰ ggml_tensor * tensor,     uint8_t ਮੁੱਲ, size_t ਵਿੱਥ, size_t ਮਾਪ);
        ਖਾਲੀ         (*ਸੈੱਟ_tensor)   (ggml_backend_ਬਫਰ_t ਬਫਰ,       ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
        ਖਾਲੀ         (*get_tensor)   (ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor,       ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
        // (optional) 2d data copies
        ਖਾਲੀ         (*ਸੈੱਟ_tensor_2d)(ggml_backend_ਬਫਰ_t ਬਫਰ,       ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);
        ਖਾਲੀ         (*get_tensor_2d)(ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor,       ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_ਬਫਰ_t ਬਫਰ, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst);
        // clear the entire buffer
        ਖਾਲੀ         (*clear)        (ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t ਮੁੱਲ);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        ਖਾਲੀ         (*reset)        (ggml_backend_ਬਫਰ_t ਬਫਰ);
    };

    ਬਣਤਰ ggml_backend_ਬਫਰ {
        ਬਣਤਰ ggml_backend_ਬਫਰ_i  iface;
        ggml_backend_ਬਫਰ_type_t    buft;
        ਖਾਲੀ * context;
        size_t ਮਾਪ;
        ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage;
    };

    GGML_API ggml_backend_ਬਫਰ_t ggml_backend_ਬਫਰ_init(
                   ggml_backend_ਬਫਰ_type_t buft,
            ਬਣਤਰ ggml_backend_ਬਫਰ_i      iface,
                   ਖਾਲੀ *                     context,
                   size_t                     ਮਾਪ);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_ਬਫਰ_copy_tensor(ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_ਬਫਰ_t ggml_backend_multi_ਬਫਰ_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_ਬਫਰ_is_multi_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ);
    GGML_API ਖਾਲੀ                  ggml_backend_multi_ਬਫਰ_ਸੈੱਟ_usage(ggml_backend_ਬਫਰ_t ਬਫਰ, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage);
    GGML_API ਖਾਲੀ                  ggml_backend_meta_ਬਫਰ_ਸੈੱਟ_usage (ggml_backend_ਬਫਰ_t ਬਫਰ, ਗਿਣਤੀ ggml_backend_ਬਫਰ_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_ਬਫਰ_is_meta(ggml_backend_ਬਫਰ_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_ਬਫਰ_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t ਸੂਚਕਾਂਕ);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        ਖਾਲੀ (*ਜੋੜੋ_alloc_dep)(ਖਾਲੀ * user_ਅੰਕੜੇ, ਬਣਤਰ ggml_tensor * tensor, ਬਣਤਰ ggml_tensor * until);
        ਖਾਲੀ * user_ਅੰਕੜੇ;
    };

    ਬਣਤਰ ggml_backend_i {
        ਅਚਲ ਅੱਖਰ * (*get_name)(ggml_backend_t backend);

        ਖਾਲੀ (*ਮੁਕਤ)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        ਖਾਲੀ (*ਸੈੱਟ_tensor_async)   (ggml_backend_t backend,       ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
        ਖਾਲੀ (*get_tensor_async)   (ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor,       ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ);
        ਖਾਲੀ (*ਸੈੱਟ_tensor_2d_async)(ggml_backend_t backend,       ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);
        ਖਾਲੀ (*get_tensor_2d_async)(ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_tensor * tensor,       ਖਾਲੀ * ਅੰਕੜੇ, size_t ਵਿੱਥ, size_t ਮਾਪ, size_t n_copies, size_t stride_tensor, size_t stride_ਅੰਕੜੇ);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, ਅਚਲ ਬਣਤਰ ggml_tensor * src, ਬਣਤਰ ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        ਖਾਲੀ (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_ਗ੍ਰਾਫ_plan_t (*ਗ੍ਰਾਫ_plan_create) (ggml_backend_t backend, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph);
        ਖਾਲੀ                      (*ਗ੍ਰਾਫ_plan_free)   (ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        ਖਾਲੀ                      (*ਗ੍ਰਾਫ_plan_update) (ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan, ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph);
        // compute the graph with the plan
        ਗਿਣਤੀ ggml_ਸਥਿਤੀ          (*ਗ੍ਰਾਫ_plan_compute)(ggml_backend_t backend, ggml_backend_ਗ੍ਰਾਫ_plan_t plan);

        // compute graph (always async if supported by the backend)
        ਗਿਣਤੀ ggml_ਸਥਿਤੀ          (*ਗ੍ਰਾਫ_compute)     (ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        ਖਾਲੀ (*event_ਰਿਕਾਰਡ)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        ਖਾਲੀ (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        ਖਾਲੀ                      (*ਗ੍ਰਾਫ_optimize)    (ggml_backend_t backend, ਬਣਤਰ ggml_cgraph * cgraph, ਬਣਤਰ ggml_backend_ਗ੍ਰਾਫ_optimize_params * params);
    };

    ਬਣਤਰ ggml_backend {
        ggml_guid_t guid;
        ਬਣਤਰ ggml_backend_i iface;
        ggml_backend_dev_t device;
        ਖਾਲੀ * context;
    };

    ਬਣਤਰ ggml_backend_event {
        ਬਣਤਰ ggml_backend_device * device;
        ਖਾਲੀ * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    ਬਣਤਰ ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        ਅਚਲ ਅੱਖਰ * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        ਅਚਲ ਅੱਖਰ * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        ਖਾਲੀ         (*get_ਸਮ੍ਰਿਤੀ)(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ);

        // device type
        ਗਿਣਤੀ ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        ਖਾਲੀ (*get_props)(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params);

        // preferred buffer type
        ggml_backend_ਬਫਰ_type_t (*get_ਬਫਰ_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_ਬਫਰ_type_t (*get_host_ਬਫਰ_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_ਬਫਰ_t (*ਬਫਰ_from_host_ptr)(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t ਮਾਪ, size_t max_tensor_ਮਾਪ);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        ਖਾਲੀ                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        ਖਾਲੀ                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    ਬਣਤਰ ggml_backend_device {
        ਬਣਤਰ ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        ਖਾਲੀ * context;
    };

    //
    // Backend (reg)
    //

    ਬਣਤਰ ggml_backend_reg_i {
        ਅਚਲ ਅੱਖਰ * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_ਗਿਣਤੀ)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        ਖਾਲੀ * (*get_proc_ਪਤਾ)(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name);
    };

    ਬਣਤਰ ggml_backend_reg {
        ਪੂਰਨ api_ਸੰਸਕਰਣ; // initialize to GGML_BACKEND_API_VERSION
        ਬਣਤਰ ggml_backend_reg_i iface;
        ਖਾਲੀ * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ ggml_backend_reg_t (*ggml_backend_init_t)(ਖਾਲੀ);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਪੂਰਨ                (*ggml_backend_score_t)(ਖਾਲੀ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            ਬਾਹਰੀ "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(ਖਾਲੀ); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(ਖਾਲੀ) {                 \
                ਵਾਪਸ reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            ਬਾਹਰੀ "C" {                                   \
            GGML_BACKEND_API ਪੂਰਨ ggml_backend_score(ਖਾਲੀ); \
            }                                              \
            ਪੂਰਨ ggml_backend_score(ਖਾਲੀ) {                 \
                ਵਾਪਸ score_fn();                         \
            }
#    ਨਹੀਂਤਾਂ
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(ਖਾਲੀ);  \
            ggml_backend_reg_t                  ggml_backend_init(ਖਾਲੀ) { \
                ਵਾਪਸ reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API ਪੂਰਨ ggml_backend_score(ਖਾਲੀ);  \
            ਪੂਰਨ                  ggml_backend_score(ਖਾਲੀ) { \
                ਵਾਪਸ score_fn();                          \
            }
#    endif
#ਨਹੀਂਤਾਂ
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ  __cplusplus
}
#ਜੇ_ਅੰਤ
