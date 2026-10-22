#pragma once

// ggml-backend internal header

#અન્તર્ભાવ "ggml-backend.h"

#ifdef  __cplusplus
બાહ્ય "C" {
#endif

    #define GGML_BACKEND_API_VERSION 3

    //
    // Backend buffer type
    //

    સંરચના ggml_backend_buffer_type_i {
        અચલ અક્ષર *          (*get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_buffer_t (*alloc_buffer_n)  (ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)    (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, અચલ સંરચના ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_size_n)(ggml_backend_buffer_type_t buft, સંરચના ggml_tensor ** tensors, પૂર્ણાંક n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
    };

    સંરચના ggml_backend_buffer_type {
        સંરચના ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        શૂન્ય * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_size_may_expand(ગણના ggml_op op);

    //
    // Backend buffer
    //

    સંરચના ggml_backend_buffer_i {
        // (optional) free the buffer
        શૂન્ય         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        શૂન્ય *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        ગણના ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, સંરચના ggml_tensor * tensor);
        // tensor data access
        શૂન્ય         (*memset_tensor)(ggml_backend_buffer_t buffer,       સંરચના ggml_tensor * tensor,     uint8_t કિંમત, size_t offset, size_t size);
        શૂન્ય         (*set_tensor)   (ggml_backend_buffer_t buffer,       સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size);
        શૂન્ય         (*get_tensor)   (ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * tensor,       શૂન્ય * data, size_t offset, size_t size);
        // (optional) 2d data copies
        શૂન્ય         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        શૂન્ય         (*get_tensor_2d)(ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * tensor,       શૂન્ય * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst);
        // clear the entire buffer
        શૂન્ય         (*clear)        (ggml_backend_buffer_t buffer, uint8_t કિંમત);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        શૂન્ય         (*reset)        (ggml_backend_buffer_t buffer);
    };

    સંરચના ggml_backend_buffer {
        સંરચના ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        શૂન્ય * context;
        size_t size;
        ગણના ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            સંરચના ggml_backend_buffer_i      iface,
                   શૂન્ય *                     context,
                   size_t                     size);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_buffer_copy_tensor(અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API શૂન્ય                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, ગણના ggml_backend_buffer_usage usage);
    GGML_API શૂન્ય                  ggml_backend_meta_buffer_set_usage (ggml_backend_buffer_t buffer, ગણના ggml_backend_buffer_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    સંરચના ggml_backend_graph_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        શૂન્ય (*add_alloc_dep)(શૂન્ય * user_data, સંરચના ggml_tensor * tensor, સંરચના ggml_tensor * until);
        શૂન્ય * user_data;
    };

    સંરચના ggml_backend_i {
        અચલ અક્ષર * (*get_name)(ggml_backend_t backend);

        શૂન્ય (*મુક્ત)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        શૂન્ય (*set_tensor_async)   (ggml_backend_t backend,       સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size);
        શૂન્ય (*get_tensor_async)   (ggml_backend_t backend, અચલ સંરચના ggml_tensor * tensor,       શૂન્ય * data, size_t offset, size_t size);
        શૂન્ય (*set_tensor_2d_async)(ggml_backend_t backend,       સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        શૂન્ય (*get_tensor_2d_async)(ggml_backend_t backend, અચલ સંરચના ggml_tensor * tensor,       શૂન્ય * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, અચલ સંરચના ggml_tensor * src, સંરચના ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        શૂન્ય (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, અચલ સંરચના ggml_cgraph * cgraph);
        શૂન્ય                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        શૂન્ય                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, અચલ સંરચના ggml_cgraph * cgraph);
        // compute the graph with the plan
        ગણના ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        ગણના ggml_status          (*graph_compute)     (ggml_backend_t backend, સંરચના ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        શૂન્ય (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        શૂન્ય (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        શૂન્ય                      (*graph_optimize)    (ggml_backend_t backend, સંરચના ggml_cgraph * cgraph, સંરચના ggml_backend_graph_optimize_params * params);
    };

    સંરચના ggml_backend {
        ggml_guid_t guid;
        સંરચના ggml_backend_i iface;
        ggml_backend_dev_t device;
        શૂન્ય * context;
    };

    સંરચના ggml_backend_event {
        સંરચના ggml_backend_device * device;
        શૂન્ય * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    સંરચના ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        અચલ અક્ષર * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        અચલ અક્ષર * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        શૂન્ય         (*get_memory)(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total);

        // device type
        ગણના ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        શૂન્ય (*get_props)(ggml_backend_dev_t dev, સંરચના ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, અચલ અક્ષર * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        શૂન્ય                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        શૂન્ય                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    સંરચના ggml_backend_device {
        સંરચના ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        શૂન્ય * context;
    };

    //
    // Backend (reg)
    //

    સંરચના ggml_backend_reg_i {
        અચલ અક્ષર * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_ગણતરી)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        શૂન્ય * (*get_proc_address)(ggml_backend_reg_t reg, અચલ અક્ષર * name);
    };

    સંરચના ggml_backend_reg {
        પૂર્ણાંક api_version; // initialize to GGML_BACKEND_API_VERSION
        સંરચના ggml_backend_reg_i iface;
        શૂન્ય * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    પ્રકારવ્યાખ્યા ggml_backend_reg_t (*ggml_backend_init_t)(શૂન્ય);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    પ્રકારવ્યાખ્યા પૂર્ણાંક                (*ggml_backend_score_t)(શૂન્ય);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            બાહ્ય "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(શૂન્ય); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(શૂન્ય) {                 \
                પાછા reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            બાહ્ય "C" {                                   \
            GGML_BACKEND_API પૂર્ણાંક ggml_backend_score(શૂન્ય); \
            }                                              \
            પૂર્ણાંક ggml_backend_score(શૂન્ય) {                 \
                પાછા score_fn();                         \
            }
#    નહીંતર
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(શૂન્ય);  \
            ggml_backend_reg_t                  ggml_backend_init(શૂન્ય) { \
                પાછા reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API પૂર્ણાંક ggml_backend_score(શૂન્ય);  \
            પૂર્ણાંક                  ggml_backend_score(શૂન્ય) { \
                પાછા score_fn();                          \
            }
#    endif
#નહીંતર
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
