#సూచన once

// ggml-backend internal header

#అన్తర్భావ "ggml-backend.h"

#నిర్వచితమైతే  __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

    #నిర్వచించు GGML_BACKEND_API_సంస్కరణ 3

    //
    // Backend buffer type
    //

    నిర్మాణం ggml_backend_బఫరు_type_i {
        స్థిరరాశి అక్షరం *          (*get_name)        (ggml_backend_బఫరు_type_t buft);
        // allocate a buffer of this type
        ggml_backend_బఫరు_t (*alloc_బఫరు)    (ggml_backend_బఫరు_type_t buft, size_t కొలత);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_బఫరు_t (*alloc_బఫరు_n)  (ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_బఫరు_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_కొలత)    (ggml_backend_బఫరు_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_కొలత)  (ggml_backend_బఫరు_type_t buft, స్థిరరాశి నిర్మాణం ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_కొలత_n)(ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_బఫరు_type_t buft);
    };

    నిర్మాణం ggml_backend_బఫరు_type {
        నిర్మాణం ggml_backend_బఫరు_type_i  iface;
        ggml_backend_dev_t device;
        శూన్య * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_కొలత_may_expand(గణన ggml_op op);

    //
    // Backend buffer
    //

    నిర్మాణం ggml_backend_బఫరు_i {
        // (optional) free the buffer
        శూన్య         (*free_బఫరు)  (ggml_backend_బఫరు_t బఫరు);
        // base address of the buffer
        శూన్య *       (*get_పునాది)     (ggml_backend_బఫరు_t బఫరు);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        గణన ggml_స్థితి (*init_tensor)(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor);
        // tensor data access
        శూన్య         (*memset_tensor)(ggml_backend_బఫరు_t బఫరు,       నిర్మాణం ggml_tensor * tensor,     uint8_t విలువ, size_t స్థానభ్రంశం, size_t కొలత);
        శూన్య         (*సమితి_tensor)   (ggml_backend_బఫరు_t బఫరు,       నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
        శూన్య         (*get_tensor)   (ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * tensor,       శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
        // (optional) 2d data copies
        శూన్య         (*సమితి_tensor_2d)(ggml_backend_బఫరు_t బఫరు,       నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);
        శూన్య         (*get_tensor_2d)(ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * tensor,       శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_బఫరు_t బఫరు, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst);
        // clear the entire buffer
        శూన్య         (*clear)        (ggml_backend_బఫరు_t బఫరు, uint8_t విలువ);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        శూన్య         (*reset)        (ggml_backend_బఫరు_t బఫరు);
    };

    నిర్మాణం ggml_backend_బఫరు {
        నిర్మాణం ggml_backend_బఫరు_i  iface;
        ggml_backend_బఫరు_type_t    buft;
        శూన్య * context;
        size_t కొలత;
        గణన ggml_backend_బఫరు_usage usage;
    };

    GGML_API ggml_backend_బఫరు_t ggml_backend_బఫరు_init(
                   ggml_backend_బఫరు_type_t buft,
            నిర్మాణం ggml_backend_బఫరు_i      iface,
                   శూన్య *                     context,
                   size_t                     కొలత);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_బఫరు_copy_tensor(స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_బఫరు_t ggml_backend_multi_బఫరు_alloc_బఫరు(ggml_backend_బఫరు_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_బఫరు_is_multi_బఫరు(ggml_backend_బఫరు_t బఫరు);
    GGML_API శూన్య                  ggml_backend_multi_బఫరు_సమితి_usage(ggml_backend_బఫరు_t బఫరు, గణన ggml_backend_బఫరు_usage usage);
    GGML_API శూన్య                  ggml_backend_meta_బఫరు_సమితి_usage (ggml_backend_బఫరు_t బఫరు, గణన ggml_backend_బఫరు_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_బఫరు_is_meta(ggml_backend_బఫరు_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_బఫరు_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t సూచీ);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        శూన్య (*కూడు_alloc_dep)(శూన్య * user_దత్తాంశం, నిర్మాణం ggml_tensor * tensor, నిర్మాణం ggml_tensor * until);
        శూన్య * user_దత్తాంశం;
    };

    నిర్మాణం ggml_backend_i {
        స్థిరరాశి అక్షరం * (*get_name)(ggml_backend_t backend);

        శూన్య (*విడుదల)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        శూన్య (*సమితి_tensor_async)   (ggml_backend_t backend,       నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
        శూన్య (*get_tensor_async)   (ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * tensor,       శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత);
        శూన్య (*సమితి_tensor_2d_async)(ggml_backend_t backend,       నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);
        శూన్య (*get_tensor_2d_async)(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * tensor,       శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t కొలత, size_t n_copies, size_t stride_tensor, size_t stride_దత్తాంశం);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి నిర్మాణం ggml_tensor * src, నిర్మాణం ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        శూన్య (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_లేఖాచిత్రం_plan_t (*లేఖాచిత్రం_plan_create) (ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph);
        శూన్య                      (*లేఖాచిత్రం_plan_free)   (ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        శూన్య                      (*లేఖాచిత్రం_plan_update) (ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan, స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph);
        // compute the graph with the plan
        గణన ggml_స్థితి          (*లేఖాచిత్రం_plan_compute)(ggml_backend_t backend, ggml_backend_లేఖాచిత్రం_plan_t plan);

        // compute graph (always async if supported by the backend)
        గణన ggml_స్థితి          (*లేఖాచిత్రం_compute)     (ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        శూన్య (*event_రికార్డు)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        శూన్య (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) sort/optimize the nodes in the graph
        శూన్య                      (*లేఖాచిత్రం_optimize)    (ggml_backend_t backend, నిర్మాణం ggml_cgraph * cgraph, నిర్మాణం ggml_backend_లేఖాచిత్రం_optimize_params * params);
    };

    నిర్మాణం ggml_backend {
        ggml_guid_t guid;
        నిర్మాణం ggml_backend_i iface;
        ggml_backend_dev_t device;
        శూన్య * context;
    };

    నిర్మాణం ggml_backend_event {
        నిర్మాణం ggml_backend_device * device;
        శూన్య * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    నిర్మాణం ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        స్థిరరాశి అక్షరం * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        స్థిరరాశి అక్షరం * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        శూన్య         (*get_జ్ఞాపకం)(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం);

        // device type
        గణన ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        శూన్య (*get_props)(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params);

        // preferred buffer type
        ggml_backend_బఫరు_type_t (*get_బఫరు_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_బఫరు_type_t (*get_host_బఫరు_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_బఫరు_t (*బఫరు_from_host_ptr)(ggml_backend_dev_t dev, శూన్య * ptr, size_t కొలత, size_t max_tensor_కొలత);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        శూన్య                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        శూన్య                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    నిర్మాణం ggml_backend_device {
        నిర్మాణం ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        శూన్య * context;
    };

    //
    // Backend (reg)
    //

    నిర్మాణం ggml_backend_reg_i {
        స్థిరరాశి అక్షరం * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_లెక్క)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t సూచీ);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        శూన్య * (*get_proc_చిరునామా)(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name);
    };

    నిర్మాణం ggml_backend_reg {
        పూర్ణాంకం api_సంస్కరణ; // initialize to GGML_BACKEND_API_VERSION
        నిర్మాణం ggml_backend_reg_i iface;
        శూన్య * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    రకనిర్వచనం ggml_backend_reg_t (*ggml_backend_init_t)(శూన్య);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    రకనిర్వచనం పూర్ణాంకం                (*ggml_backend_score_t)(శూన్య);

#నిర్వచితమైతే GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            బాహ్య "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(శూన్య); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(శూన్య) {                 \
                తిరిగి reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            బాహ్య "C" {                                   \
            GGML_BACKEND_API పూర్ణాంకం ggml_backend_score(శూన్య); \
            }                                              \
            పూర్ణాంకం ggml_backend_score(శూన్య) {                 \
                తిరిగి score_fn();                         \
            }
#    లేకపోతే
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(శూన్య);  \
            ggml_backend_reg_t                  ggml_backend_init(శూన్య) { \
                తిరిగి reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API పూర్ణాంకం ggml_backend_score(శూన్య);  \
            పూర్ణాంకం                  ggml_backend_score(శూన్య) { \
                తిరిగి score_fn();                          \
            }
#    endif
#లేకపోతే
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#అయితే_ముగింపు

#నిర్వచితమైతే  __cplusplus
}
#అయితే_ముగింపు
