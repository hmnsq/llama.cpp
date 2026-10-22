#جيڪڏهن_وصف_نه_ٿيل HTP_OPS_H
#وصف_ڪريو HTP_OPS_H

#ضمن <assert.h>

// ggml-common.h must be included prio to this header

تعداد htp_حالت {
    HTP_حالت_OK             = 1,
    HTP_حالت_اندروني_ERR   = 2,
    HTP_حالت_NO_SUPPORT     = 3,
    HTP_حالت_INVAL_PARAMS   = 4,
    HTP_حالت_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
تعداد htp_معلومات_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q4_1   = 3,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_Q2_K   = 10,
    HTP_TYPE_Q3_K   = 11,
    HTP_TYPE_Q4_K   = 12,
    HTP_TYPE_Q5_K   = 13,
    HTP_TYPE_Q6_K   = 14,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_BF16   = 30,
    HTP_TYPE_MXFP4  = 39,
    HTP_TYPE_Q1_0   = 41,
    HTP_TYPE_Q2_0   = 42,

    HTP_TYPE_INVALID
};

// Constants for internal types
#وصف_ڪريو QK_Q8_0_TILED  128  // 32x32 Q8_0 tiled layout



// Do not reorder first 4 (used as an index)
تعداد htp_op_ڪوڊ {
    HTP_OP_MUL = 0,
    HTP_OP_جوڙيو = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_NX,
    HTP_OP_MUL_MAT_ID_NX,
    HTP_OP_MUL_MAT_جوڙيو,
    HTP_OP_RMS_NORM,
    HTP_OP_RMS_NORM_MUL,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_UNARY_TANH,
    HTP_OP_UNARY_ABS,
    HTP_OP_UNARY_LOG,
    HTP_OP_UNARY_RELU,
    HTP_OP_UNARY_قدم,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_GLU_GEGLU_تيز,
    HTP_OP_SOFTMAX,
    HTP_OP_جوڙيو_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_سيٽ_صفون,
    HTP_OP_GET_صفون,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_CPY_FENCE,
    HTP_OP_ARGSORT,
    HTP_OP_چوٽي_K,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_جوڙ,
    HTP_OP_جوڙ_صفون,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_FILL,
    HTP_OP_DIAG,
    HTP_OP_SOLVE_TRI,
    HTP_OP_L2_NORM,
    HTP_OP_GATED_DELTA_NET,
    HTP_OP_TRI,
    HTP_OP_PAD,
    HTP_OP_NORM,
    HTP_OP_CONCAT,
    HTP_OP_CLAMP,
    HTP_OP_LEAKY_RELU,
    HTP_OP_IM2_کانو,
    HTP_OP_FENCE,
    HTP_OP_ALLREDUCE,
    HTP_OP_ALLREDUCE_جوڙيو,
    HTP_OP_GLU_SWIGLU_CLAMP,
    HTP_OP_MDEV_گروهه,
    HTP_OP_ROLL,
    HTP_OP_ARGMAX,
    HTP_OP_UNARY_GELU_ERF,
    HTP_OP_GLU_GEGLU_ERF,
    HTP_OP_POOL_2D,
    HTP_OP_POOL_1D,
    HTP_OP_UNARY_GELU_تيز,
    HTP_OP_IM2_کانو_3_D,
    HTP_OP_HADAMARD,

    HTP_OP_INVALID
};

#وصف_ڪريو HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#وصف_ڪريو HTP_OP_MAX_INPUTS  10   // aka GGML_MAX_SRCS
#وصف_ڪريو HTP_OP_MAX_OUTPUTS 4
#وصف_ڪريو HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
#وصف_ڪريو HTP_OP_MAX_KERN_PARAMS 32

#وصف_ڪريو HTP_OP_MAX_BUFS    16
#وصف_ڪريو HTP_OP_MAX_TENSORS 8192 // must stay under 64K (uint16)

#وصف_ڪريو HTP_FENCE_TIMEOUT  (1000000000ULL)
#وصف_ڪريو HTP_FENCE_جڳهه_سائيز 128

#وصف_ڪريو HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#وصف_ڪريو HTP_MMAP_MAX_VMEM  (2147483648u)

تعداد htp_tensor_flags {
    HTP_TENSOR_وزن  = (1U << 0), // Tensor buffer model weight data (not compute)
    HTP_TENSOR_TILED   = (1U << 1), // Tensor is in tiled format
    HTP_TENSOR_FENCE   = (1U << 2)  // Tensor is synchronization fence (explicitly managed)
};

تعداد htp_buf_flags {
    HTP_BUF_EXTENDED = (1U << 0),
};

// Tensor descriptor
بنية htp_tensor {
    uint64_t معلومات;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t سائيز;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
بنية htp_buf_desc {
    uint64_t بنياد;     // base address
    uint64_t سائيز;     // total size
    uint32_t flags;    // HTP_BUF_*
    uint32_t fd;       // file descriptor
};

تعداد htp_op_flags {
    HTP_OPFLAGS_STUB  = (1U << 0),
};

// Op descriptor
بنية htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices
    uint16_t pad[2];                    // padding to align to 64 bits
};

#جيڪڏهن_وصف_نه_ٿيل HTP_MAX_NTHREADS
#وصف_ڪريو HTP_MAX_NTHREADS 10
#جيڪڏهن_پورو

#وصف_ڪريو HTP_TRACE_MAX_EVENTS 256

تعداد htp_profiler_گهڻائي {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

// DCVS/HMX power profile requested for the session.
تعداد htp_power_گهڻائي {
    HTP_POWER_گهڻائي_هيٺيون_POWER_SAVER            = 0,
    HTP_POWER_گهڻائي_POWER_SAVER                = 1,
    HTP_POWER_گهڻائي_مٿاهون_POWER_SAVER           = 2,
    HTP_POWER_گهڻائي_هيٺيون_متوازن               = 3,
    HTP_POWER_گهڻائي_متوازن                   = 4,
    HTP_POWER_گهڻائي_مٿاهون_PERFORMANCE           = 5,
    HTP_POWER_گهڻائي_SUSTAINED_مٿاهون_PERFORMANCE = 6,
    HTP_POWER_گهڻائي_مدو                      = 7,

    HTP_POWER_گهڻائي_ڳڻپ
};

تعداد htp_trace_event_id {
    HTP_TRACE_EVT_DMA                 = 0,
    HTP_TRACE_EVT_L2FLUSH             = 1,
    HTP_TRACE_EVT_INIT                = 2,
    HTP_TRACE_EVT_BUFF                = 3,
    HTP_TRACE_EVT_FENCE               = 4,

    HTP_TRACE_EVT_HVX_COMP            = 20,
    HTP_TRACE_EVT_HVX_A_QUANT         = 21,
    HTP_TRACE_EVT_HVX_A_PREP          = 22,
    HTP_TRACE_EVT_HVX_W_DEQUANT       = 23,
    HTP_TRACE_EVT_HVX_W_PREP          = 24,
    HTP_TRACE_EVT_HVX_O_PROC          = 25,
    HTP_TRACE_EVT_HVX_FA_QK           = 26,
    HTP_TRACE_EVT_HVX_FA_SFM          = 27,
    HTP_TRACE_EVT_HVX_FA_Q_PREP       = 28,
    HTP_TRACE_EVT_HVX_FA_K_PREP       = 29,
    HTP_TRACE_EVT_HVX_FA_V_PREP       = 30,

    HTP_TRACE_EVT_HVX_GDN_PREP        = 31,
    HTP_TRACE_EVT_HVX_GDN_SOLVE       = 32,
    HTP_TRACE_EVT_HVX_GDN_V_PREP      = 33,
    HTP_TRACE_EVT_HVX_GDN_D_PREP      = 34,
    HTP_TRACE_EVT_HVX_GDN_ٻاهريون         = 35,
    HTP_TRACE_EVT_HVX_GDN_STATE       = 36,
    HTP_TRACE_EVT_HVX_GDN_REM         = 37,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

بنية htp_trace_desc {
    uint32_t چڪرن;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#وصف_ڪريو HTP_PROF_PMU_NCNT 8

// Profile descriptor
بنية htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t چڪرن_شروعات;           // Start cycle counter
    uint32_t چڪرن_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

بنية htp_opbatch_req {
    uint64_t seq;         // Sequence number
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

بنية htp_opbatch_rsp {
    uint64_t seq;            // Sequence number
    uint64_t چڪرن_شروعات;   // Start cycle counter
    uint64_t چڪرن_stop;    // Stop cycle counter
    uint32_t حالت;         // HTP_STATUS_...
    uint32_t n_bufs;         // Number of buffers
    uint32_t n_tensors;      // Number of tensors
    uint32_t n_ops;          // Number of op profile descriptors
    uint32_t usecs;          // Number of usec
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#جيڪڏهن_پورو /* HTP_OPS_H */
