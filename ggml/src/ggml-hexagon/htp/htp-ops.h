#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OPS_H

#ಅನ್ತರ್ಭಾವ <assert.h>

// ggml-common.h must be included prio to this header

ಎಣಿಕೆ htp_ಸ್ಥಿತಿ {
    HTP_ಸ್ಥಿತಿ_OK             = 1,
    HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR   = 2,
    HTP_ಸ್ಥಿತಿ_NO_SUPPORT     = 3,
    HTP_ಸ್ಥಿತಿ_INVAL_PARAMS   = 4,
    HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
ಎಣಿಕೆ htp_ದತ್ತಾಂಶ_type {
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
#ವ್ಯಾಖ್ಯಾನಿಸು QK_Q8_0_TILED  128  // 32x32 Q8_0 tiled layout



// Do not reorder first 4 (used as an index)
ಎಣಿಕೆ htp_op_ಸಂಕೇತ {
    HTP_OP_MUL = 0,
    HTP_OP_ಕೂಡಿಸು = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_MUL_MAT_NX,
    HTP_OP_MUL_MAT_ID_NX,
    HTP_OP_MUL_MAT_ಕೂಡಿಸು,
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
    HTP_OP_UNARY_ಹೆಜ್ಜೆ,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_GLU_GEGLU_ಶೀಘ್ರ,
    HTP_OP_SOFTMAX,
    HTP_OP_ಕೂಡಿಸು_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು,
    HTP_OP_GET_ಅಡ್ಡಸಾಲುಗಳು,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_CPY_FENCE,
    HTP_OP_ARGSORT,
    HTP_OP_ತುದಿ_K,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_ಮೊತ್ತ,
    HTP_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು,
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
    HTP_OP_IM2_ಲಂಬಸಾಲು,
    HTP_OP_FENCE,
    HTP_OP_ALLREDUCE,
    HTP_OP_ALLREDUCE_ಕೂಡಿಸು,
    HTP_OP_GLU_SWIGLU_CLAMP,
    HTP_OP_MDEV_ಗುಂಪು,
    HTP_OP_ROLL,
    HTP_OP_ARGMAX,
    HTP_OP_UNARY_GELU_ERF,
    HTP_OP_GLU_GEGLU_ERF,
    HTP_OP_POOL_2D,
    HTP_OP_POOL_1D,
    HTP_OP_UNARY_GELU_ಶೀಘ್ರ,
    HTP_OP_IM2_ಲಂಬಸಾಲು_3_D,
    HTP_OP_HADAMARD,

    HTP_OP_INVALID
};

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_INPUTS  10   // aka GGML_MAX_SRCS
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_OUTPUTS 4
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_KERN_PARAMS 32

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_BUFS    16
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_TENSORS 8192 // must stay under 64K (uint16)

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FENCE_TIMEOUT  (1000000000ULL)
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_FENCE_ಸ್ಥಾನ_ಗಾತ್ರ 128

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_OP_MAX_VMEM_DEFAULT (3355443200u)

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MMAP_MAX_VMEM  (2147483648u)

ಎಣಿಕೆ htp_tensor_flags {
    HTP_TENSOR_ತೂಕ  = (1U << 0), // Tensor buffer model weight data (not compute)
    HTP_TENSOR_TILED   = (1U << 1), // Tensor is in tiled format
    HTP_TENSOR_FENCE   = (1U << 2)  // Tensor is synchronization fence (explicitly managed)
};

ಎಣಿಕೆ htp_buf_flags {
    HTP_BUF_EXTENDED = (1U << 0),
};

// Tensor descriptor
ರಚನೆ htp_tensor {
    uint64_t ದತ್ತಾಂಶ;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t ಗಾತ್ರ_2;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint32_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint16_t ti;                   // Tensor index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
ರಚನೆ htp_buf_desc {
    uint64_t ತಳ;     // base address
    uint64_t ಗಾತ್ರ_2;     // total size
    uint32_t flags;    // HTP_BUF_*
    uint32_t fd;       // file descriptor
};

ಎಣಿಕೆ htp_op_flags {
    HTP_OPFLAGS_STUB  = (1U << 0),
};

// Op descriptor
ರಚನೆ htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    int32_t  kernel_params[HTP_OP_MAX_KERN_PARAMS]; // generic blob for host-precomputed parameters
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst[HTP_OP_MAX_OUTPUTS];   // Output tensor indices
    uint16_t pad[2];                    // padding to align to 64 bits
};

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_MAX_NTHREADS
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MAX_NTHREADS 10
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_TRACE_MAX_EVENTS 256

ಎಣಿಕೆ htp_profiler_ಬಹುಲಕ {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
    HTP_PROF_TRACE    = 3,
};

// DCVS/HMX power profile requested for the session.
ಎಣಿಕೆ htp_power_ಬಹುಲಕ {
    HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_POWER_SAVER            = 0,
    HTP_POWER_ಬಹುಲಕ_POWER_SAVER                = 1,
    HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_POWER_SAVER           = 2,
    HTP_POWER_ಬಹುಲಕ_ಕೆಳಗಿನ_ಸಮತೋಲಿತ               = 3,
    HTP_POWER_ಬಹುಲಕ_ಸಮತೋಲಿತ                   = 4,
    HTP_POWER_ಬಹುಲಕ_ಮೇಲಿನ_PERFORMANCE           = 5,
    HTP_POWER_ಬಹುಲಕ_SUSTAINED_ಮೇಲಿನ_PERFORMANCE = 6,
    HTP_POWER_ಬಹುಲಕ_ಅವಧಿ                      = 7,

    HTP_POWER_ಬಹುಲಕ_ಎಣಿಕೆ
};

ಎಣಿಕೆ htp_trace_event_id {
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
    HTP_TRACE_EVT_HVX_GDN_ಹೊರಹೋಗು         = 35,
    HTP_TRACE_EVT_HVX_GDN_STATE       = 36,
    HTP_TRACE_EVT_HVX_GDN_REM         = 37,

    HTP_TRACE_EVT_HMX_COMP            = 40,
};

ರಚನೆ htp_trace_desc {
    uint32_t ಚಕ್ರಗಳು;  // lower 32-bits of cycle counter
    uint16_t id;      // Event ID
    uint16_t info;    // bit 15: is_stop. bits 14-0: tile/chunk index or other metadata.
};

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_PROF_PMU_NCNT 8

// Profile descriptor
ರಚನೆ htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t ಚಕ್ರಗಳು_ಆರಂಭ;           // Start cycle counter
    uint32_t ಚಕ್ರಗಳು_stop;            // Stop cycle counter
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

ರಚನೆ htp_opbatch_req {
    uint64_t seq;         // Sequence number
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t n_traces;    // Number of trace descriptors per thread
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

ರಚನೆ htp_opbatch_rsp {
    uint64_t seq;            // Sequence number
    uint64_t ಚಕ್ರಗಳು_ಆರಂಭ;   // Start cycle counter
    uint64_t ಚಕ್ರಗಳು_stop;    // Stop cycle counter
    uint32_t ಸ್ಥಿತಿ_2;         // HTP_STATUS_...
    uint32_t n_bufs;         // Number of buffers
    uint32_t n_tensors;      // Number of tensors
    uint32_t n_ops;          // Number of op profile descriptors
    uint32_t usecs;          // Number of usec
    uint32_t n_traces[HTP_MAX_NTHREADS + 1];
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#ಆದರೆ_ಅಂತ್ಯ /* HTP_OPS_H */
