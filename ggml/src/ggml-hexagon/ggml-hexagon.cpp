#define _USE_MATH_DEFINES

#شامل <assert.h>
#شامل <inttypes.h>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>
#شامل <time.h>

#شامل <atomic>
#شامل <memory>
#شامل <chrono>
#شامل <mutex>
#شامل <thread>
#شامل <cstddef>
#شامل <stdexcept>
#شامل <string>
#شامل <sstream>
#شامل <iomanip>
#شامل <unordered_set>
#شامل <unordered_map>
#شامل <regex>
#شامل <queue>
#شامل <deque>
#شامل <algorithm>
#شامل <cmath>
#شامل <initializer_list>

#ifndef M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#endif

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#وگرنه
#    include <semaphore.h>
#    include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"

#شامل <AEEStdErr.h>
#شامل <dspqueue.h>
#شامل <rpcmem.h>

#define GGML_COMMON_IMPL_CPP
#شامل "ggml-backend-impl.h"
#شامل "ggml-alloc.h"
#شامل "ggml-common.h"
#شامل "ggml-hexagon.h"
#شامل "ggml-impl.h"
#شامل "ggml-quants.h"
#شامل "htp-opnode.h"
#شامل "htp-ops.h"
#شامل "htp/matmul-ops.h"
#شامل "htp/binary-ops.h"
#شامل "htp/flash-attn-ops.h"
#شامل "htp/unary-ops.h"
#شامل "htp/get-rows-ops.h"
#شامل "htp/set-rows-ops.h"
#شامل "htp/softmax-ops.h"
#شامل "htp/pool-ops.h"
#شامل "htp/rope-ops.h"
#شامل "htp/ssm-conv.h"
#شامل "htp/gated-delta-net-ops.h"
#شامل "htp/argsort-ops.h"
#شامل "htp/concat-ops.h"
#شامل "htp/cpy-ops.h"
#شامل "htp/im2col-ops.h"
#شامل "htp_iface.h"
#شامل "htp-drv.h"
#شامل "htp-repack.h"

using intvec  = std::vector<صحیح>;
using uintvec = std::vector<unsigned صحیح>;
using u32vec  = std::vector<uint32_t>;

#define GGML_HEXAGON_MAX_SESSIONS          16

#define GGML_HEXAGON_FENCE_SLOT_SIZE       128

ساختار ggml_hexagon_device_config {
    صحیح         physical_idx = 0;
    صحیح         virtual_idx  = 0;
    صحیح         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_group;
};

ایستا ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

ایستا صحیح    opt_arch    = 0; // autodetect
ایستا size_t opt_ndev    = 1;
ایستا size_t opt_nhvx    = 0; // use all
ایستا صحیح    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
ایستا size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
ایستا صحیح    opt_etm     = 0;
ایستا صحیح    opt_verbose = 0;
ایستا صحیح    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
ایستا منطقی   opt_hostbuf = نادرست;
ایستا منطقی   opt_dma64   = نادرست;
ایستا صحیح    opt_power_mode = HTP_POWER_MODE_BURST; // DCVS/HMX power profile requested at session start

ایستا ثابت std::pair<ثابت نویسه *, صحیح> htp_power_mode_names[] = {
    { "low_power_saver",            HTP_POWER_MODE_کم_POWER_SAVER },
    { "power_saver",                HTP_POWER_MODE_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_MODE_بالا_POWER_SAVER },
    { "low_balanced",               HTP_POWER_MODE_کم_BALANCED },
    { "balanced",                   HTP_POWER_MODE_BALANCED },
    { "high_performance",           HTP_POWER_MODE_بالا_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_MODE_SUSTAINED_بالا_PERFORMANCE },
    { "burst",                      HTP_POWER_MODE_BURST },
    { "default",                    HTP_POWER_MODE_BURST }, // flip this to change the default mode
};

ایستا صحیح htp_power_mode_from_str(ثابت نویسه * s) {
    برای (auto & e : htp_power_mode_names) {
#اگر defined(_MSC_VER)
        اگر (_stricmp(s, e.first) == 0) {
#وگرنه
        اگر (strcasecmp(s, e.first) == 0) {
#endif
            بازگشت e.second;
        }
    }
    بازگشت -1;
}

ایستا ثابت نویسه * htp_power_mode_to_str(صحیح mode) {
    برای (auto & e : htp_power_mode_names) {
        اگر (e.second == mode) {
            بازگشت e.first;
        }
    }
    بازگشت "unknown";
}

ایستا size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
ایستا size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
ایستا size_t opt_mbuf_total  = 0;                        // total buffer space limit (0 = unconstrained)

ایستا صحیح    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ایستا صحیح    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
ایستا صحیح    opt_fa_head_split = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
ایستا صحیح    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
ایستا صحیح    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
ایستا صحیح    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
ایستا u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

ایستا صحیح opt_opbatch  = 1280; // max number of ops in a batch
ایستا صحیح opt_opqueue  = 8;   // max number of pending batches
ایستا صحیح opt_optrace  = 0;    // trace buffer size per thread (0 means default)
ایستا صحیح opt_oppoll   = 0;    // polling for batch completions
ایستا صحیح opt_opfusion  = 1;    // enable/disable op fusion
ایستا صحیح opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

شمارش ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_ADD = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_ADD   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

ایستا درونخطی منطقی ggml_hexagon_is_fusion_enabled(صحیح flag) {
    اگر (opt_opfusion <= 0) بازگشت نادرست;
    اگر (opt_opfusion == 1) بازگشت درست; // 1 enables all
    بازگشت (opt_opfusion & flag) != 0;
}

ایستا std::regex* opt_opfilter = پوچ; // regex of ops to not claim

#define HEX_VERBOSE(...) \
    اگر (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

ایستا ثابت نویسه * status_to_str(uint32_t status) {
    انتخاب (status) {
        حالت HTP_STATUS_OK:
            بازگشت "OK";
        حالت HTP_STATUS_NO_SUPPORT:
            بازگشت "NO-SUPPORT";
        حالت HTP_STATUS_INVAL_PARAMS:
            بازگشت "INVAL-PARAMS";
        حالت HTP_STATUS_VTCM_TOO_SMALL:
            بازگشت "VTCM-TOO-SMALL";
        حالت HTP_STATUS_INTERNAL_ERR:
            بازگشت "INTERNAL-ERROR";
        پیشفرض:
            بازگشت "UNKNOWN";
    }
}

// ** debug helpers

ایستا تهی ggml_hexagon_dump_op_exec(ثابت std::string &sess_name, ثابت htp_opnode & گره, ثابت uint32_t req_flags) {
    اگر (!opt_verbose) بازگشت;

    htp_opformat fmt(گره);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                گره.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

ایستا تهی ggml_hexagon_dump_op_supp(ثابت std::string &sess_name, ثابت ساختار ggml_tensor * op, منطقی supp) {
    اگر (!opt_verbose) بازگشت;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

ایستا ثابت نویسه * htp_event_name(uint16_t id) {
    انتخاب (id) {
        حالت HTP_TRACE_EVT_DMA:            بازگشت "DMA";
        حالت HTP_TRACE_EVT_HVX_COMP:       بازگشت "HVX_COMP";
        حالت HTP_TRACE_EVT_HVX_A_QUANT:    بازگشت "HVX_A_QUANT";
        حالت HTP_TRACE_EVT_HVX_A_PREP:     بازگشت "HVX_A_PREP";
        حالت HTP_TRACE_EVT_HVX_W_DEQUANT:  بازگشت "HVX_W_DEQUANT";
        حالت HTP_TRACE_EVT_HVX_W_PREP:     بازگشت "HVX_W_PREP";
        حالت HTP_TRACE_EVT_HVX_O_PROC:     بازگشت "HVX_O_PROC";
        حالت HTP_TRACE_EVT_HVX_FA_QK:      بازگشت "HVX_QK_FA";
        حالت HTP_TRACE_EVT_HVX_FA_SFM:     بازگشت "HVX_SFM_FA";
        حالت HTP_TRACE_EVT_HVX_FA_Q_PREP:  بازگشت "HVX_Q_PREP";
        حالت HTP_TRACE_EVT_HVX_FA_K_PREP:  بازگشت "HVX_K_PREP";
        حالت HTP_TRACE_EVT_HVX_FA_V_PREP:  بازگشت "HVX_V_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_PREP:   بازگشت "HVX_GDN_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_SOLVE:  بازگشت "HVX_GDN_SOLVE";
        حالت HTP_TRACE_EVT_HVX_GDN_V_PREP: بازگشت "HVX_GDN_V_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_D_PREP: بازگشت "HVX_GDN_D_PREP";
        حالت HTP_TRACE_EVT_HVX_GDN_OUT:    بازگشت "HVX_GDN_OUT";
        حالت HTP_TRACE_EVT_HVX_GDN_STATE:  بازگشت "HVX_GDN_STATE";
        حالت HTP_TRACE_EVT_HVX_GDN_REM:    بازگشت "HVX_GDN_REM";
        حالت HTP_TRACE_EVT_HMX_COMP:       بازگشت "HMX_COMP";
        حالت HTP_TRACE_EVT_L2FLUSH:        بازگشت "L2FLUSH";
        حالت HTP_TRACE_EVT_INIT:           بازگشت "INIT";
        حالت HTP_TRACE_EVT_BUFF:           بازگشت "BUFF";
        حالت HTP_TRACE_EVT_FENCE:          بازگشت "FENCE";
        پیشفرض:                           بازگشت "UNKNOWN";
    }
}

ایستا تهی ggml_hexagon_dump_op_prof(ثابت std::string &sess_name, ثابت htp_opnode & گره, ثابت htp_prof_desc & pd) {
    اگر (!opt_profile) بازگشت;

    uint32_t op_usec = pd.usecs;
    uint32_t op_cycles = pd.cycles_stop - pd.cycles_start;
    ثابت uint32_t * pmu = pd.pmu;

    نویسه pmu_str[256] = "";
    اگر (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, اندازه(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(گره);
    اعشاری mhz = op_usec > 0 ? (اعشاری) op_cycles / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            گره.op_name().c_str(), fmt.names, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_cycles, pd.cycles_start, mhz, pmu_str);
}

ایستا تهی ggml_hexagon_dump_batch_prof(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp) {
    uint64_t batch_cycles = rsp.cycles_stop - rsp.cycles_start;
    اعشاری batch_mhz = rsp.usecs > 0 ? (اعشاری) batch_cycles / rsp.usecs : 0.0f;

    نویسه evt_str[256] = "----";
    اگر (opt_profile == 3) {
        snprintf(evt_str, اندازه(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (unsigned long long) batch_cycles, (unsigned long long) rsp.cycles_start, batch_mhz);
}

ایستا تهی ggml_hexagon_dump_trace_events(ثابت std::string & sess_name, ثابت htp_opbatch_rsp & rsp,
                                           ثابت htp_trace_desc * trace_events, uint32_t n_traces) {
    اگر (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        برای (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        برای (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            برای (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                ثابت auto & e = trace_events[t * n_traces + idx];
                منطقی is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.cycles);
            }
        }
    }
}

شمارش ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_WEIGHT    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

ساختار ggml_hexagon_session;

ایستا تهی ggml_hexagon_precompute_matmul_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_mm_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_fused_matmul_add_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * src2,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_mm_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_unary_params(
    ثابت ساختار ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_unary_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_precompute_دودویی_params(
    ثابت ساختار ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_دودویی_kernel_params * kparams,
    ثابت ساختار ggml_tensor * src2 = nullptr
);

ایستا تهی ggml_hexagon_precompute_get_rows_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_get_rows_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_set_rows_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_set_rows_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_softmax_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_softmax_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_rope_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_rope_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_ssm_conv_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_ssm_conv_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_gated_delta_net_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_gdn_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_sort_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    منطقی is_بالاترین_k,
    ساختار htp_sort_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_pool_2d_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_pool_2d_kernel_params * kparams,
    منطقی is_pool_1d
);

ایستا منطقی ggml_hexagon_precompute_concat_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_concat_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_precompute_cpy_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_copy_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_precompute_im2col_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_im2col_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_precompute_im2col_3d_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_im2col_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_fused_mmnx_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * act,
    int32_t n_weights,
    ساختار htp_mm_kernel_params * kparams
);

ایستا تهی ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * act,
    ثابت ساختار ggml_tensor * dst,
    int32_t n_weights,
    ساختار htp_mm_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_precompute_allreduce_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقی has_add,
    منطقی is_row_bcast,
    منطقی is_shard_ok,
    ساختار htp_allreduce_kernel_params * kparams
);

ایستا منطقی ggml_hexagon_rows_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride);
ایستا منطقی ggml_hexagon_matmul_can_collapse(ثابت ساختار ggml_tensor * src0, ثابت ساختار ggml_tensor * src1, ثابت ساختار ggml_tensor * dst);
ایستا ggml_tensor ggml_hexagon_tensor_collapse_rows(ثابت ساختار ggml_tensor * t);

ایستا منطقی mm_is_hmx_eligible(ثابت ggml_tensor * t);
ایستا htp_op_code op_remap_to_htp(ثابت ggml_tensor * t);
ایستا منطقی is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت ساختار htp_mm_kernel_params * kparams);
ایستا منطقی is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت ساختار htp_mm_kernel_params * kparams);
ایستا منطقی is_mergeable_mul_mat(ثابت ggml_tensor * t);
ایستا منطقی is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);
ایستا منطقی is_mergeable_mul_mat_id(ثابت ggml_tensor * t);
ایستا منطقی is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2);

// ** backend sessions

ساختار ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_size { 0 };
    uint32_t             flags { 0 };
};

ایستا درونخطی منطقی ggml_hexagon_tensor_is_fuseable(ثابت ساختار ggml_tensor * t) {
    اگر (!t->extra) بازگشت نادرست;
    auto extra = (ثابت ساختار ggml_hexagon_tensor_extra *) t->extra;
    بازگشت (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

ایستا درونخطی منطقی ggml_hexagon_can_rt_repack(ثابت ساختار ggml_tensor * t) {
    اگر (opt_rt_repack == 0 || !htp_is_tiled_weight_tensor(t)) {
        بازگشت نادرست;
    }
    اگر (t->buffer && ggml_backend_buft_is_host(t->buffer->buft)) {
        بازگشت نادرست;
    }
    بازگشت opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

ایستا درونخطی منطقی ggml_hexagon_tensor_is_tiled(ثابت ساختار ggml_tensor * t) {
    ثابت auto * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    اگر (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        بازگشت درست;
    }
    بازگشت ggml_hexagon_can_rt_repack(t);
}

ایستا درونخطی منطقی ggml_hexagon_needs_repack(ثابت ساختار ggml_tensor * t) {
    ثابت auto * extra = (ثابت ggml_hexagon_tensor_extra *) t->extra;
    اگر (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        بازگشت نادرست;
    }
    بازگشت ggml_hexagon_can_rt_repack(t);
}

ایستا درونخطی منطقی ggml_hexagon_tensors_overlap(ثابت ساختار ggml_tensor * a, ثابت ساختار ggml_tensor * b) {
    ثابت uintptr_t a0 = (uintptr_t) a->data;
    ثابت uintptr_t b0 = (uintptr_t) b->data;
    ثابت uintptr_t a1 = a0 + ggml_nbytes(a);
    ثابت uintptr_t b1 = b0 + ggml_nbytes(b);

    بازگشت a0 < b1 && b0 < a1;
}

ایستا درونخطی منطقی ggml_hexagon_can_row_partition(ثابت ساختار ggml_tensor * t) {
    اگر (t->ne[1] > 1 && (t->nb[1] & 127) != 0) بازگشت نادرست;
    اگر (t->ne[2] > 1 && (t->nb[2] & 127) != 0) بازگشت نادرست;
    اگر (t->ne[3] > 1 && (t->nb[3] & 127) != 0) بازگشت نادرست;
    بازگشت درست;
}

ساختار htp_opnode;

ساختار ggml_hexagon_opbatch;
ساختار ggml_hexagon_opqueue;
ساختار ggml_hexagon_shared_buffer;
ساختار ggml_hexagon_fence_buffer;
ساختار ggml_hexagon_session;
ساختار ggml_backend_hexagon_device_context;

ساختار ggml_hexagon_mdev_group {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

ساختار ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    فرار uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

ساختار ggml_hexagon_event {
    ggml_hexagon_session * sess         = nullptr;
    ggml_hexagon_session * fence_sess   = nullptr;
    فرار uint32_t *    fence_slot   = nullptr;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

ساختار ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         queue_id;
    صحیح              phys_idx;
    صحیح              virt_idx;
    منطقی             valid_session;
    منطقی             valid_handle;
    منطقی             valid_queue;
    منطقی             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_queue;

    std::unordered_map<صحیح, std::unique_ptr<ggml_hexagon_shared_buffer>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_threads   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_size   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> last_error{HTP_STATUS_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_group                mdev;
    ggml_backend_dev_t                     dev       = nullptr;
    ggml_backend_hexagon_device_context *  dev_ctx   = nullptr;
    ggml_hexagon_fence_buffer *            fence_buf = nullptr;

    ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev = nullptr, uint32_t mdev_idx = 0, uint32_t mdev_شمار = 0) noexcept(نادرست);
    ~ggml_hexagon_session() noexcept(درست);

    ثابت نویسه* c_name() ثابت { بازگشت name.c_str(); }

    تهی allocate(ثابت ggml_hexagon_device_config & config) noexcept(نادرست);
    تهی release() noexcept(درست);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    تهی      free_fence(تهی * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_slot = nullptr;
    std::unordered_map<uint64_t, فرار uint32_t *> cpy_fence_slots;

    تهی enqueue_mdev_group();
    تهی enqueue_op(ثابت htp_opnode & گره);
    تهی enqueue_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor = nullptr, uint32_t fence_seq = 0);
    تهی enqueue_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq = 0, منطقی wait = درست);
    تهی enqueue_allreduce(ثابت ggml_tensor * dst, ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
                           ثابت std::vector<ثابت ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_entry = 0, uint32_t fence_seq_exit = 0);

    تهی start_batch();
    تهی flush_sync(منطقی all = درست);
    تهی flush_async();
    تهی flush_batch(size_t min_ops = 1);
    تهی flush_peers();
    تهی flush_pending(منطقی all = درست);

    ggml_hexagon_shared_buffer * mmap_tensor(ثابت ggml_tensor * t);
    منطقی clone_buffer(ثابت ggml_hexagon_shared_buffer*);
    تهی release_buffer(ثابت ggml_hexagon_shared_buffer*);
    تهی unclone_buffer(ثابت ggml_hexagon_shared_buffer*);

    تهی add_peer(ggml_hexagon_session * peer) {
        اگر (این->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } وگرنه {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

ساختار ggml_backend_hexagon_device_context {
    صحیح                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = nullptr;

    ggml_backend_buffer_type buffer_type       = {};
    ggml_backend_buffer_type host_buffer_type  = {};
    ggml_backend_buffer_type fence_buffer_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(صحیح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    ثابت نویسه * c_name() ثابت { بازگشت config.name.c_str(); }

    ggml_hexagon_session * session() {
        اگر (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        بازگشت sess.get();
    }
};

ساختار ggml_backend_hexagon_buffer_type_context {
    ggml_backend_hexagon_buffer_type_context(ثابت std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        این->dev_ctx = dev_ctx;
        این->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

ساختار ggml_hexagon_rpcmem_block {
    uint8_t * base = nullptr;
    صحیح       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_block(size_t size) {
        base = (uint8_t *) rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        اگر (!base) {
            پرتابکن std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_to_fd(base);
        اگر (fd < 0) {
            rpcmem_free(base);
            پرتابکن std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        این->size = size;
    }

    ~ggml_hexagon_rpcmem_block() {
        اگر (base) {
            rpcmem_free(base);
        }
    }
};

ساختار ggml_hexagon_shared_buffer {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_block> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    منطقی     mapped;
    منطقی     pinned;
    منطقی     extended;

    ثابت نویسه * c_name() ثابت { بازگشت sess->c_name(); }
    uint8_t *    base()   ثابت { بازگشت mem ? mem->base : nullptr; }
    size_t       size()   ثابت { بازگشت mem ? mem->size : 0;  }
    صحیح          fd()     ثابت { بازگشت mem ? mem->fd   : -1; }

    تهی mmap(منطقی extended = نادرست) {
        اگر (!این->mem)   بازگشت;
        اگر (این->mapped) بازگشت;

        GGML_ASSERT(!این->pinned || !extended);

        این->extended = extended;
        fastrpc_map_flags flags = این->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        صحیح err = fastrpc_mmap(sess->domain_id, fd(), (تهی *) base(), 0, size(), flags);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (unsigned) err);
            پرتابکن std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (تهی *) base(), size(), fd(), pinned, extended);

        این->mapped = درست;
    }

    تهی unmap() {
        اگر (!این->mapped) بازگشت;

        اگر (!این->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        اگر (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (تهی *) base(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (تهی *) base(), size(), fd());

        این->mapped = نادرست;
    }

    تهی alloc(size_t size) {
        اگر (این->mem) بازگشت;

        این->mem = std::make_shared<ggml_hexagon_rpcmem_block>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (تهی *) base(), این->size(), fd(), (صحیح) pinned);
        اگر (این->pinned) {
            mmap();
        }
    }

    تهی آزاد() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (تهی *) base(), size(), fd());
        این->mem  = nullptr;
    }

    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, size_t size, منطقی pinned = نادرست) {
        این->sess     = sess;
        این->mapped   = نادرست;
        این->pinned   = pinned;
        این->extended = نادرست;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_offset = (size + 4095) & ~4095;
        size_t total_size   = guard_offset + 4096;
        اگر (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            total_size = (total_size + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(total_size);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_buffer(ggml_hexagon_session * sess, ثابت ggml_hexagon_shared_buffer & other) {
        این->sess   = sess;
        این->mem    = other.mem;
        این->mapped = نادرست;
        این->pinned = other.pinned;
        این->extended = other.extended;
    }

    ~ggml_hexagon_shared_buffer() {
        آزاد();
        برای (auto * extra : tensor_extra) {
            delete extra;
        }
    }
};

ساختار ggml_hexagon_fence_buffer : عمومی ggml_hexagon_shared_buffer {
    uint32_t              slot_شمار = 0;
    uint32_t              slot_head  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_buffer   backend_buffer{};

    ggml_hexagon_fence_buffer(ggml_hexagon_session * sess, ggml_backend_buffer_type_t buft, size_t size)
        : ggml_hexagon_shared_buffer(sess, size, نادرست /* pinned */),
          slot_شمار(size / GGML_HEXAGON_FENCE_SLOT_SIZE),
          slot_head(0) {
        backend_buffer.buft    = buft;
        backend_buffer.context = static_cast<ggml_hexagon_shared_buffer *>(این);
        backend_buffer.size    = size;
        mmap(نادرست);
    }

    uint8_t * alloc_slot(uint32_t n_slots = 1) {
        uint8_t * ptr = nullptr;
        اگر (n_slots == 1 && !free_slots.empty()) {
            uint32_t slot = free_slots.back();
            free_slots.pop_back();
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        } وگرنه اگر (slot_head + n_slots <= slot_شمار) {
            uint32_t slot = slot_head;
            slot_head += n_slots;
            ptr = base() + (size_t) slot * GGML_HEXAGON_FENCE_SLOT_SIZE;
        }
        اگر (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_SLOT_SIZE);
        }
        بازگشت ptr;
    }

    تهی free_slot(تهی * ptr, uint32_t n_slots = 1) {
        اگر (!ptr) بازگشت;
        uint32_t slot = ((uint8_t *) ptr - base()) / GGML_HEXAGON_FENCE_SLOT_SIZE;
        برای (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(slot + i);
        }
    }
};

درونخطی uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_slot(n_slots);
    GGML_ASSERT(ptr);
    بازگشت ptr;
}

درونخطی تهی ggml_hexagon_session::free_fence(تهی * ptr, uint32_t n_slots) {
    اگر (fence_buf) {
        fence_buf->free_slot(ptr, n_slots);
    }
}

ایستا تهی ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    sbuf->sess->unclone_buffer(sbuf);
    delete sbuf;
}

ایستا تهی * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    بازگشت sbuf->base();
}

ایستا شمارش ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(buffer->context);
    auto sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (تهی *) sbuf->base(), tensor->data, ggml_nbytes(tensor));

    auto extra = جدید ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                   ggml_tensor *         tensor,
                                                   ثابت تهی *          data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    اگر (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        اگر (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, data, offset, size, (صحیح) buffer->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy((نویسه *) tensor->data + offset, data, size);
        بازگشت;
    }

    اگر (offset == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.empty()) {
        htp_repack_tensor_tiled(tensor, data, size);
        بازگشت;
    }

    اگر (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    memcpy(extra->shadow_buf.data() + offset, data, size);
    extra->shadow_size += size;

    اگر (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ایستا تهی ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                   ثابت ggml_tensor *   tensor,
                                                   تهی *                data,
                                                   size_t                offset,
                                                   size_t                size) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, data, offset, size, (صحیح) buffer->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        memcpy(data, (ثابت نویسه *) tensor->data + offset, size);
        بازگشت;
    }

    htp_unpack_tensor_tiled(data, tensor, offset, size);
}

ایستا منطقی ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t      buffer,
                                                   ثابت ساختار ggml_tensor * src,
                                                   ساختار ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    بازگشت نادرست;

    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
}

ایستا تهی ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer,
                                                          ggml_tensor *         tensor,
                                                          ثابت تهی *          data,
                                                          size_t                offset,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    اگر (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_WEIGHT;
        اگر (htp_is_tiled_weight_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (صحیح) buffer->usage, extra->flags);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        برای (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *) tensor->data + offset + i * stride_tensor, (ثابت uint8_t *) data + i * stride_data, size);
        }
        بازگشت;
    }

    اگر (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    برای (size_t i = 0; i < n_copies; i++) {
        memcpy(extra->shadow_buf.data() + offset + i * stride_tensor, (ثابت uint8_t *) data + i * stride_data, size);
    }
    extra->shadow_size += n_copies * size;

    اگر (extra->shadow_size >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.data(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_size = 0;
    }
}

ایستا تهی ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer,
                                                      ثابت ggml_tensor *   tensor,
                                                      تهی *                data,
                                                      size_t                offset,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_data) {
    auto extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    auto sbuf  = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, (صحیح) buffer->usage);

    اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        برای (size_t i = 0; i < n_copies; i++) {
            memcpy((uint8_t *)data + i * stride_data, (ثابت uint8_t *)tensor->data + offset + i * stride_tensor, size);
        }
        بازگشت;
    }

    size_t temp_size      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t slice_size     = tensor->ne[1] * ggml_row_size(tensor->type, tensor->ne[0]);
    size_t slice_offset   = offset % slice_size;
    size_t row_size_bytes = ggml_row_size(tensor->type, tensor->ne[0]);

    GGML_ASSERT((slice_offset % row_size_bytes) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((temp_size % row_size_bytes)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((offset + temp_size) <= ggml_nbytes(tensor));

    std::vector<uint8_t> temp_buf(temp_size);
    htp_unpack_tensor_tiled(temp_buf.data(), tensor, offset, temp_size);

    برای (size_t i = 0; i < n_copies; i++) {
        memcpy((uint8_t *) data + i * stride_data, temp_buf.data() + i * stride_tensor, size);
    }
}

ایستا تهی ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto sbuf = (ggml_hexagon_shared_buffer *) buffer->context;
    auto sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (تهی *) sbuf->base(), sbuf->size());
    memset(sbuf->base(), value, sbuf->size());
}

ایستا ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ پوچ,
    /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ پوچ,
};

// ** backend buffer type

ایستا تهی ggml_backend_hexagon_host_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                        ggml_tensor *         tensor,
                                                        ثابت تهی *          data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy((نویسه *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

ایستا تهی ggml_backend_hexagon_host_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                        ثابت ggml_tensor *   tensor,
                                                        تهی *                data,
                                                        size_t                offset,
                                                        size_t                size) {
    memcpy(data, (ثابت نویسه *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

ایستا ggml_backend_buffer_i ggml_backend_hexagon_host_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
    /* .memset_tensor   = */ پوچ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_buffer_get_tensor,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_buffer_clear,
    /* .reset           = */ پوچ,
};

// ** backend buffer type

ایستا ثابت نویسه * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buffer_type) {
    بازگشت static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->name.c_str();
}

ایستا ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    اگر (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        بازگشت nullptr;
    }
    تلاش {
        ggml_hexagon_shared_buffer * sbuf = جدید ggml_hexagon_shared_buffer(sess, size, نادرست);
        بازگشت ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_buffer_interface, sbuf, size);
    } بگیر (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        بازگشت nullptr;
    }
}

ایستا ggml_backend_buffer_t ggml_backend_hexagon_host_buffer_type_alloc_buffer(
            ggml_backend_buffer_type_t buffer_type, size_t size) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type->context)->dev_ctx;
    auto sess    = dev_ctx->session();
    اگر (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        بازگشت nullptr;
    }
    تلاش {
        ggml_hexagon_shared_buffer * sbuf = جدید ggml_hexagon_shared_buffer(sess, size, نادرست);
        بازگشت ggml_backend_buffer_init(buffer_type, ggml_backend_hexagon_host_buffer_interface, sbuf, size);
    } بگیر (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        بازگشت nullptr;
    }
}

ایستا size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    بازگشت 128;  // HVX alignment
    GGML_UNUSED(buft);
}

ایستا size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ثابت ساختار ggml_tensor * t) {
    اگر (!ggml_backend_buft_is_host(buft) && htp_is_tiled_weight_tensor(t)) {
        int64_t ne0 = hex_round_up(t->ne[0], 32);
        int64_t ne1 = hex_round_up(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_size = htp_tiled_row_size(t->type, ne0) * ne1 * ne2 * ne3;
        بازگشت std::max(tiled_size, ggml_nbytes(t));
    }
    بازگشت ggml_nbytes(t);
}

ایستا size_t parse_size(ثابت نویسه * str, size_t default_unit = 1024 * 1024) {
    اگر (!str || str[0] == '\0') {
        بازگشت 0;
    }
    نویسه * end = پوچ;
    دوگانه val = strtod(str, &end);
    اگر (val < 0) {
        بازگشت 0;
    }
    اگر (end && *end) {
        وقتی (*end == ' ') end++;
        اگر (*end == 'k' || *end == 'K') {
            بازگشت (size_t) (val * 1024);
        } وگرنه اگر (*end == 'm' || *end == 'M') {
            بازگشت (size_t) (val * 1024 * 1024);
        } وگرنه اگر (*end == 'g' || *end == 'G') {
            بازگشت (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    بازگشت (size_t) (val * default_unit);
}

ایستا size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    بازگشت opt_mbuf_dyn;
    GGML_UNUSED(buft);
}

ایستا منطقی ggml_backend_hexagon_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت نادرست;
    GGML_UNUSED(buft);
}

ایستا منطقی ggml_backend_hexagon_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    بازگشت درست;
    GGML_UNUSED(buft);
}

ساختار ggml_backend_hexagon_alloc_buffer_n_plan_item {
    size_t size;
    صحیح    first;
    صحیح    last;
};

using ggml_backend_hexagon_alloc_buffer_n_plan_t = std::vector<ggml_backend_hexagon_alloc_buffer_n_plan_item>;

ایستا ثابت نویسه * ggml_hexagon_kv_layer_suffix(ثابت ساختار ggml_tensor * t) {
    اگر (strncmp(t->name, "cache_", 6) != 0) {
        بازگشت پوچ;
    }
    ثابت نویسه * p = strstr(t->name, "_l");
    اگر (!p || !isdigit((unsigned نویسه)p[2])) {
        بازگشت پوچ;
    }
    بازگشت p;
}

ساختار ggml_backend_hexagon_alloc_unit {
    size_t size;
    صحیح    first;
    صحیح    last;
};

ایستا ggml_backend_hexagon_alloc_buffer_n_plan_t ggml_backend_hexagon_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ggml_backend_hexagon_alloc_buffer_n_plan_t plan;

    ثابت size_t alignment = ggml_backend_buft_get_alignment(buft);
    ثابت size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    صحیح i = 0;
    وقتی (i < n_tensors) {
        ساختار ggml_tensor * t = tensors[i];
        size_t unit_size = 0;
        صحیح unit_first = i;
        صحیح unit_last = i + 1;

        اگر (t->data == پوچ && t->view_src == پوچ) {
            unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        ثابت نویسه * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        وقتی (unit_last < n_tensors) {
            ساختار ggml_tensor * next = tensors[unit_last];

            اگر (next->view_src != پوچ) {
                unit_last++;
                ادامه;
            }

            اگر (layer_suffix != پوچ) {
                ثابت نویسه * next_suffix = ggml_hexagon_kv_layer_suffix(next);
                اگر (next_suffix != پوچ && strcmp(layer_suffix, next_suffix) == 0) {
                    اگر (next->data == پوچ) {
                        unit_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, next), alignment);
                    }
                    unit_last++;
                    ادامه;
                }
            }

            بشکن;
        }

        units.push_back({ unit_size, unit_first, unit_last });
        i = unit_last;
    }

    size_t cur_buf_size  = 0;
    صحیح    cur_buf_first = 0;

    برای (ثابت auto & unit : units) {
        اگر (unit.size == 0) {
            ادامه;
        }

        اگر (cur_buf_size > 0 && (cur_buf_size + unit.size) > max_size) {
            plan.push_back({ cur_buf_size, cur_buf_first, unit.first });
            cur_buf_size  = 0;
            cur_buf_first = unit.first;
        }

        cur_buf_size += unit.size;
    }

    اگر (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, cur_buf_first, n_tensors });
    }

    بازگشت plan;
}

ایستا ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer_n(
        ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    برای (ثابت auto & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        اگر (buffer == پوچ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            برای (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            بازگشت پوچ;
        }

        ساختار ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        ساختار ggml_tensor * t_failed = پوچ;
        برای (صحیح j = item.first; j < item.last; j++) {
            ساختار ggml_tensor * t = tensors[j];
            اگر (t->data == پوچ) {
                اگر (t->view_src == پوچ) {
                    اگر (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                } وگرنه اگر (t->buffer == پوچ) {
                    اگر (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                }
            } وگرنه {
                اگر (t->view_src != پوچ && t->buffer == پوچ) {
                    اگر (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        بشکن;
                    }
                }
            }
        }
        اگر (t_failed != پوچ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            برای (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            بازگشت پوچ;
        }

        buffers.push_back(buffer);
    }

    اگر (buffers.empty()) {
        بازگشت پوچ;
    }

    اگر (buffers.size() == 1) {
        بازگشت buffers[0];
    }

    بازگشت ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

ایستا size_t ggml_backend_hexagon_buffer_type_get_alloc_size_n(
        ggml_backend_buffer_type_t buft, ساختار ggml_tensor ** tensors, صحیح n_tensors) {
    ثابت ggml_backend_hexagon_alloc_buffer_n_plan_t plan = ggml_backend_hexagon_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    برای (ثابت auto & item : plan) {
        total += item.size;
    }
    بازگشت total;
}

ایستا ggml_backend_buffer_type_i ggml_backend_hexagon_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_buffer_type_is_host,
};

ایستا ggml_backend_buffer_type_i ggml_backend_hexagon_host_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_hexagon_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_hexagon_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_buffer_type_get_alloc_size_n,
    /* .is_host             = */ ggml_backend_hexagon_host_buffer_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(صحیح dev_id, ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    buffer_type.device  = dev;
    buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    buffer_type.context = جدید ggml_backend_hexagon_buffer_type_context(config.name, این);

    host_buffer_type.device  = dev;
    host_buffer_type.iface   = ggml_backend_hexagon_host_buffer_type_interface;
    host_buffer_type.context = جدید ggml_backend_hexagon_buffer_type_context(config.name + "-HOST", این);

    fence_buffer_type.device  = dev;
    fence_buffer_type.iface   = ggml_backend_hexagon_buffer_type_interface;
    fence_buffer_type.context = جدید ggml_backend_hexagon_buffer_type_context(config.name + "-FENCE", این);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(host_buffer_type.context);
    delete static_cast<ggml_backend_hexagon_buffer_type_context *>(fence_buffer_type.context);
}

ایستا منطقی ggml_backend_buffer_is_hexagon(ثابت ساختار ggml_backend_buffer * b) {
    بازگشت b->buft->iface.get_alignment == ggml_backend_hexagon_buffer_type_get_alignment;
}

ساختار ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<صحیح, صحیح>                b_map; // buffer fd   to index
    std::unordered_map<ثابت ggml_tensor*, صحیح> t_map; // tensor ptr  to index
    std::unordered_multimap<تهی*, صحیح>         d_map; // tensor data to index

    unsigned صحیح n_bufs;     // num buffers in the batch
    unsigned صحیح n_tens;     // num tensors ...
    unsigned صحیح n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    unsigned صحیح n_bufs_max;
    unsigned صحیح n_tens_max;
    unsigned صحیح n_ops_max;
    size_t       b_vmem_max;

    تهی reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_size, size_t max_vmem) {
        این->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_size;
        n_tens_max = std::min<size_t>(n_ops_max + n_ops_max * HTP_OP_MAX_INPUTS, HTP_OP_MAX_TENSORS);

        b_vmem_max = max_vmem;

        ops.resize(n_ops_max);

        h_bufs.resize(n_bufs_max);
        h_tens.resize(n_tens_max);
        h_ops.resize(n_ops_max);

        b_map.reserve(n_bufs_max);
        t_map.reserve(n_tens_max);
        d_map.reserve(n_tens_max);

        GGML_LOG_INFO("ggml-hex: %s op batching: n-bufs %u n-tensors %u n-ops %u vmem %zu\n",
                sess->c_name(), n_bufs_max, n_tens_max, n_ops_max, b_vmem_max);

        reset();
    }

    منطقی empty() ثابت { بازگشت n_ops == 0; }

    // add buffer and return its index
    صحیح add_buffer(ggml_hexagon_shared_buffer * sbuf) {
        // Lookup by fd
        auto it = b_map.find(sbuf->fd());
        اگر (it != b_map.end()) { بازگشت it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        صحیح bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.base  = (uint64_t) sbuf->base();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        اگر (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (تهی*) sbuf->base(), (size_t) b.size, b_vmem);

        بازگشت bi;
    }

    منطقی same_shape(ثابت htp_tensor * h, ثابت ggml_tensor * t) ثابت {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;

        ثابت منطقی is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ثابت int64_t ne0_padded = hex_round_up(t->ne[0], 32);
        ثابت int64_t ne1_padded = hex_round_up(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_row_size(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? اندازه(ggml_fp16_t) : t->nb[0];
        شمارش ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        بازگشت (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    صحیح add_tensor(ثابت ggml_tensor * t) {
        auto extra = (ggml_hexagon_tensor_extra *) t->extra;
        auto sbuf  = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);

        // First lookup by tensor data
        auto range = d_map.equal_range(t->data);
        برای (auto it = range.first; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            اگر (same_shape(h, t)) { بازگشت it->second; }
        }

        // Lookup by tensor ptr
        auto it = t_map.find(t);
        اگر (it != t_map.end()) { بازگشت it->second; }

        // Add new tensor to the batch
        صحیح ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->data, ti});

        uint64_t t_offset = (uint8_t *) t->data - sbuf->base();
        size_t   t_size   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = add_buffer(sbuf);
        h.ti    = ti;
        h.data  = t_offset;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        ثابت منطقی is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        اگر (is_tiled) {
            اگر (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = اندازه(ggml_fp16_t);
            } وگرنه {
                h.nb[0] = t->nb[0];
            }
            ثابت int64_t ne0_padded = hex_round_up(t->ne[0], 32);
            ثابت int64_t ne1_padded = hex_round_up(t->ne[1], 32);
            h.nb[1] = htp_tiled_row_size(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_size  = h.size;
        } وگرنه {
            h.size  = t_size;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_WEIGHT) != 0) {
            h.flags |= HTP_TENSOR_WEIGHT;
        }
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        اگر ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (تهی*) t->data, (size_t) t_offset, t_size, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        بازگشت ti;
    }

    منطقی fit_op(ثابت htp_opnode & گره) ثابت {
        اگر (n_ops >= n_ops_max) بازگشت نادرست;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        صحیح seen_bufs[HTP_OP_MAX_BUFS];
        صحیح n_seen_bufs = 0;

        auto fit_tensor = [&](ثابت ggml_tensor *t) {
            اگر (!t) بازگشت;
            اگر (!t_map.count(t)) {
                extra_tens++;

                auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
                صحیح fd = sbuf->fd();
                اگر (!b_map.count(fd)) {
                    برای (صحیح i = 0; i < n_seen_bufs; i++) {
                        اگر (seen_bufs[i] == fd) بازگشت;
                    }
                    اگر (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    اگر (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        برای (ثابت auto * src : گره.get_inputs()) {
            fit_tensor(src);
        }
        برای (ثابت auto * output : گره.get_outputs()) {
            fit_tensor(output);
        }

        اگر ((extra_bufs + n_bufs) > n_bufs_max) بازگشت نادرست;
        اگر ((extra_tens + n_tens) > n_tens_max) بازگشت نادرست;
        اگر ((extra_vmem + b_vmem) > b_vmem_max) بازگشت نادرست;

        بازگشت درست;
    }

    // assumes that fit_op() was called first and returned true
    تهی add_op(ثابت htp_opnode & گره) {
        // Add new op

        unsigned صحیح n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = گره;

        htp_op_desc &o = h_ops[n];
        memcpy(o.params,        گره.گره->op_params, اندازه(گره.گره->op_params));
        memcpy(o.kernel_params, گره.kernel_params,   اندازه(o.kernel_params));
        o.opcode = گره.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        auto inputs = گره.get_inputs();
        برای (unsigned صحیح i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? add_tensor(inputs[i]) : 0xffff;
        }

        auto outputs = گره.get_outputs();
        برای (unsigned صحیح i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? add_tensor(outputs[i]) : 0xffff;
        }
    }

    تهی sort_buffers() {
        اگر (n_bufs <= 1) بازگشت;

        std::vector<صحیح> order(n_bufs);
        برای (unsigned صحیح i = 0; i < n_bufs; i++) { order[i] = (صحیح) i; }

        std::stable_sort(order.begin(), order.end(), [&](صحیح a, صحیح b) {
            بازگشت h_bufs[a].size > h_bufs[b].size;
        });

        منطقی already_sorted = درست;
        برای (unsigned صحیح i = 0; i < n_bufs; i++) {
            اگر (order[i] != (صحیح) i) {
                already_sorted = نادرست;
                بشکن;
            }
        }
        اگر (already_sorted) بازگشت;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        برای (unsigned صحیح new_bi = 0; new_bi < n_bufs; new_bi++) {
            صحیح old_bi = order[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        برای (unsigned صحیح i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        برای (unsigned صحیح i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    تهی update_mdev_group(uint32_t mdev_idx) {
        اگر (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_GROUP) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    منطقی try_fuse_common(std::initializer_list<ثابت ggml_tensor *> tensors) ثابت {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        صحیح seen_bufs[HTP_OP_MAX_BUFS];
        صحیح n_seen_bufs = 0;

        برای (ثابت auto * t : tensors) {
            اگر (!t || t_map.count(t)) {
                ادامه;
            }
            extra_tens++;
            auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
            صحیح fd = sbuf->fd();
            اگر (!b_map.count(fd)) {
                منطقی found = نادرست;
                برای (صحیح i = 0; i < n_seen_bufs; i++) {
                    اگر (seen_bufs[i] == fd) {
                        found = درست;
                        بشکن;
                    }
                }
                اگر (!found) {
                    اگر (n_seen_bufs < HTP_OP_MAX_BUFS) {
                        seen_bufs[n_seen_bufs++] = fd;
                    }
                    اگر (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        اگر ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    منطقی try_fuse_common(ثابت ggml_tensor * t1, ثابت ggml_tensor * t2) ثابت {
        بازگشت try_fuse_common({t1, t2});
    }

    منطقی try_fuse_allreduce_add(ثابت htp_opnode & گره) {
        اگر (n_ops == 0 || opt_ar_select != 2) بازگشت نادرست;
        اگر (گره.opcode != HTP_OP_ADD) بازگشت نادرست;

        htp_opnode & last_گره = ops[n_ops - 1];
        اگر (last_گره.opcode != HTP_OP_ALLREDUCE) بازگشت نادرست;

        auto * ar_kparams = (ساختار htp_allreduce_kernel_params *) last_گره.kernel_params;
        ثابت uint32_t rank    = (uint32_t) ar_kparams->rank;
        ثابت uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        ثابت ggml_tensor * ar_local = last_گره.inputs[rank];
        ثابت ggml_tensor * add_src0 = گره.src0();
        ثابت ggml_tensor * add_src1 = گره.src1();
        ثابت ggml_tensor * add_dst  = گره.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(ar_local)) بازگشت نادرست;

        ثابت ggml_tensor * res_tensor;
        اگر (add_src0 == ar_local || add_src0->data == ar_local->data) {
            res_tensor = add_src1;
        } وگرنه اگر (add_src1 == ar_local || add_src1->data == ar_local->data) {
            res_tensor = add_src0;
        } وگرنه {
            بازگشت نادرست;
        }

        اگر (ar_local->type != res_tensor->type) بازگشت نادرست;

        ثابت منطقی is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        ثابت منطقی is_row_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        اگر (!is_same_shape && !is_row_bcast) بازگشت نادرست;

        اگر (is_same_shape) {
            اگر (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                بازگشت نادرست;
            }
            اگر (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                بازگشت نادرست;
            }
        }
        اگر (ggml_is_contiguous(ar_local) != ggml_is_contiguous(add_dst)) {
            بازگشت نادرست;
        }

        // scatter is only valid for in-place add within max outputs
        ثابت منطقی is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && add_dst->data == ar_local->data;

        ساختار htp_allreduce_kernel_params new_kparams;
        اگر (!ggml_hexagon_precompute_allreduce_params(
            sess, add_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, درست, is_row_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            بازگشت نادرست;
        }

        ثابت منطقی scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

        برای (uint32_t r = 0; r < n_ranks; r++) {
            ثابت ggml_tensor * ar_src = last_گره.inputs[r];
            اگر (!ggml_hexagon_tensors_overlap(add_dst, ar_src)) {
                ادامه;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            اگر (scatter_ok && r == rank) {
                ادامه;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            بازگشت نادرست;
        }

        اگر (!try_fuse_common(res_tensor, add_dst)) {
            بازگشت نادرست;
        }

        last_گره.opcode = HTP_OP_ALLREDUCE_ADD;
        last_گره.name   = "ALLREDUCE+ADD";
        last_گره.inputs.push_back(res_tensor);
        last_گره.outputs.clear();
        last_گره.outputs.push_back(add_dst);
        last_گره.fused.push_back(گره.گره);
        memcpy(last_گره.kernel_params, &new_kparams, اندازه(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_ADD;
        memcpy(o.kernel_params, &new_kparams, اندازه(new_kparams));

        o.src[2 * n_ranks] = add_tensor(res_tensor);
        اگر (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            برای (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            برای (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } وگرنه {
            o.dst[0] = add_tensor(add_dst);
            برای (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (صحیح) new_kparams.mode, (صحیح) new_kparams.n_dsts);
        بازگشت درست;
    }

    منطقی try_fuse_rms_norm_mul(ثابت htp_opnode & گره) {
        اگر (n_ops == 0) بازگشت نادرست;
        اگر (گره.opcode != HTP_OP_MUL) بازگشت نادرست;

        htp_opnode & last_گره = ops[n_ops - 1];
        اگر (last_گره.opcode != HTP_OP_RMS_NORM) بازگشت نادرست;

        ثابت ggml_tensor * mul_src0 = گره.src0();
        ثابت ggml_tensor * mul_src1 = گره.src1();
        ثابت ggml_tensor * rms_out  = last_گره.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(rms_out)) بازگشت نادرست;

        ثابت ggml_tensor * weight;
        اگر (mul_src0 == rms_out || mul_src0->data == rms_out->data) {
            weight = mul_src1;
        } وگرنه اگر (mul_src1 == rms_out || mul_src1->data == rms_out->data) {
            weight = mul_src0;
        } وگرنه {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * src0 = last_گره.src0();

        اگر (src0->ne[0] != weight->ne[0] || src0->ne[0] != گره.dst()->ne[0]) {
            بازگشت نادرست;
        }

        ثابت منطقی is_row_bcast = (weight->ne[1] == 1 && weight->ne[2] == 1 && weight->ne[3] == 1);
        ثابت منطقی is_same_shape = (src0->ne[0] == weight->ne[0] && src0->ne[1] == weight->ne[1] &&
                                    src0->ne[2] == weight->ne[2] && src0->ne[3] == weight->ne[3]);
        اگر (!is_row_bcast && !is_same_shape) بازگشت نادرست;

        اگر (!ggml_are_same_shape(src0, گره.dst())) {
            بازگشت نادرست;
        }
        اگر (ggml_is_contiguous(src0) != ggml_is_contiguous(گره.dst())) {
            بازگشت نادرست;
        }

        ساختار htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, weight, گره.dst(), &new_kparams
        );

        اگر ((size_t) new_kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_size, sess->vtcm_size);
            بازگشت نادرست;
        }

        اگر (!try_fuse_common(weight, گره.dst())) {
            بازگشت نادرست;
        }

        last_گره.opcode = HTP_OP_RMS_NORM_MUL;
        last_گره.name   = "RMS_NORM+MUL";
        last_گره.inputs.clear();
        last_گره.inputs.push_back(src0);
        last_گره.inputs.push_back(weight);
        last_گره.outputs.clear();
        last_گره.outputs.push_back(گره.dst());
        last_گره.fused.push_back(گره.گره);
        memcpy(last_گره.kernel_params, &new_kparams, اندازه(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        memcpy(o.kernel_params, &new_kparams, اندازه(new_kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(weight);
        برای (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(گره.dst());
        برای (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        بازگشت درست;
    }

    منطقی try_fuse_mul_mat_add(ثابت htp_opnode & گره) {
        اگر (n_ops == 0) بازگشت نادرست;
        اگر (گره.opcode != HTP_OP_ADD) بازگشت نادرست;

        htp_opnode & last_گره = ops[n_ops - 1];
        اگر (last_گره.opcode != HTP_OP_MUL_MAT) بازگشت نادرست;

        ثابت ggml_tensor * add_src0 = گره.src0();
        ثابت ggml_tensor * add_src1 = گره.src1();
        ثابت ggml_tensor * mm_out   = last_گره.dst();

        اگر (!ggml_hexagon_tensor_is_fuseable(mm_out)) بازگشت نادرست;

        ثابت ggml_tensor * src2;
        اگر (add_src0 == mm_out || add_src0->data == mm_out->data) {
            src2 = add_src1;
        } وگرنه اگر (add_src1 == mm_out || add_src1->data == mm_out->data) {
            src2 = add_src0;
        } وگرنه {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * src0 = last_گره.src0();
        ثابت ggml_tensor * src1 = last_گره.src1();

        اگر (src2->type != GGML_TYPE_F32) بازگشت نادرست;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        اگر (!ggml_are_same_shape(mm_out, گره.dst())) بازگشت نادرست;

        ثابت ساختار htp_mm_kernel_params * orig_kparams = (ثابت ساختار htp_mm_kernel_params *) last_گره.kernel_params;
        ساختار htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_add_params(sess, src0, src1, src2, گره.dst(), &kparams);
        اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            بازگشت نادرست;
        }

        اگر (orig_kparams->collapse != kparams.collapse) {
            بازگشت نادرست;
        }

        ثابت صحیح src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ثابت منطقی can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        اگر (!can_fuse) بازگشت نادرست;

        اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
            بازگشت نادرست;
        }

        اگر (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            اگر (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_threads < orig_kparams->n_act_threads) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_threads, kparams.n_act_threads);
                بازگشت نادرست;
            }
        }

        اگر (!try_fuse_common(src2, گره.dst())) {
            بازگشت نادرست;
        }

        last_گره.opcode = HTP_OP_MUL_MAT_ADD;
        last_گره.name   = "MUL_MAT+ADD";
        last_گره.inputs.clear();
        last_گره.inputs.push_back(src0);
        last_گره.inputs.push_back(src1);
        last_گره.inputs.push_back(src2);
        last_گره.outputs.clear();
        last_گره.outputs.push_back(گره.dst());
        last_گره.fused.push_back(گره.گره);
        memcpy(last_گره.kernel_params, &kparams, اندازه(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_ADD;
        memcpy(o.kernel_params, &kparams, اندازه(kparams));

        o.src[0] = add_tensor(src0);
        o.src[1] = add_tensor(src1);
        o.src[2] = add_tensor(src2);
        برای (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = add_tensor(گره.dst());
        برای (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        بازگشت درست;
    }

    منطقی try_fuse_mul_mat_nx(ثابت htp_opnode & گره) {
        اگر (n_ops == 0 || گره.opcode != HTP_OP_MUL_MAT) بازگشت نادرست;
        اگر (!is_mergeable_mul_mat(گره.گره)) بازگشت نادرست;

        ثابت ggml_tensor * w_in = گره.src0();
        ثابت ggml_tensor * x_in = گره.src1();
        ثابت ggml_tensor * d_in = گره.dst();

        htp_opnode & last_گره = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        اگر (last_گره.opcode == HTP_OP_MUL_MAT_NX) {
            ثابت uint32_t curr_n = (uint32_t) last_گره.outputs.size();
            اگر (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                بازگشت نادرست;
            }

            ثابت ggml_tensor * w0 = last_گره.inputs[0];
            ثابت ggml_tensor * x  = last_گره.inputs[curr_n];

            اگر (x_in != x || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                بازگشت نادرست;
            }
            اگر (!last_گره.fused.empty() && (mm_is_hmx_eligible(last_گره.fused[0]) != mm_is_hmx_eligible(گره.گره))) {
                بازگشت نادرست;
            }

            ثابت ساختار htp_mm_kernel_params * orig_kparams = (ثابت ساختار htp_mm_kernel_params *) last_گره.kernel_params;
            ثابت منطقی collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_in, x, d_in);
            اگر (orig_kparams->collapse && !collapse) {
                بازگشت نادرست;
            }

            ساختار htp_mm_kernel_params kparams;
            اگر (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } وگرنه {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            اگر (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                بازگشت نادرست;
            }
            اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                بازگشت نادرست;
            }

            اگر (!try_fuse_common(w_in, d_in)) {
                بازگشت نادرست;
            }

            last_گره.inputs[curr_n] = w_in;
            last_گره.inputs.push_back(x);
            last_گره.outputs.push_back(d_in);
            last_گره.fused.push_back(گره.گره);
            memcpy(last_گره.kernel_params, &kparams, اندازه(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, اندازه(kparams));

            برای (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = add_tensor(last_گره.inputs[s]);
            }
            برای (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            برای (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_گره.outputs[d]);
            }
            برای (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            بازگشت درست;
        }

        // Case 2: last_node is single MUL_MAT
        اگر (last_گره.opcode == HTP_OP_MUL_MAT) {
            اگر (!is_mergeable_mul_mat_pair(last_گره.گره, گره.گره)) {
                بازگشت نادرست;
            }

            ثابت ggml_tensor * w0 = last_گره.src0();
            ثابت ggml_tensor * x  = last_گره.src1();
            ثابت ggml_tensor * w1 = گره.src0();
            ثابت ggml_tensor * dst_0 = last_گره.dst();
            ثابت ggml_tensor * dst_1 = گره.dst();

            ثابت ساختار htp_mm_kernel_params * orig_kparams = (ثابت ساختار htp_mm_kernel_params *) last_گره.kernel_params;
            ثابت منطقی collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            اگر (orig_kparams->collapse && !collapse) {
                بازگشت نادرست;
            }

            ساختار htp_mm_kernel_params kparams;
            اگر (collapse) {
                ثابت ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_rows(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } وگرنه {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            اگر (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                بازگشت نادرست;
            }
            اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                بازگشت نادرست;
            }

            اگر (!try_fuse_common(w1, گره.dst())) {
                بازگشت نادرست;
            }

            last_گره.opcode = HTP_OP_MUL_MAT_NX;
            last_گره.name   = "MUL_MAT_NX";
            last_گره.inputs.clear();
            last_گره.inputs.push_back(w0);
            last_گره.inputs.push_back(w1);
            last_گره.inputs.push_back(x);
            last_گره.outputs.clear();
            last_گره.outputs.push_back(dst_0);
            last_گره.outputs.push_back(dst_1);
            last_گره.fused.push_back(گره.گره);
            memcpy(last_گره.kernel_params, &kparams, اندازه(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            memcpy(o.kernel_params, &kparams, اندازه(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            برای (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            برای (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            بازگشت درست;
        }

        بازگشت نادرست;
    }

    منطقی try_fuse_mul_mat_id_nx(ثابت htp_opnode & گره) {
        اگر (n_ops == 0 || گره.opcode != HTP_OP_MUL_MAT_ID) بازگشت نادرست;
        اگر (!is_mergeable_mul_mat_id(گره.گره)) بازگشت نادرست;

        ثابت ggml_tensor * w_in   = گره.src0();
        ثابت ggml_tensor * x_in   = گره.src1();
        ثابت ggml_tensor * ids_in = گره.گره->src[2];
        ثابت ggml_tensor * d_in   = گره.dst();

        htp_opnode & last_گره = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        اگر (last_گره.opcode == HTP_OP_MUL_MAT_ID_NX) {
            ثابت uint32_t curr_n = (uint32_t) last_گره.outputs.size();
            اگر (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                بازگشت نادرست;
            }

            ثابت ggml_tensor * w0  = last_گره.inputs[0];
            ثابت ggml_tensor * x   = last_گره.inputs[curr_n];
            ثابت ggml_tensor * ids = last_گره.inputs[curr_n + 1];

            اگر (x_in != x || ids_in != ids || w_in->type != w0->type || w_in->ne[0] != w0->ne[0] || w_in->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_in) != ggml_hexagon_tensor_is_tiled(w0)) {
                بازگشت نادرست;
            }
            اگر (!last_گره.fused.empty() && (mm_is_hmx_eligible(last_گره.fused[0]) != mm_is_hmx_eligible(گره.گره))) {
                بازگشت نادرست;
            }

            ساختار htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_in, curr_n + 1, &kparams);
            اگر (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                بازگشت نادرست;
            }
            اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                بازگشت نادرست;
            }

            اگر (!try_fuse_common(w_in, d_in)) {
                بازگشت نادرست;
            }

            last_گره.inputs[curr_n] = w_in;
            last_گره.inputs[curr_n + 1] = x;
            last_گره.inputs.push_back(ids);
            last_گره.outputs.push_back(d_in);
            last_گره.fused.push_back(گره.گره);
            memcpy(last_گره.kernel_params, &kparams, اندازه(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            memcpy(o.kernel_params, &kparams, اندازه(kparams));

            برای (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = add_tensor(last_گره.inputs[s]);
            }
            برای (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            برای (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = add_tensor(last_گره.outputs[d]);
            }
            برای (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            بازگشت درست;
        }

        // Case 2: last_node is single MUL_MAT_ID
        اگر (last_گره.opcode == HTP_OP_MUL_MAT_ID) {
            اگر (!is_mergeable_mul_mat_id_pair(last_گره.گره, گره.گره)) {
                بازگشت نادرست;
            }

            ثابت ggml_tensor * w0  = last_گره.src0();
            ثابت ggml_tensor * x   = last_گره.src1();
            ثابت ggml_tensor * ids = last_گره.گره->src[2];
            ثابت ggml_tensor * w1  = گره.src0();

            ساختار htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, گره.dst(), 2, &kparams);
            اگر (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                بازگشت نادرست;
            }
            اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
                بازگشت نادرست;
            }

            اگر (!try_fuse_common(w1, گره.dst())) {
                بازگشت نادرست;
            }

            ثابت ggml_tensor * dst_0 = last_گره.dst();
            ثابت ggml_tensor * dst_1 = گره.dst();

            last_گره.opcode = HTP_OP_MUL_MAT_ID_NX;
            last_گره.name   = "MUL_MAT_ID_NX";
            last_گره.inputs.clear();
            last_گره.inputs.push_back(w0);
            last_گره.inputs.push_back(w1);
            last_گره.inputs.push_back(x);
            last_گره.inputs.push_back(ids);
            last_گره.outputs.clear();
            last_گره.outputs.push_back(dst_0);
            last_گره.outputs.push_back(dst_1);
            last_گره.fused.push_back(گره.گره);
            memcpy(last_گره.kernel_params, &kparams, اندازه(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            memcpy(o.kernel_params, &kparams, اندازه(kparams));

            o.src[0] = add_tensor(w0);
            o.src[1] = add_tensor(w1);
            o.src[2] = add_tensor(x);
            o.src[3] = add_tensor(ids);
            برای (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = add_tensor(dst_0);
            o.dst[1] = add_tensor(dst_1);
            برای (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            بازگشت درست;
        }

        بازگشت نادرست;
    }

    منطقی try_fuse_gdn_cpy(ثابت htp_opnode & گره) {
        اگر (n_ops == 0 || گره.opcode != HTP_OP_CPY) بازگشت نادرست;

        htp_opnode & last_گره = ops[n_ops - 1];
        اگر (last_گره.opcode != HTP_OP_GATED_DELTA_NET) بازگشت نادرست;
        اگر (last_گره.outputs.size() != 1) بازگشت نادرست;

        ثابت ggml_tensor * gdn_out  = last_گره.dst();
        ثابت ggml_tensor * cpy_گره = گره.گره;
        ثابت ggml_tensor * cpy_src  = گره.src0();
        ثابت ggml_tensor * cpy_dst  = گره.dst();

        اگر (!cpy_src || !cpy_dst || !cpy_dst->data) بازگشت نادرست;
        اگر (gdn_out->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) بازگشت نادرست;
        اگر ((gdn_out->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_گره->flags & GGML_TENSOR_FLAG_OUTPUT)) بازگشت نادرست;

        ثابت ggml_tensor * v = last_گره.گره->src[2];
        اگر (!v) بازگشت نادرست;

        ثابت int64_t S_v      = v->ne[0];
        ثابت int64_t H        = v->ne[1];
        ثابت int64_t n_tokens = v->ne[2];
        ثابت int64_t n_seqs   = v->ne[3];
        ثابت int64_t K        = ggml_get_op_params_i32(last_گره.گره, 0);
        ثابت size_t  tail_off = (size_t) S_v * H * n_tokens * n_seqs * اندازه(اعشاری);

        ثابت int64_t D         = S_v * S_v * H;
        ثابت int64_t n_written = std::min<int64_t>(n_tokens, K);

        اگر (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_out && cpy_src->view_src->data != gdn_out->data) ||
            cpy_src->view_offs != tail_off || !ggml_is_contiguous(cpy_src)) {
            بازگشت نادرست;
        }

        اگر (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != اندازه(اعشاری)) {
            بازگشت نادرست;
        }
        اگر (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * اندازه(اعشاری)) {
            بازگشت نادرست;
        }
        اگر (n_written > 1) {
            اگر (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * اندازه(اعشاری)) {
                بازگشت نادرست;
            }
        }

        اگر (!try_fuse_common({cpy_dst})) {
            بازگشت نادرست;
        }

        last_گره.name += "+CPY";
        last_گره.outputs.push_back(cpy_dst);
        last_گره.fused.push_back(گره.گره);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = add_tensor(cpy_dst);
        برای (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        بازگشت درست;
    }

    منطقی try_fuse(ثابت htp_opnode & گره) {
        اگر (!opt_opfusion) بازگشت نادرست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_ADD) && try_fuse_allreduce_add(گره)) بازگشت درست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(گره))  بازگشت درست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ADD)   && try_fuse_mul_mat_add(گره))   بازگشت درست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(گره))    بازگشت درست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(گره)) بازگشت درست;
        اگر (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(گره))       بازگشت درست;
        بازگشت نادرست;
    }
};

ساختار ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

ساختار ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_buffer *shm_buf;
    size_t                      shm_blk_size;
    size_t                      depth;

    using opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       start_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_size, size_t depth) : depth(depth) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_size;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_size = 0;
        اگر (opt_profile == 3) {
            tr_size = (HTP_MAX_NTHREADS + 1) * opt_optrace * اندازه(htp_trace_desc);
        }

        shm_blk_size = اندازه(htp_buf_desc)  * n_bufs    +
                       اندازه(htp_tensor)    * n_tensors +
                       اندازه(htp_op_desc)   * n_ops     +
                       اندازه(htp_prof_desc) * n_ops     +
                       tr_size;

        shm_buf = جدید ggml_hexagon_shared_buffer(sess, shm_blk_size * depth, درست /* pinned */);

        op_cache.resize(depth);
        start_usec.resize(depth, 0);

        اگر (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_size, depth, shm_buf->size(), shm_blk_size);
        }
    }

    ~ggml_hexagon_opqueue() {
        delete shm_buf;
    }

    size_t shm_size() ثابت { بازگشت shm_buf ? shm_buf->size() : 0; }

    // push new batch
    منطقی افزودن(htp_opbatch_req& req, dspqueue_buffer& dbuf, ثابت ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(اندازه(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(اندازه(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(اندازه(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(اندازه(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(اندازه(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(اندازه(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        اگر (seq - shm_buf->sess->batch_rsp_seq > depth) { بازگشت نادرست; }

        ثابت uint32_t slot = (uint32_t) ((seq - 1) % depth);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[slot]   = op_batch->ops;
        start_usec[slot] = ggml_time_us();

        ثابت size_t b_size = اندازه(htp_buf_desc)  * req.n_bufs;
        ثابت size_t t_size = اندازه(htp_tensor)    * req.n_tensors;
        ثابت size_t o_size = اندازه(htp_op_desc)   * req.n_ops;
        ثابت size_t p_size = اندازه(htp_prof_desc) * req.n_ops;

        size_t tr_size = 0;
        اگر (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * req.n_traces * اندازه(htp_trace_desc);
        } وگرنه {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->base() + ((size_t) slot * shm_blk_size);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        dbuf.offset   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->base();
        dbuf.size     = b_size + t_size + o_size + p_size + tr_size;

        GGML_ASSERT(dbuf.size <= shm_blk_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_size;
        uint8_t * t_ptr = m_ptr; m_ptr += t_size;
        uint8_t * o_ptr = m_ptr;

        memcpy(b_ptr, (تهی *) op_batch->h_bufs.data(), b_size);
        memcpy(t_ptr, (تهی *) op_batch->h_tens.data(), t_size);
        memcpy(o_ptr, (تهی *) op_batch->h_ops.data(),  o_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_size, t_size, o_size, (size_t) dbuf.size);

        اگر (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            برای (unsigned صحیح i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (تهی *) b[i].base, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            برای (unsigned صحیح i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (unsigned long long) t[i].data, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        بازگشت درست;
    }

    تهی برداشت(htp_opbatch_rsp rsp, dspqueue_buffer dbuf) {
        ثابت uint32_t slot = (uint32_t) ((rsp.seq - 1) % depth);

        ثابت size_t b_size = اندازه(htp_buf_desc)  * rsp.n_bufs;
        ثابت size_t t_size = اندازه(htp_tensor)    * rsp.n_tensors;
        ثابت size_t o_size = اندازه(htp_op_desc)   * rsp.n_ops;
        ثابت size_t p_size = اندازه(htp_prof_desc) * rsp.n_ops;

        size_t tr_size = 0;
        uint32_t n_traces = 0;
        اگر (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_size = (HTP_MAX_NTHREADS + 1) * n_traces * اندازه(htp_trace_desc);
        }

        ثابت size_t m_size = b_size + t_size + o_size + p_size + tr_size;
        GGML_ASSERT(m_size <= shm_blk_size);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (unsigned long long) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_size, t_size, o_size);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_size + t_size + o_size);

        اگر (rsp.n_ops > 0) {
            auto & ops = op_cache[slot];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            ثابت htp_prof_desc * pd = (ثابت htp_prof_desc *) p_ptr;
            ثابت htp_trace_desc * trace_events = nullptr;
            اگر (opt_profile == 3) {
                trace_events = (ثابت htp_trace_desc *) (p_ptr + p_size);
            }

            اگر (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            برای (uint32_t i = 0; i < rsp.n_ops; i++) {
                اگر (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            اگر (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

تهی ggml_hexagon_session::flush_peers() {
    auto vpeers = std::move(virt_peers);
    virt_peers.clear();
    برای (auto * peer : vpeers) {
        peer->flush_sync();
    }

    auto ppeers = std::move(phys_peers);
    phys_peers.clear();
    برای (auto * peer : ppeers) {
        peer->flush_async();
    }

    برای (auto & sub : این->mdev.sessions) {
        sub->flush_peers();
    }
}

تهی ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

تهی ggml_hexagon_session::flush_pending(منطقی all) {
    برای (auto & sub : این->mdev.sessions) {
        sub->flush_pending(all);
        اگر (sub->last_error > HTP_STATUS_OK) {
            این->last_error = sub->last_error.load();
        }
    }

    وقتی (این->batch_rsp_seq < این->batch_req_seq) {
        ساختار htp_opbatch_rsp rsp;
        uint32_t               rsp_size;
        uint32_t               flags;

        ساختار dspqueue_buffer dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        ثابت uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        صحیح err = dspqueue_read(این->queue, &flags, 1, &n_dbufs, &dbuf, اندازه(rsp), &rsp_size, (uint8_t *) &rsp, timeo);
        اگر (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            ادامه;
        }

        اگر (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (unsigned) err);
        }

        // Basic sanity checks
        اگر (rsp_size != اندازه(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", این->c_name(), rsp_size, n_dbufs);
        }

        اگر (rsp.status > HTP_STATUS_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", این->c_name(), status_to_str(rsp.status));
            این->last_error = rsp.status;
            برای (auto & sub : این->mdev.sessions) {
                sub->last_error = rsp.status;
            }
        }

        op_queue->برداشت(rsp, dbuf);

        GGML_ASSERT(rsp.seq == این->batch_rsp_seq + 1);
        این->batch_rsp_seq = rsp.seq;

        اگر (!all) بشکن;
    }
}

تهی ggml_hexagon_session::flush_sync(منطقی all) {
    flush_async();
    flush_pending(all);
}

تهی ggml_hexagon_session::start_batch() {
    اگر (این->mdev.count > 1) {
        enqueue_mdev_group();
    }
}

تهی ggml_hexagon_session::flush_batch(size_t min_ops) {
    اگر (op_batch->n_ops < min_ops) { بازگشت; }

    op_batch->sort_buffers();

    htp_opbatch_req req {};
    dspqueue_buffer dbuf{};

    ثابت uint64_t seq = ++این->batch_req_seq;

    op_batch->update_mdev_group(این->mdev.idx);

    اگر (!op_queue->افزودن(req, dbuf, op_batch, seq)) {
        flush_pending(نادرست);
        op_queue->افزودن(req, dbuf, op_batch, seq);
    }

    برای (auto & sub : این->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_buffer sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_group(sub->mdev.idx);

        اگر (!sub->op_queue->افزودن(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(نادرست);
            sub->op_queue->افزودن(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        صحیح err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, اندازه(sub_req), (ثابت uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (unsigned) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", این->c_name(), dbuf.ptr, dbuf.size);

    صحیح err = dspqueue_write(این->queue, 0, 1, &dbuf, اندازه(req), (ثابت uint8_t*) &req, DSPQUEUE_TIMEOUT);
    اگر (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", این->c_name(), (unsigned) err);
    }

    op_batch->reset();
}

تهی ggml_hexagon_session::enqueue_op(ثابت htp_opnode & گره) {
    auto clone_tensor_buffer = [این](ثابت ggml_tensor * t) {
        auto sbuf = این->mmap_tensor(t);
        اگر (!sbuf) بازگشت;
        اگر (sbuf->sess != این) {
            این->clone_buffer(sbuf);
        }
        برای (auto & sub : این->mdev.sessions) {
            sub->clone_buffer(sbuf);
        }
    };

    برای (auto t : گره.get_inputs()) {
        clone_tensor_buffer(t);
    }
    برای (auto t : گره.get_outputs()) {
        clone_tensor_buffer(t);
    }

    اگر (opt_opfusion && op_batch->try_fuse(گره)) {
        بازگشت;
    }

    اگر (!op_batch->fit_op(گره)) {
        flush_async();
    }

    اگر (op_batch->empty()) {
        start_batch();
    }

    اگر (!op_batch->fit_op(گره)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->add_op(گره);
}

تهی ggml_hexagon_session::enqueue_mdev_group() {
    htp_opnode group_گره(HTP_OP_MDEV_GROUP);

    uint8_t * fence_slot = این->mdev_fence_slot;

    ایستا ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.buffer       = &این->fence_buf->backend_buffer;
    dummy_t.extra        = &fence_extra;
    dummy_t.data         = (تهی *) fence_slot;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.ne[1]        = (int64_t) این->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_SLOT_SIZE;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) این->mdev.idx;

    ggml_tensor * گره = group_گره.add_dummy(dummy_t);
    گره->src[0] = گره;
    group_گره.init(گره);
    group_گره.outputs.clear();
    group_گره.name = "MDEV_GROUP";

    برای (auto & sub : این->mdev.sessions) {
        sub->clone_buffer(این->fence_buf);
    }

    op_batch->add_op(group_گره);
}

تهی ggml_hexagon_session::enqueue_cpy(ثابت ggml_tensor * src, ggml_tensor * dst, ثابت ggml_tensor * sync_tensor, uint32_t fence_seq) {
    ثابت منطقی with_fence = sync_tensor != nullptr;
    htp_opnode cpy_گره(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* گره = cpy_گره.add_dummy(*dst);
    گره->op     = GGML_OP_CPY;
    گره->src[0] = const_cast<ggml_tensor *>(src);
    گره->src[1] = with_fence ? cpy_گره.add_dummy(*sync_tensor) : nullptr;
    اگر (with_fence) {
        گره->op_params[0] = (int32_t) fence_seq;
    }

    cpy_گره.init(گره);
    اگر (with_fence) {
        cpy_گره.name = "CPY+FENCE";
    }
    ثابت منطقی ok = ggml_hexagon_precompute_cpy_params(این, گره, (ساختار htp_copy_kernel_params *) cpy_گره.kernel_params);
    ثابت auto * kparams = (ثابت ساختار htp_copy_kernel_params *) cpy_گره.kernel_params;
    اگر (ok && !with_fence && kparams->total_elems == 0) {
        بازگشت;
    }
    این->enqueue_op(cpy_گره);
}

تهی ggml_hexagon_session::enqueue_fence(ثابت ggml_tensor * sync_tensor, uint32_t fence_seq, منطقی wait) {
    htp_opnode sync_گره(HTP_OP_FENCE);

    ggml_tensor* گره = sync_گره.add_dummy(*sync_tensor);
    گره->op           = GGML_OP_NONE;
    گره->src[0]       = گره;
    گره->op_params[0] = (int32_t) fence_seq;
    گره->op_params[1] = wait ? 0 : 1;

    sync_گره.init(گره);
    sync_گره.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    این->enqueue_op(sync_گره);
}

ایستا منطقی ggml_hexagon_precompute_allreduce_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    منطقی has_add,
    منطقی is_row_bcast,
    منطقی is_shard_ok,
    ساختار htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_row_bcast = (has_add && is_row_bcast) ? 1 : 0;

    ثابت uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    ثابت uint32_t elem_size = (dst->type == GGML_TYPE_F16) ? اندازه(ggml_fp16_t) : اندازه(اعشاری);
    ثابت منطقی is_contiguous = ggml_is_contiguous(dst);

    ثابت uint32_t ne0 = (uint32_t) dst->ne[0];
    ثابت uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    ثابت منطقی use_1d = is_contiguous && !(has_add && is_row_bcast && ne1 > 1);

    اگر (has_add) {
        // sharded reduce-scatter for contiguous in-place add
        اگر (opt_ar_scatter && is_shard_ok && use_1d && !is_row_bcast) {
            ثابت uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_start = (int32_t) rank_elem_start;
            kparams->rank_nelem      = (int32_t) (rank_elem_end - rank_elem_start);
        } وگرنه اگر (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } وگرنه {
            kparams->n_dsts          = 1;
            kparams->rank_elem_start = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } وگرنه {
        kparams->n_dsts = (int32_t) n_ranks;
        اگر (use_1d) {
            ثابت uint32_t rank_chunk_elems = hex_round_up((nelem + n_ranks - 1) / n_ranks, 128);
            ثابت uint32_t rank_elem_start  = (std::min)(rank * rank_chunk_elems, nelem);
            ثابت uint32_t rank_elem_end    = (std::min)(rank_elem_start + rank_chunk_elems, nelem);
            ثابت uint32_t rank_nelem       = rank_elem_end - rank_elem_start;
            kparams->rank_elem_start        = (int32_t) rank_elem_start;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } وگرنه {
            ثابت uint32_t rank_chunk_rows = (ne1 + n_ranks - 1) / n_ranks;
            ثابت uint32_t rank_r0         = (std::min)(rank * rank_chunk_rows, ne1);
            ثابت uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_rows, ne1);
            ثابت uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_start       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_FULL;

    اگر (use_1d) {
        ثابت uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nelem / 128));
        kparams->n_threads = n_threads;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_شمار(n_ranks, n_threads, has_add, is_row_bcast);

        uint32_t block_elems = 65536;
        اگر (block_elems > rank_nelem / n_threads && rank_nelem / n_threads > 128) {
            block_elems = hex_round_up(rank_nelem / (n_threads * 2), 128);
        }
        block_elems = (std::max)(128u, block_elems);

        kparams->block_elems          = block_elems;
        kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        وقتی ((size_t) kparams->vtcm_size > sess->vtcm_size && block_elems > 128) {
            ثابت size_t max_bytes_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2);
            block_elems = (uint32_t) hex_align_down((size_t) (max_bytes_per_buf / elem_size), 128);
            اگر (block_elems < 128) بشکن;
            kparams->block_elems          = block_elems;
            kparams->vtcm_size_per_thread = 2 * block_elems * elem_size;
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
        }

        اگر (sess->vtcm_size < (size_t) kparams->vtcm_size || block_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            بازگشت نادرست;
        }

        kparams->elems_per_thread = hex_round_up((rank_nelem + n_threads - 1) / n_threads, block_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        بازگشت درست;
    } وگرنه {
        ثابت uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, (std::max)(1u, rank_nrows));
        kparams->n_threads = n_threads;
        ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_buffer_شمار(n_ranks, n_threads, has_add, is_row_bcast);

        ثابت uint32_t row_bytes = ne0 * elem_size;
        ثابت uint32_t row_size_aligned = (uint32_t) hex_align_up(row_bytes, 128);
        kparams->row_size_aligned = row_size_aligned;

        ثابت uint32_t nrows_per_thread = (rank_nrows + n_threads - 1) / n_threads;
        uint32_t block_rows = (std::min)(128u, nrows_per_thread);
        block_rows = (std::max)(1u, block_rows);
        kparams->block_elems = block_rows;

        kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
        kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;

        وقتی ((size_t) kparams->vtcm_size > sess->vtcm_size && block_rows > 1) {
            ثابت size_t max_rows_per_buf = sess->vtcm_size / (n_vtcm_buffers * 2 * row_size_aligned);
            block_rows = (std::max)(1u, (uint32_t) max_rows_per_buf);
            kparams->block_elems          = block_rows;
            kparams->vtcm_size_per_thread = 2 * (block_rows * row_size_aligned);
            kparams->vtcm_size            = n_vtcm_buffers * kparams->vtcm_size_per_thread;
            اگر (max_rows_per_buf == 0) بشکن;
        }

        اگر (sess->vtcm_size < (size_t) kparams->vtcm_size || block_rows < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_size, sess->vtcm_size);
            بازگشت نادرست;
        }

        kparams->elems_per_thread = nrows_per_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        بازگشت درست;
    }
}

تهی ggml_hexagon_session::enqueue_allreduce(
    ثابت ggml_tensor * dst,
    ثابت std::vector<ثابت ggml_tensor *> & src_tensors,
    ثابت std::vector<ثابت ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_entry,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_گره(HTP_OP_ALLREDUCE);

    ggml_tensor* گره = ar_گره.add_dummy(*dst);
    گره->op           = GGML_OP_NONE;
    گره->op_params[0] = (int32_t) fence_seq_entry;
    گره->op_params[1] = (int32_t) fence_seq_exit;

    ar_گره.init(گره);

    ar_گره.inputs.clear();
    برای (size_t i = 0; i < src_tensors.size(); i++) {
        ar_گره.inputs.push_back(src_tensors[i]);
    }
    برای (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_گره.inputs.push_back(ar_گره.add_dummy(*sync_tensors[i]));
    }

    ar_گره.outputs.clear();
    برای (size_t i = 0; i < src_tensors.size(); i++) {
        ar_گره.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        این, dst, rank, n_ranks, نادرست, نادرست, /*is_shard_ok=*/ نادرست,
        (ساختار htp_allreduce_kernel_params *) ar_گره.kernel_params);

    ar_گره.name = "ALLREDUCE";
    این->enqueue_op(ar_گره);
}

ggml_hexagon_shared_buffer * ggml_hexagon_session::mmap_tensor(ثابت ggml_tensor * t) {
    اگر (!t) بازگشت nullptr;

    auto sbuf = static_cast<ggml_hexagon_shared_buffer *>(t->buffer->context);
    اگر (!sbuf->mapped) {
        ثابت منطقی is_weight = ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        ثابت منطقی extended  = opt_dma64 && is_weight;
        sbuf->mmap(extended);
    }
    بازگشت sbuf;
}

منطقی ggml_hexagon_session::clone_buffer(ثابت ggml_hexagon_shared_buffer * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    اگر (sbuf->sess == این) بازگشت درست;

    auto mem = sbuf->mem;
    صحیح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    اگر (این->cloned_buffers.find(fd) != این->cloned_buffers.end()) بازگشت درست;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", این->name.c_str(),
                sbuf->c_name(), sbuf->base(), sbuf->size(), fd);

    auto clone = std::make_unique<ggml_hexagon_shared_buffer>(این, *sbuf);
    تلاش {
        clone->mmap(sbuf->extended);
    } بگیر (ثابت std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", این->c_name(), exc.what());
        بازگشت نادرست;
    }

    این->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(این);
    بازگشت درست;
}

تهی ggml_hexagon_session::release_buffer(ثابت ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    صحیح   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    auto it = این->cloned_buffers.find(fd);
    اگر (it != این->cloned_buffers.end()) {
        auto clone = std::move(it->second);
        این->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(این);
}

تهی ggml_hexagon_session::unclone_buffer(ثابت ggml_hexagon_shared_buffer * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    auto mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    برای (auto * sess : sessions) {
        sess->release_buffer(sbuf);
    }
}

ایستا size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_buffer *> sbufs;

    ثابت size_t MiB = 1024 * 1024;
    ثابت size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t step = 256u * MiB;

    تلاش {
        sbufs.push_back(جدید ggml_hexagon_shared_buffer(sess, GiB, درست)); vmem += GiB;
        sbufs.push_back(جدید ggml_hexagon_shared_buffer(sess, GiB, درست)); vmem += GiB;
        sbufs.push_back(جدید ggml_hexagon_shared_buffer(sess, GiB, درست)); vmem += GiB;

        وقتی (1) {
            sbufs.push_back(جدید ggml_hexagon_shared_buffer(sess, step, درست));
            vmem += step;
        }
    } بگیر (...) { }

    برای (auto b : sbufs) { delete b; }

    بازگشت vmem - step; // backoff to account for overhead from internal mappings
}

تهی ggml_hexagon_session::allocate(ثابت ggml_hexagon_device_config & config) noexcept(نادرست) {
    صحیح phys_idx = config.physical_idx;
    صحیح virt_idx = config.virtual_idx;

    این->valid_session = نادرست;
    این->valid_handle  = نادرست;
    این->valid_queue   = نادرست;
    این->valid_iface   = نادرست;

    این->name          = config.name;
    این->phys_idx      = phys_idx;
    این->virt_idx      = virt_idx;
    این->domain_id     = config.domain_id;
    این->session_id    = 0;
    این->batch_req_seq = 0;
    این->batch_rsp_seq = 0;
    این->last_error    = HTP_STATUS_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", این->name.c_str(), این->domain_id, phys_idx, virt_idx);

    اگر (config.domain_id < 0 || config.domain_name.empty()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        پرتابکن std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    ثابت std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    اگر (virt_idx > 0) {
        ساختار remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<نویسه *>(dom_name.c_str());
        n.session_name     = const_cast<نویسه *>(این->name.c_str());
        n.session_name_len = این->name.size();
        n.session_id       = virt_idx;

        صحیح err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (تهی *) &n, اندازه(n));
        اگر (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           این->c_name(), phys_idx, virt_idx, err);
            پرتابکن std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        این->session_id    = n.session_id;
        این->domain_id     = n.effective_domain_id;
        این->valid_session = درست;
    } وگرنه {
        ساختار remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<نویسه *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        صحیح err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (تهی *) &eff, اندازه(eff));
        اگر (err == AEE_SUCCESS) {
            این->domain_id = eff.effective_domain_id;
        } وگرنه {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           این->name.c_str(), err, این->domain_id);
        }
    }

    // Enable unsigned modules
    {
        ساختار remote_rpc_control_unsigned_module u;
        u.domain = این->domain_id;
        u.enable = 1;
        صحیح err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (تهی *) &u, اندازه(u));
        اگر (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", این->c_name(), err);
            پرتابکن std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    نویسه session_uri[256];
    {
        نویسه htp_uri[256];
        snprintf(htp_uri, اندازه(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        ساختار remote_rpc_get_uri u = {};
        u.session_id      = این->session_id;
        u.domain_name     = const_cast<نویسه *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<نویسه *>(htp_uri);
        u.module_uri_len  = strlen(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = اندازه(session_uri);

        صحیح err = remote_session_control(FASTRPC_GET_URI, (تهی *) &u, اندازه(u));
        اگر (err != AEE_SUCCESS) {
            snprintf(session_uri, اندازه(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), این->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          این->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    صحیح err = htp_iface_open(session_uri, &این->handle);
    اگر (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", این->c_name(), session_uri, err);
        پرتابکن std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    این->valid_handle = درست;

    // Query HW info and resolve session options
    {
        unsigned صحیح hw_n_threads = 0;
        unsigned صحیح hw_n_hvx     = 0;
        unsigned صحیح hw_n_hmx     = 0;
        unsigned long long hw_vtcm_size = 0;
        صحیح hw_err = htp_iface_hwinfo(این->handle, &hw_n_threads, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_size);
        اگر (hw_err == 0) {
            ثابت uint32_t max_n_threads = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_threads);
            این->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_threads) : max_n_threads;
            این->n_hvx     = این->n_threads;
            این->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            این->vtcm_size = (uint64_t)hw_vtcm_size;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          این->c_name(), این->n_threads, این->n_hvx, این->n_hmx,
                          (unsigned long long)(این->vtcm_size / (1024 * 1024)));
        } وگرنه {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", این->c_name(), hw_err);
            ثابت uint32_t default_n_threads = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            این->n_threads = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_threads;
            این->n_hvx     = این->n_threads;
            این->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            این->vtcm_size = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        ساختار remote_rpc_control_latency l;
        l.enable = 1;

        صحیح err = remote_handle64_control(این->handle, DSPRPC_CONTROL_LATENCY, (تهی *) &l, اندازه(l));
        اگر (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (unsigned) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", این->c_name(),
                  این->session_id, این->domain_id, session_uri, (unsigned long) این->handle);

    ثابت size_t req_q_size = (اندازه(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    ثابت size_t rsp_q_size = (اندازه(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(این->domain_id,
                          0,              // Flags
                          req_q_size,     // Request  queue size (in bytes)
                          rsp_q_size,     // Response queue size (in bytes)
                          nullptr,        // Read packet callback (we handle reads explicitly)
                          nullptr,        // Error callback (we handle errors during reads)
                          (تهی *) این,  // Callback context
                          &queue);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", این->name.c_str(), (unsigned) err);
        پرتابکن std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    این->valid_queue = درست;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &این->queue_id);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (unsigned) err);
        پرتابکن std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    اگر (opt_etm) {
        err = htp_iface_etm(این->handle, 1);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    // Allocate buffers and state for op batching
    این->op_queue = جدید ggml_hexagon_opqueue(این, opt_opbatch, opt_opqueue);

    این->fence_buf = جدید ggml_hexagon_fence_buffer(این, &dev_ctx->fence_buffer_type, 64 * 1024);
    اگر (این->mdev.count > 1) {
        این->mdev_fence_slot = این->alloc_fence(این->mdev.count);
    }

    اگر (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(این);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", این->c_name(), opt_vmem);
    }
    ثابت size_t shm_size = این->op_queue->shm_size();
    این->max_vmem = (opt_vmem > shm_size) ? (opt_vmem - shm_size) : opt_vmem;

    این->op_batch = جدید ggml_hexagon_opbatch(این, opt_opbatch, این->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_start(این->handle, این->session_id, این->queue_id, این->n_threads, opt_nhmx, این->max_vmem, (uint32_t) opt_power_mode);
    اگر (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", این->c_name(), (unsigned) err);
        پرتابکن std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    این->valid_iface = درست;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", این->c_name(), htp_power_mode_to_str(opt_power_mode));

    اگر (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(این->handle, opt_profile, &pmu_conf);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (unsigned) err);
        }
    }
}

تهی ggml_hexagon_session::release() noexcept(درست) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", این->name.c_str());

    این->mdev.sessions.clear();

    صحیح err;

    اگر (این->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(این->handle);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (unsigned) err);
        }
    }

    delete این->op_batch;
    delete این->op_queue;
    برای (auto & it : این->cpy_fence_slots) {
        free_fence((تهی *) it.second, 1);
    }
    این->cpy_fence_slots.clear();

    اگر (این->fence_buf) {
        unclone_buffer(این->fence_buf);
        delete این->fence_buf;
        این->fence_buf = nullptr;
    }
    وقتی (!این->cloned_buffers.empty()) {
        release_buffer(این->cloned_buffers.begin()->second.get());
    }

    اگر (opt_etm) {
        err = htp_iface_etm(این->handle, 0);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (unsigned) err);
        }
    }

    اگر (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(این->handle, 0, &pmu_conf);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (unsigned) err);
        }
    }

    اگر (این->valid_queue) {
        err = dspqueue_close(queue);
        اگر (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (unsigned) err);
        }
    }

    اگر (این->valid_handle) {
        htp_iface_close(این->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(ثابت ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_شمار) noexcept(نادرست) {
    این->dev        = dev;
    این->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    این->mdev.idx   = mdev_idx;
    این->mdev.count = mdev_شمار > 0 ? mdev_شمار : (uint32_t) (1 + config.mdev_group.size());
    op_batch         = nullptr;
    op_queue         = nullptr;
    fence_buf        = nullptr;
    fence_seq        = ((uintptr_t)این) & 0xFFFF;

    تلاش {
        allocate(config);
        اگر (این->mdev.idx == 0 && !config.mdev_group.empty()) {
            برای (size_t i = 0; i < config.mdev_group.size(); i++) {
                این->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_group[i], این->dev, (uint32_t) (i + 1), این->mdev.count));
            }
        }
    } بگیر (ثابت std::exception & exc) {
        release();
        پرتابکن;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(درست) {
    release();
}

// ** backend interface

ایستا منطقی ggml_hexagon_flash_attn_is_hmx_eligible(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * q,
    ثابت ساختار ggml_tensor * k,
    ثابت ساختار ggml_tensor * v,
    ثابت ساختار ggml_tensor * sinks
) {
    اگر (sess->n_hmx == 0) {
        بازگشت نادرست;
    }

    اگر (opt_fa_select < 2) {
        بازگشت نادرست;
    }

    اگر ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        بازگشت نادرست;
    }

    ثابت uint32_t DK = q->ne[0];
    ثابت uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    اگر (DK % 8 != 0 || DV % 8 != 0) {
        بازگشت نادرست;
    }

    GGML_UNUSED(sinks);

    // Explicit force mode
    اگر (opt_fa_select > 2) {
        بازگشت درست;
    }

    ثابت uint32_t M = q->ne[1];

    // Prefill or batched decode
    اگر (M > 1) {
        بازگشت درست;
    }

    ثابت uint32_t n_head = q->ne[2];
    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G = n_kv_heads > 0 ? n_head / n_kv_heads : 1;
    ثابت uint32_t S = k->ne[1];
    ثابت uint32_t total_threads = sess->n_threads * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    اگر (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < total_threads))) {
        بازگشت درست;
    }

    // Tile alignment for 32-row HMX tiles
    ثابت منطقی is_tile_aligned = (G > 0 && (32 % G == 0));
    اگر (!is_tile_aligned) {
        اگر (sess->n_threads >= 6) {
            بازگشت نادرست;
        }
        بازگشت S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    اگر (DK == 256) {
        s_cross = 4096;
    } وگرنه اگر (DK <= 64) {
        s_cross = (sess->n_threads >= 8) ? 2048 : ((sess->n_threads >= 6) ? 768 : 512);
    } وگرنه {
        s_cross = (sess->n_threads >= 8) ? 1024 : 512;
    }

    بازگشت S >= s_cross;
}

ایستا منطقی ggml_hexagon_precompute_flash_attn_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_fa_kernel_params * kparams
) {
    اگر (opt_fa_select < 1) {
        بازگشت نادرست;
    }

    memset(kparams, 0, اندازه(*kparams));

    ثابت ساختار ggml_tensor * q    = op->src[0];
    ثابت ساختار ggml_tensor * k    = op->src[1];
    ثابت ساختار ggml_tensor * v    = op->src[2];
    ثابت ساختار ggml_tensor * mask = op->src[3];

    ثابت uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ثابت uint32_t neq1 = q->ne[1];  // n_tokens
    ثابت uint32_t neq2 = q->ne[2];  // n_heads

    ثابت uint32_t nek1 = k->ne[1];  // kv_len

    ثابت uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ثابت uint32_t DK = neq0;
    ثابت uint32_t DV = nev0;

    ثابت uint32_t n_kv_heads = k->ne[2];
    ثابت uint32_t G          = neq2 / n_kv_heads;

    اعشاری scale         = 1.0f;
    اعشاری max_bias      = 0.0f;
    اعشاری logit_softcap = 0.0f;
    memcpy(&scale,         &op->op_params[0], اندازه(اعشاری));
    memcpy(&max_bias,      &op->op_params[1], اندازه(اعشاری));
    memcpy(&logit_softcap, &op->op_params[2], اندازه(اعشاری));

    اگر (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->head_split = (opt_fa_head_split != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    ثابت uint32_t n_head = q->ne[2];
    kparams->n_head_log2 = 1u << (uint32_t) std::floor(std::log2(n_head));
    kparams->m0 = std::pow(2.0f, -(max_bias) / kparams->n_head_log2);
    kparams->m1 = std::pow(2.0f, -(max_bias / 2.0f) / kparams->n_head_log2);

    // Check HMX eligibility
    ثابت ساختار ggml_tensor * sinks = op->src[4];
    اگر (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        ثابت uint32_t DK_pad = hex_round_up(DK, 64);
        ثابت uint32_t DV_pad = hex_round_up(DV, 64);
        size_t Br = 0, Bc = 0;
        صحیح ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_size, sess->n_threads, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);
        اگر (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            اگر (logit_softcap == 0.0f) {
                kparams->scale = scale * (اعشاری) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } وگرنه {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (اعشاری) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_blocks = (nek1 + Bc - 1) / Bc;
            kparams->n_threads = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? sess->n_threads : 1;

            kparams->u.hmx.g_br = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_blocks >= 3 && sess->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != nullptr, n_head);

            ثابت size_t row_vec_bytes = hex_align_up(Bc * اندازه(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128; // HVX vector is 128 bytes

            ثابت size_t m_سطر_bytes = hex_align_up(Bc * اندازه(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_سطر_bytes / اندازه(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask != nullptr && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_مقادیر(G);
            اگر (mask) {
                kparams->src3_div2 = init_fastdiv_مقادیر(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_مقادیر(mask->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            بازگشت درست;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_blocks = (k->ne[1] + 64 - 1) / 64;
    kparams->n_threads = sess->n_threads;

    ثابت size_t size_q_row_padded = hex_round_up(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    ثابت size_t size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    ثابت size_t size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);

    kparams->vtcm_size = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), mask != nullptr, sinks != nullptr, n_head, sess->n_threads);

    kparams->u.hvx.size_q_row_padded = size_q_row_padded;
    kparams->u.hvx.size_k_row_padded = size_k_row_padded;
    kparams->u.hvx.size_v_row_padded = size_v_row_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_مقادیر(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_مقادیر(q->ne[1]);
    kparams->broadcast_rk2 = init_fastdiv_مقادیر(q->ne[2]/k->ne[2]);
    kparams->broadcast_rk3 = init_fastdiv_مقادیر(q->ne[3]/k->ne[3]);
    kparams->broadcast_rv2 = init_fastdiv_مقادیر(q->ne[2]/v->ne[2]);
    kparams->broadcast_rv3 = init_fastdiv_مقادیر(q->ne[3]/v->ne[3]);
    اگر (mask) {
        kparams->src3_div2 = init_fastdiv_مقادیر(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_مقادیر(mask->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + sess->n_threads - 1) / sess->n_threads;

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_flash_attn_ext(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * src3 = op->src[3];
    ثابت ساختار ggml_tensor * src4 = op->src[4];
    ثابت ساختار ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    اگر ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        بازگشت نادرست;
    }

    اگر (src3 && src3->type != GGML_TYPE_F16) {  // mask
        بازگشت نادرست;
    }

    اگر (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        بازگشت نادرست;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        بازگشت نادرست;
    }

    ساختار htp_fa_kernel_params kparams;
    اگر (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        بازگشت نادرست;
    }

    اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_size, sess->vtcm_size);
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_gated_delta_net(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    اگر (opt_gdn_select < 1) {
        بازگشت نادرست;
    }

    ثابت ساختار ggml_tensor * q     = op->src[0];
    ثابت ساختار ggml_tensor * k     = op->src[1];
    ثابت ساختار ggml_tensor * v     = op->src[2];
    ثابت ساختار ggml_tensor * g     = op->src[3];
    ثابت ساختار ggml_tensor * beta  = op->src[4];
    ثابت ساختار ggml_tensor * state = op->src[5];
    ثابت ساختار ggml_tensor * dst   = op;

    اگر (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    ثابت int64_t S_v      = v->ne[0];
    ثابت int64_t H        = v->ne[1];
    ثابت int64_t n_tokens = v->ne[2];
    ثابت int64_t n_seqs   = v->ne[3];
    ثابت int64_t K        = ggml_get_op_params_i32(op, 0);

    اگر (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        بازگشت نادرست;
    }
    اگر (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        بازگشت نادرست;
    }
    اگر ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        بازگشت نادرست;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    اگر (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        بازگشت نادرست;
    }
    اگر (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        بازگشت نادرست;
    }

    ثابت uint32_t total_rows = (uint32_t) (H * n_seqs);
    ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ثابت منطقی can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    اگر (can_use_hmx) {
        ساختار htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        اگر (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, درست, &n_heads_batch)) {
            بازگشت نادرست;
        }
    } وگرنه {
        ساختار htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_threads);
        اگر (layout.total_bytes > sess->vtcm_size) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_matmul_is_hmx_eligible(
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    منطقی is_matmul_id,
    منطقی is_batched
) {
    GGML_UNUSED(dst);

    اگر (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        بازگشت نادرست;
    }

    ثابت صحیح ne00  = src0->ne[0];
    ثابت صحیح ne01  = src0->ne[1];
    ثابت صحیح ne11  = src1->ne[1];
    ثابت صحیح ne12  = src1->ne[2];
    ثابت صحیح wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    اگر (ne01 % HTP_TENSOR_TILE_N_ROWS != 0 && is_matmul_id) {
        بازگشت نادرست;
    }

    // HMX supports F16, F32, and repack quantized types.
    اگر (!htp_is_hmx_weight_type((ggml_type) wtype)) {
        بازگشت نادرست;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    اگر (ne00 % HTP_TENSOR_TILE_N_COLS != 0) {
        بازگشت نادرست;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    اگر (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        بازگشت نادرست;
    }
    اگر (!is_matmul_id && is_batched && is_tiled) {
        بازگشت نادرست;
    }

    // HMX assumes contiguous row-major layout.
    اگر (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        بازگشت نادرست;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    ثابت صحیح m = is_matmul_id ? ne12 : ne11;
    اگر (m <= HTP_MM_HMX_MIN_NROWS) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(dst);
}

ایستا منطقی ggml_hexagon_precompute_hmx_mm_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * act,
    ثابت ساختار ggml_tensor * dst,
    صحیح wtype,
    صحیح ne00_padded,
    صحیح ne01_padded,
    صحیح ne02,
    صحیح ne11,
    صحیح ne12,
    صحیح ne11_padded,
    منطقی is_matmul_id,
    منطقی is_batched,
    size_t src2_size,
    size_t vtcm_budget,
    ساختار htp_mm_kernel_params * kparams
) {
    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح aligned_tile_size = is_tiled ? htp_tensor_get_tile_padded_size(wtype) : 0;
    ثابت صحیح n_threads = (صحیح)sess->n_threads;
    ثابت صحیح ne10 = act->ne[0];

    صحیح m_for_solver = ne11;
    صحیح m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    اگر (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        ثابت منطقی dst_row_split = dst ? ggml_hexagon_can_row_partition(dst) : درست;
        ثابت منطقی act_row_split = ggml_hexagon_can_row_partition(act);
        اگر (dst_row_split && act_row_split) {
            m_for_solver = (ne11 + (صحیح) sess->mdev.count - 1) / (صحیح) sess->mdev.count;
            m_for_solver_padded = hex_round_up(std::max(m_for_solver, 32), 32);
        }
    }
    ثابت منطقی pipeline = is_matmul_id ? نادرست : htp_mm_hmx_pipeline(m_for_solver);

    ثابت منطقی is_batched_val = is_matmul_id ? نادرست : is_batched;
    ثابت صحیح group_size = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_size = 0;
    منطقی use_grouped = نادرست;
    صحیح act_threads_selected = 0;

    اگر (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && group_size > 1) {
        // Try grouped path first
        اگر (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, group_size, n_threads, pipeline, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            use_grouped = درست;
        }
    }

    اگر (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        صحیح m_id_rows = 0;
        اگر (dst && is_matmul_id) {
            ثابت صحیح n_experts = ne02 > 0 ? ne02 : 1;
            ثابت size_t total_expert_rows = (size_t) dst->ne[1] * dst->ne[2];
            صحیح m_per_expert = (صحیح) ((total_expert_rows + n_experts - 1) / n_experts);
            اگر (sess->mdev.count > 1 && ggml_hexagon_can_row_partition(dst)) {
                m_per_expert = (m_per_expert + (صحیح) sess->mdev.count - 1) / (صحیح) sess->mdev.count;
            }
            m_id_rows = hex_round_up(std::max(m_per_expert, 32), 32);
        }
        ثابت uint32_t cost_m = is_matmul_id ? (uint32_t) m_id_rows : (uint32_t) m_for_solver;
        اگر (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows, ne01_padded, m_for_solver_padded, cost_m, n_threads, pipeline, is_matmul_id, aligned_tile_size, src2_size, vtcm_budget, &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            بازگشت نادرست;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_threads = n_threads;
    kparams->n_act_threads = act_threads_selected;
    kparams->tile_size = is_tiled ? htp_tensor_get_tile_size(wtype) : 0;
    kparams->aligned_tile_size = aligned_tile_size;
    kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
    kparams->vtcm_size = vtcm_size;
    kparams->vtcm_src0_size = 0;
    kparams->div_n_act_threads = init_fastdiv_مقادیر(act_threads_selected);
    kparams->div_ne00_padded   = init_fastdiv_مقادیر(ne00_padded);
    kparams->vtcm_act_size     = 0;
    kparams->vtcm_bias_size    = (int32_t) src2_size;
    kparams->vtcm_dst_size     = 0;

    اگر (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } وگرنه {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    بازگشت درست;
}

ایستا تهی ggml_hexagon_precompute_hvx_mm_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    صحیح wtype,
    منطقی is_tiled,
    صحیح ne02,
    صحیح ne03,
    صحیح ne10,
    صحیح ne11,
    صحیح ne12,
    صحیح ne13,
    منطقی is_matmul_id,
    ثابت size_t src2_row_size,
    size_t vtcm_budget,
    ساختار htp_mm_kernel_params * kparams
) {
    اگر (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    }

    kparams->n_hmx = 0;
    kparams->n_threads = sess->n_threads;

    GGML_UNUSED(ne02);
    GGML_UNUSED(ne03);

    ثابت صحیح src1_nrows = ne11 * ne12 * ne13;

    اگر (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_size = htp_tensor_get_tile_size(wtype);
        kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);

        ثابت منطقی k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        اگر (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (صحیح) sess->n_threads)
                                  ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);

            ساختار htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            برای (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                    0, src0->nb[1], kparams->act_row_size, 0, d, درست, نادرست
                );
                اگر (L.total_bytes <= vtcm_budget) {
                    best_n_prefetch = d;
                    بشکن;
                }
            }
            اگر (L.total_bytes > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                بازگشت;
            }
            kparams->n_prefetch     = best_n_prefetch;
            kparams->vtcm_size      = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = 0;
            kparams->vtcm_dst_size  = L.dst_bytes;
            بازگشت;
        } وگرنه {
            منطقی try_tiled = (k_align && opt_mm_select >= 1);
            اگر (try_tiled) {
                kparams->act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                اگر (src1_nrows < (صحیح) sess->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } وگرنه {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                ساختار htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                برای (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, d, نادرست, نادرست
                    );
                    اگر (L.total_bytes <= vtcm_budget) {
                        best_n_prefetch = d;
                        بشکن;
                    }
                }

                uint32_t m_chunk = 0;
                اگر (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, best_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = best_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_size      = L.total_bytes;
                    kparams->vtcm_src0_size = L.src0_bytes;
                    kparams->vtcm_act_size  = L.act_bytes;
                    kparams->vtcm_bias_size = L.bias_bytes;
                    kparams->vtcm_dst_size  = L.dst_bytes;
                    بازگشت;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            بازگشت;
        }
    } وگرنه اگر (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    } وگرنه اگر (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        ساختار htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            بازگشت;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    } وگرنه اگر (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        ساختار htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            بازگشت;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    } وگرنه اگر (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        ساختار htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        اگر (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_row_size, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_row_size = hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = L.total_bytes;
            kparams->vtcm_src0_size = L.src0_bytes;
            kparams->vtcm_act_size  = L.act_bytes;
            kparams->vtcm_bias_size = L.bias_bytes;
            kparams->vtcm_dst_size = L.dst_bytes;
            kparams->n_prefetch = 16;
            بازگشت;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    } وگرنه {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    }
}

ایستا تهی ggml_hexagon_precompute_matmul_params_impl(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ثابت size_t src2_row_size,
    ثابت size_t src2_size,
    ساختار htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت صحیح ne00 = src0->ne[0];
    ثابت صحیح ne01 = src0->ne[1];
    ثابت صحیح ne02 = src0->ne[2];
    ثابت صحیح ne03 = src0->ne[3];

    ثابت صحیح ne10 = src1->ne[0];
    ثابت صحیح ne11 = src1->ne[1];
    ثابت صحیح ne12 = src1->ne[2];
    ثابت صحیح ne13 = src1->ne[3];

    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحیح ne00_padded = is_tiled ? hex_round_up(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    ثابت صحیح ne11_padded = hex_round_up(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    ثابت صحیح ne01_tiled  = hex_round_up(ne01, 32);

    ثابت منطقی is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقی is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    ثابت size_t vtcm_budget = sess->vtcm_size;

    // Check HMX eligibility and try precomputing HMX parameters
    منطقی hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    اگر (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        اگر (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_size, vtcm_budget, kparams)) {
            برو finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_row_size, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_مقادیر(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_مقادیر(ne11);
    kparams->div_r2       = init_fastdiv_مقادیر(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_مقادیر(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_مقادیر(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
ایستا منطقی ggml_hexagon_rows_stride(ثابت int64_t * ne, ثابت size_t * nb, size_t * stride) {
    size_t s = 0, next = 0;
    برای (صحیح i = 1; i < GGML_MAX_DIMS; i++) {
        اگر (ne[i] == 1) ادامه;
        اگر (s == 0) { s = nb[i]; next = s * ne[i]; ادامه; }
        اگر (nb[i] != next) بازگشت نادرست;
        next *= ne[i];
    }
    *stride = s ? s : nb[1];
    بازگشت درست;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
ایستا منطقی ggml_hexagon_matmul_can_collapse(ثابت ساختار ggml_tensor * src0, ثابت ساختار ggml_tensor * src1, ثابت ساختار ggml_tensor * dst) {
    size_t s1, sd;
    بازگشت (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_ADD) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_size(src1->type) && ggml_hexagon_rows_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_size(dst->type) && ggml_hexagon_rows_stride(dst->ne, dst->nb, &sd);
}

ایستا منطقی ggml_hexagon_matmul_add_can_collapse(
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * src2,
    ثابت ساختار ggml_tensor * dst
) {
    اگر (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        بازگشت نادرست;
    }
    اگر (!src2) {
        بازگشت درست;
    }
    ثابت int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    اگر (src2_nrows == 1) {
        بازگشت درست;
    }
    size_t s2;
    بازگشت src2->nb[0] == ggml_type_size(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_rows_stride(src2->ne, src2->nb, &s2);
}

ایستا ggml_tensor ggml_hexagon_tensor_collapse_rows(ثابت ساختار ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_rows_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    بازگشت c;
}

ایستا تهی ggml_hexagon_precompute_matmul_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_mm_kernel_params * kparams
) {
    اگر (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        بازگشت;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

ایستا تهی ggml_hexagon_precompute_fused_matmul_add_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * src2,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_mm_kernel_params * kparams
) {
    اگر (ggml_hexagon_matmul_add_can_collapse(src0, src1, src2, dst)) {
        ثابت ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_rows(src1);
        ثابت ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_rows(dst);
        ثابت ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_rows(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        ثابت ساختار ggml_tensor * p_src2 = src2 ? &src2_collapsed : nullptr;
        ثابت size_t src2_size = p_src2 ? hex_round_up(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_size, kparams);
        kparams->collapse = 1;
        بازگشت;
    }
    ثابت size_t src2_size = src2 ? hex_round_up(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_size, kparams);
}

ایستا منطقی ggml_hexagon_precompute_دودویی_params(
    ثابت ساختار ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_دودویی_kernel_params * kparams,
    ثابت ساختار ggml_tensor * src2
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت size_t elem_size = ggml_type_size(src0->type);
    ثابت size_t src0_row_size = src0->ne[0] * elem_size;
    ثابت size_t src1_row_size = src1->ne[0] * elem_size;
    ثابت size_t dst_row_size  = dst->ne[0]  * elem_size;

    ثابت size_t src0_row_size_aligned = hex_round_up(src0_row_size, 128);
    ثابت size_t src1_row_size_aligned = hex_round_up(src1_row_size, 128);
    ثابت size_t dst_row_size_aligned  = hex_round_up(dst_row_size,  128);

    اگر (!htp_دودویی_strides_safe(src0, src1, dst, elem_size)) {
        بازگشت نادرست;
    }

    ثابت منطقی is_add_id = op == HTP_OP_ADD_ID;
    ثابت منطقی is_scalar = !is_add_id && src1->ne[0] == 1;
    ثابت منطقی is_row_bcast = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    ثابت منطقی is_same_shape = !is_add_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    ثابت منطقی is_complex   = !is_add_id && !is_scalar && !is_same_shape && !is_row_bcast && (src1->ne[0] == src0->ne[0]);
    ثابت منطقی is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    ثابت منطقی is_scalar_broadcast = !is_add_id && (ggml_nelements(src1) == 1);

    اگر (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
        ثابت uint32_t total_elems = (uint32_t) ggml_nelements(src0);
        ثابت uint32_t n_threads = sess->n_threads;
        ثابت uint32_t max_chunk_elems = 32768 / elem_size;
        ثابت uint32_t min_chunk_elems = 256;
        ثابت uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
        ثابت uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
        ثابت uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

        kparams->kernel_type           = HTP_دودویی_KERNEL_CHUNKED;
        kparams->n_threads             = n_threads;
        kparams->rows_per_buffer       = 1;
        kparams->src0_row_size_aligned = src0_row_size_aligned;
        kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
        kparams->dst_row_size_aligned  = dst_row_size_aligned;
        kparams->src1_size             = 0;
        kparams->chunk_size            = chunk_size;
        kparams->chunk_bytes           = chunk_bytes;
        kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

        ساختار htp_دودویی_vtcm_layout L;
        htp_دودویی_vtcm_layout_build(&L, kparams, sess->vtcm_size);
        اگر (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
            بازگشت نادرست;
        }

        kparams->vtcm_size = L.total_bytes;
        بازگشت درست;
    }

    شمارش htp_دودویی_kernel_type kernel_type;
    size_t src1_size = 0;
    size_t src2_size = 0;

    اگر (is_add_id) {
        GGML_ASSERT(src2 != nullptr);
        kernel_type = HTP_دودویی_KERNEL_ADD_ID;
        src1_size = hex_round_up(src1->ne[1] * src1_row_size_aligned, 128);
        src2_size = hex_round_up(ggml_nelements(src2) * اندازه(int32_t), 128);
        اگر (src1_size + src2_size > sess->vtcm_size) {
            بازگشت نادرست;
        }
    } وگرنه اگر (is_row_bcast) {
        kernel_type = HTP_دودویی_KERNEL_ROW_BCAST;
        src1_size = src1_row_size_aligned;
    } وگرنه اگر (is_scalar) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size, 128);
        اگر (src1_size > sess->vtcm_size) {
            بازگشت نادرست;
        }
        kernel_type = HTP_دودویی_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } وگرنه اگر (is_same_shape) {
        kernel_type = HTP_دودویی_KERNEL_SAME_SHAPE;
    } وگرنه اگر (is_complex) {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        اگر (src1_size > sess->vtcm_size) {
            بازگشت نادرست;
        }
        kernel_type = HTP_دودویی_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } وگرنه {
        src1_size = hex_round_up(ggml_nelements(src1) * elem_size + 128, 128);
        اگر (src1_size > sess->vtcm_size) {
            بازگشت نادرست;
        }
        kernel_type = HTP_دودویی_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_threads = sess->n_threads;
    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;
    kparams->src1_size             = src1_size;
    kparams->src2_size             = src2_size;

    ثابت منطقی src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    ثابت منطقی src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    ثابت منطقی dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    ثابت منطقی src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    ثابت منطقی src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    ثابت منطقی dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    منطقی split_at_ne01 = نادرست;
    اگر (src0->ne[2] > 1) {
        اگر (kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE) {
            split_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } وگرنه اگر (kernel_type == HTP_دودویی_KERNEL_ROW_BCAST) {
            split_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } وگرنه {
            split_at_ne01 = درست;
        }
    }
    kparams->split_at_ne01 = split_at_ne01 ? 1 : 0;

    منطقی split_at_ne02 = نادرست;
    اگر (src0->ne[3] > 1) {
        اگر (kernel_type == HTP_دودویی_KERNEL_SAME_SHAPE) {
            split_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } وگرنه اگر (kernel_type == HTP_دودویی_KERNEL_ROW_BCAST) {
            split_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } وگرنه {
            split_at_ne02 = درست;
        }
    }
    kparams->split_at_ne02 = split_at_ne02 ? 1 : 0;

    ساختار htp_دودویی_vtcm_layout L;
    htp_دودویی_vtcm_layout_build(&L, kparams, sess->vtcm_size);
    اگر (L.rows_per_buffer == 0 || L.total_bytes > sess->vtcm_size) {
        اگر (!is_add_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_broadcast)) {
            ثابت uint32_t total_elems = (uint32_t) ggml_nelements(src0);
            ثابت uint32_t n_threads = sess->n_threads;
            ثابت uint32_t max_chunk_elems = 32768 / elem_size;
            ثابت uint32_t min_chunk_elems = 256;
            ثابت uint32_t target_chunk_elems = hex_round_up((total_elems + (2 * n_threads) - 1) / (2 * n_threads), 32);
            ثابت uint32_t chunk_size = (std::min)(max_chunk_elems, (std::max)(target_chunk_elems, min_chunk_elems));
            ثابت uint32_t chunk_bytes = hex_round_up(chunk_size * elem_size, 128);

            kparams->kernel_type           = HTP_دودویی_KERNEL_CHUNKED;
            kparams->n_threads             = n_threads;
            kparams->rows_per_buffer       = 1;
            kparams->src1_row_size_aligned = is_scalar_broadcast ? 0 : src1_row_size_aligned;
            kparams->src1_size             = 0;
            kparams->chunk_size            = chunk_size;
            kparams->chunk_bytes           = chunk_bytes;
            kparams->is_scalar             = is_scalar_broadcast ? 1 : 0;

            htp_دودویی_vtcm_layout_build(&L, kparams, sess->vtcm_size);
            اگر (L.total_bytes == 0 || L.total_bytes > sess->vtcm_size) {
                بازگشت نادرست;
            }

            kparams->vtcm_size = L.total_bytes;
            بازگشت درست;
        }
        بازگشت نادرست;
    }

    kparams->rows_per_buffer = L.rows_per_buffer;
    kparams->vtcm_size       = L.total_bytes;

    بازگشت درست;
}

ایستا تهی ggml_hexagon_precompute_unary_params(
    ثابت ساختار ggml_hexagon_session * sess,
    uint32_t op,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads  = (std::min)((uint32_t)sess->n_threads, src0_nrows);

    kparams->n_threads = n_threads;

    ثابت size_t elem_size = ggml_type_size(src0->type);

    ثابت size_t src0_data_row_size = src0->ne[0] * elem_size;
    ثابت size_t dst_data_row_size  = dst->ne[0]  * ggml_type_size(dst->type);

    ثابت size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ثابت size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    kparams->src0_row_size_aligned = src0_row_size_aligned;
    kparams->dst_row_size_aligned  = dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    منطقی broadcast_weight = نادرست;

    اگر (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = src1->ne[0] * ggml_type_size(src1->type);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight;

    ساختار htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                broadcast_weight, n_threads, sess->vtcm_size, elem_size,
                                &col_tile, &vtcm_row_per_thread);

    kparams->col_tile = col_tile;
    kparams->vtcm_row_per_thread = vtcm_row_per_thread;
    kparams->vtcm_size = L.total_bytes;

    kparams->vtcm_src0_size_per_thread = L.src0_bytes;
    kparams->vtcm_src1_size_per_thread = L.src1_bytes;
    kparams->vtcm_dst_size_per_thread  = L.dst_bytes;

    kparams->vtcm_src0_size = L.src0_bytes * n_threads;
    kparams->vtcm_src1_size = L.src1_bytes * n_threads;
    kparams->vtcm_dst_size  = L.dst_bytes * n_threads;

    kparams->block = col_tile ? 0 : ((L.src0_bytes / 2) / src0_row_size_aligned);

    ثابت uint32_t tiles_per_row = col_tile > 0 ? (src0->ne[0] + col_tile - 1) / col_tile : 1;
    kparams->div_ne01  = init_fastdiv_مقادیر(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_مقادیر(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_مقادیر(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_مقادیر(tiles_per_row);
}

ایستا تهی ggml_hexagon_precompute_get_rows_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_get_rows_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne02 = src0->ne[2];
    ثابت uint32_t ne03 = src0->ne[3];

    ثابت uint32_t ne10 = src1->ne[0];
    ثابت uint32_t ne11 = src1->ne[1];
    ثابت uint32_t ne12 = src1->ne[2];
    ثابت uint32_t nr = ne10 * ne11 * ne12;

    ثابت منطقی tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    اگر (tiled) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_TILED;
    } وگرنه اگر (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_SAMETYPE;
    } وگرنه {
        kparams->kernel_type = HTP_GET_ROWS_KERNEL_FLAT;
    }

    ثابت uint32_t total_tasks = nr;

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, total_tasks);

    ساختار htp_get_rows_vtcm_layout vtcm_layout = {};
    وقتی (kparams->n_threads > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, kparams->n_threads);
        اگر (vtcm_layout.total_bytes <= sess->vtcm_size) {
            بشکن;
        }
        --kparams->n_threads;
    }

    اگر (kparams->n_threads == 0 && total_tasks > 0) {
        htp_get_rows_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, total_tasks, src1->type, 1);
    }

    kparams->vtcm_size = (total_tasks == 0) ? 0 : vtcm_layout.total_bytes;
    kparams->tasks_per_thread = kparams->n_threads > 0 ? (total_tasks + kparams->n_threads - 1) / kparams->n_threads : 0;
    kparams->total_tasks = total_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_مقادیر(ne10) : fastdiv_مقادیر{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_مقادیر(ne10 * ne11) : fastdiv_مقادیر{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_مقادیر(ne02) : fastdiv_مقادیر{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_مقادیر(ne03) : fastdiv_مقادیر{0, 0};
}

ایستا تهی ggml_hexagon_precompute_set_rows_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0, //مقادیر
    ثابت ساختار ggml_tensor * src1, // indices
    ثابت ساختار ggml_tensor * dst,  // destination
    ساختار htp_set_rows_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت uint32_t nr = src0->ne[1];

    kparams->n_threads = (std::min)((uint32_t)sess->n_threads, nr);
    kparams->tasks_per_thread = (nr + kparams->n_threads - 1) / kparams->n_threads;
    kparams->total_tasks = nr;

    kparams->div_ne11 = init_fastdiv_مقادیر(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_مقادیر(src1->ne[2]);
    kparams->div_tasks_per_thread = init_fastdiv_مقادیر(kparams->tasks_per_thread);
    kparams->div_ne02 = init_fastdiv_مقادیر(src0->ne[2]);

    ساختار htp_set_rows_vtcm_layout vtcm_layout;
    htp_set_rows_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_threads);
    kparams->vtcm_size = vtcm_layout.total_bytes;
}

ایستا تهی ggml_hexagon_precompute_softmax_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);

    اعشاری scale = 1.0f;
    اعشاری max_bias = 0.0f;
    memcpy(&scale,    &op->op_params[0], اندازه(اعشاری));
    memcpy(&max_bias, &op->op_params[1], اندازه(اعشاری));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    ثابت uint32_t n_head = src0->ne[2];
    ثابت uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    kparams->n_head      = n_head;
    kparams->n_head_log2 = n_head_log2;

    اگر (max_bias > 0.0f && n_head_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_head_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    } وگرنه {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != nullptr) ? 1 : 0;
    kparams->use_f16  = (src1 != nullptr && src1->type == GGML_TYPE_F16) ? 1 : 0;

    ثابت uint32_t ne00 = src0->ne[0];
    ثابت uint32_t ne10 = src1 ? src1->ne[0] : 1;

    ساختار htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_threads);

    kparams->n_threads                 = n_threads;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_per_thread     = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size                 = (uint32_t) layout.total_bytes;
    kparams->vtcm_src0_size_per_thread = (uint32_t) layout.src0_bytes_per_thread;
    kparams->vtcm_src1_size_per_thread = (uint32_t) layout.src1_bytes_per_thread;
    kparams->vtcm_dst_size_per_thread  = (uint32_t) layout.dst_bytes_per_thread;
    kparams->src0_row_size_aligned     = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_row_size_aligned     = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_row_size_aligned      = (uint32_t) layout.dst_spad_half_size;
    kparams->src0_spad_half_size       = (uint32_t) layout.src0_spad_half_size;
    kparams->src1_spad_half_size       = (uint32_t) layout.src1_spad_half_size;
    kparams->dst_spad_half_size        = (uint32_t) layout.dst_spad_half_size;
    اگر (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } وگرنه اگر (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F16;
    } وگرنه {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_MASK_F32;
    }

    اگر (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_مقادیر(src0->ne[1]);
    اگر (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_مقادیر(src0->ne[2]);
    ثابت uint32_t ne12 = src1 ? src1->ne[2] : 1;
    ثابت uint32_t ne13 = src1 ? src1->ne[3] : 1;
    اگر (ne12 > 0) kparams->div_ne12 = init_fastdiv_مقادیر(ne12);
    اگر (ne13 > 0) kparams->div_ne13 = init_fastdiv_مقادیر(ne13);
}

ایستا تهی ggml_hexagon_precompute_rope_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * dst  = op;

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ساختار htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);

    kparams->n_threads              = n_threads;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_per_thread  = (src0_nrows + n_threads - 1) / n_threads;
    kparams->vtcm_size              = (uint32_t) layout.total_bytes;
    kparams->spad_per_thread        = (uint32_t) layout.bytes_per_thread;
    kparams->theta_cache_offset     = (uint32_t) layout.theta_cache_size_aligned;
    kparams->src0_row_size_aligned  = (uint32_t) layout.src0_row_size_aligned;
    kparams->freq_factors_offset    = (uint32_t) (layout.bytes_per_thread * n_threads);
    kparams->freq_factors_size      = (uint32_t) layout.freq_factors_size_aligned;

    اگر (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_مقادیر(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_مقادیر(dst->ne[1]);
    }
}

ایستا تهی ggml_hexagon_precompute_ssm_conv_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * src1,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت uint32_t d_conv  = (uint32_t) src1->ne[0];
    ثابت uint32_t d_inner = (uint32_t) src0->ne[1];
    ثابت uint32_t n_t     = (uint32_t) dst->ne[1];
    ثابت uint32_t n_s     = (uint32_t) dst->ne[2];
    ثابت uint32_t ncs     = (uint32_t) src0->ne[0];

    ثابت uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, (d_inner + 31) / 32);

    kparams->n_threads = n_threads;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    ثابت uint32_t raw_rpt = (d_inner + n_threads - 1) / n_threads;
    ثابت uint32_t d_inner_per_thread = hex_round_up(raw_rpt, 32);

    ثابت uint32_t src1_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * اندازه(اعشاری), 128) + 128;
    ثابت uint32_t src1_T_bytes   = hex_round_up(d_conv * d_inner_per_thread * اندازه(اعشاری), 128);
    ثابت uint32_t vtcm_src1_per_thread = src1_raw_bytes + src1_T_bytes;

    uint32_t vtcm_src0_per_thread = 0;
    uint32_t vtcm_dst_per_thread  = 0;

    اگر (n_t == 1) {
        kparams->d_inner_tile = d_inner_per_thread;

        ثابت uint32_t src0_tile_raw_bytes = hex_round_up(d_inner_per_thread * d_conv * اندازه(اعشاری), 128);
        ثابت uint32_t src0_T_bytes        = hex_round_up(d_conv * d_inner_per_thread * اندازه(اعشاری), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw_bytes + src0_T_bytes;

        ثابت uint32_t dst_tile_bytes = hex_round_up(d_inner_per_thread * اندازه(اعشاری), 128);
        vtcm_dst_per_thread = 2 * dst_tile_bytes;
    } وگرنه {
        ثابت size_t vtcm_budget = (sess->vtcm_size > 0 ? sess->vtcm_size / n_threads : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        ثابت uint32_t src0_block_T    = hex_round_up(ncs * 32 * اندازه(اعشاری), 128);
        ثابت size_t   fixed_bytes     = vtcm_src1_per_thread + src0_block_T;
        ثابت size_t   avail_for_tiles = vtcm_budget > fixed_bytes ? vtcm_budget - fixed_bytes : (128 * 1024);

        uint32_t target_max_tile = hex_round_up((d_inner_per_thread + 3) / 4, 32);
        target_max_tile = (std::max)(target_max_tile, 32u);
        target_max_tile = (std::min)(target_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * اندازه(اعشاری)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        اگر (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        اگر (d_inner_tile > target_max_tile) {
            d_inner_tile = target_max_tile;
        }
        اگر (d_inner_tile > d_inner_per_thread) {
            d_inner_tile = d_inner_per_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        ثابت uint32_t src0_tile_raw = hex_round_up(d_inner_tile * ncs * اندازه(اعشاری), 128);
        vtcm_src0_per_thread = 2 * src0_tile_raw + src0_block_T;

        vtcm_dst_per_thread = 2 * hex_round_up(d_inner_tile * n_t * اندازه(اعشاری), 128);
    }

    kparams->vtcm_src0_size_per_thread = vtcm_src0_per_thread;
    kparams->vtcm_src1_size_per_thread = vtcm_src1_per_thread;
    kparams->vtcm_dst_size_per_thread  = vtcm_dst_per_thread;

    kparams->vtcm_src0_size = vtcm_src0_per_thread * n_threads;
    kparams->vtcm_src1_size = vtcm_src1_per_thread * n_threads;
    kparams->vtcm_dst_size  = vtcm_dst_per_thread  * n_threads;
    kparams->vtcm_size      = kparams->vtcm_src0_size + kparams->vtcm_src1_size + kparams->vtcm_dst_size;
}

ایستا تهی ggml_hexagon_precompute_gated_delta_net_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت ساختار ggml_tensor * q     = op->src[0];
    ثابت ساختار ggml_tensor * k     = op->src[1];
    ثابت ساختار ggml_tensor * v     = op->src[2];
    ثابت ساختار ggml_tensor * g     = op->src[3];
    ثابت ساختار ggml_tensor * state = op->src[5];

    ثابت uint32_t S_v      = (uint32_t) v->ne[0];
    ثابت uint32_t H        = (uint32_t) v->ne[1];
    ثابت uint32_t n_tokens = (uint32_t) v->ne[2];
    ثابت uint32_t n_seqs   = (uint32_t) v->ne[3];
    ثابت uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    ثابت uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    ثابت uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    ثابت uint32_t total_rows = H * n_seqs;
    ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, total_rows);

    ثابت منطقی can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_tokens >= HTP_GDN_MIN_TOKENS) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ساختار htp_gdn_hmx_vtcm_layout hmx_layout;
    ساختار htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    اگر (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_SIZE, total_rows, sess->vtcm_size, n_threads, درست, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_size      = HTP_GDN_CHUNK_SIZE;
        kparams->n_chunks        = (n_tokens + HTP_GDN_CHUNK_SIZE - 1) / HTP_GDN_CHUNK_SIZE;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_size       = (uint32_t) hmx_layout.total_bytes;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_bytes;
        kparams->vtcm_per_thread = (uint32_t) (hmx_layout.total_bytes / (n_threads > 0 ? n_threads : 1));
    } وگرنه {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_threads);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_per_thread = (uint32_t) hvx_layout.bytes_per_thread;
        kparams->vtcm_size       = (uint32_t) hvx_layout.total_bytes;
    }

    kparams->n_threads           = n_threads;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_tokens            = n_tokens;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->total_rows          = total_rows;
    kparams->rows_per_thread     = (total_rows + kparams->n_threads - 1) / kparams->n_threads;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / اندازه(اعشاری));
    kparams->state_size_per_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((اعشاری) S_v);

    اگر (H > 0)                  kparams->div_H         = init_fastdiv_مقادیر(H);
    اگر (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_مقادیر((uint32_t) q->ne[1]);
    اگر (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_مقادیر((uint32_t) k->ne[1]);
    اگر (rq3 > 0)                kparams->div_rq3       = init_fastdiv_مقادیر(rq3);
    اگر (rk3 > 0)                kparams->div_rk3       = init_fastdiv_مقادیر(rk3);
    اگر (kparams->n_threads > 0) kparams->div_n_threads = init_fastdiv_مقادیر(kparams->n_threads);
}

ایستا تهی ggml_hexagon_precompute_sort_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    منطقی is_بالاترین_k,
    ساختار htp_sort_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    ثابت uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t ne00       = src0->ne[0];
    ثابت uint32_t k          = dst->ne[0];

    int32_t order = GGML_SORT_ORDER_DESC;
    اگر (!is_بالاترین_k) {
        order = ((ثابت int32_t *) op->op_params)[0];
    }

    ثابت uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ثابت size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ساختار htp_sort_vtcm_layout layout;
    منطقی ok = htp_sort_solve_layout(&layout, ne00, total_rows, k, n_threads_max, vtcm_budget, is_بالاترین_k);
    GGML_ASSERT(ok);

    kparams->n_threads         = (int32_t) layout.n_threads;
    kparams->total_rows        = (int32_t) total_rows;
    kparams->row_start         = 0;
    kparams->row_end           = (int32_t) total_rows;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->order             = order;
    kparams->is_بالاترین_k          = is_بالاترین_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_size         = (int32_t) layout.total_bytes;
    kparams->phase1_slot_size  = (int32_t) layout.phase1_slot_size;
    kparams->merge_مقادیر_off  = (int32_t) layout.merge_مقادیر_off;
    kparams->merge_indices_off = (int32_t) layout.merge_indices_off;
    kparams->merge_elems       = (int32_t) layout.merge_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

ایستا تهی ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    ثابت uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    ثابت int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    اگر (numer_hi < 0) {
        *ox_hi = 0;
    } وگرنه {
        ثابت uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    اگر (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

ایستا منطقی ggml_hexagon_pool_shape_fits(
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * dst
) {
    بازگشت ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

ایستا تهی ggml_hexagon_precompute_pool_2d_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0,
    ثابت ساختار ggml_tensor * dst,
    ساختار htp_pool_2d_kernel_params * kparams,
    منطقی is_pool_1d
) {
    memset(kparams, 0, اندازه(*kparams));

    ثابت uint32_t src_x = (uint32_t) src0->ne[0];
    ثابت uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    ثابت uint32_t dst_x = (uint32_t) dst->ne[0];
    ثابت uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    ثابت uint32_t src_plane_bytes = src_x * src_y * اندازه(اعشاری);
    ثابت uint32_t dst_plane_bytes = dst_x * dst_y * اندازه(اعشاری);
    ثابت uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    ثابت uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ساختار htp_pool_vtcm_layout layout;
    ثابت منطقی ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_threads, sess->vtcm_size);
    GGML_ASSERT(ok);

    kparams->src_x = src_x;
    kparams->src_y = src_y;
    kparams->dst_x = dst_x;
    kparams->dst_y = dst_y;
    kparams->kernel_x = (uint32_t) ggml_get_op_params_i32(dst, 1);
    kparams->kernel_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 2);
    kparams->stride_x = (uint32_t) ggml_get_op_params_i32(dst, is_pool_1d ? 2 : 3);
    kparams->stride_y = is_pool_1d ? 1 : (uint32_t) ggml_get_op_params_i32(dst, 4);
    kparams->pad_x = ggml_get_op_params_i32(dst, is_pool_1d ? 3 : 5);
    kparams->pad_y = is_pool_1d ? 0 : ggml_get_op_params_i32(dst, 6);
    kparams->src_plane_bytes = src_plane_bytes;
    kparams->dst_plane_bytes = dst_plane_bytes;
    kparams->src_plane_bytes_aligned = (uint32_t) layout.src_spad_half_size;
    kparams->dst_plane_bytes_aligned = (uint32_t) layout.dst_spad_half_size;
    kparams->n_threads = n_threads;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->block_path = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_شمار = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    ثابت منطقی narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_path = (kparams->fast_path && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (اعشاری) (kparams->kernel_x * kparams->kernel_y);
}

ایستا منطقی ggml_hexagon_precompute_concat_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        بازگشت نادرست;
    }

    صحیح dim = ((ثابت int32_t *) op->op_params)[0];
    اگر (dim < 0 || dim >= GGML_MAX_DIMS) {
        بازگشت نادرست;
    }
    kparams->dim = dim;

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }
    اگر (src0->type != dst->type || src1->type != dst->type) {
        بازگشت نادرست;
    }

    ثابت uint32_t type_size = ggml_type_size(dst->type);

    برای (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
        ثابت int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        اگر (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            بازگشت نادرست;
        }
    }

    ثابت uint32_t max_gather_ne0 = 128 / type_size;
    ثابت uint32_t total_rows = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    ثابت منطقی b_dense = (src1->nb[0] == type_size);
    ثابت منطقی gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_size &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_size
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_size);

    اگر (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_size && dst->nb[1] == (size_t) dst->ne[0] * type_size &&
        src0->nb[0] == type_size &&
        (src1->nb[0] == type_size || src1->nb[1] == type_size) &&
        gather_strides_ok &&
        htp_gather_rows_tensor_outer_contiguous(dst) &&
        htp_gather_rows_tensor_outer_contiguous(src0) &&
        htp_gather_rows_tensor_outer_contiguous(src1)) {

        اگر (htp_gather_rows_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_size, total_rows,
                                         sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_ROWS;
            kparams->n_threads   = (uint8_t) sess->n_threads;
            kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
            بازگشت درست;
        }
    }

    ثابت منطقی dma_strides_ok = (src0->nb[0] == type_size && src1->nb[0] == type_size && dst->nb[0] == type_size);

    اگر (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_threads   = 1;
        بازگشت درست;
    }

    ثابت منطقی is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    ثابت منطقی is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    ثابت منطقی transposed_rows_ok = (src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size);

    اگر (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_rows_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        ثابت uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 8;
        ساختار htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_size, n_threads);

        اگر (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
            بازگشت نادرست;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_threads                         = n_threads;
        kparams->vtcm_size                         = layout.total_bytes;
        kparams->u.transposed.spad0_size_per_thread = layout.src0_spad_size_per_thread;
        kparams->u.transposed.spad1_size_per_thread = layout.src1_spad_size_per_thread;
        بازگشت درست;
    }

    بازگشت نادرست;
}

ایستا منطقی ggml_hexagon_precompute_cpy_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!src0 || !dst) {
        بازگشت نادرست;
    }

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }
    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    ثابت int64_t nelem_src = ggml_nelements(src0);
    ثابت int64_t nelem_dst = ggml_nelements(dst);
    اگر (nelem_src != nelem_dst || nelem_src < 0) {
        بازگشت نادرست;
    }

    ثابت uint32_t src_type_size = ggml_type_size(src0->type);
    ثابت uint32_t dst_type_size = ggml_type_size(dst->type);

    kparams->src0_type_size = (uint8_t) src_type_size;
    kparams->dst_type_size  = (uint8_t) dst_type_size;
    kparams->total_elems    = (uint32_t) nelem_src;

    اگر (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        بازگشت درست;
    }

    اگر (nelem_src == 1) {
        اگر (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            بازگشت درست;
        }
        اگر ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            بازگشت درست;
        }
        بازگشت نادرست;
    }

    ثابت منطقی sametype = (src0->type == dst->type);

    منطقی same_extents = درست;
    برای (صحیح d = 0; d < GGML_MAX_DIMS; d++) {
        اگر (src0->ne[d] != dst->ne[d]) {
            same_extents = نادرست;
            بشکن;
        }
    }

    ثابت منطقی transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_size) || (dst->nb[0] != dst_type_size) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_size) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_size);
    ثابت منطقی sameshape  = same_extents && !transposed;

    ثابت منطقی src_is_contiguous = ggml_is_contiguous(src0);
    ثابت منطقی dst_is_contiguous = ggml_is_contiguous(dst);

    اگر (sametype) {
        اگر (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            بازگشت درست;
        }

        ثابت uint32_t max_gather_ne0 = 128 / src_type_size;
        ثابت uint32_t total_rows = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        اگر ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_size && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_size &&
            total_rows > 1 &&
            htp_gather_rows_tensor_outer_contiguous(src0)) {

            اگر (htp_gather_rows_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, نادرست,
                                             src_type_size, total_rows,
                                             sess->n_threads, sess->mdev.count, sess->vtcm_size)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_ROWS;
                kparams->n_threads   = (uint8_t) sess->n_threads;
                kparams->vtcm_size   = kparams->u.gather.slice * sess->n_threads;
                بازگشت درست;
            }
        }

        اگر (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->total_rows  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            بازگشت درست;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_threads   = sess->n_threads > 0 ? (uint8_t) sess->n_threads : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_مقادیر((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_مقادیر((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_مقادیر((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_مقادیر((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_مقادیر((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_مقادیر((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        بازگشت درست;
    }

    اگر (!sameshape) {
        بازگشت نادرست;
    }

    ثابت منطقی valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    اگر (!valid_conversion) {
        بازگشت نادرست;
    }

    ثابت uint32_t n_threads = sess->n_threads > 0 ? (uint32_t) sess->n_threads : 4;
    ساختار htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_size, dst_type_size, n_threads);

    اگر (sess->vtcm_size > 0 && layout.total_bytes > sess->vtcm_size) {
        بازگشت نادرست;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->total_rows              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_threads               = (uint8_t) n_threads;
    kparams->vtcm_size               = layout.total_bytes;
    kparams->u.convert.src0_buf_size = layout.src0_buf_size;
    kparams->u.convert.dst_buf_size  = layout.dst_buf_size;
    kparams->u.convert.spad0_size_per_thread = layout.spad0_size_per_thread;
    kparams->u.convert.spad1_size_per_thread = layout.spad1_size_per_thread;
    kparams->u.convert.src0_row_stride = layout.src0_row_stride;
    kparams->u.convert.dst_row_stride  = layout.dst_row_stride;
    kparams->u.convert.blk_rows        = layout.blk_rows;
    kparams->u.convert.div_ne01      = init_fastdiv_مقادیر((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_مقادیر((uint32_t) (src0->ne[2] * src0->ne[1]));

    بازگشت درست;
}

ایستا تهی ggml_hexagon_precompute_fused_mmnx_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0, // W0
    ثابت ساختار ggml_tensor * act,  // x
    int32_t n_weights,
    ساختار htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->n_threads = sess->n_threads;

    ثابت صحیح ne00 = src0->ne[0];
    ثابت صحیح ne01 = src0->ne[1];
    ثابت صحیح ne02 = src0->ne[2];
    ثابت صحیح ne03 = src0->ne[3];

    ثابت صحیح ne10 = act->ne[0];
    ثابت صحیح ne11 = act->ne[1];
    ثابت صحیح ne12 = act->ne[2];
    ثابت صحیح ne13 = act->ne[3];

    ثابت منطقی is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ثابت صحیح wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    ثابت صحیح ne00_padded = is_tiled ? hex_round_up(ne00, 32) : ne00;
    ثابت صحیح ne11_padded = hex_round_up(ne11, 32);
    ثابت صحیح ne01_tiled  = hex_round_up(ne01, 32);

    ثابت size_t vtcm_budget = sess->vtcm_size;
    ثابت منطقی is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    منطقی hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    اگر (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, nullptr, نادرست, is_batched)) {
        اگر (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, nullptr, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, نادرست, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            برو finalize;
        }
    }

    اگر (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        بازگشت;
    }

    {
        ثابت صحیح act_nrows = ne11 * ne12 * ne13;
        ثابت size_t act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        ثابت size_t src0_row_size = src0->nb[1];

        ثابت uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t best_n_prefetch = 2;
        برای (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            ساختار htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
                0, src0_row_size, act_row_size, 0, d, نادرست, درست
            );
            اگر (L.total_bytes <= sess->vtcm_size) {
                best_n_prefetch = d;
                بشکن;
            }
        }

        ساختار htp_mm_hvx_vtcm_layout L;
        منطقی try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_ROW, wtype, ne10, act_nrows, sess->n_threads,
            0, src0_row_size, act_row_size, 0, best_n_prefetch, نادرست, درست
        );

        اگر (try_tiled && L.total_bytes <= sess->vtcm_size) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->tile_size         = htp_tensor_get_tile_size(wtype);
            kparams->aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
            kparams->act_row_size      = act_row_size;
            kparams->vtcm_src0_size    = L.src0_bytes;
            kparams->vtcm_act_size     = L.act_bytes;
            kparams->vtcm_bias_size    = 0;
            kparams->vtcm_dst_size     = L.dst_bytes;
            kparams->vtcm_size         = L.total_bytes;
            kparams->n_prefetch        = best_n_prefetch;
            kparams->n_weights         = n_weights;
        } وگرنه {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            بازگشت;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_مقادیر(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_مقادیر(ne11);
    kparams->div_r2       = init_fastdiv_مقادیر(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_مقادیر(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_مقادیر(ne12);
}

ایستا تهی ggml_hexagon_precompute_fused_mmidnx_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * src0, // W0
    ثابت ساختار ggml_tensor * act,  // x
    ثابت ساختار ggml_tensor * dst,  // dst0
    int32_t n_weights,
    ساختار htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

ایستا منطقی ggml_hexagon_op_mul_mat_use_fwht(ثابت ساختار ggml_tensor * dst) {
    اگر (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        بازگشت نادرست;
    }

    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    ثابت int64_t n = src1->ne[0];
    اگر (n < 32 || (n & (n - 1)) != 0) {
        بازگشت نادرست;
    }

    بازگشت src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_rows(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

ایستا منطقی ggml_hexagon_supported_mul_mat(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * dst) {
    ثابت ساختار ggml_tensor * src0 = dst->src[0];
    ثابت ساختار ggml_tensor * src1 = dst->src[1];

    اگر (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        اگر (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            بازگشت نادرست;
        }
        ساختار htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, nullptr, dst, &ukparams);
        بازگشت ukparams.vtcm_row_per_thread > 0 && (size_t) ukparams.vtcm_size <= sess->vtcm_size;
    }

    اگر (dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        بازگشت نادرست;
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q2_K: {
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                بازگشت نادرست;
            }

            ثابت int64_t qk = ggml_blck_size(src0->type);
            اگر (src0->ne[0] % qk) {
                بازگشت نادرست;
            }

            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                بازگشت نادرست;
            }
            بشکن;
        }

        حالت GGML_TYPE_F16:
            اگر (src0->nb[1] < src0->nb[0]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                بازگشت نادرست;
            }
            بشکن;

        حالت GGML_TYPE_F32:
            اگر (src1->type != GGML_TYPE_F32) {
                بازگشت نادرست;
            }
            اگر (src0->nb[1] < src0->nb[0]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                بازگشت نادرست;
            }
            بشکن;

        حالت GGML_TYPE_BF16:
            اگر (src0->nb[1] < src0->nb[0]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                بازگشت نادرست;
            }
            اگر (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                بازگشت نادرست;
            }
            بشکن;

        پیشفرض:
            بازگشت نادرست;
    }

    ساختار htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_mul_mat_id(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    اگر (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        بازگشت نادرست;
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q2_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_IQ4_NL:
        حالت GGML_TYPE_MXFP4:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q2_K:
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                بازگشت نادرست;
            }

            اگر (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
                بازگشت نادرست;
            }
            بشکن;

        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_F32:
        حالت GGML_TYPE_BF16:
            اگر (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                بازگشت نادرست;
            }
            اگر (src0->ne[0] % 32 != 0) {
                بازگشت نادرست;
            }
            بشکن;

        پیشفرض:
            بازگشت نادرست;
    }

    ساختار htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_size, sess->vtcm_size);
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_دودویی(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type == GGML_TYPE_F32) {
        اگر (src1->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
        اگر (dst->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
    }
    وگرنه اگر (src0->type == GGML_TYPE_F16) {
        اگر (src1->type != GGML_TYPE_F16) {
            بازگشت نادرست;
        }
        اگر (dst->type != GGML_TYPE_F16) {
            بازگشت نادرست;
        }
    }
    وگرنه {
        بازگشت نادرست;
    }

    اگر (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        بازگشت نادرست;
    }
    اگر (!ggml_are_same_shape(src0, dst)) {
        بازگشت نادرست;
    }
    اگر (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        بازگشت نادرست;
    }

    ساختار htp_دودویی_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_دودویی_params(sess, op_remap_to_htp(op), src0, src1, dst, &kparams);
}

ایستا منطقی ggml_hexagon_supported_add_id(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!src2) {
        بازگشت نادرست;
    }
    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }
    اگر (!ggml_are_same_shape(src0, dst)) {
        بازگشت نادرست;
    }
    اگر (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        بازگشت نادرست;
    }
    اگر (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        بازگشت نادرست;
    }
    اگر (src0->nb[0] != اندازه(اعشاری) || src1->nb[0] != اندازه(اعشاری) || dst->nb[0] != اندازه(اعشاری) || src2->nb[0] != اندازه(int32_t)) {
        بازگشت نادرست;
    }

    // REVISIT: add support for non-contiguous tensors
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }
    اگر (src2->nb[1] < src2->ne[0] * اندازه(int32_t)) {
        بازگشت نادرست;
    }

    ساختار htp_دودویی_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_دودویی_params(sess, HTP_OP_ADD_ID, src0, src1, dst, &kparams, src2);
}

ایستا منطقی ggml_hexagon_supported_unary(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        بازگشت نادرست;
    }
    اگر (dst->type != src0->type) {
        بازگشت نادرست;
    }
    اگر (!ggml_is_contiguous_rows(src0)) {
        بازگشت نادرست;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    اگر (src0->type == GGML_TYPE_F16) {
        انتخاب (op->op) {
            حالت GGML_OP_NORM:
            حالت GGML_OP_RMS_NORM:
            حالت GGML_OP_L2_NORM:
            حالت GGML_OP_SCALE:
            حالت GGML_OP_CLAMP:
            حالت GGML_OP_SQR:
            حالت GGML_OP_SQRT:
            حالت GGML_OP_LOG:
                بشکن;
            حالت GGML_OP_UNARY:
                اگر (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_STEP) {
                    بازگشت نادرست;
                }
                بشکن;
            پیشفرض:
                بازگشت نادرست;
        }
    }

    اگر (!ggml_are_same_shape(src0, dst)) {
        بازگشت نادرست;
    }

    // dst must be contiguous; src0 may be non-contiguous
    اگر (!ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_sum(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_sum_rows(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    // TODO: add support for non-contigiuos tensors
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_argmax(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (dst->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_pool_2d(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    اگر (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        بازگشت نادرست;
    }
    اگر (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        بازگشت نادرست;
    }
    اگر (!ggml_hexagon_pool_shape_fits(src0, op)) {
        بازگشت نادرست;
    }

    ثابت int32_t kernel_x = params[1];
    ثابت int32_t kernel_y = params[2];
    ثابت int32_t stride_x = params[3];
    ثابت int32_t stride_y = params[4];
    ثابت int32_t pad_x    = params[5];
    ثابت int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    اگر (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        بازگشت نادرست;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ساختار htp_pool_vtcm_layout layout;
    بازگشت htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_threads, sess->vtcm_size);
}

ایستا منطقی ggml_hexagon_supported_pool_1d(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت int32_t * params = op->op_params;

    اگر (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        بازگشت نادرست;
    }
    اگر (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        بازگشت نادرست;
    }
    اگر (!ggml_hexagon_pool_shape_fits(src0, op)) {
        بازگشت نادرست;
    }

    ثابت int32_t kernel = params[1];
    ثابت int32_t stride = params[2];
    ثابت int32_t pad    = params[3];
    اگر (kernel <= 0 || stride <= 0 || pad < 0) {
        بازگشت نادرست;
    }

    ثابت uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    ثابت uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, planes > 0 ? planes : 1);

    ساختار htp_pool_vtcm_layout layout;
    بازگشت htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_threads, sess->vtcm_size);
}

ایستا منطقی ggml_hexagon_supported_activations(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        بازگشت نادرست;
    }
    اگر (dst->type != src0->type) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous_1(src0)) {
        بازگشت نادرست;
    }
    اگر (!ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    اگر (src1) {
        اگر (src1->type != src0->type) {
            بازگشت نادرست;
        }
        اگر (!ggml_are_same_shape(src0, src1)) {
            بازگشت نادرست;
        }
        اگر (!ggml_is_contiguous_1(src1)) {
            بازگشت نادرست;
        }
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_softmax(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src2) {
        بازگشت نادرست;  // FIXME: add support for sinks
    }

    اگر (src0->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (src0->ne[2] > 512) {
        بازگشت نادرست;
    }

    اگر (src1) {
        اگر (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            بازگشت نادرست;
        }
        اگر (src0->ne[0] != src1->ne[0]) {
            بازگشت نادرست;
        }
        اگر (src1->ne[1] < src0->ne[1]) {
            بازگشت نادرست;
        }
        اگر (src0->ne[2] % src1->ne[2] != 0) {
            بازگشت نادرست;
        }
        اگر (src0->ne[3] % src1->ne[3] != 0) {
            بازگشت نادرست;
        }
    }

    اگر (src1) {
        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            بازگشت نادرست;
        }
    } وگرنه {
        اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            بازگشت نادرست;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    ثابت int64_t ne0 = src0->ne[0];
    اگر (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        بازگشت نادرست;
    }

    // HVX vector size constraints for softmax
    #define SOFTMAX_MAX_ROW_SIZE 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    اگر (ne0 > SOFTMAX_MAX_ROW_SIZE) {
        بازگشت نادرست;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads  = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ساختار htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != nullptr, src1 && src1->type == GGML_TYPE_F16, n_threads);
    اگر (layout.total_bytes > sess->vtcm_size) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_set_rows(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0]; //مقادیر
    ثابت ساختار ggml_tensor * src1 = op->src[1]; // indices
    ثابت ساختار ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    اگر (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        بازگشت نادرست;
    }

    اگر (src0->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        بازگشت نادرست;
    }

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_get_rows(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0]; //مقادیر
    ثابت ساختار ggml_tensor * src1 = op->src[1]; // indices
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous_rows(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(dst)) {
        بازگشت نادرست;
    }

    اگر (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        بازگشت نادرست;
    }

    اگر (src0->type == dst->type) {
        اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            بازگشت نادرست;
        }
    } وگرنه اگر (src0->type == GGML_TYPE_I32) {
        بازگشت نادرست;
    } وگرنه اگر (dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    اگر (ggml_is_empty(op)) {
        بازگشت درست;
    }

    ساختار htp_get_rows_kernel_params kparams;
    ggml_hexagon_precompute_get_rows_params(sess, src0, src1, dst, &kparams);
    اگر (kparams.n_threads == 0 || (size_t) kparams.vtcm_size > sess->vtcm_size) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_argsort(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0]; //مقادیر
    ثابت ساختار ggml_tensor * dst  = op;         // indices

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    ثابت uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ثابت size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ساختار htp_sort_vtcm_layout layout;
    اگر (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, نادرست)) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_بالاترین_k(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0]; //مقادیر
    ثابت ساختار ggml_tensor * dst  = op;         // indices

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    ثابت uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت uint32_t n_threads_max = sess->n_threads > 0 ? sess->n_threads : 4;
    ثابت size_t vtcm_budget = sess->vtcm_size > 0 ? sess->vtcm_size : (8 * 1024 * 1024);

    ساختار htp_sort_vtcm_layout layout;
    اگر (!htp_sort_solve_layout(&layout, src0->ne[0], total_rows, dst->ne[0], n_threads_max, vtcm_budget, درست)) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_rope(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * src2 = op->src[2];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!ggml_are_same_shape(src0, dst)) {
        بازگشت نادرست;
    }

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        بازگشت نادرست;
    }

    اگر (src0->ne[0] <= 0) {
        بازگشت نادرست;
    }

    ثابت uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    اگر (src0_nrows == 0) {
        بازگشت نادرست;
    }

    ثابت int32_t * op_params = &op->op_params[0];
    ثابت صحیح n_dims = op_params[1];
    ثابت صحیح mode   = op_params[2];
    ثابت صحیح n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    اگر (n_dims < 0 || n_dims % 2 != 0) {
        بازگشت نادرست;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    اگر (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        بازگشت نادرست;
    }

    اعشاری freq_base;
    memcpy(&freq_base, op_params + 5, اندازه(اعشاری));
    اگر (freq_base < 0.0f) {
        بازگشت نادرست;
    }

    اگر (mode != GGML_ROPE_TYPE_NORMAL &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        بازگشت نادرست;
    }

    ثابت منطقی is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    اگر (mode == GGML_ROPE_TYPE_VISION) {
        اگر (n_dims != (صحیح) (src0->ne[0] / 2) || n_offs != 0) {
            بازگشت نادرست;
        }
    }

    اگر (is_mrope) {
        ثابت int32_t * sections = op_params + 11;
        اگر (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            بازگشت نادرست;
        }
    }

    ثابت int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    اگر (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        بازگشت نادرست;
    }

    اگر (src2) {
        اگر (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            بازگشت نادرست;
        }
        اگر (src2->ne[0] < (n_dims / 2)) {
            بازگشت نادرست;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    اگر (src0->nb[0] != اندازه(اعشاری) || dst->nb[0] != اندازه(اعشاری)) {
        بازگشت نادرست;
    }
    اگر (src0->nb[1] < src0->ne[0] * اندازه(اعشاری) || dst->nb[1] < dst->ne[0] * اندازه(اعشاری)) {
        بازگشت نادرست;
    }

    ثابت uint32_t n_threads = (std::min)((uint32_t) sess->n_threads, src0_nrows);
    ثابت uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    ساختار htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_threads, n_freq_factors);
    اگر (layout.total_bytes > sess->vtcm_size) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_ssm_conv(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    // Only support FP32 for now
    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    // Check IO tensor shapes and dims
    اگر (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        بازگشت نادرست; // src0 should be effectively 3D
    }

    ثابت صحیح d_conv  = src1->ne[0];
    ثابت صحیح d_inner = src0->ne[1];
    ثابت صحیح n_t = dst->ne[1];
    ثابت صحیح n_s = dst->ne[2];

    اگر (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        بازگشت نادرست;
    }
    اگر (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        بازگشت نادرست;
    }
    اگر (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        بازگشت نادرست;
    }
    اگر (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        بازگشت نادرست;
    }
    اگر (src0->nb[0] != اندازه(اعشاری) || src1->nb[0] != اندازه(اعشاری) || dst->nb[0] != اندازه(اعشاری)) {
        بازگشت نادرست;
    }
    اگر (src0->nb[1] != src0->ne[0] * اندازه(اعشاری) || src1->nb[1] != src1->ne[0] * اندازه(اعشاری)) {
        بازگشت نادرست;
    }

    ساختار htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    اگر ((size_t) kparams.vtcm_size > sess->vtcm_size) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

ایستا منطقی im2col_3d_calc_output_size(uint32_t درون_داد, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    اگر (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        بازگشت نادرست;
    }

    ثابت int64_t numerator = (int64_t) درون_داد + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    اگر (numerator < 0) {
        بازگشت نادرست;
    }

    ثابت uint64_t size = (uint64_t) (numerator / stride) + 1;
    اگر (size == 0 || size > UINT32_MAX) {
        بازگشت نادرست;
    }

    *output = (uint32_t) size;
    بازگشت درست;
}

ایستا منطقی ggml_hexagon_precompute_im2col_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        بازگشت نادرست;
    }

    اگر (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        بازگشت نادرست;
    }

    اگر (src1->nb[0] != اندازه(اعشاری) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    ثابت int32_t * op_params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = op_params[0];
    ثابت int32_t s1 = op_params[1];
    ثابت int32_t p0 = op_params[2];
    ثابت int32_t p1 = op_params[3];
    ثابت int32_t d0 = op_params[4];
    ثابت int32_t d1 = op_params[5];
    ثابت منطقی is_2D = (op_params[6] == 1);

    اگر (s0 <= 0 || d0 <= 0 || p0 < 0) {
        بازگشت نادرست;
    }
    اگر (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        بازگشت نادرست;
    }

    ثابت uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    ثابت uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    ثابت uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    ثابت uint32_t IW = (uint32_t) src1->ne[0];
    ثابت uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    ثابت uint32_t OW = (uint32_t) dst->ne[1];

    اگر (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        بازگشت نادرست;
    }

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    ثابت uint64_t total_rows_64   = (uint64_t) N * OH;
    اگر (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX) {
        بازگشت نادرست;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t total_rows   = (uint32_t) total_rows_64;

    اگر (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        بازگشت نادرست;
    }
    اگر (is_2D && dst->ne[2] != OH) {
        بازگشت نادرست;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? اندازه(ggml_fp16_t) : اندازه(اعشاری);

    منطقی is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    اگر (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    ثابت uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ثابت uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ثابت uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    اگر (is_patchembed) {
        ثابت uint64_t src_row_bytes_raw = (uint64_t) IC * KH * IW * اندازه(اعشاری);
        ثابت uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        اگر (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ثابت uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ثابت uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ساختار htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            اگر (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = layout.total_bytes;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
                kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = 1;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = 1;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = 1;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                بازگشت درست;
            }
        }
    }

    برای (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        اگر (Wb > UINT32_MAX) {
            اگر (owb == 1) بشکن;
            ادامه;
        }

        ثابت uint64_t src_row_bytes_raw = (uint64_t) IC * KH * Wb * اندازه(اعشاری);
        ثابت uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        اگر (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            اگر (owb == 1) بشکن;
            ادامه;
        }

        ثابت uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ثابت uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ساختار htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        اگر (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_size            = layout.total_bytes;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
            kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = (OW + owb - 1) / owb;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = 1;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = 1;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = 1;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            بازگشت درست;
        }
        اگر (owb == 1) بشکن;
    }

    بازگشت نادرست;
}

ایستا منطقی ggml_hexagon_precompute_im2col_3d_params(
    ثابت ساختار ggml_hexagon_session * sess,
    ثابت ساختار ggml_tensor * op,
    ساختار htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, اندازه(*kparams));
    kparams->kernel_type = HTP_IM2COL_KERNEL_UNSUPPORTED;

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * src1 = op->src[1];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (!src0 || !src1 || !dst) {
        بازگشت نادرست;
    }

    اگر (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        بازگشت نادرست;
    }

    اگر (src1->nb[0] != اندازه(اعشاری) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    ثابت int32_t * params = (ثابت int32_t *) op->op_params;
    ثابت int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    ثابت int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    ثابت int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    ثابت int32_t ic = params[9];

    اگر (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        بازگشت نادرست;
    }

    اگر (src1->ne[3] % ic != 0) {
        بازگشت نادرست;
    }

    ثابت uint32_t IC = (uint32_t) ic;
    ثابت uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    اگر (N == 0) {
        بازگشت نادرست;
    }

    uint32_t OW, OH, OD;
    اگر (!im2col_3d_calc_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        بازگشت نادرست;
    }

    ثابت uint32_t KD = (uint32_t) src0->ne[2];
    ثابت uint32_t KH = (uint32_t) src0->ne[1];
    ثابت uint32_t KW = (uint32_t) src0->ne[0];
    ثابت uint32_t ID = (uint32_t) src1->ne[2];
    ثابت uint32_t IH = (uint32_t) src1->ne[1];
    ثابت uint32_t IW = (uint32_t) src1->ne[0];

    ثابت uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    ثابت uint64_t total_rows_64   = (uint64_t) N * OD * OH;
    ثابت uint64_t dst_ne3_64      = (uint64_t) OD * N;

    اگر (patch_stride_64 > UINT32_MAX || total_rows_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        بازگشت نادرست;
    }
    ثابت uint32_t patch_stride = (uint32_t) patch_stride_64;
    ثابت uint32_t total_rows   = (uint32_t) total_rows_64;
    ثابت uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    اگر (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        بازگشت نادرست;
    }

    ثابت uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? اندازه(ggml_fp16_t) : اندازه(اعشاری);

    ثابت منطقی is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    ثابت uint32_t max_threads = (sess && sess->n_threads > 0) ? (uint32_t) sess->n_threads : 8;
    ثابت uint32_t n_threads   = (total_rows < max_threads) ? total_rows : max_threads;
    ثابت uint32_t vtcm_size   = (sess && sess->vtcm_size > 0) ? sess->vtcm_size : 0;

    اگر (is_patchembed) {
        ثابت uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * IW * اندازه(اعشاری);
        ثابت uint64_t dst_row_bytes_raw = (uint64_t) OW * patch_stride * dst_elem;
        اگر (src_row_bytes_raw <= UINT32_MAX && dst_row_bytes_raw <= UINT32_MAX) {
            ثابت uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
            ثابت uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

            ساختار htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

            اگر (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
                kparams->kernel_type          = HTP_IM2COL_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_threads            = (uint8_t) n_threads;
                kparams->vtcm_size            = layout.total_bytes;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
                kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
                kparams->src_row_bytes        = src_row_bytes;
                kparams->dst_row_bytes        = dst_row_bytes;
                kparams->owb                  = OW;
                kparams->wb                   = IW;
                kparams->nbpr                 = 1;
                kparams->N                    = N;
                kparams->IC                   = IC;
                kparams->ID                   = ID;
                kparams->IH                   = IH;
                kparams->IW                   = IW;
                kparams->KD                   = KD;
                kparams->KH                   = KH;
                kparams->KW                   = KW;
                kparams->OD                   = OD;
                kparams->OH                   = OH;
                kparams->OW                   = OW;
                kparams->patch_stride         = patch_stride;
                kparams->total_rows           = total_rows;
                بازگشت درست;
            }
        }
    }

    برای (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        ثابت uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        اگر (Wb > UINT32_MAX) {
            اگر (owb == 1) بشکن;
            ادامه;
        }

        ثابت uint64_t src_row_bytes_raw = (uint64_t) IC * KD * KH * Wb * اندازه(اعشاری);
        ثابت uint64_t dst_row_bytes_raw = (uint64_t) owb * patch_stride * dst_elem;
        اگر (src_row_bytes_raw > UINT32_MAX || dst_row_bytes_raw > UINT32_MAX) {
            اگر (owb == 1) بشکن;
            ادامه;
        }

        ثابت uint32_t src_row_bytes = hex_round_up((uint32_t) src_row_bytes_raw, 256);
        ثابت uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_row_bytes_raw, 256);

        ساختار htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_row_bytes, dst_row_bytes, n_threads);

        اگر (vtcm_size == 0 || layout.total_bytes <= vtcm_size) {
            kparams->kernel_type          = HTP_IM2COL_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_threads            = (uint8_t) n_threads;
            kparams->vtcm_size            = layout.total_bytes;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_bytes_per_thread = layout.src_bytes_per_thread;
            kparams->dst_bytes_per_thread = layout.dst_bytes_per_thread;
            kparams->src_row_bytes        = src_row_bytes;
            kparams->dst_row_bytes        = dst_row_bytes;
            kparams->owb                  = owb;
            kparams->wb                   = (uint32_t) Wb;
            kparams->nbpr                 = (OW + owb - 1) / owb;
            kparams->N                    = N;
            kparams->IC                   = IC;
            kparams->ID                   = ID;
            kparams->IH                   = IH;
            kparams->IW                   = IW;
            kparams->KD                   = KD;
            kparams->KH                   = KH;
            kparams->KW                   = KW;
            kparams->OD                   = OD;
            kparams->OH                   = OH;
            kparams->OW                   = OW;
            kparams->patch_stride         = patch_stride;
            kparams->total_rows           = total_rows;
            بازگشت درست;
        }
        اگر (owb == 1) بشکن;
    }

    بازگشت نادرست;
}

ایستا منطقی ggml_hexagon_supported_im2col(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ساختار htp_im2col_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

ایستا منطقی ggml_hexagon_supported_im2col_3d(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ساختار htp_im2col_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

ایستا منطقی ggml_hexagon_supported_pad(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    ثابت int32_t lp0 = ((ثابت int32_t *) op->op_params)[0];
    ثابت int32_t rp0 = ((ثابت int32_t *) op->op_params)[1];
    ثابت int32_t circular = ((ثابت int32_t *) op->op_params)[8];

    اگر (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_cumsum(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_diag(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    // diag only supports F32 currently
    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    // Input must have ne[1] == 1 (vector input)
    اگر (src0->ne[1] != 1) {
        بازگشت نادرست;
    }

    // Output must be square in first two dimensions
    اگر (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_solve_tri(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * src0 = op->src[0]; // A
    ثابت ساختار ggml_tensor * src1 = op->src[1]; // B
    ثابت ساختار ggml_tensor * dst  = op;         // X

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (src0->ne[0] != src0->ne[1]) {
        بازگشت نادرست;
    }

    اگر (src0->ne[1] != src1->ne[1]) {
        بازگشت نادرست;
    }

    اگر (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        بازگشت نادرست;
    }

    اگر (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        بازگشت نادرست;
    }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_tri(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32) { بازگشت نادرست; }
    اگر (dst->type  != GGML_TYPE_F32) { بازگشت نادرست; }
    اگر (!ggml_are_same_shape(src0, dst)) { بازگشت نادرست; }
    اگر (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { بازگشت نادرست; }

    بازگشت درست;

    GGML_UNUSED(sess);
}

ایستا ثابت نویسه * ggml_backend_hexagon_name(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    بازگشت sess->c_name();
}

ایستا تهی ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    delete backend;
}

ایستا htp_op_code op_remap_to_htp(ثابت ggml_tensor * t) {
    انتخاب (t->op) {
        حالت GGML_OP_FLASH_ATTN_EXT:  بازگشت HTP_OP_FLASH_ATTN_EXT;
        حالت GGML_OP_MUL_MAT:
            اگر (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                بازگشت HTP_OP_HADAMARD;
            }
            بازگشت HTP_OP_MUL_MAT;
        حالت GGML_OP_MUL_MAT_ID:      بازگشت HTP_OP_MUL_MAT_ID;
        حالت GGML_OP_MUL:             بازگشت HTP_OP_MUL;
        حالت GGML_OP_ADD:             بازگشت HTP_OP_ADD;
        حالت GGML_OP_ADD_ID:          بازگشت HTP_OP_ADD_ID;
        حالت GGML_OP_SUB:             بازگشت HTP_OP_SUB;
        حالت GGML_OP_DIV:             بازگشت HTP_OP_DIV;
        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:             بازگشت HTP_OP_CPY;
        حالت GGML_OP_CONT:            بازگشت HTP_OP_CPY;
        حالت GGML_OP_GET_ROWS:        بازگشت HTP_OP_GET_ROWS;
        حالت GGML_OP_SET_ROWS:        بازگشت HTP_OP_SET_ROWS;
        حالت GGML_OP_SUM:             بازگشت HTP_OP_SUM;
        حالت GGML_OP_SUM_ROWS:        بازگشت HTP_OP_SUM_ROWS;
        حالت GGML_OP_POOL_2D:         بازگشت HTP_OP_POOL_2D;
        حالت GGML_OP_POOL_1D:         بازگشت HTP_OP_POOL_1D;
        حالت GGML_OP_ARGSORT:         بازگشت HTP_OP_ARGSORT;
        حالت GGML_OP_بالاترین_K:           بازگشت HTP_OP_بالاترین_K;
        حالت GGML_OP_ARGMAX:          بازگشت HTP_OP_ARGMAX;
        حالت GGML_OP_NORM:            بازگشت HTP_OP_NORM;
        حالت GGML_OP_L2_NORM:         بازگشت HTP_OP_L2_NORM;
        حالت GGML_OP_RMS_NORM:        بازگشت HTP_OP_RMS_NORM;
        حالت GGML_OP_CONCAT:          بازگشت HTP_OP_CONCAT;
        حالت GGML_OP_SCALE:           بازگشت HTP_OP_SCALE;
        حالت GGML_OP_CLAMP:           بازگشت HTP_OP_CLAMP;
        حالت GGML_OP_LEAKY_RELU:      بازگشت HTP_OP_LEAKY_RELU;
        حالت GGML_OP_SQR:             بازگشت HTP_OP_SQR;
        حالت GGML_OP_SQRT:            بازگشت HTP_OP_SQRT;
        حالت GGML_OP_LOG:             بازگشت HTP_OP_UNARY_LOG;
        حالت GGML_OP_SOFT_MAX:        بازگشت HTP_OP_SOFTMAX;
        حالت GGML_OP_SSM_CONV:        بازگشت HTP_OP_SSM_CONV;
        حالت GGML_OP_GATED_DELTA_NET: بازگشت HTP_OP_GATED_DELTA_NET;
        حالت GGML_OP_ROPE:            بازگشت HTP_OP_ROPE;
        حالت GGML_OP_REPEAT:          بازگشت HTP_OP_REPEAT;
        حالت GGML_OP_CUMSUM:          بازگشت HTP_OP_CUMSUM;
        حالت GGML_OP_FILL:            بازگشت HTP_OP_FILL;
        حالت GGML_OP_DIAG:            بازگشت HTP_OP_DIAG;
        حالت GGML_OP_SOLVE_TRI:       بازگشت HTP_OP_SOLVE_TRI;
        حالت GGML_OP_TRI:             بازگشت HTP_OP_TRI;
        حالت GGML_OP_PAD:             بازگشت HTP_OP_PAD;
        حالت GGML_OP_IM2COL:          بازگشت HTP_OP_IM2COL;
        حالت GGML_OP_IM2COL_3D:       بازگشت HTP_OP_IM2COL_3D;
        حالت GGML_OP_ROLL:            بازگشت HTP_OP_ROLL;

        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(t)) {
                حالت GGML_UNARY_OP_SILU:       بازگشت HTP_OP_UNARY_SILU;
                حالت GGML_UNARY_OP_GELU:       بازگشت HTP_OP_UNARY_GELU;
                حالت GGML_UNARY_OP_GELU_QUICK: بازگشت HTP_OP_UNARY_GELU_QUICK;
                حالت GGML_UNARY_OP_GELU_ERF:   بازگشت HTP_OP_UNARY_GELU_ERF;
                حالت GGML_UNARY_OP_SIGMOID:    بازگشت HTP_OP_UNARY_SIGMOID;
                حالت GGML_UNARY_OP_NEG:        بازگشت HTP_OP_UNARY_NEG;
                حالت GGML_UNARY_OP_EXP:        بازگشت HTP_OP_UNARY_EXP;
                حالت GGML_UNARY_OP_SOFTPLUS:   بازگشت HTP_OP_UNARY_SOFTPLUS;
                حالت GGML_UNARY_OP_TANH:       بازگشت HTP_OP_UNARY_TANH;
                حالت GGML_UNARY_OP_ABS:        بازگشت HTP_OP_UNARY_ABS;
                حالت GGML_UNARY_OP_RELU:       بازگشت HTP_OP_UNARY_RELU;
                حالت GGML_UNARY_OP_STEP:       بازگشت HTP_OP_UNARY_STEP;
            پیشفرض:
                بشکن;
            }
            بشکن;

        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(t)) {
                حالت GGML_GLU_OP_SWIGLU:       بازگشت HTP_OP_GLU_SWIGLU;
                حالت GGML_GLU_OP_SWIGLU_OAI:   بازگشت HTP_OP_GLU_SWIGLU_OAI;
                حالت GGML_GLU_OP_SWIGLU_CLAMP: بازگشت HTP_OP_GLU_SWIGLU_CLAMP;
                حالت GGML_GLU_OP_GEGLU:        بازگشت HTP_OP_GLU_GEGLU;
                حالت GGML_GLU_OP_GEGLU_QUICK:  بازگشت HTP_OP_GLU_GEGLU_QUICK;
                حالت GGML_GLU_OP_GEGLU_ERF:    بازگشت HTP_OP_GLU_GEGLU_ERF;
                پیشفرض: بشکن;
            }
            بشکن;

        پیشفرض:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    بازگشت HTP_OP_INVALID;
}

ایستا درونخطی منطقی op_is_compute(ggml_tensor *گره)
{
    بازگشت !ggml_op_is_empty(گره->op) && !ggml_is_empty(گره) && (گره->flags & GGML_TENSOR_FLAG_COMPUTE);
}

ایستا منطقی mm_is_hmx_eligible(ثابت ggml_tensor * t) {
    اگر (opt_nhmx == 0) { بازگشت نادرست; }

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];

    اگر (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        ثابت ggml_tensor src1_c = ggml_hexagon_tensor_collapse_rows(src1);
        بازگشت ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, نادرست, نادرست);
    }

    ثابت منطقی is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    ثابت منطقی is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    بازگشت ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

ایستا منطقی is_supported_mul_mat_nx_kernel(ثابت ggml_tensor * src0, ثابت ساختار htp_mm_kernel_params * kparams) {
    اگر (kparams->n_hmx) {
        بازگشت kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    اگر (!ggml_hexagon_tensor_is_tiled(src0)) {
        بازگشت نادرست;
    }

    بازگشت kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW;
}

ایستا منطقی is_supported_mul_mat_id_nx_kernel(ثابت ggml_tensor * src0, ثابت ساختار htp_mm_kernel_params * kparams) {
    اگر (kparams->n_hmx) {
        بازگشت kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    اگر (!ggml_hexagon_tensor_is_tiled(src0)) {
        بازگشت نادرست;
    }

    بازگشت kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK;
}

ایستا منطقی is_mergeable_mul_mat(ثابت ggml_tensor * t) {
    اگر (t->op != GGML_OP_MUL_MAT) بازگشت نادرست;
    اگر (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) بازگشت نادرست;

    ثابت ggml_tensor * src0 = t->src[0];
    ثابت ggml_tensor * src1 = t->src[1];
    اگر (src0->ne[2] != 1 || src0->ne[3] != 1) بازگشت نادرست;

    اگر (mm_is_hmx_eligible(t)) {
        بازگشت htp_is_hmx_weight_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    اگر (src1->type != GGML_TYPE_F32) بازگشت نادرست;

    بازگشت ggml_hexagon_tensor_is_tiled(src0);
}

ایستا منطقی is_mergeable_mul_mat_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    اگر (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        بازگشت نادرست;
    }
    اگر (n1->src[1] != n2->src[1]) {
        بازگشت نادرست;
    }
    اگر (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        بازگشت نادرست;
    }
    اگر (n1->src[0]->type != n2->src[0]->type) {
        بازگشت نادرست;
    }
    اگر (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        بازگشت نادرست;
    }
    اگر (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        بازگشت نادرست;
    }
    بازگشت درست;
}

ایستا منطقی is_mergeable_mul_mat_id(ثابت ggml_tensor * t) {
    اگر (t->op != GGML_OP_MUL_MAT_ID) بازگشت نادرست;

    ثابت ggml_tensor * src0 = t->src[0];
    اگر (mm_is_hmx_eligible(t)) {
        بازگشت htp_is_hmx_weight_type(src0->type);
    }

    بازگشت ggml_hexagon_tensor_is_tiled(src0);
}

ایستا منطقی is_mergeable_mul_mat_id_pair(ثابت ggml_tensor * n1, ثابت ggml_tensor * n2) {
    اگر (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        بازگشت نادرست;
    }
    اگر (n1->src[1] != n2->src[1]) {
        بازگشت نادرست;
    }
    اگر (n1->src[2] != n2->src[2]) {
        بازگشت نادرست;
    }
    اگر (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        بازگشت نادرست;
    }
    اگر (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        بازگشت نادرست;
    }
    اگر (n1->src[0]->type != n2->src[0]->type) {
        بازگشت نادرست;
    }
    اگر (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        بازگشت نادرست;
    }
    اگر (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        بازگشت نادرست;
    }
    بازگشت درست;
}

ایستا ggml_status ggml_backend_hexagon_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    اگر (sess->last_error > HTP_STATUS_OK) {
        بازگشت GGML_STATUS_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    ثابت std::vector<htp_opnode> * nodes_ptr = nullptr;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    منطقی cache_hit = (graph->uid != 0 && sess->cached_uid == graph->uid);
    اگر (cache_hit) {
        nodes_ptr = &sess->cached_nodes;
    } وگرنه {
        // Tag fusable tensors in graph
        برای (صحیح i = 0; i < graph->n_nodes; i++) {
            auto * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            اگر (!extra) ادامه;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            اگر (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } وگرنه اگر ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                اگر ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_ADD && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_ADD })) ||
                    ggml_گره_has_n_uses(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        برای (صحیح i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            اگر (!op_is_compute(n)) {
                ادامه;
            }

            htp_opnode گره(HTP_OP_INVALID, n);
            گره.opcode = op_remap_to_htp(n);
            اگر (گره.opcode == HTP_OP_MUL_MAT || گره.opcode == HTP_OP_MUL_MAT_ID || گره.opcode == HTP_OP_GET_ROWS) {
                ggml_tensor * src0 = گره.گره->src[0];
                اگر (ggml_hexagon_needs_repack(src0)) {
                    اگر (opt_verbose) {
                        نویسه dims_buf[64];
                        نویسه strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, اندازه(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, اندازه(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    memcpy(raw.data(), src0->data, raw.size());
                    htp_repack_tensor_tiled(src0, raw.data(), raw.size());
                    auto * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            اگر (گره.opcode == HTP_OP_MUL_MAT || گره.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    گره.گره->src[0], گره.گره->src[1], گره.گره,
                    (ساختار htp_mm_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_MUL || گره.opcode == HTP_OP_ADD || گره.opcode == HTP_OP_ADD_ID || گره.opcode == HTP_OP_SUB || گره.opcode == HTP_OP_DIV) {
                ثابت ggml_tensor * src1 = گره.گره->src[1];
                ثابت ggml_tensor * src2 = (گره.opcode == HTP_OP_ADD_ID) ? گره.گره->src[2] : nullptr;
                GGML_ASSERT(ggml_hexagon_precompute_دودویی_params(sess,
                    گره.opcode, گره.گره->src[0], src1, گره.گره,
                    (ساختار htp_دودویی_kernel_params *) گره.kernel_params, src2));
            } وگرنه اگر (گره.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    گره.گره,
                    (ساختار htp_fa_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (htp_op_is_unary(گره.opcode)) {
                auto inputs = گره.get_inputs();
                ثابت ساختار ggml_tensor * src0 = inputs[0];
                ثابت ساختار ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : nullptr;
                ggml_hexagon_precompute_unary_params(sess,
                    گره.opcode, src0, src1, گره.dst(),
                    (ساختار htp_unary_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_GET_ROWS) {
                ggml_hexagon_precompute_get_rows_params(sess,
                    گره.گره->src[0], گره.گره->src[1], گره.dst(),
                    (ساختار htp_get_rows_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_SET_ROWS) {
                ggml_hexagon_precompute_set_rows_params(sess,
                    گره.گره->src[0], گره.گره->src[1], گره.dst(),
                    (ساختار htp_set_rows_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    گره.گره,
                    (ساختار htp_rope_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    گره.گره->src[0], گره.گره->src[1], گره.dst(),
                    (ساختار htp_ssm_conv_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    گره.گره,
                    (ساختار htp_softmax_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    گره.گره,
                    (ساختار htp_gdn_kernel_params *)گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_ARGSORT || گره.opcode == HTP_OP_بالاترین_K) {
                ggml_hexagon_precompute_sort_params(sess,
                    گره.گره,
                    گره.opcode == HTP_OP_بالاترین_K,
                    (ساختار htp_sort_kernel_params *) گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_POOL_2D || گره.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, گره.گره->src[0], گره.dst(),
                    (ساختار htp_pool_2d_kernel_params *)گره.kernel_params,
                    گره.opcode == HTP_OP_POOL_1D);
            } وگرنه اگر (گره.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    گره.گره,
                    (ساختار htp_concat_kernel_params *) گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_CPY || گره.opcode == HTP_OP_CPY_FENCE) {
                ثابت منطقی ok = ggml_hexagon_precompute_cpy_params(sess,
                    گره.گره,
                    (ساختار htp_copy_kernel_params *) گره.kernel_params
                );
                ثابت auto * kparams = (ثابت ساختار htp_copy_kernel_params *) گره.kernel_params;
                اگر (ok && گره.opcode == HTP_OP_CPY && kparams->total_elems == 0) {
                    ادامه;
                }
            } وگرنه اگر (گره.opcode == HTP_OP_IM2COL) {
                ggml_hexagon_precompute_im2col_params(sess,
                    گره.گره,
                    (ساختار htp_im2col_kernel_params *) گره.kernel_params
                );
            } وگرنه اگر (گره.opcode == HTP_OP_IM2COL_3D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    گره.گره,
                    (ساختار htp_im2col_kernel_params *) گره.kernel_params
                );
            }
            computed_nodes.push_back(std::move(گره));
        }

        اگر (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } وگرنه {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    برای (ثابت auto & گره : *nodes_ptr) {
        sess->enqueue_op(گره);
    }

    اگر (sess->last_error > HTP_STATUS_OK) {
        بازگشت GGML_STATUS_FAILED;
    }

    بازگشت GGML_STATUS_SUCCESS;
}

ایستا تهی ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    اگر (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
}

شمارش ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

ساختار ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

ساختار ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    تهی reset() {
        ranges.clear();
    }

    تهی add(ثابت ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    منطقی check(ثابت ggml_hexagon_mem_range & mr) ثابت {
        برای (ثابت auto & cmp : ranges) {
            اگر (mr.pb != cmp.pb) {
                ادامه;
            }
            اگر (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                ادامه;
            }
            اگر (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                بازگشت نادرست;
            }
        }
        بازگشت درست;
    }
};

ایستا ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(ثابت ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    ثابت ggml_tensor * base = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    اگر (tensor->buffer) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } وگرنه {
        mr = {
            /*.pb =*/ (uint64_t) base,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    بازگشت mr;
}

ایستا تهی ggml_hexagon_mem_ranges_add_گره(ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & گره) {
    اگر (گره.is_empty()) بازگشت;

    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (گره.گره->src[i]) {
            mrs.add(ggml_hexagon_mem_range_from_tensor(گره.گره->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    برای (ثابت auto * fused : گره.fused) {
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (fused->src[i]) {
                mrs.add(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.add(ggml_hexagon_mem_range_from_tensor(گره.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ایستا منطقی ggml_hexagon_mem_ranges_check_گره(ثابت ggml_hexagon_mem_ranges & mrs, ثابت htp_opnode & گره) {
    اگر (گره.is_empty()) بازگشت درست;

    برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
        اگر (گره.گره->src[i]) {
            اگر (!mrs.check(ggml_hexagon_mem_range_from_tensor(گره.گره->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                بازگشت نادرست;
            }
        }
    }
    برای (ثابت auto * fused : گره.fused) {
        برای (صحیح i = 0; i < GGML_MAX_SRC; i++) {
            اگر (fused->src[i]) {
                اگر (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    بازگشت نادرست;
                }
            }
        }
    }
    بازگشت mrs.check(ggml_hexagon_mem_range_from_tensor(گره.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

ایستا std::vector<صحیح> ggml_hexagon_graph_optimize_reorder(ثابت std::vector<htp_opnode> & nodes) {
    ثابت صحیح n = nodes.size();

    std::vector<صحیح> res;
    res.reserve(n);

    std::vector<منطقی> used(n, نادرست);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    برای (صحیح i0 = 0; i0 < n; i0++) {
        اگر (used[i0]) {
            ادامه;
        }

        ثابت auto & node0 = nodes[i0];

        اگر (!node0.stackable()) {
            res.push_back(i0);
            used[i0] = درست;
            ادامه;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr صحیح N_FORWARD = 16;

        std::vector<صحیح> پشته;
        پشته.push_back(i0);

        mrs.reset();

        برای (صحیح i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            اگر (used[i1]) {
                ادامه;
            }

            ثابت auto & node1 = nodes[i1];

            اگر (node1.stackable() && node1.same_درون_داد(node0) && ggml_hexagon_mem_ranges_check_گره(mrs, node1)) {
                پشته.push_back(i1);
            } وگرنه {
                ggml_hexagon_mem_ranges_add_گره(mrs, node1);
            }
        }

        برای (صحیح idx : پشته) {
            res.push_back(idx);
            used[idx] = درست;
        }
    }

    بازگشت res;
}

ایستا تهی ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);

    ثابت صحیح n = gf->n_nodes;

    constexpr صحیح MAX_FUSE = 16;

    شمارش ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    برای (صحیح i = 0; i < n; i++) {
        htp_opnode گره(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        اگر (گره.op() == GGML_OP_ADD ||
            گره.op() == GGML_OP_NORM ||
            گره.op() == GGML_OP_RMS_NORM) {
            ops[0] = گره.op();

            صحیح f = i + 1;
            وقتی (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                اگر (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    بشکن;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            برای (; f > 1; f--) {
                اگر (ggml_can_fuse(gf, i, ops, f)) {
                    بشکن;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            برای (صحیح k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                گره.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(گره));
    }

    ثابت auto order = ggml_hexagon_graph_optimize_reorder(nodes);

    // unfuse
    {
        صحیح j = 0;
        برای (ثابت auto i : order) {
            ثابت auto & گره = nodes[i];

            gf->nodes[j++] = گره.گره;

            برای (auto * fused : گره.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_UNUSED(backend);
}

ایستا uint64_t ggml_hexagon_session_کلید(ثابت ggml_hexagon_session * sess) {
    بازگشت ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

ایستا منطقی ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    auto sbuf_dst = sess_dst->mmap_tensor(dst);

    اگر (!sess_src->clone_buffer(sbuf_dst)) { بازگشت نادرست; }

    ثابت uint64_t src_کلید = ggml_hexagon_session_کلید(sess_src);
    auto & fence_slot = sess_dst->cpy_fence_slots[src_کلید];
    اگر (!fence_slot) {
        fence_slot = (فرار uint32_t *) sess_dst->alloc_fence(1);
    }

    اگر (!sess_src->clone_buffer(sess_dst->fence_buf)) { بازگشت نادرست; }

    اگر (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    ایستا ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.buffer = &sess_dst->fence_buf->backend_buffer;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.data   = (تهی *) fence_slot;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = اندازه(int32_t);
    fence_tensor.nb[1]  = اندازه(int32_t);
    fence_tensor.nb[2]  = اندازه(int32_t);
    fence_tensor.nb[3]  = اندازه(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->enqueue_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->enqueue_fence(&fence_tensor, fence_seq, /* wait = */ درست);

    sess_dst->add_peer(sess_src);

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    auto sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    اگر (!sess_dst->clone_buffer(sbuf_src)) { بازگشت نادرست; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->enqueue_cpy(src, dst);
    sess_dst->add_peer(sess_src);

    بازگشت درست;
}

ایستا منطقی ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ثابت ggml_tensor * src, ggml_tensor * dst) {
    اگر (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        بازگشت نادرست;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    اگر (!dst->extra) {
        ggml_backend_buffer_init_tensor(dst->buffer, dst);
    }

    auto * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    ثابت auto * src_extra = static_cast<ثابت ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    auto sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    auto sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    اگر (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->enqueue_cpy(src, dst);
        بازگشت درست;
    }

    اگر (sess_src->phys_idx != sess_dst->phys_idx)
        بازگشت ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    بازگشت ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

ایستا ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = جدید ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_slot = (فرار uint32_t *) sess->alloc_fence(1);

    ایستا ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.buffer = &sess->fence_buf->backend_buffer;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.data   = (تهی *) hex_event->fence_slot;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = اندازه(int32_t);
    hex_event->fence_tensor.nb[1]  = اندازه(int32_t);
    hex_event->fence_tensor.nb[2]  = اندازه(int32_t);
    hex_event->fence_tensor.nb[3]  = اندازه(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (تهی *)hex_event, (تهی *)hex_event->fence_slot);

    بازگشت جدید ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

ایستا تهی ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    اگر (hex_event->seq == 0) {
        بازگشت;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (تهی *)hex_event, hex_event->seq, (تهی *)hex_event->fence_slot);

    auto * fence = reinterpret_cast<ثابت فرار std::atomic<uint32_t> *>(hex_event->fence_slot);

    اگر ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    وقتی (درست) {
        اگر ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t status = fence[1].load(std::memory_order_acquire);
            اگر (status > HTP_STATUS_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), status_to_str(status));
            }
            بشکن;
        }
        std::this_thread::yield();
    }
}

ایستا تهی ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (تهی *)hex_event);
    hex_event->fence_sess->free_fence((تهی *) hex_event->fence_slot, 1);
    delete hex_event;
    delete event;
}

ایستا تهی ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    auto * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

ایستا تهی ggml_backend_hexagon_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    اگر (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ نادرست);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (تهی *)hex_event, hex_event->seq, (تهی *)hex_event->fence_slot);
}

ایستا تهی ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    auto hex_event = static_cast<ggml_hexagon_event *>(event->context);

    اگر (hex_event->seq == 0) {
        بازگشت;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (تهی *)hex_event, hex_event->seq, (تهی *)hex_event->fence_slot);

    // same physical NPU runs sequentially in FIFO order
    اگر (sess->phys_idx == hex_event->sess->phys_idx) {
        اگر (sess != hex_event->sess) {
            sess->add_peer(hex_event->sess);
        }
        بازگشت;
    }

    sess->clone_buffer(hex_event->fence_sess->fence_buf);
    sess->add_peer(hex_event->sess);
    sess->enqueue_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ درست);
}

ایستا تهی ggml_backend_hexagon_set_tensor_async(ggml_backend_t backend, ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (صحیح) tensor->buffer->usage : -1);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

ایستا تهی ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, ثابت ساختار ggml_tensor * tensor, تهی * data, size_t offset, size_t size) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, tensor->buffer ? (صحیح) tensor->buffer->usage : -1);
    sess->flush_sync();
    اگر (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get(tensor, data, offset, size);
}

ایستا تهی ggml_backend_hexagon_set_tensor_2d_async(ggml_backend_t backend,
                                                     ساختار ggml_tensor * tensor,
                                                     ثابت تهی * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (صحیح) tensor->buffer->usage : -1);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ایستا تهی ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     ثابت ساختار ggml_tensor * tensor,
                                                     تهی * data,
                                                     size_t offset,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_data) {
    auto sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, data, offset, size, n_copies, stride_tensor, stride_data, tensor->buffer ? (صحیح) tensor->buffer->usage : -1);
    sess->flush_sync();
    اگر (sess->last_error > HTP_STATUS_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), status_to_str(sess->last_error));
    }
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

ایستا ساختار ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ پوچ,
    /* .graph_plan_free         = */ پوچ,
    /* .graph_plan_update       = */ پوچ,
    /* .graph_plan_compute      = */ پوچ,
    /* .graph_compute           = */ ggml_backend_hexagon_graph_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_record,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_graph_optimize,
};

ایستا ggml_guid_t ggml_backend_hexagon_guid() {
    ایستا ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    بازگشت &guid;
}

منطقی ggml_backend_is_hexagon(ggml_backend_t backend) {
    بازگشت backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

ایستا ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, ثابت نویسه * params) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    بازگشت جدید ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

ایستا ثابت نویسه * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    بازگشت dev_ctx->c_name();

    GGML_UNUSED(dev);
}

ایستا ثابت نویسه * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    بازگشت "Hexagon";
    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * آزاد, size_t * total) {
    *آزاد  = opt_mbuf_total;
    *total = opt_mbuf_total;

    GGML_UNUSED(dev);
}

ایستا شمارش ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    بازگشت GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

ایستا تهی ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, ساختار ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ درست,
        /* .host_buffer           = */ نادرست,
        /* .buffer_from_host_ptr  = */ نادرست,
        /* .events                = */ درست,
        /* .mmap_support          = */ نادرست,
    };
}

ایستا ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    بازگشت &dev_ctx->buffer_type;
}

ایستا ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    اگر (!opt_hostbuf) {
        بازگشت پوچ;
    }
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    بازگشت &dev_ctx->host_buffer_type;
}

ایستا منطقی ggml_hexagon_supported_cpy(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ساختار htp_copy_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ایستا منطقی ggml_hexagon_supported_cont(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ساختار htp_copy_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

ایستا منطقی ggml_hexagon_supported_repeat(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    GGML_UNUSED(sess);
    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    // Support f32 and f16
    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) بازگشت نادرست;

    // src and dst must be the same type
    اگر (src0->type != dst->type) بازگشت نادرست;

    // dst dims must be multiples of src dims
    اگر (dst->ne[0] % src0->ne[0] != 0) بازگشت نادرست;
    اگر (dst->ne[1] % src0->ne[1] != 0) بازگشت نادرست;
    اگر (dst->ne[2] % src0->ne[2] != 0) بازگشت نادرست;
    اگر (dst->ne[3] % src0->ne[3] != 0) بازگشت نادرست;

    // require contiguous tensors (no transposition)
    اگر (ggml_is_transposed(src0) || ggml_is_transposed(dst)) بازگشت نادرست;

    بازگشت درست;
}

ایستا منطقی ggml_hexagon_supported_concat(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ساختار htp_concat_kernel_params kparams;
    بازگشت ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

ایستا منطقی ggml_hexagon_supported_fill(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    ثابت ساختار ggml_tensor * dst = op;

    اگر (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        بازگشت نادرست;
    }

    بازگشت درست;
    GGML_UNUSED(sess);
}

ایستا منطقی ggml_hexagon_supported_roll(ثابت ساختار ggml_hexagon_session * sess, ثابت ساختار ggml_tensor * op) {
    GGML_UNUSED(sess);

    ثابت ساختار ggml_tensor * src0 = op->src[0];
    ثابت ساختار ggml_tensor * dst  = op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    اگر (!ggml_are_same_shape(src0, dst)) {
        بازگشت نادرست;
    }

    اگر (src0->nb[0] != ggml_type_size(src0->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        بازگشت نادرست;
    }

    اگر (!ggml_is_contiguous(dst)) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
ایستا منطقی ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    auto sess    = dev_ctx->session();

    // reject ops that match the filter
    اگر (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        بازگشت نادرست;
    }

    منطقی supp = نادرست;
    انتخاب (op->op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_PERMUTE:
        حالت GGML_OP_TRANSPOSE:
            supp = درست;
            بشکن;

        حالت GGML_OP_MUL:
        حالت GGML_OP_ADD:
        حالت GGML_OP_SUB:
        حالت GGML_OP_DIV:
            supp = ggml_hexagon_supported_دودویی(sess, op);
            بشکن;

        حالت GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            بشکن;

        حالت GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            بشکن;

        حالت GGML_OP_ADD_ID:
            supp = ggml_hexagon_supported_add_id(sess, op);
            بشکن;

        حالت GGML_OP_NORM:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_RMS_NORM:
        حالت GGML_OP_SCALE:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            بشکن;

        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            بشکن;

        حالت GGML_OP_SUM:
            supp = ggml_hexagon_supported_sum(sess, op);
            بشکن;

        حالت GGML_OP_SUM_ROWS:
            supp = ggml_hexagon_supported_sum_rows(sess, op);
            بشکن;

        حالت GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            بشکن;

        حالت GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            بشکن;

        حالت GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            بشکن;

        حالت GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            بشکن;

        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(op)) {
                حالت GGML_UNARY_OP_NEG:
                حالت GGML_UNARY_OP_EXP:
                حالت GGML_UNARY_OP_SIGMOID:
                حالت GGML_UNARY_OP_SOFTPLUS:
                حالت GGML_UNARY_OP_TANH:
                حالت GGML_UNARY_OP_ABS:
                حالت GGML_UNARY_OP_SILU:
                حالت GGML_UNARY_OP_GELU:
                حالت GGML_UNARY_OP_GELU_QUICK:
                حالت GGML_UNARY_OP_GELU_ERF:
                حالت GGML_UNARY_OP_RELU:
                حالت GGML_UNARY_OP_STEP:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    بشکن;
                پیشفرض:
                    supp = نادرست;
                    بشکن;
            }
            بشکن;

        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(op)) {
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_SWIGLU_OAI:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                حالت GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    بشکن;
                حالت GGML_GLU_OP_GEGLU_QUICK:
                حالت GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    بشکن;
                پیشفرض:
                    supp = نادرست;
                    بشکن;
            }
            بشکن;

        حالت GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            بشکن;

        حالت GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            بشکن;

        حالت GGML_OP_SET_ROWS:
            supp = ggml_hexagon_supported_set_rows(sess, op);
            بشکن;

        حالت GGML_OP_GET_ROWS:
            supp = ggml_hexagon_supported_get_rows(sess, op);
            بشکن;

        حالت GGML_OP_DUP:
        حالت GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            بشکن;

        حالت GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            بشکن;

        حالت GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            بشکن;

        حالت GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            بشکن;

        حالت GGML_OP_بالاترین_K:
            supp = ggml_hexagon_supported_بالاترین_k(sess, op);
            بشکن;

        حالت GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            بشکن;

        حالت GGML_OP_IM2COL:
            supp = ggml_hexagon_supported_im2col(sess, op);
            بشکن;

        حالت GGML_OP_IM2COL_3D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            بشکن;

        حالت GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            بشکن;

        حالت GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            بشکن;

        حالت GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            بشکن;

        حالت GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            بشکن;

        حالت GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            بشکن;

        حالت GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            بشکن;

        حالت GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            بشکن;

        حالت GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            بشکن;

        حالت GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            بشکن;

        پیشفرض:
            بشکن;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    بازگشت supp;
}

ایستا منطقی ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    منطقی supp = (buft == &dev_ctx->host_buffer_type) || (buft == &dev_ctx->buffer_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    بازگشت supp;
}

ایستا ثابت ساختار ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ پوچ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ پوچ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    برای (size_t i = 0; i < opt_ndev; i++) {
        ثابت auto & cfg = opt_device_configs[i];
        اگر (cfg.mdev_group.empty()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } وگرنه {
            std::string peers_str;
            برای (ثابت auto & p : cfg.mdev_group) {
                اگر (!peers_str.empty()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = جدید ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    برای (size_t i = 0; i < opt_ndev; i++) {
        auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        delete dev_ctx;
    }
}

ایستا ثابت نویسه * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    بازگشت "HTP";
    GGML_UNUSED(reg);
}

ایستا size_t ggml_backend_hexagon_reg_get_device_شمار(ggml_backend_reg_t reg) {
    بازگشت opt_ndev;
    GGML_UNUSED(reg);
}

ایستا ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    اگر (index >= opt_ndev || !hreg->devices[index].context) {
        بازگشت nullptr;
    }

    بازگشت &hreg->devices[index];
}

// ** communication context for tensor-split allreduce

ایستا تهی * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    اگر (n_backends < 2 || n_backends > 4) {
        بازگشت nullptr;
    }

    برای (size_t i = 0; i < n_backends; ++i) {
        اگر (!ggml_backend_is_hexagon(backends[i])) {
            بازگشت nullptr;
        }
    }

    برای (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        برای (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            اگر (sess_i->phys_idx == sess_j->phys_idx) {
                بازگشت nullptr;
            }
        }
    }

    auto * ctx = جدید ggml_backend_hexagon_comm_context();
    ctx->backends.assign(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    ایستا ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    برای (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (فرار uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].buffer = &sess_i->fence_buf->backend_buffer;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].data   = (تهی *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = اندازه(int32_t);
        ctx->fence_tensors[i].nb[1]  = اندازه(int32_t);
        ctx->fence_tensors[i].nb[2]  = اندازه(int32_t);
        ctx->fence_tensors[i].nb[3]  = اندازه(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    بازگشت ctx;
}

ایستا تهی ggml_backend_hexagon_comm_free(تهی * comm_ctx_v) {
    اگر (!comm_ctx_v) بازگشت;
    auto * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    برای (size_t i = 0; i < ctx->n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((تهی *) ctx->fence_slots[i], 1);
    }
    delete ctx;
}

ایستا منطقی ggml_backend_hexagon_comm_allreduce_tensor(تهی * comm_ctx_v, ساختار ggml_tensor ** tensors) {
    اگر (opt_ar_select == 0 || !comm_ctx_v) بازگشت نادرست;
    auto * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    ثابت size_t n_backends = comm_ctx->n_backends;

    اگر (n_backends < 2 || n_backends > 4) بازگشت نادرست;

    برای (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        برای (size_t j = i + 1; j < n_backends; j++) {
            auto sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            اگر (sess_i->phys_idx == sess_j->phys_idx) {
                بازگشت نادرست;
            }
        }
    }

    برای (size_t i = 0; i < n_backends; i++) {
        اگر (!tensors[i] || !tensors[i]->buffer || !ggml_backend_buffer_is_hexagon(tensors[i]->buffer)) {
            بازگشت نادرست;
        }
        اگر (tensors[i]->type != tensors[0]->type) {
            بازگشت نادرست;
        }
        اگر (!ggml_is_contiguous(tensors[i])) {
            بازگشت نادرست;
        }
        اگر (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            بازگشت نادرست;
        }
    }

    اگر (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }

    برای (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        ساختار htp_allreduce_kernel_params kparams;
        اگر (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, نادرست, نادرست, /*is_shard_ok=*/ نادرست, &kparams)) {
            بازگشت نادرست;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    برای (size_t i = 1; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        اگر ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    اگر (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_entry = max_seq;
    اگر (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    برای (size_t i = 0; i < n_backends; i++) {
        auto sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<ثابت ggml_tensor *> data_tensors(n_backends);
    std::vector<ثابت ggml_tensor *> sync_tensors(n_backends);
    برای (size_t i = 0; i < n_backends; i++) {
        data_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    برای (size_t r = 0; r < n_backends; r++) {
        auto sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->enqueue_allreduce(tensors[r], data_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_entry, fence_seq_exit);
        برای (size_t j = 0; j < n_backends; j++) {
            اگر (r != j) {
                sess->add_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    بازگشت درست;
}

ایستا ggml_backend_buffer_type_t ggml_backend_hexagon_split_buffer_type(صحیح main_device, ثابت اعشاری * tensor_split) {
    GGML_UNUSED(tensor_split);
    auto reg = ggml_backend_hexagon_reg();
    auto dev = ggml_backend_reg_dev_get(reg, main_device);
    اگر (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    اگر (!dev) بازگشت nullptr;
    auto dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    بازگشت &dev_ctx->buffer_type;
}

ایستا تهی * ggml_backend_hexagon_get_proc_address(ggml_backend_reg_t reg, ثابت نویسه * name) {
    GGML_UNUSED(reg);
    اگر (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        بازگشت (تهی *) ggml_backend_hexagon_split_buffer_type;
    }
    اگر (strcmp(name, "ggml_backend_comm_init") == 0) {
        بازگشت (تهی *) ggml_backend_hexagon_comm_init;
    }
    اگر (strcmp(name, "ggml_backend_comm_free") == 0) {
        بازگشت (تهی *) ggml_backend_hexagon_comm_free;
    }
    اگر (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        بازگشت (تهی *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    بازگشت پوچ;
}

template<typename T> std::vector<T> str_to_vec(ثابت نویسه* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    وقتی (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, nullptr, 0));
    }

    بازگشت v;
}

template<typename T, صحیح BASE=10> std::string vec_to_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(BASE) << std::showbase;
    برای (auto i : v) { ss << i << ','; }
    auto str = ss.str(); str.pop_back(); // drop last comma
    بازگشت str;
}

ایستا تهی ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, منطقی discovery_supported, ثابت std::unordered_map<صحیح, fastrpc_domain> & cdsp_map) {
    اگر (discovery_supported) {
        auto it = cdsp_map.find(cfg.physical_idx);
        اگر (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } وگرنه {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } وگرنه {
        انتخاب (cfg.physical_idx) {
            حالت 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                بشکن;
            حالت 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                بشکن;
            پیشفرض:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                بشکن;
        }
    }
    برای (auto & sub_cfg : cfg.mdev_group) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
ایستا تهی ggml_hexagon_discover_devices() {
    std::unordered_map<صحیح, fastrpc_domain> cdsp_map;
    منطقی discovery_supported = نادرست;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = nullptr;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_SET_TYPE(0, FASTRPC_NSP);

    صحیح err = remote_system_request(&domain_info);
    اگر (err == AEE_SUCCESS && domain_info.sys.رقم_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.رقم_domains);
        domain_info.sys.domains     = domains.data();
        domain_info.sys.max_domains = (صحیح) domains.size();

        err = remote_system_request(&domain_info);
        اگر (err == AEE_SUCCESS) {
            discovery_supported = درست;
            ثابت صحیح n_domains = std::min(domain_info.sys.رقم_domains, (صحیح) domains.size());
            برای (صحیح i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (صحیح) domains[i].type, domains[i].id, domains[i].name, domains[i].status, domains[i].instance_id);
                اگر (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (صحیح) domains[i].type);
                    ادامه;
                }
                اگر (!domains[i].status) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    ادامه;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } وگرنه {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (unsigned) err);
        }
    } وگرنه اگر (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (unsigned) err);
    }

    // Populate domain IDs and names for all configured devices
    برای (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

ایستا تهی ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((unsigned صحیح) HTP_TYPE_Q4_0 == (unsigned صحیح) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q4_1 == (unsigned صحیح) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q8_0 == (unsigned صحیح) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_MXFP4 == (unsigned صحیح) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_IQ4_NL == (unsigned صحیح) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q4_K == (unsigned صحیح) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q5_K == (unsigned صحیح) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q6_K == (unsigned صحیح) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q3_K == (unsigned صحیح) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q2_K == (unsigned صحیح) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q1_0 == (unsigned صحیح) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_Q2_0 == (unsigned صحیح) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((unsigned صحیح) HTP_TYPE_BF16 == (unsigned صحیح) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    ثابت نویسه * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    ثابت نویسه * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    ثابت نویسه * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    ثابت نویسه * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    ثابت نویسه * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    ثابت نویسه * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    ثابت نویسه * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    ثابت نویسه * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    ثابت نویسه * str_etm      = getenv("GGML_HEXAGON_ETM");
    ثابت نویسه * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    ثابت نویسه * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    ثابت نویسه * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    ثابت نویسه * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    ثابت نویسه * str_fa_head_split = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    ثابت نویسه * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    ثابت نویسه * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    ثابت نویسه * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    ثابت نویسه * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    ثابت نویسه * str_arch     = getenv("GGML_HEXAGON_ARCH");
    ثابت نویسه * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    ثابت نویسه * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    ثابت نویسه * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    ثابت نویسه * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    ثابت نویسه * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    ثابت نویسه * str_power_mode = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    اگر (!str_arch) {
        صحیح err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        اگر (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } وگرنه {
            اگر (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } وگرنه اگر (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } وگرنه {
        اگر (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, پوچ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    auto RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? جدید std::regex(str_opfilter, RE_ICASE) : پوچ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, پوچ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, پوچ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, پوچ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  پوچ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    اگر (str_power_mode) {
        صحیح m = htp_power_mode_from_str(str_power_mode);
        اگر (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_mode);
        } وگرنه {
            opt_power_mode = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, پوچ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_head_split = str_fa_head_split ? atoi(str_fa_head_split)       : opt_fa_head_split;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    اگر (str_mbuf) {
        ثابت نویسه * p = str_mbuf;
        برای (صحیح idx = 0; idx < 3 && p && *p; idx++) {
            وقتی (*p == ' ') p++;
            ثابت نویسه * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            وقتی (len > 0 && p[len - 1] == ' ') len--;
            اگر (len > 0) {
                std::string token(p, len);
                اگر (idx == 0) opt_mbuf_dyn    = parse_size(token.c_str());
                اگر (idx == 1) opt_mbuf_static = parse_size(token.c_str());
                اگر (idx == 2) opt_mbuf_total  = parse_size(token.c_str());
            }
            اگر (!comma) بشکن;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_size(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    ثابت نویسه * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    اگر (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    اگر (str_devices && str_devices[0] != '\0') {
        منطقی is_single_عدد = درست;
        برای (صحیح i = 0; str_devices[i] != '\0'; i++) {
            اگر (!isdigit((unsigned نویسه)str_devices[i])) {
                is_single_عدد = نادرست;
                بشکن;
            }
        }
        اگر (is_single_عدد) {
            صحیح n = atoi(str_devices);
            اگر (n < 1) n = 1;
            اگر (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            برای (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (صحیح)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_group.clear();
            }
        } وگرنه {
            std::string s_devices(str_devices);
            std::vector<std::string> اقلام;
            std::string curr_item;
            صحیح bracket_depth = 0;
            برای (نویسه ch : s_devices) {
                اگر (ch == '[') {
                    bracket_depth++;
                    curr_item += ch;
                } وگرنه اگر (ch == ']') {
                    اگر (bracket_depth > 0) bracket_depth--;
                    curr_item += ch;
                } وگرنه اگر (ch == ',' && bracket_depth == 0) {
                    size_t s = curr_item.find_first_not_of(" \t\r\n");
                    size_t e = curr_item.find_last_not_of(" \t\r\n");
                    اگر (s != std::string::npos) {
                        اقلام.push_back(curr_item.substr(s, e - s + 1));
                    }
                    curr_item.clear();
                } وگرنه {
                    curr_item += ch;
                }
            }
            size_t s = curr_item.find_first_not_of(" \t\r\n");
            size_t e = curr_item.find_last_not_of(" \t\r\n");
            اگر (s != std::string::npos) {
                اقلام.push_back(curr_item.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            برای (ثابت auto & item : اقلام) {
                size_t b_open  = item.find('[');
                size_t b_close = item.rfind(']');

                اگر (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = item.substr(0, b_open);
                    std::string content  = item.substr(b_open + 1, b_close - b_open - 1);

                    صحیح virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    اگر (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        تلاش {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } بگیر (...) {
                            virt = 0;
                        }
                    } وگرنه {
                        size_t dev_colon = dev_name.find(':');
                        اگر (dev_colon != std::string::npos) {
                            تلاش {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } بگیر (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<صحیح> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    وقتی (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        اگر (ps == std::string::npos) ادامه;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        اگر (dash_pos != std::string::npos) {
                            تلاش {
                                صحیح p_start = std::stoi(p_part.substr(0, dash_pos));
                                صحیح p_end   = std::stoi(p_part.substr(dash_pos + 1));
                                برای (صحیح p = p_start; p <= p_end; p++) {
                                    اگر (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } بگیر (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } وگرنه {
                            تلاش {
                                صحیح p = std::stoi(p_part);
                                اگر (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } بگیر (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    اگر (phys_list.empty()) {
                        phys_list.push_back(0);
                    }

                    اگر (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        auto & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_group.clear();

                        برای (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_group.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } وگرنه {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } وگرنه اگر (item.rfind("HTP", 0) == 0) {
                    std::string rest = item.substr(3);
                    size_t colon_pos = rest.find(':');
                    صحیح phys = 0;
                    صحیح virt = 0;
                    تلاش {
                        اگر (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } وگرنه {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } بگیر (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", item.c_str());
                        ادامه;
                    }

                    اگر (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_group.clear();
                        opt_ndev++;
                    } وگرنه {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, item.c_str());
                    }
                } وگرنه {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", item.c_str());
                }
            }
        }
    } وگرنه {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_group.clear();
    }

#اگر defined(__ANDROID__)
    اگر (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#endif

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    اگر (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            auto v  = str_to_vec<uint32_t>(str_profile);
            انتخاب (v.size()) {
                حالت 1:  opt_profile = v[0]; بازگشت opt_pmu_evt; // mode with default pmu events
                حالت 8:  opt_profile = 2;    بازگشت v;           // mode with custom  pmu events
                پیشفرض: opt_profile = 0;    بازگشت {};          // garbage input
            }}();
        اگر (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_to_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = جدید ggml_hexagon_registry(reg);
}

ایستا ثابت ساختار ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(تهی) {
    ایستا منطقی initialized = نادرست;

    ایستا ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ پوچ };

    {
        ایستا std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        اگر (!initialized) {
            auto nErr = htpdrv_init();
            اگر (nErr != AEE_SUCCESS) {
                بازگشت پوچ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = درست;
    }

    بازگشت &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
