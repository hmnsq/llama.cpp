#నిర్వచించు USE_MATH_నిర్వచిస్తుంది

#అన్తర్భావ <assert.h>
#అన్తర్భావ <inttypes.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <time.h>

#అన్తర్భావ <atomic>
#అన్తర్భావ <memory>
#అన్తర్భావ <chrono>
#అన్తర్భావ <mutex>
#అన్తర్భావ <thread>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <stdexcept>
#అన్తర్భావ <string>
#అన్తర్భావ <sstream>
#అన్తర్భావ <iomanip>
#అన్తర్భావ <unordered_set>
#అన్తర్భావ <unordered_map>
#అన్తర్భావ <regex>
#అన్తర్భావ <queue>
#అన్తర్భావ <deque>
#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <initializer_list>

#నిర్వచితం_కాకపోతే M_LOG2E
#    define M_LOG2E 1.44269504088896340736
#అయితే_ముగింపు

#నిర్వచితమైతే _WIN32
#    define WIN32_LEAN_AND_సగటు
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <windows.h>
#    include <sal.h>
#లేకపోతే
#    include <semaphore.h>
#    include <unistd.h>
#అయితే_ముగింపు

#సూచన clang diagnostic ignored "-Wnested-anon-types"
#సూచన clang diagnostic ignored "-Wlanguage-extension-token"
#సూచన clang diagnostic ignored "-Wgnu-anonymous-struct"
#సూచన clang diagnostic ignored "-Wmicrosoft-enum-value"

#అన్తర్భావ <AEEStdErr.h>
#అన్తర్భావ <dspqueue.h>
#అన్తర్భావ <rpcmem.h>

#నిర్వచించు GGML_COMMON_IMPL_CPP
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-alloc.h"
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-hexagon.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-quants.h"
#అన్తర్భావ "htp-opnode.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp/matmul-ops.h"
#అన్తర్భావ "htp/binary-ops.h"
#అన్తర్భావ "htp/flash-attn-ops.h"
#అన్తర్భావ "htp/unary-ops.h"
#అన్తర్భావ "htp/get-rows-ops.h"
#అన్తర్భావ "htp/set-rows-ops.h"
#అన్తర్భావ "htp/softmax-ops.h"
#అన్తర్భావ "htp/pool-ops.h"
#అన్తర్భావ "htp/rope-ops.h"
#అన్తర్భావ "htp/ssm-conv.h"
#అన్తర్భావ "htp/gated-delta-net-ops.h"
#అన్తర్భావ "htp/argsort-ops.h"
#అన్తర్భావ "htp/concat-ops.h"
#అన్తర్భావ "htp/cpy-ops.h"
#అన్తర్భావ "htp/im2col-ops.h"
#అన్తర్భావ "htp_iface.h"
#అన్తర్భావ "htp-drv.h"
#అన్తర్భావ "htp-repack.h"

ఉపయోగించి intvec  = std::vector<పూర్ణాంకం>;
ఉపయోగించి uintvec = std::vector<గుర్తులేని పూర్ణాంకం>;
ఉపయోగించి u32vec  = std::vector<uint32_t>;

#నిర్వచించు GGML_HEXAGON_MAX_SESSIONS          16

#నిర్వచించు GGML_HEXAGON_FENCE_స్థానం_కొలత       128

నిర్మాణం ggml_hexagon_device_config {
    పూర్ణాంకం         physical_idx = 0;
    పూర్ణాంకం         virtual_idx  = 0;
    పూర్ణాంకం         domain_id    = 0;
    std::string domain_name;
    std::string name;

    std::vector<ggml_hexagon_device_config> mdev_సమూహం;
};

స్థిర ggml_hexagon_device_config opt_device_configs[GGML_HEXAGON_MAX_SESSIONS];

స్థిర పూర్ణాంకం    opt_arch    = 0; // autodetect
స్థిర size_t opt_ndev    = 1;
స్థిర size_t opt_nhvx    = 0; // use all
స్థిర పూర్ణాంకం    opt_nhmx    = 1; // when set, enable HMX; when 0, use HVX only
స్థిర size_t opt_vmem    = HTP_OP_MAX_VMEM_DEFAULT;  // max available va space for buffer mappings
స్థిర పూర్ణాంకం    opt_etm     = 0;
స్థిర పూర్ణాంకం    opt_verbose = 0;
స్థిర పూర్ణాంకం    opt_profile = 0; // profiling mode (0-disabled, 1-basic, 2-pmu)
స్థిర తార్కిక   opt_hostbuf = అబద్ధం;
స్థిర తార్కిక   opt_dma64   = అబద్ధం;
స్థిర పూర్ణాంకం    opt_power_బహుళకం = HTP_POWER_బహుళకం_వ్యవధి; // DCVS/HMX power profile requested at session start

స్థిర స్థిరరాశి std::pair<స్థిరరాశి అక్షరం *, పూర్ణాంకం> htp_power_బహుళకం_పేర్లు[] = {
    { "low_power_saver",            HTP_POWER_బహుళకం_దిగువ_POWER_SAVER },
    { "power_saver",                HTP_POWER_బహుళకం_POWER_SAVER },
    { "high_power_saver",           HTP_POWER_బహుళకం_ఎగువ_POWER_SAVER },
    { "low_balanced",               HTP_POWER_బహుళకం_దిగువ_సమతుల్య },
    { "balanced",                   HTP_POWER_బహుళకం_సమతుల్య },
    { "high_performance",           HTP_POWER_బహుళకం_ఎగువ_PERFORMANCE },
    { "sustained_high_performance", HTP_POWER_బహుళకం_SUSTAINED_ఎగువ_PERFORMANCE },
    { "burst",                      HTP_POWER_బహుళకం_వ్యవధి },
    { "default",                    HTP_POWER_బహుళకం_వ్యవధి }, // flip this to change the default mode
};

స్థిర పూర్ణాంకం htp_power_బహుళకం_from_str(స్థిరరాశి అక్షరం * s) {
    కొరకు (స్వయంచాలక & e : htp_power_బహుళకం_పేర్లు) {
#ఒకవేళ defined(_MSC_VER)
        ఒకవేళ (_stricmp(s, e.మొదటి) == 0) {
#లేకపోతే
        ఒకవేళ (strcasecmp(s, e.మొదటి) == 0) {
#అయితే_ముగింపు
            తిరిగి e.second;
        }
    }
    తిరిగి -1;
}

స్థిర స్థిరరాశి అక్షరం * htp_power_బహుళకం_గమ్యం_str(పూర్ణాంకం mode) {
    కొరకు (స్వయంచాలక & e : htp_power_బహుళకం_పేర్లు) {
        ఒకవేళ (e.second == mode) {
            తిరిగి e.మొదటి;
        }
    }
    తిరిగి "unknown";
}

స్థిర size_t opt_mbuf_dyn    = 512ul * 1024 * 1024;      // max dynamic (compute) buffer size
స్థిర size_t opt_mbuf_static = 1ul * 1024 * 1024 * 1024; // max static (weight/KV) buffer size
స్థిర size_t opt_mbuf_మొత్తం  = 0;                        // total buffer space limit (0 = unconstrained)

స్థిర పూర్ణాంకం    opt_mm_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
స్థిర పూర్ణాంకం    opt_fa_select  = 2; // 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
స్థిర పూర్ణాంకం    opt_fa_తల_విభజించు = 1; // 1 = partition flash_attn by KV heads in multicore (default on), 0 = token-based (original)
స్థిర పూర్ణాంకం    opt_gdn_select = 2; // 2 = HMX -> HVX, 1 = HVX, 0 = CPU (unsupported)
స్థిర పూర్ణాంకం    opt_ar_select  = 2; // 2 = fused ALLREDUCE+ADD (default), 1 = unfused ALLREDUCE, 0 = fallback to CPY+FENCE
స్థిర పూర్ణాంకం    opt_ar_scatter = 1; // 1 = reduce-scatter the fused ALLREDUCE+ADD (default), 0 = full reduction

// Default PMU events, if profiling with PMU (mode=2) is enabled
// See https://docs.qualcomm.com/doc/80-N2040-60/topic/pmu-events.html
//     https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-pmu-events.html
స్థిర u32vec opt_pmu_evt { 0x3, 0x111, 0x100, 0x105, 0x240, 0x256, 0x7D, 0x8C };

స్థిర పూర్ణాంకం opt_opbatch  = 1280; // max number of ops in a batch
స్థిర పూర్ణాంకం opt_opqueue  = 8;   // max number of pending batches
స్థిర పూర్ణాంకం opt_optrace  = 0;    // trace buffer size per thread (0 means default)
స్థిర పూర్ణాంకం opt_oppoll   = 0;    // polling for batch completions
స్థిర పూర్ణాంకం opt_opfusion  = 1;    // enable/disable op fusion
స్థిర పూర్ణాంకం opt_rt_repack = 1;    // runtime repacking: 0 = disabled, 1 = quantized only (default), 2 = always

గణన ggml_hexagon_fusion_flags {
    GGML_HEXAGON_FUSE_ALLREDUCE_కూడు = (1 << 1), // 2
    GGML_HEXAGON_FUSE_RMS_NORM_MUL  = (1 << 2), // 4
    GGML_HEXAGON_FUSE_MUL_MAT_కూడు   = (1 << 3), // 8
    GGML_HEXAGON_FUSE_MUL_MAT_NX    = (1 << 4), // 16
    GGML_HEXAGON_FUSE_MUL_MAT_ID_NX = (1 << 5), // 32
    GGML_HEXAGON_FUSE_GDN_CPY       = (1 << 6), // 64
};

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_is_fusion_enabled(పూర్ణాంకం flag) {
    ఒకవేళ (opt_opfusion <= 0) తిరిగి అబద్ధం;
    ఒకవేళ (opt_opfusion == 1) తిరిగి నిజం; // 1 enables all
    తిరిగి (opt_opfusion & flag) != 0;
}

స్థిర std::regex* opt_opfilter = ఖాళీ; // regex of ops to not claim

#నిర్వచించు HEX_VERBOSE(...) \
    ఒకవేళ (opt_verbose) GGML_LOG_DEBUG(__VA_ARGS__)

స్థిర స్థిరరాశి అక్షరం * స్థితి_గమ్యం_str(uint32_t స్థితి_2) {
    ఎంపిక (స్థితి_2) {
        స్థితి HTP_స్థితి_OK:
            తిరిగి "OK";
        స్థితి HTP_స్థితి_NO_SUPPORT:
            తిరిగి "NO-SUPPORT";
        స్థితి HTP_స్థితి_INVAL_PARAMS:
            తిరిగి "INVAL-PARAMS";
        స్థితి HTP_స్థితి_VTCM_TOO_SMALL:
            తిరిగి "VTCM-TOO-SMALL";
        స్థితి HTP_స్థితి_అంతర్గత_ERR:
            తిరిగి "INTERNAL-ERROR";
        అప్రమేయ:
            తిరిగి "UNKNOWN";
    }
}

// ** debug helpers

స్థిర శూన్య ggml_hexagon_dump_op_exec(స్థిరరాశి std::string &sess_name, స్థిరరాశి htp_opnode & కణుపు, స్థిరరాశి uint32_t req_flags) {
    ఒకవేళ (!opt_verbose) తిరిగి;

    htp_opformat fmt(కణుపు);
    GGML_LOG_DEBUG("ggml-hex: %s execute-op %s|%s|%s|%s|%s|%s|%s|flags 0x%x\n", sess_name.c_str(),
                కణుపు.op_name().c_str(), fmt.పేర్లు, fmt.dims, fmt.types, fmt.strides, fmt.buffs, fmt.kparams, req_flags);
}

స్థిర శూన్య ggml_hexagon_dump_op_supp(స్థిరరాశి std::string &sess_name, స్థిరరాశి నిర్మాణం ggml_tensor * op, తార్కిక supp) {
    ఒకవేళ (!opt_verbose) తిరిగి;

    htp_opformat fmt(htp_opformat(htp_opnode(HTP_OP_INVALID, const_cast<ggml_tensor*>(op))));
    GGML_LOG_DEBUG("ggml-hex: %s supports-op %s|%s|%s|%s|%s|%s|%s\n", sess_name.c_str(),
                ggml_op_desc(op), fmt.పేర్లు, fmt.dims, fmt.types, fmt.strides, fmt.buffs, supp ? "yes" : "no");
}

స్థిర స్థిరరాశి అక్షరం * htp_event_name(uint16_t id) {
    ఎంపిక (id) {
        స్థితి HTP_TRACE_EVT_DMA:            తిరిగి "DMA";
        స్థితి HTP_TRACE_EVT_HVX_COMP:       తిరిగి "HVX_COMP";
        స్థితి HTP_TRACE_EVT_HVX_A_QUANT:    తిరిగి "HVX_A_QUANT";
        స్థితి HTP_TRACE_EVT_HVX_A_PREP:     తిరిగి "HVX_A_PREP";
        స్థితి HTP_TRACE_EVT_HVX_W_DEQUANT:  తిరిగి "HVX_W_DEQUANT";
        స్థితి HTP_TRACE_EVT_HVX_W_PREP:     తిరిగి "HVX_W_PREP";
        స్థితి HTP_TRACE_EVT_HVX_O_PROC:     తిరిగి "HVX_O_PROC";
        స్థితి HTP_TRACE_EVT_HVX_FA_QK:      తిరిగి "HVX_QK_FA";
        స్థితి HTP_TRACE_EVT_HVX_FA_SFM:     తిరిగి "HVX_SFM_FA";
        స్థితి HTP_TRACE_EVT_HVX_FA_Q_PREP:  తిరిగి "HVX_Q_PREP";
        స్థితి HTP_TRACE_EVT_HVX_FA_K_PREP:  తిరిగి "HVX_K_PREP";
        స్థితి HTP_TRACE_EVT_HVX_FA_V_PREP:  తిరిగి "HVX_V_PREP";
        స్థితి HTP_TRACE_EVT_HVX_GDN_PREP:   తిరిగి "HVX_GDN_PREP";
        స్థితి HTP_TRACE_EVT_HVX_GDN_SOLVE:  తిరిగి "HVX_GDN_SOLVE";
        స్థితి HTP_TRACE_EVT_HVX_GDN_V_PREP: తిరిగి "HVX_GDN_V_PREP";
        స్థితి HTP_TRACE_EVT_HVX_GDN_D_PREP: తిరిగి "HVX_GDN_D_PREP";
        స్థితి HTP_TRACE_EVT_HVX_GDN_బహిర్గతం:    తిరిగి "HVX_GDN_OUT";
        స్థితి HTP_TRACE_EVT_HVX_GDN_STATE:  తిరిగి "HVX_GDN_STATE";
        స్థితి HTP_TRACE_EVT_HVX_GDN_REM:    తిరిగి "HVX_GDN_REM";
        స్థితి HTP_TRACE_EVT_HMX_COMP:       తిరిగి "HMX_COMP";
        స్థితి HTP_TRACE_EVT_L2FLUSH:        తిరిగి "L2FLUSH";
        స్థితి HTP_TRACE_EVT_INIT:           తిరిగి "INIT";
        స్థితి HTP_TRACE_EVT_BUFF:           తిరిగి "BUFF";
        స్థితి HTP_TRACE_EVT_FENCE:          తిరిగి "FENCE";
        అప్రమేయ:                           తిరిగి "UNKNOWN";
    }
}

స్థిర శూన్య ggml_hexagon_dump_op_prof(స్థిరరాశి std::string &sess_name, స్థిరరాశి htp_opnode & కణుపు, స్థిరరాశి htp_prof_desc & pd) {
    ఒకవేళ (!opt_profile) తిరిగి;

    uint32_t op_usec = pd.usecs;
    uint32_t op_చక్రాలు = pd.చక్రాలు_stop - pd.చక్రాలు_ప్రారంభం;
    స్థిరరాశి uint32_t * pmu = pd.pmu;

    అక్షరం pmu_str[256] = "";
    ఒకవేళ (opt_profile == 2) {
        static_assert(HTP_PROF_PMU_NCNT == 8, "current implementation assumes 8 PMU counters");
        snprintf(pmu_str, పరిమాణం(pmu_str), " pmu [%u,%u,%u,%u,%u,%u,%u,%u]",
                pmu[0], pmu[1], pmu[2], pmu[3], pmu[4], pmu[5], pmu[6], pmu[7]);
    }

    htp_opformat fmt(కణుపు);
    దశాంశం mhz = op_usec > 0 ? (దశాంశం) op_చక్రాలు / op_usec : 0.0f;
    GGML_LOG_DEBUG("ggml-hex: %s profile-op %s|%s|%s|%s|%s|%s|usec %u cycles %u start %u mhz %.1f%s\n", sess_name.c_str(),
            కణుపు.op_name().c_str(), fmt.పేర్లు, fmt.dims, fmt.types, fmt.strides, fmt.kparams, op_usec, op_చక్రాలు, pd.చక్రాలు_ప్రారంభం, mhz, pmu_str);
}

స్థిర శూన్య ggml_hexagon_dump_batch_prof(స్థిరరాశి std::string & sess_name, స్థిరరాశి htp_opbatch_rsp & rsp) {
    uint64_t batch_చక్రాలు = rsp.చక్రాలు_stop - rsp.చక్రాలు_ప్రారంభం;
    దశాంశం batch_mhz = rsp.usecs > 0 ? (దశాంశం) batch_చక్రాలు / rsp.usecs : 0.0f;

    అక్షరం evt_str[256] = "----";
    ఒకవేళ (opt_profile == 3) {
        snprintf(evt_str, పరిమాణం(evt_str), "evt-cnt %u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
                rsp.n_traces[0], rsp.n_traces[1], rsp.n_traces[2], rsp.n_traces[3],
                rsp.n_traces[4], rsp.n_traces[5], rsp.n_traces[6], rsp.n_traces[7],
                rsp.n_traces[8], rsp.n_traces[9], rsp.n_traces[10]);
    }

    GGML_LOG_DEBUG("ggml-hex: %s profile-op OPBATCH|----|n-ops %u|%s|----|----|usec %u cycles %llu start %llu mhz %.1f\n",
                   sess_name.c_str(), rsp.n_ops, evt_str, rsp.usecs, (గుర్తులేని దీర్ఘ దీర్ఘ) batch_చక్రాలు, (గుర్తులేని దీర్ఘ దీర్ఘ) rsp.చక్రాలు_ప్రారంభం, batch_mhz);
}

స్థిర శూన్య ggml_hexagon_dump_trace_events(స్థిరరాశి std::string & sess_name, స్థిరరాశి htp_opbatch_rsp & rsp,
                                           స్థిరరాశి htp_trace_desc * trace_events, uint32_t n_traces) {
    ఒకవేళ (opt_profile == 3 && trace_events) {
        uint32_t valid_cnt[HTP_MAX_NTHREADS + 1] = {0};
        కొరకు (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            uint32_t count = rsp.n_traces[t];
            valid_cnt[t] = count > n_traces ? n_traces : count;
        }

        కొరకు (uint32_t t = 0; t <= HTP_MAX_NTHREADS; t++) {
            కొరకు (uint32_t idx = 0; idx < valid_cnt[t]; idx++) {
                స్థిరరాశి స్వయంచాలక & e = trace_events[t * n_traces + idx];
                తార్కిక is_stop = (e.info & 0x8000) != 0;
                uint16_t info = e.info & 0x7FFF;
                GGML_LOG_DEBUG("ggml-hex: %s trace-evt %s: thread %u info %u %s %u\n",
                               sess_name.c_str(), htp_event_name(e.id), t, info, is_stop ? "stop" : "start", e.చక్రాలు);
            }
        }
    }
}

గణన ggml_hexagon_tensor_flags {
    GGML_HEXAGON_TENSOR_TILED     = (1 << 0),
    GGML_HEXAGON_TENSOR_బరువు    = (1 << 1),
    GGML_HEXAGON_TENSOR_FENCE     = (1 << 2),
    GGML_HEXAGON_TENSOR_FUSEABLE  = (1 << 3),
};

నిర్మాణం ggml_hexagon_session;

స్థిర శూన్య ggml_hexagon_precompute_matmul_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_mm_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_fused_matmul_కూడు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * src2,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_mm_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_unary_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    uint32_t op,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_unary_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_precompute_ద్విమాన_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    uint32_t op,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_ద్విమాన_kernel_params * kparams,
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = శూన్యసూచిక
);

స్థిర శూన్య ggml_hexagon_precompute_get_అడ్డువరుసలు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_సమితి_అడ్డువరుసలు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_softmax_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_softmax_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_rope_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_rope_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_ssm_conv_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_ssm_conv_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_gated_delta_net_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_gdn_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_క్రమీకరించు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    తార్కిక is_శిఖరం_k,
    నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_pool_2d_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_pool_2d_kernel_params * kparams,
    తార్కిక is_pool_1d
);

స్థిర తార్కిక ggml_hexagon_precompute_concat_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_concat_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_precompute_cpy_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_copy_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_precompute_im2col_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_im2col_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_precompute_im2col_3d_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_im2col_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_fused_mmnx_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * act,
    int32_t n_weights,
    నిర్మాణం htp_mm_kernel_params * kparams
);

స్థిర శూన్య ggml_hexagon_precompute_fused_mmidnx_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * act,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    int32_t n_weights,
    నిర్మాణం htp_mm_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_precompute_allreduce_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    తార్కిక has_కూడు,
    తార్కిక is_అడ్డువరుస_bcast,
    తార్కిక is_shard_ok,
    నిర్మాణం htp_allreduce_kernel_params * kparams
);

స్థిర తార్కిక ggml_hexagon_అడ్డువరుసలు_stride(స్థిరరాశి int64_t * ne, స్థిరరాశి size_t * nb, size_t * stride);
స్థిర తార్కిక ggml_hexagon_matmul_can_collapse(స్థిరరాశి నిర్మాణం ggml_tensor * src0, స్థిరరాశి నిర్మాణం ggml_tensor * src1, స్థిరరాశి నిర్మాణం ggml_tensor * dst);
స్థిర ggml_tensor ggml_hexagon_tensor_collapse_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_tensor * t);

స్థిర తార్కిక mm_is_hmx_eligible(స్థిరరాశి ggml_tensor * t);
స్థిర htp_op_సంకేతం op_remap_గమ్యం_htp(స్థిరరాశి ggml_tensor * t);
స్థిర తార్కిక is_supported_mul_mat_nx_kernel(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams);
స్థిర తార్కిక is_supported_mul_mat_id_nx_kernel(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams);
స్థిర తార్కిక is_mergeable_mul_mat(స్థిరరాశి ggml_tensor * t);
స్థిర తార్కిక is_mergeable_mul_mat_pair(స్థిరరాశి ggml_tensor * n1, స్థిరరాశి ggml_tensor * n2);
స్థిర తార్కిక is_mergeable_mul_mat_id(స్థిరరాశి ggml_tensor * t);
స్థిర తార్కిక is_mergeable_mul_mat_id_pair(స్థిరరాశి ggml_tensor * n1, స్థిరరాశి ggml_tensor * n2);

// ** backend sessions

నిర్మాణం ggml_hexagon_tensor_extra {
    std::vector<uint8_t> shadow_buf;
    size_t               shadow_కొలత { 0 };
    uint32_t             flags { 0 };
};

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_tensor_is_fuseable(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (!t->extra) తిరిగి అబద్ధం;
    స్వయంచాలక extra = (స్థిరరాశి నిర్మాణం ggml_hexagon_tensor_extra *) t->extra;
    తిరిగి (extra->flags & GGML_HEXAGON_TENSOR_FUSEABLE) != 0;
}

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_can_rt_repack(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (opt_rt_repack == 0 || !htp_is_tiled_బరువు_tensor(t)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (t->బఫరు && ggml_backend_buft_is_host(t->బఫరు->buft)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి opt_rt_repack > 1 || ggml_is_quantized(t->type);
}

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_tensor_is_tiled(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    స్థిరరాశి స్వయంచాలక * extra = (స్థిరరాశి ggml_hexagon_tensor_extra *) t->extra;
    ఒకవేళ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        తిరిగి నిజం;
    }
    తిరిగి ggml_hexagon_can_rt_repack(t);
}

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_needs_repack(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    స్థిరరాశి స్వయంచాలక * extra = (స్థిరరాశి ggml_hexagon_tensor_extra *) t->extra;
    ఒకవేళ (extra && (extra->flags & GGML_HEXAGON_TENSOR_TILED)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి ggml_hexagon_can_rt_repack(t);
}

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_tensors_overlap(స్థిరరాశి నిర్మాణం ggml_tensor * a, స్థిరరాశి నిర్మాణం ggml_tensor * b) {
    స్థిరరాశి uintptr_t a0 = (uintptr_t) a->దత్తాంశం;
    స్థిరరాశి uintptr_t b0 = (uintptr_t) b->దత్తాంశం;
    స్థిరరాశి uintptr_t a1 = a0 + ggml_nbytes(a);
    స్థిరరాశి uintptr_t b1 = b0 + ggml_nbytes(b);

    తిరిగి a0 < b1 && b0 < a1;
}

స్థిర అంతఃవరుస తార్కిక ggml_hexagon_can_అడ్డువరుస_విభజన(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (t->ne[1] > 1 && (t->nb[1] & 127) != 0) తిరిగి అబద్ధం;
    ఒకవేళ (t->ne[2] > 1 && (t->nb[2] & 127) != 0) తిరిగి అబద్ధం;
    ఒకవేళ (t->ne[3] > 1 && (t->nb[3] & 127) != 0) తిరిగి అబద్ధం;
    తిరిగి నిజం;
}

నిర్మాణం htp_opnode;

నిర్మాణం ggml_hexagon_opbatch;
నిర్మాణం ggml_hexagon_opqueue;
నిర్మాణం ggml_hexagon_shared_బఫరు;
నిర్మాణం ggml_hexagon_fence_బఫరు;
నిర్మాణం ggml_hexagon_session;
నిర్మాణం ggml_backend_hexagon_device_context;

నిర్మాణం ggml_hexagon_mdev_సమూహం {
    uint32_t idx   = 0;
    uint32_t count = 1;
    std::vector<std::unique_ptr<ggml_hexagon_session>> sessions;
};

నిర్మాణం ggml_backend_hexagon_comm_context {
    std::vector<ggml_backend_t> backends;
    size_t                      n_backends = 0;
    అస్థిర uint32_t *         fence_slots[GGML_HEXAGON_MAX_SESSIONS] = {};
    ggml_tensor                 fence_tensors[GGML_HEXAGON_MAX_SESSIONS] = {};
};

నిర్మాణం ggml_hexagon_event {
    ggml_hexagon_session * sess         = శూన్యసూచిక;
    ggml_hexagon_session * fence_sess   = శూన్యసూచిక;
    అస్థిర uint32_t *    fence_స్థానం   = శూన్యసూచిక;
    ggml_tensor            fence_tensor = {};
    uint32_t               seq          = 0;
};

నిర్మాణం ggml_hexagon_session {
    std::string      name;
    remote_handle64  handle;
    dspqueue_t       queue;
    uint32_t         session_id;
    uint32_t         domain_id;
    uint64_t         వరుస_id;
    పూర్ణాంకం              phys_idx;
    పూర్ణాంకం              virt_idx;
    తార్కిక             valid_session;
    తార్కిక             valid_handle;
    తార్కిక             valid_వరుస;
    తార్కిక             valid_iface;

    ggml_hexagon_opbatch* op_batch;
    ggml_hexagon_opqueue* op_వరుస;

    std::unordered_map<పూర్ణాంకం, std::unique_ptr<ggml_hexagon_shared_బఫరు>> cloned_buffers;
    std::unordered_set<ggml_hexagon_session *>                           virt_peers;
    std::unordered_set<ggml_hexagon_session *>                           phys_peers;

    uint32_t n_దారాలు   = 0;
    uint32_t n_hvx       = 0;
    uint32_t n_hmx       = 0;
    uint64_t vtcm_కొలత   = 0;
    size_t   max_vmem    = 0;
    uint32_t fence_seq   = 0;

    std::atomic<uint64_t> batch_req_seq{0};
    std::atomic<uint64_t> batch_rsp_seq{0};
    std::atomic<uint32_t> చివరి_error{HTP_స్థితి_OK};

    uint64_t                cached_uid = 0;
    std::vector<htp_opnode> cached_nodes;

    ggml_hexagon_mdev_సమూహం                mdev;
    ggml_backend_dev_t                     dev       = శూన్యసూచిక;
    ggml_backend_hexagon_device_context *  dev_ctx   = శూన్యసూచిక;
    ggml_hexagon_fence_బఫరు *            fence_buf = శూన్యసూచిక;

    ggml_hexagon_session(స్థిరరాశి ggml_hexagon_device_config & config, ggml_backend_dev_t dev = శూన్యసూచిక, uint32_t mdev_idx = 0, uint32_t mdev_లెక్క = 0) noexcept(అబద్ధం);
    ~ggml_hexagon_session() noexcept(నిజం);

    స్థిరరాశి అక్షరం* c_name() స్థిరరాశి { తిరిగి name.c_str(); }

    శూన్య allocate(స్థిరరాశి ggml_hexagon_device_config & config) noexcept(అబద్ధం);
    శూన్య release() noexcept(నిజం);

    uint8_t * alloc_fence(uint32_t n_slots = 1);
    శూన్య      free_fence(శూన్య * ptr, uint32_t n_slots = 1);

    uint8_t *                                         mdev_fence_స్థానం = శూన్యసూచిక;
    std::unordered_map<uint64_t, అస్థిర uint32_t *> cpy_fence_slots;

    శూన్య వరుసలో_చేర్చు_mdev_సమూహం();
    శూన్య వరుసలో_చేర్చు_op(స్థిరరాశి htp_opnode & కణుపు);
    శూన్య వరుసలో_చేర్చు_cpy(స్థిరరాశి ggml_tensor * src, ggml_tensor * dst, స్థిరరాశి ggml_tensor * sync_tensor = శూన్యసూచిక, uint32_t fence_seq = 0);
    శూన్య వరుసలో_చేర్చు_fence(స్థిరరాశి ggml_tensor * sync_tensor, uint32_t fence_seq = 0, తార్కిక wait = నిజం);
    శూన్య వరుసలో_చేర్చు_allreduce(స్థిరరాశి ggml_tensor * dst, స్థిరరాశి std::vector<స్థిరరాశి ggml_tensor *> & src_tensors,
                           స్థిరరాశి std::vector<స్థిరరాశి ggml_tensor *> & sync_tensors, uint32_t rank, uint32_t n_ranks,
                           uint32_t fence_seq_నమోదు = 0, uint32_t fence_seq_exit = 0);

    శూన్య ప్రారంభం_batch();
    శూన్య flush_sync(తార్కిక all = నిజం);
    శూన్య flush_async();
    శూన్య flush_batch(size_t min_ops = 1);
    శూన్య flush_peers();
    శూన్య flush_pending(తార్కిక all = నిజం);

    ggml_hexagon_shared_బఫరు * mmap_tensor(స్థిరరాశి ggml_tensor * t);
    తార్కిక clone_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు*);
    శూన్య release_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు*);
    శూన్య unclone_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు*);

    శూన్య కూడు_peer(ggml_hexagon_session * peer) {
        ఒకవేళ (ఇది->phys_idx == peer->phys_idx) {
            virt_peers.insert(peer);
        } లేకపోతే {
            phys_peers.insert(peer);
        }
    }
};

// ** backend buffers

నిర్మాణం ggml_backend_hexagon_device_context {
    పూర్ణాంకం                        dev_id;
    ggml_hexagon_device_config config;
    ggml_backend_dev_t         dev = శూన్యసూచిక;

    ggml_backend_బఫరు_type బఫరు_type       = {};
    ggml_backend_బఫరు_type host_బఫరు_type  = {};
    ggml_backend_బఫరు_type fence_బఫరు_type = {};

    std::unique_ptr<ggml_hexagon_session> sess;

    ggml_backend_hexagon_device_context(పూర్ణాంకం dev_id, స్థిరరాశి ggml_hexagon_device_config & config, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_device_context();

    స్థిరరాశి అక్షరం * c_name() స్థిరరాశి { తిరిగి config.name.c_str(); }

    ggml_hexagon_session * session() {
        ఒకవేళ (!sess) {
            sess = std::make_unique<ggml_hexagon_session>(config, dev);
        }
        తిరిగి sess.get();
    }
};

నిర్మాణం ggml_backend_hexagon_బఫరు_type_context {
    ggml_backend_hexagon_బఫరు_type_context(స్థిరరాశి std::string & name, ggml_backend_hexagon_device_context * dev_ctx) {
        ఇది->dev_ctx = dev_ctx;
        ఇది->name    = name;
    }

    ggml_backend_hexagon_device_context * dev_ctx;
    std::string                           name;
};

నిర్మాణం ggml_hexagon_rpcmem_ఖండం {
    uint8_t * పునాది = శూన్యసూచిక;
    పూర్ణాంకం       fd   = -1;
    size_t    size = 0;

    std::unordered_set<ggml_hexagon_session *> mapped_clones;

    ggml_hexagon_rpcmem_ఖండం(size_t size) {
        పునాది = (uint8_t *) rpcmem_alloc2(RPCMEM_కుప్ప_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size);
        ఒకవేళ (!పునాది) {
            విసురు std::runtime_error("ggml-hex: rpcmem_alloc failed");
        }
        fd = rpcmem_గమ్యం_fd(పునాది);
        ఒకవేళ (fd < 0) {
            rpcmem_free(పునాది);
            విసురు std::runtime_error("ggml-hex: rpcmem_to_fd failed");
        }
        ఇది->size = size;
    }

    ~ggml_hexagon_rpcmem_ఖండం() {
        ఒకవేళ (పునాది) {
            rpcmem_free(పునాది);
        }
    }
};

నిర్మాణం ggml_hexagon_shared_బఫరు {
    ggml_hexagon_session *                     sess;
    std::shared_ptr<ggml_hexagon_rpcmem_ఖండం> mem;
    std::vector<ggml_hexagon_tensor_extra *>   tensor_extra;
    తార్కిక     mapped;
    తార్కిక     pinned;
    తార్కిక     extended;

    స్థిరరాశి అక్షరం * c_name() స్థిరరాశి { తిరిగి sess->c_name(); }
    uint8_t *    పునాది()   స్థిరరాశి { తిరిగి mem ? mem->పునాది : శూన్యసూచిక; }
    size_t       size()   స్థిరరాశి { తిరిగి mem ? mem->size : 0;  }
    పూర్ణాంకం          fd()     స్థిరరాశి { తిరిగి mem ? mem->fd   : -1; }

    శూన్య mmap(తార్కిక extended = అబద్ధం) {
        ఒకవేళ (!ఇది->mem)   తిరిగి;
        ఒకవేళ (ఇది->mapped) తిరిగి;

        GGML_ASSERT(!ఇది->pinned || !extended);

        ఇది->extended = extended;
        fastrpc_map_flags flags = ఇది->pinned ? FASTRPC_MAP_FD : (extended ? FASTRPC_MAP_FD_DELAYED_EXTENDED : FASTRPC_MAP_FD_DELAYED);

        పూర్ణాంకం err = fastrpc_mmap(sess->domain_id, fd(), (శూన్య *) పునాది(), 0, size(), flags);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: %s buffer mapping failed : domain_id %d size %zu fd %d error 0x%08x\n", sess->c_name(),
                    sess->domain_id, size(), fd(), (గుర్తులేని) err);
            విసురు std::runtime_error("ggml-hex: fastrpc_mmap failed (see log for details)");
        }

        HEX_VERBOSE("ggml-hex: %s mapped buffer: base %p size %zu fd %d pinned %u extended %u\n",
                sess->c_name(), (శూన్య *) పునాది(), size(), fd(), pinned, extended);

        ఇది->mapped = నిజం;
    }

    శూన్య unmap() {
        ఒకవేళ (!ఇది->mapped) తిరిగి;

        ఒకవేళ (!ఇది->pinned && mem) {
            // HTP might still hold a reference, tell it drop it
            htp_iface_munmap(sess->handle, fd());
        }

        ఒకవేళ (mem) {
            fastrpc_munmap(sess->domain_id, fd(), (శూన్య *) పునాది(), size());
        }

        HEX_VERBOSE("ggml-hex: %s unmapped buffer: base %p size %zu fd %d\n", sess->c_name(),
                (శూన్య *) పునాది(), size(), fd());

        ఇది->mapped = అబద్ధం;
    }

    శూన్య alloc(size_t size) {
        ఒకవేళ (ఇది->mem) తిరిగి;

        ఇది->mem = std::make_shared<ggml_hexagon_rpcmem_ఖండం>(size);

        HEX_VERBOSE("ggml-hex: %s allocated buffer: base %p size %zu fd %d pinned %d\n", sess->c_name(),
                    (శూన్య *) పునాది(), ఇది->size(), fd(), (పూర్ణాంకం) pinned);
        ఒకవేళ (ఇది->pinned) {
            mmap();
        }
    }

    శూన్య విడుదల() {
        unmap();
        // The memory is freed when the shared_ptr refcount drops to 0.
        HEX_VERBOSE("ggml-hex: %s release ref on buffer: base %p size %zu fd %d\n", sess->c_name(),
                    (శూన్య *) పునాది(), size(), fd());
        ఇది->mem  = శూన్యసూచిక;
    }

    ggml_hexagon_shared_బఫరు(ggml_hexagon_session * sess, size_t size, తార్కిక pinned = అబద్ధం) {
        ఇది->sess     = sess;
        ఇది->mapped   = అబద్ధం;
        ఇది->pinned   = pinned;
        ఇది->extended = అబద్ధం;

        // Size adjustment inside the buffer class: 4K aligned data size + 4K guard page
        size_t guard_స్థానభ్రంశం = (size + 4095) & ~4095;
        size_t మొత్తం_కొలత   = guard_స్థానభ్రంశం + 4096;
        ఒకవేళ (!pinned && opt_dma64) {
            constexpr size_t extended_align = 2 * 1024 * 1024;
            మొత్తం_కొలత = (మొత్తం_కొలత + extended_align - 1) & ~(extended_align - 1);
        }

        alloc(మొత్తం_కొలత);
    }

    // Clone constructor for cross-session mapping
    ggml_hexagon_shared_బఫరు(ggml_hexagon_session * sess, స్థిరరాశి ggml_hexagon_shared_బఫరు & other) {
        ఇది->sess   = sess;
        ఇది->mem    = other.mem;
        ఇది->mapped = అబద్ధం;
        ఇది->pinned = other.pinned;
        ఇది->extended = other.extended;
    }

    ~ggml_hexagon_shared_బఫరు() {
        విడుదల();
        కొరకు (స్వయంచాలక * extra : tensor_extra) {
            తొలగించు extra;
        }
    }
};

నిర్మాణం ggml_hexagon_fence_బఫరు : బహిరంగ ggml_hexagon_shared_బఫరు {
    uint32_t              స్థానం_లెక్క = 0;
    uint32_t              స్థానం_తల  = 0;
    std::vector<uint32_t> free_slots;
    ggml_backend_బఫరు   backend_బఫరు{};

    ggml_hexagon_fence_బఫరు(ggml_hexagon_session * sess, ggml_backend_బఫరు_type_t buft, size_t size)
        : ggml_hexagon_shared_బఫరు(sess, size, అబద్ధం /* pinned */),
          స్థానం_లెక్క(size / GGML_HEXAGON_FENCE_స్థానం_కొలత),
          స్థానం_తల(0) {
        backend_బఫరు.buft    = buft;
        backend_బఫరు.context = static_cast<ggml_hexagon_shared_బఫరు *>(ఇది);
        backend_బఫరు.size    = size;
        mmap(అబద్ధం);
    }

    uint8_t * alloc_స్థానం(uint32_t n_slots = 1) {
        uint8_t * ptr = శూన్యసూచిక;
        ఒకవేళ (n_slots == 1 && !free_slots.ఖాళీలు()) {
            uint32_t స్థానం = free_slots.back();
            free_slots.pop_back();
            ptr = పునాది() + (size_t) స్థానం * GGML_HEXAGON_FENCE_స్థానం_కొలత;
        } లేకపోతే ఒకవేళ (స్థానం_తల + n_slots <= స్థానం_లెక్క) {
            uint32_t స్థానం = స్థానం_తల;
            స్థానం_తల += n_slots;
            ptr = పునాది() + (size_t) స్థానం * GGML_HEXAGON_FENCE_స్థానం_కొలత;
        }
        ఒకవేళ (ptr) {
            memset(ptr, 0, (size_t) n_slots * GGML_HEXAGON_FENCE_స్థానం_కొలత);
        }
        తిరిగి ptr;
    }

    శూన్య free_స్థానం(శూన్య * ptr, uint32_t n_slots = 1) {
        ఒకవేళ (!ptr) తిరిగి;
        uint32_t స్థానం = ((uint8_t *) ptr - పునాది()) / GGML_HEXAGON_FENCE_స్థానం_కొలత;
        కొరకు (uint32_t i = 0; i < n_slots; i++) {
            free_slots.push_back(స్థానం + i);
        }
    }
};

అంతఃవరుస uint8_t * ggml_hexagon_session::alloc_fence(uint32_t n_slots) {
    uint8_t * ptr = fence_buf->alloc_స్థానం(n_slots);
    GGML_ASSERT(ptr);
    తిరిగి ptr;
}

అంతఃవరుస శూన్య ggml_hexagon_session::free_fence(శూన్య * ptr, uint32_t n_slots) {
    ఒకవేళ (fence_buf) {
        fence_buf->free_స్థానం(ptr, n_slots);
    }
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(బఫరు->context);
    sbuf->sess->unclone_బఫరు(sbuf);
    తొలగించు sbuf;
}

స్థిర శూన్య * ggml_backend_hexagon_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(బఫరు->context);
    తిరిగి sbuf->పునాది();
}

స్థిర గణన ggml_స్థితి ggml_backend_hexagon_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(బఫరు->context);
    స్వయంచాలక sess = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s init-tensor %s : base %p data %p nbytes %zu\n", sess->c_name(),
                tensor->name, (శూన్య *) sbuf->పునాది(), tensor->దత్తాంశం, ggml_nbytes(tensor));

    స్వయంచాలక extra = కొత్త ggml_hexagon_tensor_extra();
    sbuf->tensor_extra.push_back(extra);

    tensor->extra = extra;

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                   ggml_tensor *         tensor,
                                                   స్థిరరాశి శూన్య *          దత్తాంశం,
                                                   size_t                స్థానభ్రంశం,
                                                   size_t                size) {
    స్వయంచాలక extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    స్వయంచాలక sbuf  = (ggml_hexagon_shared_బఫరు *) బఫరు->context;
    స్వయంచాలక sess  = sbuf->sess;

    ఒకవేళ (ggml_backend_బఫరు_get_usage(బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_బరువు;
        ఒకవేళ (htp_is_tiled_బరువు_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
        sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, (పూర్ణాంకం) బఫరు->usage, extra->flags);

    ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);
        తిరిగి;
    }

    ఒకవేళ (స్థానభ్రంశం == 0 && size == ggml_nbytes(tensor) && extra->shadow_buf.ఖాళీలు()) {
        htp_repack_tensor_tiled(tensor, దత్తాంశం, size);
        తిరిగి;
    }

    ఒకవేళ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    జ్ఞాపకం_నకలు(extra->shadow_buf.దత్తాంశం() + స్థానభ్రంశం, దత్తాంశం, size);
    extra->shadow_కొలత += size;

    ఒకవేళ (extra->shadow_కొలత >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.దత్తాంశం(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_కొలత = 0;
    }
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                   స్థిరరాశి ggml_tensor *   tensor,
                                                   శూన్య *                దత్తాంశం,
                                                   size_t                స్థానభ్రంశం,
                                                   size_t                size) {
    స్వయంచాలక extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    స్వయంచాలక sbuf  = (ggml_hexagon_shared_బఫరు *) బఫరు->context;
    స్వయంచాలక sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor %s : data %p offset %zu size %zu usage %d flags 0x%x\n",
            sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, (పూర్ణాంకం) బఫరు->usage, extra->flags);

    ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size);
        తిరిగి;
    }

    htp_unpack_tensor_tiled(దత్తాంశం, tensor, స్థానభ్రంశం, size);
}

స్థిర తార్కిక ggml_backend_hexagon_బఫరు_cpy_tensor(ggml_backend_బఫరు_t      బఫరు,
                                                   స్థిరరాశి నిర్మాణం ggml_tensor * src,
                                                   నిర్మాణం ggml_tensor *       dst) {
    // we might optimize this later, for now take the slow path (ie get/set_tensor)
    తిరిగి అబద్ధం;

    GGML_వాడని(బఫరు);
    GGML_వాడని(src);
    GGML_వాడని(dst);
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_సమితి_tensor_2d(ggml_backend_బఫరు_t బఫరు,
                                                          ggml_tensor *         tensor,
                                                          స్థిరరాశి శూన్య *          దత్తాంశం,
                                                          size_t                స్థానభ్రంశం,
                                                          size_t                size,
                                                          size_t                n_copies,
                                                          size_t                stride_tensor,
                                                          size_t                stride_దత్తాంశం) {
    స్వయంచాలక extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    స్వయంచాలక sbuf  = (ggml_hexagon_shared_బఫరు *) బఫరు->context;
    స్వయంచాలక sess  = sbuf->sess;

    ఒకవేళ (ggml_backend_బఫరు_get_usage(బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS) {
        extra->flags |= GGML_HEXAGON_TENSOR_బరువు;
        ఒకవేళ (htp_is_tiled_బరువు_tensor(tensor)) {
            extra->flags |= GGML_HEXAGON_TENSOR_TILED;
        }
    }

    HEX_VERBOSE("ggml-hex: %s set-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d flags 0x%x\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం, (పూర్ణాంకం) బఫరు->usage, extra->flags);

    ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            జ్ఞాపకం_నకలు((uint8_t *) tensor->దత్తాంశం + స్థానభ్రంశం + i * stride_tensor, (స్థిరరాశి uint8_t *) దత్తాంశం + i * stride_దత్తాంశం, size);
        }
        తిరిగి;
    }

    ఒకవేళ (extra->shadow_buf.size() < ggml_nbytes(tensor)) {
        extra->shadow_buf.resize(ggml_nbytes(tensor));
    }
    కొరకు (size_t i = 0; i < n_copies; i++) {
        జ్ఞాపకం_నకలు(extra->shadow_buf.దత్తాంశం() + స్థానభ్రంశం + i * stride_tensor, (స్థిరరాశి uint8_t *) దత్తాంశం + i * stride_దత్తాంశం, size);
    }
    extra->shadow_కొలత += n_copies * size;

    ఒకవేళ (extra->shadow_కొలత >= ggml_nbytes(tensor)) {
        htp_repack_tensor_tiled(tensor, extra->shadow_buf.దత్తాంశం(), extra->shadow_buf.size());
        extra->shadow_buf.clear();
        extra->shadow_buf.shrink_to_fit();
        extra->shadow_కొలత = 0;
    }
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_get_tensor_2d(ggml_backend_బఫరు_t బఫరు,
                                                      స్థిరరాశి ggml_tensor *   tensor,
                                                      శూన్య *                దత్తాంశం,
                                                      size_t                స్థానభ్రంశం,
                                                      size_t                size,
                                                      size_t                n_copies,
                                                      size_t                stride_tensor,
                                                      size_t                stride_దత్తాంశం) {
    స్వయంచాలక extra = (ggml_hexagon_tensor_extra *)  tensor->extra;
    స్వయంచాలక sbuf  = (ggml_hexagon_shared_బఫరు *) బఫరు->context;
    స్వయంచాలక sess  = sbuf->sess;

    HEX_VERBOSE("ggml-hex: %s get-tensor-2d %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం, (పూర్ణాంకం) బఫరు->usage);

    ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) == 0) {
        కొరకు (size_t i = 0; i < n_copies; i++) {
            జ్ఞాపకం_నకలు((uint8_t *)దత్తాంశం + i * stride_దత్తాంశం, (స్థిరరాశి uint8_t *)tensor->దత్తాంశం + స్థానభ్రంశం + i * stride_tensor, size);
        }
        తిరిగి;
    }

    size_t తాత్కాలిక_కొలత      = n_copies > 0 ? (n_copies - 1) * stride_tensor + size : 0;
    size_t ముక్క_కొలత     = tensor->ne[1] * ggml_అడ్డువరుస_కొలత(tensor->type, tensor->ne[0]);
    size_t ముక్క_స్థానభ్రంశం   = స్థానభ్రంశం % ముక్క_కొలత;
    size_t అడ్డువరుస_కొలత_బైట్లు = ggml_అడ్డువరుస_కొలత(tensor->type, tensor->ne[0]);

    GGML_ASSERT((ముక్క_స్థానభ్రంశం % అడ్డువరుస_కొలత_బైట్లు) == 0 && "offset must be aligned to row boundary");
    GGML_ASSERT((తాత్కాలిక_కొలత % అడ్డువరుస_కొలత_బైట్లు)    == 0 && "temp_size must be a multiple of row size");
    GGML_ASSERT((స్థానభ్రంశం + తాత్కాలిక_కొలత) <= ggml_nbytes(tensor));

    std::vector<uint8_t> తాత్కాలిక_buf(తాత్కాలిక_కొలత);
    htp_unpack_tensor_tiled(తాత్కాలిక_buf.దత్తాంశం(), tensor, స్థానభ్రంశం, తాత్కాలిక_కొలత);

    కొరకు (size_t i = 0; i < n_copies; i++) {
        జ్ఞాపకం_నకలు((uint8_t *) దత్తాంశం + i * stride_దత్తాంశం, తాత్కాలిక_buf.దత్తాంశం() + i * stride_tensor, size);
    }
}

స్థిర శూన్య ggml_backend_hexagon_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    స్వయంచాలక sbuf = (ggml_hexagon_shared_బఫరు *) బఫరు->context;
    స్వయంచాలక sess = sbuf->sess;
    HEX_VERBOSE("ggml-hex: %s clear-buff base %p size %zu\n", sess->c_name(), (శూన్య *) sbuf->పునాది(), sbuf->size());
    memset(sbuf->పునాది(), value, sbuf->size());
}

స్థిర ggml_backend_బఫరు_i ggml_backend_hexagon_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_hexagon_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_hexagon_బఫరు_init_tensor,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_hexagon_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_hexagon_బఫరు_సమితి_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_hexagon_బఫరు_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_hexagon_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

// ** backend buffer type

స్థిర శూన్య ggml_backend_hexagon_host_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                        ggml_tensor *         tensor,
                                                        స్థిరరాశి శూన్య *          దత్తాంశం,
                                                        size_t                స్థానభ్రంశం,
                                                        size_t                size) {
    జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);
    GGML_వాడని(బఫరు);
}

స్థిర శూన్య ggml_backend_hexagon_host_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                        స్థిరరాశి ggml_tensor *   tensor,
                                                        శూన్య *                దత్తాంశం,
                                                        size_t                స్థానభ్రంశం,
                                                        size_t                size) {
    జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size);
    GGML_వాడని(బఫరు);
}

స్థిర ggml_backend_బఫరు_i ggml_backend_hexagon_host_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_hexagon_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_hexagon_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_hexagon_బఫరు_init_tensor,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_hexagon_host_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_hexagon_host_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_hexagon_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_hexagon_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

// ** backend buffer type

స్థిర స్థిరరాశి అక్షరం * ggml_backend_hexagon_బఫరు_type_name(ggml_backend_బఫరు_type_t బఫరు_type) {
    తిరిగి static_cast<ggml_backend_hexagon_బఫరు_type_context *>(బఫరు_type->context)->name.c_str();
}

స్థిర ggml_backend_బఫరు_t ggml_backend_hexagon_బఫరు_type_alloc_బఫరు(
            ggml_backend_బఫరు_type_t బఫరు_type, size_t size) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_బఫరు_type_context *>(బఫరు_type->context)->dev_ctx;
    స్వయంచాలక sess    = dev_ctx->session();
    ఒకవేళ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        తిరిగి శూన్యసూచిక;
    }
    ప్రయత్నించు {
        ggml_hexagon_shared_బఫరు * sbuf = కొత్త ggml_hexagon_shared_బఫరు(sess, size, అబద్ధం);
        తిరిగి ggml_backend_బఫరు_init(బఫరు_type, ggml_backend_hexagon_బఫరు_interface, sbuf, size);
    } పట్టుకో (స్థిరరాశి std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate device buffer context: %s\n", dev_ctx->c_name(), exc.what());
        తిరిగి శూన్యసూచిక;
    }
}

స్థిర ggml_backend_బఫరు_t ggml_backend_hexagon_host_బఫరు_type_alloc_బఫరు(
            ggml_backend_బఫరు_type_t బఫరు_type, size_t size) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_బఫరు_type_context *>(బఫరు_type->context)->dev_ctx;
    స్వయంచాలక sess    = dev_ctx->session();
    ఒకవేళ (sess && sess->max_vmem && size > sess->max_vmem) {
        GGML_LOG_ERROR("ggml-hex: %s host buffer size %zu exceeds max_vmem %zu\n",
                       dev_ctx->c_name(), size, sess->max_vmem);
        తిరిగి శూన్యసూచిక;
    }
    ప్రయత్నించు {
        ggml_hexagon_shared_బఫరు * sbuf = కొత్త ggml_hexagon_shared_బఫరు(sess, size, అబద్ధం);
        తిరిగి ggml_backend_బఫరు_init(బఫరు_type, ggml_backend_hexagon_host_బఫరు_interface, sbuf, size);
    } పట్టుకో (స్థిరరాశి std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s failed to allocate host buffer context: %s\n", dev_ctx->c_name(), exc.what());
        తిరిగి శూన్యసూచిక;
    }
}

స్థిర size_t ggml_backend_hexagon_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి 128;  // HVX alignment
    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (!ggml_backend_buft_is_host(buft) && htp_is_tiled_బరువు_tensor(t)) {
        int64_t ne0 = hex_విడత_పైకి(t->ne[0], 32);
        int64_t ne1 = hex_విడత_పైకి(t->ne[1], 32);
        int64_t ne2 = t->ne[2];
        int64_t ne3 = t->ne[3];
        size_t tiled_కొలత = htp_tiled_అడ్డువరుస_కొలత(t->type, ne0) * ne1 * ne2 * ne3;
        తిరిగి std::max(tiled_కొలత, ggml_nbytes(t));
    }
    తిరిగి ggml_nbytes(t);
}

స్థిర size_t parse_కొలత(స్థిరరాశి అక్షరం * str, size_t default_unit = 1024 * 1024) {
    ఒకవేళ (!str || str[0] == '\0') {
        తిరిగి 0;
    }
    అక్షరం * end = ఖాళీ;
    ద్విగుణ val = strtod(str, &end);
    ఒకవేళ (val < 0) {
        తిరిగి 0;
    }
    ఒకవేళ (end && *end) {
        అయితే (*end == ' ') end++;
        ఒకవేళ (*end == 'k' || *end == 'K') {
            తిరిగి (size_t) (val * 1024);
        } లేకపోతే ఒకవేళ (*end == 'm' || *end == 'M') {
            తిరిగి (size_t) (val * 1024 * 1024);
        } లేకపోతే ఒకవేళ (*end == 'g' || *end == 'G') {
            తిరిగి (size_t) (val * 1024 * 1024 * 1024);
        }
    }
    తిరిగి (size_t) (val * default_unit);
}

స్థిర size_t ggml_backend_hexagon_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    తిరిగి opt_mbuf_dyn;
    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_hexagon_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;
    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_hexagon_host_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి నిజం;
    GGML_వాడని(buft);
}

నిర్మాణం ggml_backend_hexagon_alloc_బఫరు_n_plan_అంశం {
    size_t size;
    పూర్ణాంకం    మొదటి;
    పూర్ణాంకం    చివరి;
};

ఉపయోగించి ggml_backend_hexagon_alloc_బఫరు_n_plan_t = std::vector<ggml_backend_hexagon_alloc_బఫరు_n_plan_అంశం>;

స్థిర స్థిరరాశి అక్షరం * ggml_hexagon_kv_layer_suffix(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    ఒకవేళ (అక్షరశ్రేణి_పోలిక(t->name, "cache_", 6) != 0) {
        తిరిగి ఖాళీ;
    }
    స్థిరరాశి అక్షరం * p = అక్షరశ్రేణి_శోధన(t->name, "_l");
    ఒకవేళ (!p || !అంకెనా((గుర్తులేని అక్షరం)p[2])) {
        తిరిగి ఖాళీ;
    }
    తిరిగి p;
}

నిర్మాణం ggml_backend_hexagon_alloc_unit {
    size_t size;
    పూర్ణాంకం    మొదటి;
    పూర్ణాంకం    చివరి;
};

స్థిర ggml_backend_hexagon_alloc_బఫరు_n_plan_t ggml_backend_hexagon_alloc_బఫరు_n_plan(
        ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    ggml_backend_hexagon_alloc_బఫరు_n_plan_t plan;

    స్థిరరాశి size_t alignment = ggml_backend_buft_get_alignment(buft);
    స్థిరరాశి size_t max_size  = opt_mbuf_static > 0 ? opt_mbuf_static : SIZE_MAX;

    std::vector<ggml_backend_hexagon_alloc_unit> units;

    పూర్ణాంకం i = 0;
    అయితే (i < n_tensors) {
        నిర్మాణం ggml_tensor * t = tensors[i];
        size_t unit_కొలత = 0;
        పూర్ణాంకం unit_మొదటి = i;
        పూర్ణాంకం unit_చివరి = i + 1;

        ఒకవేళ (t->దత్తాంశం == ఖాళీ && t->view_src == ఖాళీ) {
            unit_కొలత += GGML_PAD(ggml_backend_buft_get_alloc_కొలత(buft, t), alignment);
        }

        స్థిరరాశి అక్షరం * layer_suffix = ggml_hexagon_kv_layer_suffix(t);

        అయితే (unit_చివరి < n_tensors) {
            నిర్మాణం ggml_tensor * తదుపరి = tensors[unit_చివరి];

            ఒకవేళ (తదుపరి->view_src != ఖాళీ) {
                unit_చివరి++;
                కొనసాగు;
            }

            ఒకవేళ (layer_suffix != ఖాళీ) {
                స్థిరరాశి అక్షరం * తదుపరి_suffix = ggml_hexagon_kv_layer_suffix(తదుపరి);
                ఒకవేళ (తదుపరి_suffix != ఖాళీ && strcmp(layer_suffix, తదుపరి_suffix) == 0) {
                    ఒకవేళ (తదుపరి->దత్తాంశం == ఖాళీ) {
                        unit_కొలత += GGML_PAD(ggml_backend_buft_get_alloc_కొలత(buft, తదుపరి), alignment);
                    }
                    unit_చివరి++;
                    కొనసాగు;
                }
            }

            ఆపు;
        }

        units.push_back({ unit_కొలత, unit_మొదటి, unit_చివరి });
        i = unit_చివరి;
    }

    size_t cur_buf_కొలత  = 0;
    పూర్ణాంకం    cur_buf_మొదటి = 0;

    కొరకు (స్థిరరాశి స్వయంచాలక & unit : units) {
        ఒకవేళ (unit.size == 0) {
            కొనసాగు;
        }

        ఒకవేళ (cur_buf_కొలత > 0 && (cur_buf_కొలత + unit.size) > max_size) {
            plan.push_back({ cur_buf_కొలత, cur_buf_మొదటి, unit.మొదటి });
            cur_buf_కొలత  = 0;
            cur_buf_మొదటి = unit.మొదటి;
        }

        cur_buf_కొలత += unit.size;
    }

    ఒకవేళ (cur_buf_కొలత > 0) {
        plan.push_back({ cur_buf_కొలత, cur_buf_మొదటి, n_tensors });
    }

    తిరిగి plan;
}

స్థిర ggml_backend_బఫరు_t ggml_backend_hexagon_బఫరు_type_alloc_బఫరు_n(
        ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    స్థిరరాశి ggml_backend_hexagon_alloc_బఫరు_n_plan_t plan = ggml_backend_hexagon_alloc_బఫరు_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_బఫరు_t> buffers;
    buffers.reserve(plan.size());

    కొరకు (స్థిరరాశి స్వయంచాలక & అంశం : plan) {
        ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు(buft, అంశం.size);
        ఒకవేళ (బఫరు == ఖాళీ) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), అంశం.size);
            కొరకు (ggml_backend_బఫరు_t b : buffers) {
                ggml_backend_బఫరు_free(b);
            }
            తిరిగి ఖాళీ;
        }

        నిర్మాణం ggml_tallocr tallocr = ggml_tallocr_new(బఫరు);

        నిర్మాణం ggml_tensor * t_failed = ఖాళీ;
        కొరకు (పూర్ణాంకం j = అంశం.మొదటి; j < అంశం.చివరి; j++) {
            నిర్మాణం ggml_tensor * t = tensors[j];
            ఒకవేళ (t->దత్తాంశం == ఖాళీ) {
                ఒకవేళ (t->view_src == ఖాళీ) {
                    ఒకవేళ (ggml_tallocr_alloc(&tallocr, t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                } లేకపోతే ఒకవేళ (t->బఫరు == ఖాళీ) {
                    ఒకవేళ (ggml_backend_view_init(t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                }
            } లేకపోతే {
                ఒకవేళ (t->view_src != ఖాళీ && t->బఫరు == ఖాళీ) {
                    ఒకవేళ (ggml_backend_view_init(t) != GGML_స్థితి_SUCCESS) {
                        t_failed = t;
                        ఆపు;
                    }
                }
            }
        }
        ఒకవేళ (t_failed != ఖాళీ) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            కొరకు (ggml_backend_బఫరు_t b : buffers) {
                ggml_backend_బఫరు_free(b);
            }
            ggml_backend_బఫరు_free(బఫరు);
            తిరిగి ఖాళీ;
        }

        buffers.push_back(బఫరు);
    }

    ఒకవేళ (buffers.ఖాళీలు()) {
        తిరిగి ఖాళీ;
    }

    ఒకవేళ (buffers.size() == 1) {
        తిరిగి buffers[0];
    }

    తిరిగి ggml_backend_multi_బఫరు_alloc_బఫరు(buffers.దత్తాంశం(), buffers.size());
}

స్థిర size_t ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత_n(
        ggml_backend_బఫరు_type_t buft, నిర్మాణం ggml_tensor ** tensors, పూర్ణాంకం n_tensors) {
    స్థిరరాశి ggml_backend_hexagon_alloc_బఫరు_n_plan_t plan = ggml_backend_hexagon_alloc_బఫరు_n_plan(buft, tensors, n_tensors);

    size_t మొత్తం = 0;
    కొరకు (స్థిరరాశి స్వయంచాలక & అంశం : plan) {
        మొత్తం += అంశం.size;
    }
    తిరిగి మొత్తం;
}

స్థిర ggml_backend_బఫరు_type_i ggml_backend_hexagon_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_బఫరు_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_బఫరు_type_alloc_బఫరు_n,
    /* .get_alignment       = */ ggml_backend_hexagon_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత_n,
    /* .is_host             = */ ggml_backend_hexagon_బఫరు_type_is_host,
};

స్థిర ggml_backend_బఫరు_type_i ggml_backend_hexagon_host_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_hexagon_బఫరు_type_name,
    /* .alloc_buffer        = */ ggml_backend_hexagon_host_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ggml_backend_hexagon_బఫరు_type_alloc_బఫరు_n,
    /* .get_alignment       = */ ggml_backend_hexagon_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_hexagon_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ggml_backend_hexagon_బఫరు_type_get_alloc_కొలత_n,
    /* .is_host             = */ ggml_backend_hexagon_host_బఫరు_type_is_host,
};

ggml_backend_hexagon_device_context::ggml_backend_hexagon_device_context(పూర్ణాంకం dev_id, స్థిరరాశి ggml_hexagon_device_config & config, ggml_backend_dev_t dev)
    : dev_id(dev_id), config(config), dev(dev) {
    బఫరు_type.device  = dev;
    బఫరు_type.iface   = ggml_backend_hexagon_బఫరు_type_interface;
    బఫరు_type.context = కొత్త ggml_backend_hexagon_బఫరు_type_context(config.name, ఇది);

    host_బఫరు_type.device  = dev;
    host_బఫరు_type.iface   = ggml_backend_hexagon_host_బఫరు_type_interface;
    host_బఫరు_type.context = కొత్త ggml_backend_hexagon_బఫరు_type_context(config.name + "-HOST", ఇది);

    fence_బఫరు_type.device  = dev;
    fence_బఫరు_type.iface   = ggml_backend_hexagon_బఫరు_type_interface;
    fence_బఫరు_type.context = కొత్త ggml_backend_hexagon_బఫరు_type_context(config.name + "-FENCE", ఇది);
}

ggml_backend_hexagon_device_context::~ggml_backend_hexagon_device_context() {
    తొలగించు static_cast<ggml_backend_hexagon_బఫరు_type_context *>(బఫరు_type.context);
    తొలగించు static_cast<ggml_backend_hexagon_బఫరు_type_context *>(host_బఫరు_type.context);
    తొలగించు static_cast<ggml_backend_hexagon_బఫరు_type_context *>(fence_బఫరు_type.context);
}

స్థిర తార్కిక ggml_backend_బఫరు_is_hexagon(స్థిరరాశి నిర్మాణం ggml_backend_బఫరు * b) {
    తిరిగి b->buft->iface.get_alignment == ggml_backend_hexagon_బఫరు_type_get_alignment;
}

నిర్మాణం ggml_hexagon_opbatch {
    ggml_hexagon_session*            sess;

    std::vector<htp_opnode>          ops;       // htp_opnode of ops

    std::vector<htp_buf_desc>        h_bufs;    // htp buffer descriptors
    std::vector<htp_tensor>          h_tens;    // htp tensor descriptors
    std::vector<htp_op_desc>         h_ops;     // htp op descriptors

    std::unordered_map<పూర్ణాంకం, పూర్ణాంకం>                b_map; // buffer fd   to index
    std::unordered_map<స్థిరరాశి ggml_tensor*, పూర్ణాంకం> t_map; // tensor ptr  to index
    std::unordered_multimap<శూన్య*, పూర్ణాంకం>         d_map; // tensor data to index

    గుర్తులేని పూర్ణాంకం n_bufs;     // num buffers in the batch
    గుర్తులేని పూర్ణాంకం n_tens;     // num tensors ...
    గుర్తులేని పూర్ణాంకం n_ops;      // num ops ...
    size_t       b_vmem;     // sum of non-extended buffer sizes

    గుర్తులేని పూర్ణాంకం n_bufs_max;
    గుర్తులేని పూర్ణాంకం n_tens_max;
    గుర్తులేని పూర్ణాంకం n_ops_max;
    size_t       b_vmem_max;

    శూన్య reset() {
        n_bufs = 0;
        n_tens = 0;
        n_ops  = 0;
        b_vmem = 0;

        b_map.clear();
        t_map.clear();
        d_map.clear();
        ops.resize(n_ops_max);
    }

    ggml_hexagon_opbatch(ggml_hexagon_session *sess, size_t batch_కొలత, size_t max_vmem) {
        ఇది->sess = sess;

        n_bufs_max = HTP_OP_MAX_BUFS;
        n_ops_max  = batch_కొలత;
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

    తార్కిక ఖాళీలు() స్థిరరాశి { తిరిగి n_ops == 0; }

    // add buffer and return its index
    పూర్ణాంకం కూడు_బఫరు(ggml_hexagon_shared_బఫరు * sbuf) {
        // Lookup by fd
        స్వయంచాలక it = b_map.find(sbuf->fd());
        ఒకవేళ (it != b_map.end()) { తిరిగి it->second; }

        // Add new buffer to the batch
        GGML_ASSERT(n_bufs < HTP_OP_MAX_BUFS);
        పూర్ణాంకం bi = n_bufs++;

        b_map.insert({sbuf->fd(), bi});

        htp_buf_desc &b = h_bufs[bi];
        b.పునాది  = (uint64_t) sbuf->పునాది();
        b.fd    = sbuf->fd();
        b.size  = sbuf->size();
        b.flags = sbuf->extended ? HTP_BUF_EXTENDED : 0;

        ఒకవేళ (!sbuf->extended) {
            b_vmem += b.size;
        }

        HEX_VERBOSE("ggml-hex: %s add-buffer #%u : fd %d base %p size %zu : vmem %zu\n", sess->c_name(), bi, b.fd, (శూన్య*) sbuf->పునాది(), (size_t) b.size, b_vmem);

        తిరిగి bi;
    }

    తార్కిక same_shape(స్థిరరాశి htp_tensor * h, స్థిరరాశి ggml_tensor * t) స్థిరరాశి {
        స్వయంచాలక extra = (ggml_hexagon_tensor_extra *) t->extra;

        స్థిరరాశి తార్కిక is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        స్థిరరాశి int64_t ne0_padded = hex_విడత_పైకి(t->ne[0], 32);
        స్థిరరాశి int64_t ne1_padded = hex_విడత_పైకి(t->ne[1], 32);

        int64_t nb1 = is_tiled ? (int64_t) htp_tiled_అడ్డువరుస_కొలత(t->type, ne0_padded) : t->nb[1];
        int64_t nb2 = is_tiled ? nb1 * ne1_padded : t->nb[2];
        int64_t nb3 = is_tiled ? nb2 * t->ne[2]   : t->nb[3];
        int64_t nb0 = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? పరిమాణం(ggml_fp16_t) : t->nb[0];
        గణన ggml_type expected_type = (is_tiled && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : t->type;

        తిరిగి (h->type == expected_type) &&
               (h->ne[0] == t->ne[0]) && (h->ne[1] == t->ne[1]) && (h->ne[2] == t->ne[2]) && (h->ne[3] == t->ne[3]) &&
               (h->nb[0] == nb0) && (h->nb[1] == nb1) && (h->nb[2] == nb2) && (h->nb[3] == nb3);
    }

    // add tensor and return its index
    పూర్ణాంకం కూడు_tensor(స్థిరరాశి ggml_tensor * t) {
        స్వయంచాలక extra = (ggml_hexagon_tensor_extra *) t->extra;
        స్వయంచాలక sbuf  = static_cast<ggml_hexagon_shared_బఫరు *>(t->బఫరు->context);

        // First lookup by tensor data
        స్వయంచాలక range = d_map.equal_range(t->దత్తాంశం);
        కొరకు (స్వయంచాలక it = range.మొదటి; it != range.second; ++it) {
            htp_tensor * h = &h_tens[it->second];
            ఒకవేళ (same_shape(h, t)) { తిరిగి it->second; }
        }

        // Lookup by tensor ptr
        స్వయంచాలక it = t_map.find(t);
        ఒకవేళ (it != t_map.end()) { తిరిగి it->second; }

        // Add new tensor to the batch
        పూర్ణాంకం ti = n_tens++;
        GGML_ASSERT(n_tens <= n_tens_max);

        t_map.insert({t,       ti});
        d_map.insert({t->దత్తాంశం, ti});

        uint64_t t_స్థానభ్రంశం = (uint8_t *) t->దత్తాంశం - sbuf->పునాది();
        size_t   t_కొలత   = ggml_nbytes(t);

        htp_tensor &h = h_tens[ti];
        h.bi    = కూడు_బఫరు(sbuf);
        h.ti    = ti;
        h.దత్తాంశం  = t_స్థానభ్రంశం;
        h.type  = t->type;
        h.ne[0] = t->ne[0];
        h.ne[1] = t->ne[1];
        h.ne[2] = t->ne[2];
        h.ne[3] = t->ne[3];

        స్థిరరాశి తార్కిక is_tiled = (extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0;
        ఒకవేళ (is_tiled) {
            ఒకవేళ (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_BF16) {
                h.type  = GGML_TYPE_F16;
                h.nb[0] = పరిమాణం(ggml_fp16_t);
            } లేకపోతే {
                h.nb[0] = t->nb[0];
            }
            స్థిరరాశి int64_t ne0_padded = hex_విడత_పైకి(t->ne[0], 32);
            స్థిరరాశి int64_t ne1_padded = hex_విడత_పైకి(t->ne[1], 32);
            h.nb[1] = htp_tiled_అడ్డువరుస_కొలత(t->type, ne0_padded);
            h.nb[2] = h.nb[1] * ne1_padded;
            h.nb[3] = h.nb[2] * h.ne[2];
            h.size  = h.nb[3] * h.ne[3];
            t_కొలత  = h.size;
        } లేకపోతే {
            h.size  = t_కొలత;
            h.nb[0] = t->nb[0]; h.nb[1] = t->nb[1]; h.nb[2] = t->nb[2]; h.nb[3] = t->nb[3];
        }

        h.flags = 0;
        ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_బరువు) != 0) {
            h.flags |= HTP_TENSOR_బరువు;
        }
        ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_TILED) != 0) {
            h.flags |= HTP_TENSOR_TILED;
        }
        ఒకవేళ ((extra->flags & GGML_HEXAGON_TENSOR_FENCE) != 0) {
            h.flags |= HTP_TENSOR_FENCE;
        }

        HEX_VERBOSE("ggml-hex: %s add-tensor #%u %s : bi %d data %p offset %zu size %zu flags 0x%x : %zu:%zu:%zu:%zu\n", sess->c_name(),
                ti, t->name, h.bi, (శూన్య*) t->దత్తాంశం, (size_t) t_స్థానభ్రంశం, t_కొలత, h.flags,
                (size_t) h.ne[0], (size_t) h.ne[1], (size_t) h.ne[2], (size_t) h.ne[3]);

        తిరిగి ti;
    }

    తార్కిక fit_op(స్థిరరాశి htp_opnode & కణుపు) స్థిరరాశి {
        ఒకవేళ (n_ops >= n_ops_max) తిరిగి అబద్ధం;

        // check how much extras we will need
        size_t extra_bufs = 0;
        size_t extra_vmem = 0;
        size_t extra_tens = 0;

        పూర్ణాంకం చూసిన_bufs[HTP_OP_MAX_BUFS];
        పూర్ణాంకం n_చూసిన_bufs = 0;

        స్వయంచాలక fit_tensor = [&](స్థిరరాశి ggml_tensor *t) {
            ఒకవేళ (!t) తిరిగి;
            ఒకవేళ (!t_map.count(t)) {
                extra_tens++;

                స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(t->బఫరు->context);
                పూర్ణాంకం fd = sbuf->fd();
                ఒకవేళ (!b_map.count(fd)) {
                    కొరకు (పూర్ణాంకం i = 0; i < n_చూసిన_bufs; i++) {
                        ఒకవేళ (చూసిన_bufs[i] == fd) తిరిగి;
                    }
                    ఒకవేళ (n_చూసిన_bufs < HTP_OP_MAX_BUFS) {
                        చూసిన_bufs[n_చూసిన_bufs++] = fd;
                    }
                    ఒకవేళ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        };

        కొరకు (స్థిరరాశి స్వయంచాలక * src : కణుపు.get_inputs()) {
            fit_tensor(src);
        }
        కొరకు (స్థిరరాశి స్వయంచాలక * output : కణుపు.get_outputs()) {
            fit_tensor(output);
        }

        ఒకవేళ ((extra_bufs + n_bufs) > n_bufs_max) తిరిగి అబద్ధం;
        ఒకవేళ ((extra_tens + n_tens) > n_tens_max) తిరిగి అబద్ధం;
        ఒకవేళ ((extra_vmem + b_vmem) > b_vmem_max) తిరిగి అబద్ధం;

        తిరిగి నిజం;
    }

    // assumes that fit_op() was called first and returned true
    శూన్య కూడు_op(స్థిరరాశి htp_opnode & కణుపు) {
        // Add new op

        గుర్తులేని పూర్ణాంకం n = n_ops++;
        GGML_ASSERT(n_ops <= n_ops_max);

        ops[n] = కణుపు;

        htp_op_desc &o = h_ops[n];
        జ్ఞాపకం_నకలు(o.params,        కణుపు.కణుపు->op_params, పరిమాణం(కణుపు.కణుపు->op_params));
        జ్ఞాపకం_నకలు(o.kernel_params, కణుపు.kernel_params,   పరిమాణం(o.kernel_params));
        o.opcode = కణుపు.opcode;
        o.flags  = 0;

        ggml_hexagon_dump_op_exec(sess->c_name(), ops[n], o.flags);

        స్వయంచాలక inputs = కణుపు.get_inputs();
        కొరకు (గుర్తులేని పూర్ణాంకం i=0; i < HTP_OP_MAX_INPUTS; i++) {
            o.src[i] = (i < inputs.size() && inputs[i])   ? కూడు_tensor(inputs[i]) : 0xffff;
        }

        స్వయంచాలక outputs = కణుపు.get_outputs();
        కొరకు (గుర్తులేని పూర్ణాంకం i=0; i < HTP_OP_MAX_OUTPUTS; i++) {
            o.dst[i] = (i < outputs.size() && outputs[i]) ? కూడు_tensor(outputs[i]) : 0xffff;
        }
    }

    శూన్య క్రమీకరించు_buffers() {
        ఒకవేళ (n_bufs <= 1) తిరిగి;

        std::vector<పూర్ణాంకం> క్రమం(n_bufs);
        కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_bufs; i++) { క్రమం[i] = (పూర్ణాంకం) i; }

        std::stable_sort(క్రమం.begin(), క్రమం.end(), [&](పూర్ణాంకం a, పూర్ణాంకం b) {
            తిరిగి h_bufs[a].size > h_bufs[b].size;
        });

        తార్కిక already_sorted = నిజం;
        కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_bufs; i++) {
            ఒకవేళ (క్రమం[i] != (పూర్ణాంకం) i) {
                already_sorted = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (already_sorted) తిరిగి;

        std::vector<uint16_t> remap(n_bufs);
        std::vector<htp_buf_desc> sorted_bufs(n_bufs);
        కొరకు (గుర్తులేని పూర్ణాంకం new_bi = 0; new_bi < n_bufs; new_bi++) {
            పూర్ణాంకం old_bi = క్రమం[new_bi];
            remap[old_bi] = (uint16_t) new_bi;
            sorted_bufs[new_bi] = h_bufs[old_bi];
        }

        కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_bufs; i++) {
            h_bufs[i] = sorted_bufs[i];
        }

        కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < n_tens; i++) {
            h_tens[i].bi = remap[h_tens[i].bi];
        }
    }

    శూన్య update_mdev_సమూహం(uint32_t mdev_idx) {
        ఒకవేళ (n_ops > 0 && h_ops[0].opcode == HTP_OP_MDEV_సమూహం) {
            h_ops[0].params[0] = (int32_t) mdev_idx;
        }
    }

    తార్కిక try_fuse_common(std::initializer_list<స్థిరరాశి ggml_tensor *> tensors) స్థిరరాశి {
        size_t extra_bufs = 0, extra_vmem = 0, extra_tens = 0;

        పూర్ణాంకం చూసిన_bufs[HTP_OP_MAX_BUFS];
        పూర్ణాంకం n_చూసిన_bufs = 0;

        కొరకు (స్థిరరాశి స్వయంచాలక * t : tensors) {
            ఒకవేళ (!t || t_map.count(t)) {
                కొనసాగు;
            }
            extra_tens++;
            స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(t->బఫరు->context);
            పూర్ణాంకం fd = sbuf->fd();
            ఒకవేళ (!b_map.count(fd)) {
                తార్కిక దొరికింది = అబద్ధం;
                కొరకు (పూర్ణాంకం i = 0; i < n_చూసిన_bufs; i++) {
                    ఒకవేళ (చూసిన_bufs[i] == fd) {
                        దొరికింది = నిజం;
                        ఆపు;
                    }
                }
                ఒకవేళ (!దొరికింది) {
                    ఒకవేళ (n_చూసిన_bufs < HTP_OP_MAX_BUFS) {
                        చూసిన_bufs[n_చూసిన_bufs++] = fd;
                    }
                    ఒకవేళ (!sbuf->extended) {
                        extra_vmem += sbuf->size();
                    }
                    extra_bufs += 1;
                }
            }
        }

        ఒకవేళ ((extra_bufs + n_bufs) > n_bufs_max || (extra_tens + n_tens) > n_tens_max || (extra_vmem + b_vmem) > b_vmem_max) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    తార్కిక try_fuse_common(స్థిరరాశి ggml_tensor * t1, స్థిరరాశి ggml_tensor * t2) స్థిరరాశి {
        తిరిగి try_fuse_common({t1, t2});
    }

    తార్కిక try_fuse_allreduce_కూడు(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0 || opt_ar_select != 2) తిరిగి అబద్ధం;
        ఒకవేళ (కణుపు.opcode != HTP_OP_కూడు) తిరిగి అబద్ధం;

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];
        ఒకవేళ (చివరి_కణుపు.opcode != HTP_OP_ALLREDUCE) తిరిగి అబద్ధం;

        స్వయంచాలక * ar_kparams = (నిర్మాణం htp_allreduce_kernel_params *) చివరి_కణుపు.kernel_params;
        స్థిరరాశి uint32_t rank    = (uint32_t) ar_kparams->rank;
        స్థిరరాశి uint32_t n_ranks = (uint32_t) ar_kparams->n_ranks;
        స్థిరరాశి ggml_tensor * ar_local = చివరి_కణుపు.inputs[rank];
        స్థిరరాశి ggml_tensor * కూడు_src0 = కణుపు.src0();
        స్థిరరాశి ggml_tensor * కూడు_src1 = కణుపు.src1();
        స్థిరరాశి ggml_tensor * కూడు_dst  = కణుపు.dst();

        ఒకవేళ (!ggml_hexagon_tensor_is_fuseable(ar_local)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * res_tensor;
        ఒకవేళ (కూడు_src0 == ar_local || కూడు_src0->దత్తాంశం == ar_local->దత్తాంశం) {
            res_tensor = కూడు_src1;
        } లేకపోతే ఒకవేళ (కూడు_src1 == ar_local || కూడు_src1->దత్తాంశం == ar_local->దత్తాంశం) {
            res_tensor = కూడు_src0;
        } లేకపోతే {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (ar_local->type != res_tensor->type) తిరిగి అబద్ధం;

        స్థిరరాశి తార్కిక is_same_shape = (ar_local->ne[0] == res_tensor->ne[0] && ar_local->ne[1] == res_tensor->ne[1] &&
                                    ar_local->ne[2] == res_tensor->ne[2] && ar_local->ne[3] == res_tensor->ne[3]);
        స్థిరరాశి తార్కిక is_అడ్డువరుస_bcast  = !is_same_shape && (ar_local->ne[0] == res_tensor->ne[0] && res_tensor->ne[1] == 1 &&
                                                      res_tensor->ne[2] == 1 && res_tensor->ne[3] == 1);

        ఒకవేళ (!is_same_shape && !is_అడ్డువరుస_bcast) తిరిగి అబద్ధం;

        ఒకవేళ (is_same_shape) {
            ఒకవేళ (ar_local->nb[1] != res_tensor->nb[1] || ar_local->nb[2] != res_tensor->nb[2] ||
                ar_local->nb[3] != res_tensor->nb[3]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(res_tensor)) {
                తిరిగి అబద్ధం;
            }
        }
        ఒకవేళ (ggml_is_contiguous(ar_local) != ggml_is_contiguous(కూడు_dst)) {
            తిరిగి అబద్ధం;
        }

        // scatter is only valid for in-place add within max outputs
        స్థిరరాశి తార్కిక is_shard_ok = n_ranks <= HTP_OP_MAX_OUTPUTS && కూడు_dst->దత్తాంశం == ar_local->దత్తాంశం;

        నిర్మాణం htp_allreduce_kernel_params new_kparams;
        ఒకవేళ (!ggml_hexagon_precompute_allreduce_params(
            sess, కూడు_dst, (uint32_t) ar_kparams->rank, (uint32_t) ar_kparams->n_ranks, నిజం, is_అడ్డువరుస_bcast, is_shard_ok, &new_kparams
        )) {
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: solver failed\n", sess->c_name());
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి తార్కిక scatter_ok = new_kparams.n_dsts > 1;
        new_kparams.mode = scatter_ok ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_నిండిన;

        కొరకు (uint32_t r = 0; r < n_ranks; r++) {
            స్థిరరాశి ggml_tensor * ar_src = చివరి_కణుపు.inputs[r];
            ఒకవేళ (!ggml_hexagon_tensors_overlap(కూడు_dst, ar_src)) {
                కొనసాగు;
            }
            // in-place aliasing is safe under reduce-scatter since each rank writes disjoint shards
            ఒకవేళ (scatter_ok && r == rank) {
                కొనసాగు;
            }
            HEX_VERBOSE("ggml-hex: %s skip ALLREDUCE_ADD fusion: dst overlaps allreduce src %u\n", sess->c_name(), r);
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!try_fuse_common(res_tensor, కూడు_dst)) {
            తిరిగి అబద్ధం;
        }

        చివరి_కణుపు.opcode = HTP_OP_ALLREDUCE_కూడు;
        చివరి_కణుపు.name   = "ALLREDUCE+ADD";
        చివరి_కణుపు.inputs.push_back(res_tensor);
        చివరి_కణుపు.outputs.clear();
        చివరి_కణుపు.outputs.push_back(కూడు_dst);
        చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
        జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &new_kparams, పరిమాణం(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_ALLREDUCE_కూడు;
        జ్ఞాపకం_నకలు(o.kernel_params, &new_kparams, పరిమాణం(new_kparams));

        o.src[2 * n_ranks] = కూడు_tensor(res_tensor);
        ఒకవేళ (new_kparams.mode == HTP_ALLREDUCE_SHARDED_FANOUT) {
            // fan out shard to all per-rank partial buffers
            GGML_ASSERT((uint32_t) new_kparams.n_dsts == n_ranks);
            కొరకు (uint32_t d = 0; d < n_ranks; d++) {
                GGML_ASSERT(o.src[d] != 0xffff);
                o.dst[d] = o.src[d];
            }
            కొరకు (uint32_t d = n_ranks; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        } లేకపోతే {
            o.dst[0] = కూడు_tensor(కూడు_dst);
            కొరకు (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }
        }

        HEX_VERBOSE("ggml-hex: %s fused ALLREDUCE+ADD (#%u) mode=%d n_dsts=%d\n",
                    sess->c_name(), n_ops - 1, (పూర్ణాంకం) new_kparams.mode, (పూర్ణాంకం) new_kparams.n_dsts);
        తిరిగి నిజం;
    }

    తార్కిక try_fuse_rms_norm_mul(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0) తిరిగి అబద్ధం;
        ఒకవేళ (కణుపు.opcode != HTP_OP_MUL) తిరిగి అబద్ధం;

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];
        ఒకవేళ (చివరి_కణుపు.opcode != HTP_OP_RMS_NORM) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * mul_src0 = కణుపు.src0();
        స్థిరరాశి ggml_tensor * mul_src1 = కణుపు.src1();
        స్థిరరాశి ggml_tensor * rms_బహిర్గతం  = చివరి_కణుపు.dst();

        ఒకవేళ (!ggml_hexagon_tensor_is_fuseable(rms_బహిర్గతం)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * బరువు;
        ఒకవేళ (mul_src0 == rms_బహిర్గతం || mul_src0->దత్తాంశం == rms_బహిర్గతం->దత్తాంశం) {
            బరువు = mul_src1;
        } లేకపోతే ఒకవేళ (mul_src1 == rms_బహిర్గతం || mul_src1->దత్తాంశం == rms_బహిర్గతం->దత్తాంశం) {
            బరువు = mul_src0;
        } లేకపోతే {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * src0 = చివరి_కణుపు.src0();

        ఒకవేళ (src0->ne[0] != బరువు->ne[0] || src0->ne[0] != కణుపు.dst()->ne[0]) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి తార్కిక is_అడ్డువరుస_bcast = (బరువు->ne[1] == 1 && బరువు->ne[2] == 1 && బరువు->ne[3] == 1);
        స్థిరరాశి తార్కిక is_same_shape = (src0->ne[0] == బరువు->ne[0] && src0->ne[1] == బరువు->ne[1] &&
                                    src0->ne[2] == బరువు->ne[2] && src0->ne[3] == బరువు->ne[3]);
        ఒకవేళ (!is_అడ్డువరుస_bcast && !is_same_shape) తిరిగి అబద్ధం;

        ఒకవేళ (!ggml_are_same_shape(src0, కణుపు.dst())) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ggml_is_contiguous(src0) != ggml_is_contiguous(కణుపు.dst())) {
            తిరిగి అబద్ధం;
        }

        నిర్మాణం htp_unary_kernel_params new_kparams;
        ggml_hexagon_precompute_unary_params(
            sess, HTP_OP_RMS_NORM_MUL, src0, బరువు, కణుపు.dst(), &new_kparams
        );

        ఒకవేళ ((size_t) new_kparams.vtcm_కొలత > sess->vtcm_కొలత) {
            HEX_VERBOSE("ggml-hex: %s skip RMS_NORM_MUL fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), new_kparams.vtcm_కొలత, sess->vtcm_కొలత);
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!try_fuse_common(బరువు, కణుపు.dst())) {
            తిరిగి అబద్ధం;
        }

        చివరి_కణుపు.opcode = HTP_OP_RMS_NORM_MUL;
        చివరి_కణుపు.name   = "RMS_NORM+MUL";
        చివరి_కణుపు.inputs.clear();
        చివరి_కణుపు.inputs.push_back(src0);
        చివరి_కణుపు.inputs.push_back(బరువు);
        చివరి_కణుపు.outputs.clear();
        చివరి_కణుపు.outputs.push_back(కణుపు.dst());
        చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
        జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &new_kparams, పరిమాణం(new_kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_RMS_NORM_MUL;
        జ్ఞాపకం_నకలు(o.kernel_params, &new_kparams, పరిమాణం(new_kparams));

        o.src[0] = కూడు_tensor(src0);
        o.src[1] = కూడు_tensor(బరువు);
        కొరకు (uint32_t s = 2; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = కూడు_tensor(కణుపు.dst());
        కొరకు (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused RMS_NORM+MUL (#%u)\n", sess->c_name(), n_ops - 1);
        తిరిగి నిజం;
    }

    తార్కిక try_fuse_mul_mat_కూడు(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0) తిరిగి అబద్ధం;
        ఒకవేళ (కణుపు.opcode != HTP_OP_కూడు) తిరిగి అబద్ధం;

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];
        ఒకవేళ (చివరి_కణుపు.opcode != HTP_OP_MUL_MAT) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * కూడు_src0 = కణుపు.src0();
        స్థిరరాశి ggml_tensor * కూడు_src1 = కణుపు.src1();
        స్థిరరాశి ggml_tensor * mm_బహిర్గతం   = చివరి_కణుపు.dst();

        ఒకవేళ (!ggml_hexagon_tensor_is_fuseable(mm_బహిర్గతం)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * src2;
        ఒకవేళ (కూడు_src0 == mm_బహిర్గతం || కూడు_src0->దత్తాంశం == mm_బహిర్గతం->దత్తాంశం) {
            src2 = కూడు_src1;
        } లేకపోతే ఒకవేళ (కూడు_src1 == mm_బహిర్గతం || కూడు_src1->దత్తాంశం == mm_బహిర్గతం->దత్తాంశం) {
            src2 = కూడు_src0;
        } లేకపోతే {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * src0 = చివరి_కణుపు.src0();
        స్థిరరాశి ggml_tensor * src1 = చివరి_కణుపు.src1();

        ఒకవేళ (src2->type != GGML_TYPE_F32) తిరిగి అబద్ధం;

        // the fused kernel writes the shape of the MUL_MAT output: the ADD must not broadcast it
        ఒకవేళ (!ggml_are_same_shape(mm_బహిర్గతం, కణుపు.dst())) తిరిగి అబద్ధం;

        స్థిరరాశి నిర్మాణం htp_mm_kernel_params * orig_kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) చివరి_కణుపు.kernel_params;
        నిర్మాణం htp_mm_kernel_params kparams;
        ggml_hexagon_precompute_fused_matmul_కూడు_params(sess, src0, src1, src2, కణుపు.dst(), &kparams);
        ఒకవేళ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (orig_kparams->collapse != kparams.collapse) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి పూర్ణాంకం src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        స్థిరరాశి తార్కిక can_fuse = (kparams.n_hmx > 0) || (src1_nrows == 1);
        ఒకవేళ (!can_fuse) తిరిగి అబద్ధం;

        ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
            HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)\n",
                        sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (kparams.n_hmx > 0 && orig_kparams->n_hmx > 0) {
            ఒకవేళ (kparams.m_chunk < orig_kparams->m_chunk ||
                kparams.n_chunk < orig_kparams->n_chunk ||
                kparams.n_act_దారాలు < orig_kparams->n_act_దారాలు) {
                HEX_VERBOSE("ggml-hex: %s skip MUL_MAT_ADD fusion: HMX efficiency reduced (m %d->%d, n %d->%d, th %d->%d)\n",
                            sess->c_name(), orig_kparams->m_chunk, kparams.m_chunk,
                            orig_kparams->n_chunk, kparams.n_chunk,
                            orig_kparams->n_act_దారాలు, kparams.n_act_దారాలు);
                తిరిగి అబద్ధం;
            }
        }

        ఒకవేళ (!try_fuse_common(src2, కణుపు.dst())) {
            తిరిగి అబద్ధం;
        }

        చివరి_కణుపు.opcode = HTP_OP_MUL_MAT_కూడు;
        చివరి_కణుపు.name   = "MUL_MAT+ADD";
        చివరి_కణుపు.inputs.clear();
        చివరి_కణుపు.inputs.push_back(src0);
        చివరి_కణుపు.inputs.push_back(src1);
        చివరి_కణుపు.inputs.push_back(src2);
        చివరి_కణుపు.outputs.clear();
        చివరి_కణుపు.outputs.push_back(కణుపు.dst());
        చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
        జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &kparams, పరిమాణం(kparams));

        htp_op_desc & o = h_ops[n_ops - 1];
        o.opcode = HTP_OP_MUL_MAT_కూడు;
        జ్ఞాపకం_నకలు(o.kernel_params, &kparams, పరిమాణం(kparams));

        o.src[0] = కూడు_tensor(src0);
        o.src[1] = కూడు_tensor(src1);
        o.src[2] = కూడు_tensor(src2);
        కొరకు (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
            o.src[s] = 0xffff;
        }
        o.dst[0] = కూడు_tensor(కణుపు.dst());
        కొరకు (uint32_t d = 1; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused MUL_MAT+ADD (#%u)\n", sess->c_name(), n_ops - 1);
        తిరిగి నిజం;
    }

    తార్కిక try_fuse_mul_mat_nx(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0 || కణుపు.opcode != HTP_OP_MUL_MAT) తిరిగి అబద్ధం;
        ఒకవేళ (!is_mergeable_mul_mat(కణుపు.కణుపు)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * w_లోపల = కణుపు.src0();
        స్థిరరాశి ggml_tensor * x_లోపల = కణుపు.src1();
        స్థిరరాశి ggml_tensor * d_లోపల = కణుపు.dst();

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_NX
        ఒకవేళ (చివరి_కణుపు.opcode == HTP_OP_MUL_MAT_NX) {
            స్థిరరాశి uint32_t curr_n = (uint32_t) చివరి_కణుపు.outputs.size();
            ఒకవేళ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 1 >= HTP_OP_MAX_INPUTS) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor * w0 = చివరి_కణుపు.inputs[0];
            స్థిరరాశి ggml_tensor * x  = చివరి_కణుపు.inputs[curr_n];

            ఒకవేళ (x_లోపల != x || w_లోపల->type != w0->type || w_లోపల->ne[0] != w0->ne[0] ||
                ggml_hexagon_tensor_is_tiled(w_లోపల) != ggml_hexagon_tensor_is_tiled(w0)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (!చివరి_కణుపు.fused.ఖాళీలు() && (mm_is_hmx_eligible(చివరి_కణుపు.fused[0]) != mm_is_hmx_eligible(కణుపు.కణుపు))) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి నిర్మాణం htp_mm_kernel_params * orig_kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) చివరి_కణుపు.kernel_params;
            స్థిరరాశి తార్కిక collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w_లోపల, x, d_లోపల);
            ఒకవేళ (orig_kparams->collapse && !collapse) {
                తిరిగి అబద్ధం;
            }

            నిర్మాణం htp_mm_kernel_params kparams;
            ఒకవేళ (collapse) {
                స్థిరరాశి ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, curr_n + 1, &kparams);
                kparams.collapse = 1;
            } లేకపోతే {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, curr_n + 1, &kparams);
            }
            ఒకవేళ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (!try_fuse_common(w_లోపల, d_లోపల)) {
                తిరిగి అబద్ధం;
            }

            చివరి_కణుపు.inputs[curr_n] = w_లోపల;
            చివరి_కణుపు.inputs.push_back(x);
            చివరి_కణుపు.outputs.push_back(d_లోపల);
            చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
            జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &kparams, పరిమాణం(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            జ్ఞాపకం_నకలు(o.kernel_params, &kparams, పరిమాణం(kparams));

            కొరకు (uint32_t s = 0; s <= curr_n + 1; s++) {
                o.src[s] = కూడు_tensor(చివరి_కణుపు.inputs[s]);
            }
            కొరకు (uint32_t s = curr_n + 2; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            కొరకు (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = కూడు_tensor(చివరి_కణుపు.outputs[d]);
            }
            కొరకు (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            తిరిగి నిజం;
        }

        // Case 2: last_node is single MUL_MAT
        ఒకవేళ (చివరి_కణుపు.opcode == HTP_OP_MUL_MAT) {
            ఒకవేళ (!is_mergeable_mul_mat_pair(చివరి_కణుపు.కణుపు, కణుపు.కణుపు)) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor * w0 = చివరి_కణుపు.src0();
            స్థిరరాశి ggml_tensor * x  = చివరి_కణుపు.src1();
            స్థిరరాశి ggml_tensor * w1 = కణుపు.src0();
            స్థిరరాశి ggml_tensor * dst_0 = చివరి_కణుపు.dst();
            స్థిరరాశి ggml_tensor * dst_1 = కణుపు.dst();

            స్థిరరాశి నిర్మాణం htp_mm_kernel_params * orig_kparams = (స్థిరరాశి నిర్మాణం htp_mm_kernel_params *) చివరి_కణుపు.kernel_params;
            స్థిరరాశి తార్కిక collapse = orig_kparams->collapse && ggml_hexagon_matmul_can_collapse(w1, x, dst_1);
            ఒకవేళ (orig_kparams->collapse && !collapse) {
                తిరిగి అబద్ధం;
            }

            నిర్మాణం htp_mm_kernel_params kparams;
            ఒకవేళ (collapse) {
                స్థిరరాశి ggml_tensor x_collapsed = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(x);
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, &x_collapsed, 2, &kparams);
                kparams.collapse = 1;
            } లేకపోతే {
                ggml_hexagon_precompute_fused_mmnx_params(sess, w0, x, 2, &kparams);
            }
            ఒకవేళ (!is_supported_mul_mat_nx_kernel(w0, &kparams)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
                HEX_VERBOSE("ggml-hex: %s skip NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (!try_fuse_common(w1, కణుపు.dst())) {
                తిరిగి అబద్ధం;
            }

            చివరి_కణుపు.opcode = HTP_OP_MUL_MAT_NX;
            చివరి_కణుపు.name   = "MUL_MAT_NX";
            చివరి_కణుపు.inputs.clear();
            చివరి_కణుపు.inputs.push_back(w0);
            చివరి_కణుపు.inputs.push_back(w1);
            చివరి_కణుపు.inputs.push_back(x);
            చివరి_కణుపు.outputs.clear();
            చివరి_కణుపు.outputs.push_back(dst_0);
            చివరి_కణుపు.outputs.push_back(dst_1);
            చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
            జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &kparams, పరిమాణం(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_NX;
            జ్ఞాపకం_నకలు(o.kernel_params, &kparams, పరిమాణం(kparams));

            o.src[0] = కూడు_tensor(w0);
            o.src[1] = కూడు_tensor(w1);
            o.src[2] = కూడు_tensor(x);
            కొరకు (uint32_t s = 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = కూడు_tensor(dst_0);
            o.dst[1] = కూడు_tensor(dst_1);
            కొరకు (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            తిరిగి నిజం;
        }

        తిరిగి అబద్ధం;
    }

    తార్కిక try_fuse_mul_mat_id_nx(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0 || కణుపు.opcode != HTP_OP_MUL_MAT_ID) తిరిగి అబద్ధం;
        ఒకవేళ (!is_mergeable_mul_mat_id(కణుపు.కణుపు)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * w_లోపల   = కణుపు.src0();
        స్థిరరాశి ggml_tensor * x_లోపల   = కణుపు.src1();
        స్థిరరాశి ggml_tensor * ids_లోపల = కణుపు.కణుపు->src[2];
        స్థిరరాశి ggml_tensor * d_లోపల   = కణుపు.dst();

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];

        // Case 1: last_node is already MUL_MAT_ID_NX
        ఒకవేళ (చివరి_కణుపు.opcode == HTP_OP_MUL_MAT_ID_NX) {
            స్థిరరాశి uint32_t curr_n = (uint32_t) చివరి_కణుపు.outputs.size();
            ఒకవేళ (curr_n >= HTP_OP_MAX_OUTPUTS || curr_n + 2 >= HTP_OP_MAX_INPUTS) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor * w0  = చివరి_కణుపు.inputs[0];
            స్థిరరాశి ggml_tensor * x   = చివరి_కణుపు.inputs[curr_n];
            స్థిరరాశి ggml_tensor * ids = చివరి_కణుపు.inputs[curr_n + 1];

            ఒకవేళ (x_లోపల != x || ids_లోపల != ids || w_లోపల->type != w0->type || w_లోపల->ne[0] != w0->ne[0] || w_లోపల->ne[2] != w0->ne[2] ||
                ggml_hexagon_tensor_is_tiled(w_లోపల) != ggml_hexagon_tensor_is_tiled(w0)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (!చివరి_కణుపు.fused.ఖాళీలు() && (mm_is_hmx_eligible(చివరి_కణుపు.fused[0]) != mm_is_hmx_eligible(కణుపు.కణుపు))) {
                తిరిగి అబద్ధం;
            }

            నిర్మాణం htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, d_లోపల, curr_n + 1, &kparams);
            ఒకవేళ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (!try_fuse_common(w_లోపల, d_లోపల)) {
                తిరిగి అబద్ధం;
            }

            చివరి_కణుపు.inputs[curr_n] = w_లోపల;
            చివరి_కణుపు.inputs[curr_n + 1] = x;
            చివరి_కణుపు.inputs.push_back(ids);
            చివరి_కణుపు.outputs.push_back(d_లోపల);
            చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
            జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &kparams, పరిమాణం(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            జ్ఞాపకం_నకలు(o.kernel_params, &kparams, పరిమాణం(kparams));

            కొరకు (uint32_t s = 0; s <= curr_n + 2; s++) {
                o.src[s] = కూడు_tensor(చివరి_కణుపు.inputs[s]);
            }
            కొరకు (uint32_t s = curr_n + 3; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            కొరకు (uint32_t d = 0; d <= curr_n; d++) {
                o.dst[d] = కూడు_tensor(చివరి_కణుపు.outputs[d]);
            }
            కొరకు (uint32_t d = curr_n + 1; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=%u, #%u)\n", sess->c_name(), curr_n + 1, n_ops - 1);
            తిరిగి నిజం;
        }

        // Case 2: last_node is single MUL_MAT_ID
        ఒకవేళ (చివరి_కణుపు.opcode == HTP_OP_MUL_MAT_ID) {
            ఒకవేళ (!is_mergeable_mul_mat_id_pair(చివరి_కణుపు.కణుపు, కణుపు.కణుపు)) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor * w0  = చివరి_కణుపు.src0();
            స్థిరరాశి ggml_tensor * x   = చివరి_కణుపు.src1();
            స్థిరరాశి ggml_tensor * ids = చివరి_కణుపు.కణుపు->src[2];
            స్థిరరాశి ggml_tensor * w1  = కణుపు.src0();

            నిర్మాణం htp_mm_kernel_params kparams;
            ggml_hexagon_precompute_fused_mmidnx_params(sess, w0, x, కణుపు.dst(), 2, &kparams);
            ఒకవేళ (!is_supported_mul_mat_id_nx_kernel(w0, &kparams)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
                HEX_VERBOSE("ggml-hex: %s skip ID NX fusion: VTCM needed (%d) > budget (%zu)\n",
                            sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (!try_fuse_common(w1, కణుపు.dst())) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ggml_tensor * dst_0 = చివరి_కణుపు.dst();
            స్థిరరాశి ggml_tensor * dst_1 = కణుపు.dst();

            చివరి_కణుపు.opcode = HTP_OP_MUL_MAT_ID_NX;
            చివరి_కణుపు.name   = "MUL_MAT_ID_NX";
            చివరి_కణుపు.inputs.clear();
            చివరి_కణుపు.inputs.push_back(w0);
            చివరి_కణుపు.inputs.push_back(w1);
            చివరి_కణుపు.inputs.push_back(x);
            చివరి_కణుపు.inputs.push_back(ids);
            చివరి_కణుపు.outputs.clear();
            చివరి_కణుపు.outputs.push_back(dst_0);
            చివరి_కణుపు.outputs.push_back(dst_1);
            చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);
            జ్ఞాపకం_నకలు(చివరి_కణుపు.kernel_params, &kparams, పరిమాణం(kparams));

            htp_op_desc & o = h_ops[n_ops - 1];
            o.opcode = HTP_OP_MUL_MAT_ID_NX;
            జ్ఞాపకం_నకలు(o.kernel_params, &kparams, పరిమాణం(kparams));

            o.src[0] = కూడు_tensor(w0);
            o.src[1] = కూడు_tensor(w1);
            o.src[2] = కూడు_tensor(x);
            o.src[3] = కూడు_tensor(ids);
            కొరకు (uint32_t s = 4; s < HTP_OP_MAX_INPUTS; s++) {
                o.src[s] = 0xffff;
            }
            o.dst[0] = కూడు_tensor(dst_0);
            o.dst[1] = కూడు_tensor(dst_1);
            కొరకు (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
                o.dst[d] = 0xffff;
            }

            HEX_VERBOSE("ggml-hex: %s fused MUL_MAT_ID_NX (N=2, #%u)\n", sess->c_name(), n_ops - 1);
            తిరిగి నిజం;
        }

        తిరిగి అబద్ధం;
    }

    తార్కిక try_fuse_gdn_cpy(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (n_ops == 0 || కణుపు.opcode != HTP_OP_CPY) తిరిగి అబద్ధం;

        htp_opnode & చివరి_కణుపు = ops[n_ops - 1];
        ఒకవేళ (చివరి_కణుపు.opcode != HTP_OP_GATED_DELTA_NET) తిరిగి అబద్ధం;
        ఒకవేళ (చివరి_కణుపు.outputs.size() != 1) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * gdn_బహిర్గతం  = చివరి_కణుపు.dst();
        స్థిరరాశి ggml_tensor * cpy_కణుపు = కణుపు.కణుపు;
        స్థిరరాశి ggml_tensor * cpy_src  = కణుపు.src0();
        స్థిరరాశి ggml_tensor * cpy_dst  = కణుపు.dst();

        ఒకవేళ (!cpy_src || !cpy_dst || !cpy_dst->దత్తాంశం) తిరిగి అబద్ధం;
        ఒకవేళ (gdn_బహిర్గతం->type != GGML_TYPE_F32 || cpy_src->type != GGML_TYPE_F32 || cpy_dst->type != GGML_TYPE_F32) తిరిగి అబద్ధం;
        ఒకవేళ ((gdn_బహిర్గతం->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy_కణుపు->flags & GGML_TENSOR_FLAG_OUTPUT)) తిరిగి అబద్ధం;

        స్థిరరాశి ggml_tensor * v = చివరి_కణుపు.కణుపు->src[2];
        ఒకవేళ (!v) తిరిగి అబద్ధం;

        స్థిరరాశి int64_t S_v      = v->ne[0];
        స్థిరరాశి int64_t H        = v->ne[1];
        స్థిరరాశి int64_t n_టోకెన్లు = v->ne[2];
        స్థిరరాశి int64_t n_seqs   = v->ne[3];
        స్థిరరాశి int64_t K        = ggml_get_op_params_i32(చివరి_కణుపు.కణుపు, 0);
        స్థిరరాశి size_t  తోక_off = (size_t) S_v * H * n_టోకెన్లు * n_seqs * పరిమాణం(దశాంశం);

        స్థిరరాశి int64_t D         = S_v * S_v * H;
        స్థిరరాశి int64_t n_written = std::min<int64_t>(n_టోకెన్లు, K);

        ఒకవేళ (cpy_src->op != GGML_OP_VIEW || (cpy_src->view_src != gdn_బహిర్గతం && cpy_src->view_src->దత్తాంశం != gdn_బహిర్గతం->దత్తాంశం) ||
            cpy_src->view_offs != తోక_off || !ggml_is_contiguous(cpy_src)) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (cpy_dst->ne[0] != D || cpy_dst->ne[1] != n_seqs || cpy_dst->nb[0] != పరిమాణం(దశాంశం)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (n_seqs > 1 && cpy_dst->nb[1] != (size_t) D * పరిమాణం(దశాంశం)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (n_written > 1) {
            ఒకవేళ (cpy_dst->ne[2] != n_written || cpy_dst->nb[2] != (size_t) D * n_seqs * పరిమాణం(దశాంశం)) {
                తిరిగి అబద్ధం;
            }
        }

        ఒకవేళ (!try_fuse_common({cpy_dst})) {
            తిరిగి అబద్ధం;
        }

        చివరి_కణుపు.name += "+CPY";
        చివరి_కణుపు.outputs.push_back(cpy_dst);
        చివరి_కణుపు.fused.push_back(కణుపు.కణుపు);

        htp_op_desc & o = h_ops[n_ops - 1];
        o.dst[1] = కూడు_tensor(cpy_dst);
        కొరకు (uint32_t d = 2; d < HTP_OP_MAX_OUTPUTS; d++) {
            o.dst[d] = 0xffff;
        }

        HEX_VERBOSE("ggml-hex: %s fused GATED_DELTA_NET+CPY (#%u)\n", sess->c_name(), n_ops - 1);
        తిరిగి నిజం;
    }

    తార్కిక try_fuse(స్థిరరాశి htp_opnode & కణుపు) {
        ఒకవేళ (!opt_opfusion) తిరిగి అబద్ధం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_ALLREDUCE_కూడు) && try_fuse_allreduce_కూడు(కణుపు)) తిరిగి నిజం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_RMS_NORM_MUL)  && try_fuse_rms_norm_mul(కణుపు))  తిరిగి నిజం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_కూడు)   && try_fuse_mul_mat_కూడు(కణుపు))   తిరిగి నిజం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_NX)    && try_fuse_mul_mat_nx(కణుపు))    తిరిగి నిజం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_MUL_MAT_ID_NX) && try_fuse_mul_mat_id_nx(కణుపు)) తిరిగి నిజం;
        ఒకవేళ (ggml_hexagon_is_fusion_enabled(GGML_HEXAGON_FUSE_GDN_CPY)       && try_fuse_gdn_cpy(కణుపు))       తిరిగి నిజం;
        తిరిగి అబద్ధం;
    }
};

నిర్మాణం ggml_hexagon_registry {
    ggml_hexagon_registry(ggml_backend_reg_t reg);
    ~ggml_hexagon_registry();

    ggml_backend_device devices[GGML_HEXAGON_MAX_SESSIONS];
};

నిర్మాణం ggml_hexagon_opqueue {
    // Shared buffer for storing batches
    ggml_hexagon_shared_బఫరు *shm_buf;
    size_t                      shm_blk_కొలత;
    size_t                      లోతు;

    ఉపయోగించి opvec = std::vector<htp_opnode>;

    std::vector<opvec>          op_cache;       // per batch op cache
    std::vector<uint64_t>       ప్రారంభం_usec;     // per batch start time

    ggml_hexagon_opqueue(ggml_hexagon_session *sess, size_t batch_కొలత, size_t లోతు) : లోతు(లోతు) {
        size_t n_bufs    = HTP_OP_MAX_BUFS;
        size_t n_ops     = batch_కొలత;
        size_t n_tensors = n_ops * HTP_OP_MAX_OUTPUTS + n_ops * HTP_OP_MAX_INPUTS;

        size_t tr_కొలత = 0;
        ఒకవేళ (opt_profile == 3) {
            tr_కొలత = (HTP_MAX_NTHREADS + 1) * opt_optrace * పరిమాణం(htp_trace_desc);
        }

        shm_blk_కొలత = పరిమాణం(htp_buf_desc)  * n_bufs    +
                       పరిమాణం(htp_tensor)    * n_tensors +
                       పరిమాణం(htp_op_desc)   * n_ops     +
                       పరిమాణం(htp_prof_desc) * n_ops     +
                       tr_కొలత;

        shm_buf = కొత్త ggml_hexagon_shared_బఫరు(sess, shm_blk_కొలత * లోతు, నిజం /* pinned */);

        op_cache.resize(లోతు);
        ప్రారంభం_usec.resize(లోతు, 0);

        ఒకవేళ (opt_verbose) {
            GGML_LOG_INFO("ggml-hex: %s allocated opqueue : batch-size %zu depth %zu shm-size %zu shm-block-size %zu\n",
                    sess->c_name(), batch_కొలత, లోతు, shm_buf->size(), shm_blk_కొలత);
        }
    }

    ~ggml_hexagon_opqueue() {
        తొలగించు shm_buf;
    }

    size_t shm_కొలత() స్థిరరాశి { తిరిగి shm_buf ? shm_buf->size() : 0; }

    // push new batch
    తార్కిక నెట్టు(htp_opbatch_req& req, dspqueue_బఫరు& dbuf, స్థిరరాశి ggml_hexagon_opbatch* op_batch, uint64_t seq) {
        static_assert(పరిమాణం(htp_opbatch_req) % 8 == 0, "sizeof(htp_opbatch_req) must be multiple of 8");
        static_assert(పరిమాణం(htp_opbatch_rsp) % 8 == 0, "sizeof(htp_opbatch_rsp) must be multiple of 8");
        static_assert(పరిమాణం(htp_buf_desc)    % 8 == 0, "sizeof(htp_buf_desc) must be multiple of 8");
        static_assert(పరిమాణం(htp_tensor)      % 8 == 0, "sizeof(htp_tensor) must be multiple of 8");
        static_assert(పరిమాణం(htp_op_desc)     % 8 == 0, "sizeof(htp_op_desc) must be multiple of 8");
        static_assert(పరిమాణం(htp_prof_desc)   % 8 == 0, "sizeof(htp_prof_desc) must be multiple of 8");

        ఒకవేళ (seq - shm_buf->sess->batch_rsp_seq > లోతు) { తిరిగి అబద్ధం; }

        స్థిరరాశి uint32_t స్థానం = (uint32_t) ((seq - 1) % లోతు);

        req.seq       = seq;
        req.n_bufs    = op_batch->n_bufs;
        req.n_tensors = op_batch->n_tens;
        req.n_ops     = op_batch->n_ops;

        op_cache[స్థానం]   = op_batch->ops;
        ప్రారంభం_usec[స్థానం] = ggml_సమయం_us();

        స్థిరరాశి size_t b_కొలత = పరిమాణం(htp_buf_desc)  * req.n_bufs;
        స్థిరరాశి size_t t_కొలత = పరిమాణం(htp_tensor)    * req.n_tensors;
        స్థిరరాశి size_t o_కొలత = పరిమాణం(htp_op_desc)   * req.n_ops;
        స్థిరరాశి size_t p_కొలత = పరిమాణం(htp_prof_desc) * req.n_ops;

        size_t tr_కొలత = 0;
        ఒకవేళ (opt_profile == 3) {
            req.n_traces = opt_optrace;
            tr_కొలత = (HTP_MAX_NTHREADS + 1) * req.n_traces * పరిమాణం(htp_trace_desc);
        } లేకపోతే {
            req.n_traces = 0;
        }

        dbuf.ptr      = shm_buf->పునాది() + ((size_t) స్థానం * shm_blk_కొలత);
        dbuf.fd       = shm_buf->fd();
        dbuf.flags    = DSPQUEUE_బఫరు_FLAG_FLUSH_SENDER | DSPQUEUE_బఫరు_FLAG_INVALIDATE_RECIPIENT;
        dbuf.స్థానభ్రంశం   = (uint8_t*) dbuf.ptr - (uint8_t*) shm_buf->పునాది();
        dbuf.size     = b_కొలత + t_కొలత + o_కొలత + p_కొలత + tr_కొలత;

        GGML_ASSERT(dbuf.size <= shm_blk_కొలత);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * b_ptr = m_ptr; m_ptr += b_కొలత;
        uint8_t * t_ptr = m_ptr; m_ptr += t_కొలత;
        uint8_t * o_ptr = m_ptr;

        జ్ఞాపకం_నకలు(b_ptr, (శూన్య *) op_batch->h_bufs.దత్తాంశం(), b_కొలత);
        జ్ఞాపకం_నకలు(t_ptr, (శూన్య *) op_batch->h_tens.దత్తాంశం(), t_కొలత);
        జ్ఞాపకం_నకలు(o_ptr, (శూన్య *) op_batch->h_ops.దత్తాంశం(),  o_కొలత);

        HEX_VERBOSE("ggml-hex: %s opqueue-push batch #%llu : n-bufs %u n-tensors %u n-ops %u vmem %zu : b-size %zu t-size %zu o-size %zu m-size %zu\n",
                shm_buf->sess->c_name(), (గుర్తులేని దీర్ఘ దీర్ఘ) req.seq, req.n_bufs, req.n_tensors, req.n_ops, op_batch->b_vmem,
                b_కొలత, t_కొలత, o_కొలత, (size_t) dbuf.size);

        ఒకవేళ (opt_verbose > 1) {
            htp_buf_desc *b = (htp_buf_desc*) b_ptr;
            కొరకు (గుర్తులేని పూర్ణాంకం i=0; i < req.n_bufs; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-buf #%u : fd %d base %p size %zu\n", shm_buf->sess->c_name(), i,
                            b[i].fd, (శూన్య *) b[i].పునాది, (size_t) b[i].size);
            }
            htp_tensor *t = (htp_tensor*) t_ptr;
            కొరకు (గుర్తులేని పూర్ణాంకం i=0; i < req.n_tensors; i++) {
                GGML_LOG_DEBUG("ggml-hex: %s htp-tensor #%u : bi %u offset %llu size %u : %zu:%zu:%zu:%zu\n",
                            shm_buf->sess->c_name(), i, t[i].bi, (గుర్తులేని దీర్ఘ దీర్ఘ) t[i].దత్తాంశం, t[i].size,
                            (size_t) t[i].ne[0], (size_t) t[i].ne[1], (size_t) t[i].ne[2], (size_t) t[i].ne[3]);
            }
        }

        తిరిగి నిజం;
    }

    శూన్య తీయి(htp_opbatch_rsp rsp, dspqueue_బఫరు dbuf) {
        స్థిరరాశి uint32_t స్థానం = (uint32_t) ((rsp.seq - 1) % లోతు);

        స్థిరరాశి size_t b_కొలత = పరిమాణం(htp_buf_desc)  * rsp.n_bufs;
        స్థిరరాశి size_t t_కొలత = పరిమాణం(htp_tensor)    * rsp.n_tensors;
        స్థిరరాశి size_t o_కొలత = పరిమాణం(htp_op_desc)   * rsp.n_ops;
        స్థిరరాశి size_t p_కొలత = పరిమాణం(htp_prof_desc) * rsp.n_ops;

        size_t tr_కొలత = 0;
        uint32_t n_traces = 0;
        ఒకవేళ (opt_profile == 3) {
            n_traces = opt_optrace;
            tr_కొలత = (HTP_MAX_NTHREADS + 1) * n_traces * పరిమాణం(htp_trace_desc);
        }

        స్థిరరాశి size_t m_కొలత = b_కొలత + t_కొలత + o_కొలత + p_కొలత + tr_కొలత;
        GGML_ASSERT(m_కొలత <= shm_blk_కొలత);

        HEX_VERBOSE("ggml-hex: %s opqueue-pop batch #%llu : n-bufs %u n-tensors %u n-ops %u : m-size %zu b-size %zu t-size %zu o-size %zu\n",
                shm_buf->sess->c_name(), (గుర్తులేని దీర్ఘ దీర్ఘ) rsp.seq, rsp.n_bufs, rsp.n_tensors, rsp.n_ops,
                (size_t) dbuf.size, b_కొలత, t_కొలత, o_కొలత);

        uint8_t * m_ptr = (uint8_t*) dbuf.ptr;
        uint8_t * p_ptr = m_ptr + (b_కొలత + t_కొలత + o_కొలత);

        ఒకవేళ (rsp.n_ops > 0) {
            స్వయంచాలక & ops = op_cache[స్థానం];
            GGML_ASSERT(rsp.n_ops <= ops.size());

            స్థిరరాశి htp_prof_desc * pd = (స్థిరరాశి htp_prof_desc *) p_ptr;
            స్థిరరాశి htp_trace_desc * trace_events = శూన్యసూచిక;
            ఒకవేళ (opt_profile == 3) {
                trace_events = (స్థిరరాశి htp_trace_desc *) (p_ptr + p_కొలత);
            }

            ఒకవేళ (opt_profile) {
                ggml_hexagon_dump_batch_prof(shm_buf->sess->name, rsp);
            }

            కొరకు (uint32_t i = 0; i < rsp.n_ops; i++) {
                ఒకవేళ (opt_profile) {
                    ggml_hexagon_dump_op_prof(shm_buf->sess->name, ops[i], pd[i]);
                }
            }

            ఒకవేళ (opt_profile) {
                ggml_hexagon_dump_trace_events(shm_buf->sess->name, rsp, trace_events, n_traces);
            }
        }
    }
};

శూన్య ggml_hexagon_session::flush_peers() {
    స్వయంచాలక vpeers = std::move(virt_peers);
    virt_peers.clear();
    కొరకు (స్వయంచాలక * peer : vpeers) {
        peer->flush_sync();
    }

    స్వయంచాలక ppeers = std::move(phys_peers);
    phys_peers.clear();
    కొరకు (స్వయంచాలక * peer : ppeers) {
        peer->flush_async();
    }

    కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
        sub->flush_peers();
    }
}

శూన్య ggml_hexagon_session::flush_async() {
    flush_peers();
    flush_batch();
}

శూన్య ggml_hexagon_session::flush_pending(తార్కిక all) {
    కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
        sub->flush_pending(all);
        ఒకవేళ (sub->చివరి_error > HTP_స్థితి_OK) {
            ఇది->చివరి_error = sub->చివరి_error.load();
        }
    }

    అయితే (ఇది->batch_rsp_seq < ఇది->batch_req_seq) {
        నిర్మాణం htp_opbatch_rsp rsp;
        uint32_t               rsp_కొలత;
        uint32_t               flags;

        నిర్మాణం dspqueue_బఫరు dbuf;
        uint32_t               n_dbufs;

        // Read response packet from queue
        స్థిరరాశి uint32_t timeo = opt_oppoll ? 0 : DSPQUEUE_TIMEOUT;

        పూర్ణాంకం err = dspqueue_read(ఇది->queue, &flags, 1, &n_dbufs, &dbuf, పరిమాణం(rsp), &rsp_కొలత, (uint8_t *) &rsp, timeo);
        ఒకవేళ (err == AEE_EEXPIRED || err == AEE_EWOULDBLOCK) {
            కొనసాగు;
        }

        ఒకవేళ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_read failed: 0x%08x\n", (గుర్తులేని) err);
        }

        // Basic sanity checks
        ఒకవేళ (rsp_కొలత != పరిమాణం(rsp) || n_dbufs != 1) {
            GGML_ABORT("ggml-hex: %s dspcall : bad response : size %u dspbufs %u\n", ఇది->c_name(), rsp_కొలత, n_dbufs);
        }

        ఒకవేళ (rsp.స్థితి_2 > HTP_స్థితి_OK) {
            GGML_LOG_ERROR("ggml-hex: %s dspcall : dsp-rsp %s\n", ఇది->c_name(), స్థితి_గమ్యం_str(rsp.స్థితి_2));
            ఇది->చివరి_error = rsp.స్థితి_2;
            కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
                sub->చివరి_error = rsp.స్థితి_2;
            }
        }

        op_వరుస->తీయి(rsp, dbuf);

        GGML_ASSERT(rsp.seq == ఇది->batch_rsp_seq + 1);
        ఇది->batch_rsp_seq = rsp.seq;

        ఒకవేళ (!all) ఆపు;
    }
}

శూన్య ggml_hexagon_session::flush_sync(తార్కిక all) {
    flush_async();
    flush_pending(all);
}

శూన్య ggml_hexagon_session::ప్రారంభం_batch() {
    ఒకవేళ (ఇది->mdev.count > 1) {
        వరుసలో_చేర్చు_mdev_సమూహం();
    }
}

శూన్య ggml_hexagon_session::flush_batch(size_t min_ops) {
    ఒకవేళ (op_batch->n_ops < min_ops) { తిరిగి; }

    op_batch->క్రమీకరించు_buffers();

    htp_opbatch_req req {};
    dspqueue_బఫరు dbuf{};

    స్థిరరాశి uint64_t seq = ++ఇది->batch_req_seq;

    op_batch->update_mdev_సమూహం(ఇది->mdev.idx);

    ఒకవేళ (!op_వరుస->నెట్టు(req, dbuf, op_batch, seq)) {
        flush_pending(అబద్ధం);
        op_వరుస->నెట్టు(req, dbuf, op_batch, seq);
    }

    కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
        htp_opbatch_req sub_req {};
        dspqueue_బఫరు sub_dbuf{};

        sub->batch_req_seq = seq;
        op_batch->update_mdev_సమూహం(sub->mdev.idx);

        ఒకవేళ (!sub->op_వరుస->నెట్టు(sub_req, sub_dbuf, op_batch, seq)) {
            sub->flush_pending(అబద్ధం);
            sub->op_వరుస->నెట్టు(sub_req, sub_dbuf, op_batch, seq);
        }

        HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", sub->c_name(), sub_dbuf.ptr, sub_dbuf.size);

        పూర్ణాంకం err = dspqueue_write(sub->queue, 0, 1, &sub_dbuf, పరిమాణం(sub_req), (స్థిరరాశి uint8_t*) &sub_req, DSPQUEUE_TIMEOUT);
        ఒకవేళ (err != 0) {
            GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", sub->c_name(), (గుర్తులేని) err);
        }
    }

    HEX_VERBOSE("ggml-hex: %s queue-opbatch: %p size %u\n", ఇది->c_name(), dbuf.ptr, dbuf.size);

    పూర్ణాంకం err = dspqueue_write(ఇది->queue, 0, 1, &dbuf, పరిమాణం(req), (స్థిరరాశి uint8_t*) &req, DSPQUEUE_TIMEOUT);
    ఒకవేళ (err != 0) {
        GGML_ABORT("ggml-hex: %s dspqueue_write failed: 0x%08x\n", ఇది->c_name(), (గుర్తులేని) err);
    }

    op_batch->reset();
}

శూన్య ggml_hexagon_session::వరుసలో_చేర్చు_op(స్థిరరాశి htp_opnode & కణుపు) {
    స్వయంచాలక clone_tensor_బఫరు = [ఇది](స్థిరరాశి ggml_tensor * t) {
        స్వయంచాలక sbuf = ఇది->mmap_tensor(t);
        ఒకవేళ (!sbuf) తిరిగి;
        ఒకవేళ (sbuf->sess != ఇది) {
            ఇది->clone_బఫరు(sbuf);
        }
        కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
            sub->clone_బఫరు(sbuf);
        }
    };

    కొరకు (స్వయంచాలక t : కణుపు.get_inputs()) {
        clone_tensor_బఫరు(t);
    }
    కొరకు (స్వయంచాలక t : కణుపు.get_outputs()) {
        clone_tensor_బఫరు(t);
    }

    ఒకవేళ (opt_opfusion && op_batch->try_fuse(కణుపు)) {
        తిరిగి;
    }

    ఒకవేళ (!op_batch->fit_op(కణుపు)) {
        flush_async();
    }

    ఒకవేళ (op_batch->ఖాళీలు()) {
        ప్రారంభం_batch();
    }

    ఒకవేళ (!op_batch->fit_op(కణుపు)) {
        GGML_ABORT("ggml-hex: %s op does not fit into empty batch (vmem/tensor/buffer limit exceeded)\n",
                   c_name());
    }

    op_batch->కూడు_op(కణుపు);
}

శూన్య ggml_hexagon_session::వరుసలో_చేర్చు_mdev_సమూహం() {
    htp_opnode సమూహం_కణుపు(HTP_OP_MDEV_సమూహం);

    uint8_t * fence_స్థానం = ఇది->mdev_fence_స్థానం;

    స్థిర ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    ggml_tensor dummy_t {};
    dummy_t.బఫరు       = &ఇది->fence_buf->backend_బఫరు;
    dummy_t.extra        = &fence_extra;
    dummy_t.దత్తాంశం         = (శూన్య *) fence_స్థానం;
    dummy_t.type         = GGML_TYPE_I8;
    dummy_t.ne[0]        = HTP_FENCE_స్థానం_కొలత;
    dummy_t.ne[1]        = (int64_t) ఇది->mdev.count;
    dummy_t.ne[2]        = 1;
    dummy_t.ne[3]        = 1;
    dummy_t.nb[0]        = 1;
    dummy_t.nb[1]        = HTP_FENCE_స్థానం_కొలత;
    dummy_t.nb[2]        = dummy_t.nb[1] * dummy_t.ne[1];
    dummy_t.nb[3]        = dummy_t.nb[2];
    dummy_t.op           = GGML_OP_NONE;
    dummy_t.op_params[0] = (int32_t) ఇది->mdev.idx;

    ggml_tensor * కణుపు = సమూహం_కణుపు.కూడు_dummy(dummy_t);
    కణుపు->src[0] = కణుపు;
    సమూహం_కణుపు.init(కణుపు);
    సమూహం_కణుపు.outputs.clear();
    సమూహం_కణుపు.name = "MDEV_GROUP";

    కొరకు (స్వయంచాలక & sub : ఇది->mdev.sessions) {
        sub->clone_బఫరు(ఇది->fence_buf);
    }

    op_batch->కూడు_op(సమూహం_కణుపు);
}

శూన్య ggml_hexagon_session::వరుసలో_చేర్చు_cpy(స్థిరరాశి ggml_tensor * src, ggml_tensor * dst, స్థిరరాశి ggml_tensor * sync_tensor, uint32_t fence_seq) {
    స్థిరరాశి తార్కిక with_fence = sync_tensor != శూన్యసూచిక;
    htp_opnode cpy_కణుపు(with_fence ? HTP_OP_CPY_FENCE : HTP_OP_CPY);

    ggml_tensor* కణుపు = cpy_కణుపు.కూడు_dummy(*dst);
    కణుపు->op     = GGML_OP_CPY;
    కణుపు->src[0] = const_cast<ggml_tensor *>(src);
    కణుపు->src[1] = with_fence ? cpy_కణుపు.కూడు_dummy(*sync_tensor) : శూన్యసూచిక;
    ఒకవేళ (with_fence) {
        కణుపు->op_params[0] = (int32_t) fence_seq;
    }

    cpy_కణుపు.init(కణుపు);
    ఒకవేళ (with_fence) {
        cpy_కణుపు.name = "CPY+FENCE";
    }
    స్థిరరాశి తార్కిక ok = ggml_hexagon_precompute_cpy_params(ఇది, కణుపు, (నిర్మాణం htp_copy_kernel_params *) cpy_కణుపు.kernel_params);
    స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_copy_kernel_params *) cpy_కణుపు.kernel_params;
    ఒకవేళ (ok && !with_fence && kparams->మొత్తం_elems == 0) {
        తిరిగి;
    }
    ఇది->వరుసలో_చేర్చు_op(cpy_కణుపు);
}

శూన్య ggml_hexagon_session::వరుసలో_చేర్చు_fence(స్థిరరాశి ggml_tensor * sync_tensor, uint32_t fence_seq, తార్కిక wait) {
    htp_opnode sync_కణుపు(HTP_OP_FENCE);

    ggml_tensor* కణుపు = sync_కణుపు.కూడు_dummy(*sync_tensor);
    కణుపు->op           = GGML_OP_NONE;
    కణుపు->src[0]       = కణుపు;
    కణుపు->op_params[0] = (int32_t) fence_seq;
    కణుపు->op_params[1] = wait ? 0 : 1;

    sync_కణుపు.init(కణుపు);
    sync_కణుపు.name = wait ? "FENCE_WAIT" : "FENCE_SIGNAL";
    ఇది->వరుసలో_చేర్చు_op(sync_కణుపు);
}

స్థిర తార్కిక ggml_hexagon_precompute_allreduce_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    uint32_t rank,
    uint32_t n_ranks,
    తార్కిక has_కూడు,
    తార్కిక is_అడ్డువరుస_bcast,
    తార్కిక is_shard_ok,
    నిర్మాణం htp_allreduce_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->rank         = (int32_t) rank;
    kparams->n_ranks      = (int32_t) n_ranks;
    kparams->is_అడ్డువరుస_bcast = (has_కూడు && is_అడ్డువరుస_bcast) ? 1 : 0;

    స్థిరరాశి uint32_t nelem     = (uint32_t) ggml_nelements(dst);
    స్థిరరాశి uint32_t elem_కొలత = (dst->type == GGML_TYPE_F16) ? పరిమాణం(ggml_fp16_t) : పరిమాణం(దశాంశం);
    స్థిరరాశి తార్కిక is_contiguous = ggml_is_contiguous(dst);

    స్థిరరాశి uint32_t ne0 = (uint32_t) dst->ne[0];
    స్థిరరాశి uint32_t ne1 = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    kparams->ne0 = (int32_t) ne0;
    kparams->ne1 = (int32_t) ne1;

    స్థిరరాశి తార్కిక use_1d = is_contiguous && !(has_కూడు && is_అడ్డువరుస_bcast && ne1 > 1);

    ఒకవేళ (has_కూడు) {
        // sharded reduce-scatter for contiguous in-place add
        ఒకవేళ (opt_ar_scatter && is_shard_ok && use_1d && !is_అడ్డువరుస_bcast) {
            స్థిరరాశి uint32_t rank_chunk_elems = hex_విడత_పైకి((nelem + n_ranks - 1) / n_ranks, 128);
            స్థిరరాశి uint32_t rank_elem_ప్రారంభం  = (std::min)(rank * rank_chunk_elems, nelem);
            స్థిరరాశి uint32_t rank_elem_చివర    = (std::min)(rank_elem_ప్రారంభం + rank_chunk_elems, nelem);
            kparams->n_dsts          = (int32_t) n_ranks;
            kparams->rank_elem_ప్రారంభం = (int32_t) rank_elem_ప్రారంభం;
            kparams->rank_nelem      = (int32_t) (rank_elem_చివర - rank_elem_ప్రారంభం);
        } లేకపోతే ఒకవేళ (use_1d) {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ప్రారంభం = 0;
            kparams->rank_nelem      = (int32_t) nelem;
        } లేకపోతే {
            kparams->n_dsts          = 1;
            kparams->rank_elem_ప్రారంభం = 0;
            kparams->rank_nelem      = (int32_t) ne1;
        }
    } లేకపోతే {
        kparams->n_dsts = (int32_t) n_ranks;
        ఒకవేళ (use_1d) {
            స్థిరరాశి uint32_t rank_chunk_elems = hex_విడత_పైకి((nelem + n_ranks - 1) / n_ranks, 128);
            స్థిరరాశి uint32_t rank_elem_ప్రారంభం  = (std::min)(rank * rank_chunk_elems, nelem);
            స్థిరరాశి uint32_t rank_elem_చివర    = (std::min)(rank_elem_ప్రారంభం + rank_chunk_elems, nelem);
            స్థిరరాశి uint32_t rank_nelem       = rank_elem_చివర - rank_elem_ప్రారంభం;
            kparams->rank_elem_ప్రారంభం        = (int32_t) rank_elem_ప్రారంభం;
            kparams->rank_nelem             = (int32_t) rank_nelem;
        } లేకపోతే {
            స్థిరరాశి uint32_t rank_chunk_అడ్డువరుసలు = (ne1 + n_ranks - 1) / n_ranks;
            స్థిరరాశి uint32_t rank_r0         = (std::min)(rank * rank_chunk_అడ్డువరుసలు, ne1);
            స్థిరరాశి uint32_t rank_r1         = (std::min)(rank_r0 + rank_chunk_అడ్డువరుసలు, ne1);
            స్థిరరాశి uint32_t rank_nrows      = rank_r1 - rank_r0;
            kparams->rank_elem_ప్రారంభం       = (int32_t) rank_r0;
            kparams->rank_nelem            = (int32_t) rank_nrows;
        }
    }

    kparams->mode = (kparams->n_dsts > 1) ? HTP_ALLREDUCE_SHARDED_FANOUT : HTP_ALLREDUCE_నిండిన;

    ఒకవేళ (use_1d) {
        స్థిరరాశి uint32_t rank_nelem = (uint32_t) kparams->rank_nelem;
        స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, (std::max)(1u, rank_nelem / 128));
        kparams->n_దారాలు = n_దారాలు;
        స్థిరరాశి size_t n_vtcm_buffers = htp_allreduce_vtcm_బఫరు_లెక్క(n_ranks, n_దారాలు, has_కూడు, is_అడ్డువరుస_bcast);

        uint32_t ఖండం_elems = 65536;
        ఒకవేళ (ఖండం_elems > rank_nelem / n_దారాలు && rank_nelem / n_దారాలు > 128) {
            ఖండం_elems = hex_విడత_పైకి(rank_nelem / (n_దారాలు * 2), 128);
        }
        ఖండం_elems = (std::max)(128u, ఖండం_elems);

        kparams->ఖండం_elems          = ఖండం_elems;
        kparams->vtcm_కొలత_చొప్పున_thread = 2 * ఖండం_elems * elem_కొలత;
        kparams->vtcm_కొలత            = n_vtcm_buffers * kparams->vtcm_కొలత_చొప్పున_thread;

        అయితే ((size_t) kparams->vtcm_కొలత > sess->vtcm_కొలత && ఖండం_elems > 128) {
            స్థిరరాశి size_t max_బైట్లు_చొప్పున_buf = sess->vtcm_కొలత / (n_vtcm_buffers * 2);
            ఖండం_elems = (uint32_t) hex_align_కిందికి((size_t) (max_బైట్లు_చొప్పున_buf / elem_కొలత), 128);
            ఒకవేళ (ఖండం_elems < 128) ఆపు;
            kparams->ఖండం_elems          = ఖండం_elems;
            kparams->vtcm_కొలత_చొప్పున_thread = 2 * ఖండం_elems * elem_కొలత;
            kparams->vtcm_కొలత            = n_vtcm_buffers * kparams->vtcm_కొలత_చొప్పున_thread;
        }

        ఒకవేళ (sess->vtcm_కొలత < (size_t) kparams->vtcm_కొలత || ఖండం_elems < 128) {
            HEX_VERBOSE("ggml-hex: %s allreduce 1D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_కొలత, sess->vtcm_కొలత);
            తిరిగి అబద్ధం;
        }

        kparams->elems_చొప్పున_thread = hex_విడత_పైకి((rank_nelem + n_దారాలు - 1) / n_దారాలు, ఖండం_elems);
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_1D;
        తిరిగి నిజం;
    } లేకపోతే {
        స్థిరరాశి uint32_t rank_nrows = (uint32_t) kparams->rank_nelem;
        స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, (std::max)(1u, rank_nrows));
        kparams->n_దారాలు = n_దారాలు;
        స్థిరరాశి size_t n_vtcm_buffers = htp_allreduce_vtcm_బఫరు_లెక్క(n_ranks, n_దారాలు, has_కూడు, is_అడ్డువరుస_bcast);

        స్థిరరాశి uint32_t అడ్డువరుస_బైట్లు = ne0 * elem_కొలత;
        స్థిరరాశి uint32_t అడ్డువరుస_కొలత_aligned = (uint32_t) hex_align_పైకి(అడ్డువరుస_బైట్లు, 128);
        kparams->అడ్డువరుస_కొలత_aligned = అడ్డువరుస_కొలత_aligned;

        స్థిరరాశి uint32_t nrows_చొప్పున_thread = (rank_nrows + n_దారాలు - 1) / n_దారాలు;
        uint32_t ఖండం_అడ్డువరుసలు = (std::min)(128u, nrows_చొప్పున_thread);
        ఖండం_అడ్డువరుసలు = (std::max)(1u, ఖండం_అడ్డువరుసలు);
        kparams->ఖండం_elems = ఖండం_అడ్డువరుసలు;

        kparams->vtcm_కొలత_చొప్పున_thread = 2 * (ఖండం_అడ్డువరుసలు * అడ్డువరుస_కొలత_aligned);
        kparams->vtcm_కొలత            = n_vtcm_buffers * kparams->vtcm_కొలత_చొప్పున_thread;

        అయితే ((size_t) kparams->vtcm_కొలత > sess->vtcm_కొలత && ఖండం_అడ్డువరుసలు > 1) {
            స్థిరరాశి size_t max_అడ్డువరుసలు_చొప్పున_buf = sess->vtcm_కొలత / (n_vtcm_buffers * 2 * అడ్డువరుస_కొలత_aligned);
            ఖండం_అడ్డువరుసలు = (std::max)(1u, (uint32_t) max_అడ్డువరుసలు_చొప్పున_buf);
            kparams->ఖండం_elems          = ఖండం_అడ్డువరుసలు;
            kparams->vtcm_కొలత_చొప్పున_thread = 2 * (ఖండం_అడ్డువరుసలు * అడ్డువరుస_కొలత_aligned);
            kparams->vtcm_కొలత            = n_vtcm_buffers * kparams->vtcm_కొలత_చొప్పున_thread;
            ఒకవేళ (max_అడ్డువరుసలు_చొప్పున_buf == 0) ఆపు;
        }

        ఒకవేళ (sess->vtcm_కొలత < (size_t) kparams->vtcm_కొలత || ఖండం_అడ్డువరుసలు < 1) {
            HEX_VERBOSE("ggml-hex: %s allreduce 2D solver failed to fit VTCM (%d > %zu)\n",
                        sess->c_name(), kparams->vtcm_కొలత, sess->vtcm_కొలత);
            తిరిగి అబద్ధం;
        }

        kparams->elems_చొప్పున_thread = nrows_చొప్పున_thread;
        kparams->kernel_type      = HTP_ALLREDUCE_KERNEL_DMA_2D;
        తిరిగి నిజం;
    }
}

శూన్య ggml_hexagon_session::వరుసలో_చేర్చు_allreduce(
    స్థిరరాశి ggml_tensor * dst,
    స్థిరరాశి std::vector<స్థిరరాశి ggml_tensor *> & src_tensors,
    స్థిరరాశి std::vector<స్థిరరాశి ggml_tensor *> & sync_tensors,
    uint32_t rank,
    uint32_t n_ranks,
    uint32_t fence_seq_నమోదు,
    uint32_t fence_seq_exit
) {
    htp_opnode ar_కణుపు(HTP_OP_ALLREDUCE);

    ggml_tensor* కణుపు = ar_కణుపు.కూడు_dummy(*dst);
    కణుపు->op           = GGML_OP_NONE;
    కణుపు->op_params[0] = (int32_t) fence_seq_నమోదు;
    కణుపు->op_params[1] = (int32_t) fence_seq_exit;

    ar_కణుపు.init(కణుపు);

    ar_కణుపు.inputs.clear();
    కొరకు (size_t i = 0; i < src_tensors.size(); i++) {
        ar_కణుపు.inputs.push_back(src_tensors[i]);
    }
    కొరకు (size_t i = 0; i < sync_tensors.size(); i++) {
        ar_కణుపు.inputs.push_back(ar_కణుపు.కూడు_dummy(*sync_tensors[i]));
    }

    ar_కణుపు.outputs.clear();
    కొరకు (size_t i = 0; i < src_tensors.size(); i++) {
        ar_కణుపు.outputs.push_back(src_tensors[i]);
    }

    ggml_hexagon_precompute_allreduce_params(
        ఇది, dst, rank, n_ranks, అబద్ధం, అబద్ధం, /*is_shard_ok=*/ అబద్ధం,
        (నిర్మాణం htp_allreduce_kernel_params *) ar_కణుపు.kernel_params);

    ar_కణుపు.name = "ALLREDUCE";
    ఇది->వరుసలో_చేర్చు_op(ar_కణుపు);
}

ggml_hexagon_shared_బఫరు * ggml_hexagon_session::mmap_tensor(స్థిరరాశి ggml_tensor * t) {
    ఒకవేళ (!t) తిరిగి శూన్యసూచిక;

    స్వయంచాలక sbuf = static_cast<ggml_hexagon_shared_బఫరు *>(t->బఫరు->context);
    ఒకవేళ (!sbuf->mapped) {
        స్థిరరాశి తార్కిక is_బరువు = ggml_backend_బఫరు_get_usage(t->బఫరు) == GGML_BACKEND_బఫరు_USAGE_WEIGHTS;
        స్థిరరాశి తార్కిక extended  = opt_dma64 && is_బరువు;
        sbuf->mmap(extended);
    }
    తిరిగి sbuf;
}

తార్కిక ggml_hexagon_session::clone_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు * sbuf)
{
    GGML_ASSERT(sbuf && sbuf->mem);
    ఒకవేళ (sbuf->sess == ఇది) తిరిగి నిజం;

    స్వయంచాలక mem = sbuf->mem;
    పూర్ణాంకం   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    ఒకవేళ (ఇది->cloned_buffers.find(fd) != ఇది->cloned_buffers.end()) తిరిగి నిజం;

    GGML_ASSERT(sbuf->mapped);

    HEX_VERBOSE("ggml-hex: %s clone-buffer: %s base %p size %zu fd %d\n", ఇది->name.c_str(),
                sbuf->c_name(), sbuf->పునాది(), sbuf->size(), fd);

    స్వయంచాలక clone = std::make_unique<ggml_hexagon_shared_బఫరు>(ఇది, *sbuf);
    ప్రయత్నించు {
        clone->mmap(sbuf->extended);
    } పట్టుకో (స్థిరరాశి std::exception & exc) {
        GGML_LOG_ERROR("ggml-hex: %s lazy mapping of buffer context failed: %s\n", ఇది->c_name(), exc.what());
        తిరిగి అబద్ధం;
    }

    ఇది->cloned_buffers[fd] = std::move(clone);
    mem->mapped_clones.insert(ఇది);
    తిరిగి నిజం;
}

శూన్య ggml_hexagon_session::release_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    స్వయంచాలక mem = sbuf->mem;
    పూర్ణాంకం   fd = mem->fd;

    GGML_ASSERT(fd >= 0);

    స్వయంచాలక it = ఇది->cloned_buffers.find(fd);
    ఒకవేళ (it != ఇది->cloned_buffers.end()) {
        స్వయంచాలక clone = std::move(it->second);
        ఇది->cloned_buffers.erase(it);
    }
    mem->mapped_clones.erase(ఇది);
}

శూన్య ggml_hexagon_session::unclone_బఫరు(స్థిరరాశి ggml_hexagon_shared_బఫరు * sbuf) {
    GGML_ASSERT(sbuf && sbuf->mem);

    స్వయంచాలక mem = sbuf->mem;
    std::vector<ggml_hexagon_session *> sessions(mem->mapped_clones.begin(), mem->mapped_clones.end());

    కొరకు (స్వయంచాలక * sess : sessions) {
        sess->release_బఫరు(sbuf);
    }
}

స్థిర size_t ggml_hexagon_measure_max_vmem(ggml_hexagon_session *sess) {
    // Allocate a bunch pinned buffers till failure.
    // This is kind of expensive but handy for figuring out exactly how much we can mmap on a specific device.
    // Typically we're going to allocate all/most of these buffers anyway for the model weights.

    std::vector<ggml_hexagon_shared_బఫరు *> sbufs;

    స్థిరరాశి size_t MiB = 1024 * 1024;
    స్థిరరాశి size_t GiB = MiB  * 1024;

    size_t vmem = 0;
    size_t అడుగు = 256u * MiB;

    ప్రయత్నించు {
        sbufs.push_back(కొత్త ggml_hexagon_shared_బఫరు(sess, GiB, నిజం)); vmem += GiB;
        sbufs.push_back(కొత్త ggml_hexagon_shared_బఫరు(sess, GiB, నిజం)); vmem += GiB;
        sbufs.push_back(కొత్త ggml_hexagon_shared_బఫరు(sess, GiB, నిజం)); vmem += GiB;

        అయితే (1) {
            sbufs.push_back(కొత్త ggml_hexagon_shared_బఫరు(sess, అడుగు, నిజం));
            vmem += అడుగు;
        }
    } పట్టుకో (...) { }

    కొరకు (స్వయంచాలక b : sbufs) { తొలగించు b; }

    తిరిగి vmem - అడుగు; // backoff to account for overhead from internal mappings
}

శూన్య ggml_hexagon_session::allocate(స్థిరరాశి ggml_hexagon_device_config & config) noexcept(అబద్ధం) {
    పూర్ణాంకం phys_idx = config.physical_idx;
    పూర్ణాంకం virt_idx = config.virtual_idx;

    ఇది->valid_session = అబద్ధం;
    ఇది->valid_handle  = అబద్ధం;
    ఇది->valid_వరుస   = అబద్ధం;
    ఇది->valid_iface   = అబద్ధం;

    ఇది->name          = config.name;
    ఇది->phys_idx      = phys_idx;
    ఇది->virt_idx      = virt_idx;
    ఇది->domain_id     = config.domain_id;
    ఇది->session_id    = 0;
    ఇది->batch_req_seq = 0;
    ఇది->batch_rsp_seq = 0;
    ఇది->చివరి_error    = HTP_స్థితి_OK;

    GGML_LOG_DEBUG("ggml-hex: %s allocating new session : domain %u phys-idx %u virt-idx %u\n", ఇది->name.c_str(), ఇది->domain_id, phys_idx, virt_idx);

    ఒకవేళ (config.domain_id < 0 || config.domain_name.ఖాళీలు()) {
        GGML_LOG_ERROR("ggml-hex: %s: invalid physical CDSP core %d\n", config.name.c_str(), config.physical_idx);
        విసురు std::runtime_error("ggml-hex: invalid physical CDSP core");
    }

    స్థిరరాశి std::string & dom_name = config.domain_name;

    // Create new session if virtual_idx > 0
    ఒకవేళ (virt_idx > 0) {
        నిర్మాణం remote_rpc_reserve_new_session n {};
        n.domain_name_len  = dom_name.size();
        n.domain_name      = const_cast<అక్షరం *>(dom_name.c_str());
        n.session_name     = const_cast<అక్షరం *>(ఇది->name.c_str());
        n.session_name_len = ఇది->name.size();
        n.session_id       = virt_idx;

        పూర్ణాంకం err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (శూన్య *) &n, పరిమాణం(n));
        ఒకవేళ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to reserve new session (physical %d, virtual %d) : error 0x%x\n",
                           ఇది->c_name(), phys_idx, virt_idx, err);
            విసురు std::runtime_error("ggml-hex: remote_session_control(new-sess) failed (see log for details)");
        }

        // Save the IDs
        ఇది->session_id    = n.session_id;
        ఇది->domain_id     = n.effective_domain_id;
        ఇది->valid_session = నిజం;
    } లేకపోతే {
        నిర్మాణం remote_rpc_effective_domain_id eff {};
        eff.domain_name     = const_cast<అక్షరం *>(dom_name.c_str());
        eff.domain_name_len = dom_name.size();
        eff.session_id      = 0;

        పూర్ణాంకం err = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, (శూన్య *) &eff, పరిమాణం(eff));
        ఒకవేళ (err == AEE_SUCCESS) {
            ఇది->domain_id = eff.effective_domain_id;
        } లేకపోతే {
            GGML_LOG_DEBUG("ggml-hex: %s FASTRPC_GET_EFFECTIVE_DOMAIN_ID returned 0x%x, using domain_id %d\n",
                           ఇది->name.c_str(), err, ఇది->domain_id);
        }
    }

    // Enable unsigned modules
    {
        నిర్మాణం remote_rpc_control_unsigned_module u;
        u.domain = ఇది->domain_id;
        u.enable = 1;
        పూర్ణాంకం err  = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (శూన్య *) &u, పరిమాణం(u));
        ఒకవేళ (err != AEE_SUCCESS) {
            GGML_LOG_ERROR("ggml-hex: %s failed to enable unsigned PD : error 0x%x\n", ఇది->c_name(), err);
            విసురు std::runtime_error("ggml-hex: remote_session_control(unsign) failed (see log for details)");
        }
    }

    అక్షరం session_uri[256];
    {
        అక్షరం htp_uri[256];
        snprintf(htp_uri, పరిమాణం(htp_uri), "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0", opt_arch);

        నిర్మాణం remote_rpc_get_uri u = {};
        u.session_id      = ఇది->session_id;
        u.domain_name     = const_cast<అక్షరం *>(dom_name.c_str());
        u.domain_name_len = dom_name.size();
        u.module_uri      = const_cast<అక్షరం *>(htp_uri);
        u.module_uri_len  = అక్షరశ్రేణి_పొడవు(htp_uri);
        u.uri             = session_uri;
        u.uri_len         = పరిమాణం(session_uri);

        పూర్ణాంకం err = remote_session_control(FASTRPC_GET_URI, (శూన్య *) &u, పరిమాణం(u));
        ఒకవేళ (err != AEE_SUCCESS) {
            snprintf(session_uri, పరిమాణం(session_uri), "%s&_dom=%s&_session=%u",
                     htp_uri, dom_name.c_str(), ఇది->session_id);

            GGML_LOG_WARN("ggml-hex: %s failed to get URI (physical %d, virtual %d) : error 0x%x. Falling back to single session URI: %s\n",
                          ఇది->c_name(), phys_idx, virt_idx, err, session_uri);
        }
    }

    // Open session
    పూర్ణాంకం err = htp_iface_open(session_uri, &ఇది->handle);
    ఒకవేళ (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: %s failed to open session : uri %s error 0x%x\n", ఇది->c_name(), session_uri, err);
        విసురు std::runtime_error("ggml-hex: failed to open session (see log for details)");
    }

    ఇది->valid_handle = నిజం;

    // Query HW info and resolve session options
    {
        గుర్తులేని పూర్ణాంకం hw_n_దారాలు = 0;
        గుర్తులేని పూర్ణాంకం hw_n_hvx     = 0;
        గుర్తులేని పూర్ణాంకం hw_n_hmx     = 0;
        గుర్తులేని దీర్ఘ దీర్ఘ hw_vtcm_కొలత = 0;
        పూర్ణాంకం hw_err = htp_iface_hwinfo(ఇది->handle, &hw_n_దారాలు, &hw_n_hvx, &hw_n_hmx, &hw_vtcm_కొలత);
        ఒకవేళ (hw_err == 0) {
            స్థిరరాశి uint32_t max_n_దారాలు = (std::min)((uint32_t) HTP_MAX_NTHREADS, (uint32_t) hw_n_దారాలు);
            ఇది->n_దారాలు = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) max_n_దారాలు) : max_n_దారాలు;
            ఇది->n_hvx     = ఇది->n_దారాలు;
            ఇది->n_hmx     = (opt_nhmx != 0) ? (uint32_t)hw_n_hmx : 0;
            ఇది->vtcm_కొలత = (uint64_t)hw_vtcm_కొలత;
            GGML_LOG_INFO("ggml-hex: %s hwinfo: threads %u, hvx %u, hmx %u, vtcm %llu MB\n",
                          ఇది->c_name(), ఇది->n_దారాలు, ఇది->n_hvx, ఇది->n_hmx,
                          (గుర్తులేని దీర్ఘ దీర్ఘ)(ఇది->vtcm_కొలత / (1024 * 1024)));
        } లేకపోతే {
            GGML_LOG_WARN("ggml-hex: %s failed to query hwinfo (0x%x), using defaults\n", ఇది->c_name(), hw_err);
            స్థిరరాశి uint32_t default_n_దారాలు = (std::min)(8u, (uint32_t) HTP_MAX_NTHREADS);
            ఇది->n_దారాలు = opt_nhvx > 0 ? (uint32_t) (std::min)(opt_nhvx, (size_t) HTP_MAX_NTHREADS) : default_n_దారాలు;
            ఇది->n_hvx     = ఇది->n_దారాలు;
            ఇది->n_hmx     = (opt_nhmx != 0) ? 1 : 0;
            ఇది->vtcm_కొలత = 8 * 1024 * 1024;
        }
    }

    // Enable FastRPC QoS mode
    {
        నిర్మాణం remote_rpc_control_latency l;
        l.enable = 1;

        పూర్ణాంకం err = remote_handle64_control(ఇది->handle, DSPRPC_CONTROL_LATENCY, (శూన్య *) &l, పరిమాణం(l));
        ఒకవేళ (err != 0) {
            GGML_LOG_WARN("ggml-hex: failed to enable fastrpc QOS mode: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    GGML_LOG_INFO("ggml-hex: %s new session : session-id %d domain-id %d uri %s handle 0x%lx\n", ఇది->c_name(),
                  ఇది->session_id, ఇది->domain_id, session_uri, (గుర్తులేని దీర్ఘ) ఇది->handle);

    స్థిరరాశి size_t req_q_కొలత = (పరిమాణం(htp_opbatch_req) * opt_opqueue * 2) + 1024;
    స్థిరరాశి size_t rsp_q_కొలత = (పరిమాణం(htp_opbatch_rsp) * opt_opqueue * 2) + 1024;

    // Now let's setup the DSP queue
    err = dspqueue_create(ఇది->domain_id,
                          0,              // Flags
                          req_q_కొలత,     // Request  queue size (in bytes)
                          rsp_q_కొలత,     // Response queue size (in bytes)
                          శూన్యసూచిక,        // Read packet callback (we handle reads explicitly)
                          శూన్యసూచిక,        // Error callback (we handle errors during reads)
                          (శూన్య *) ఇది,  // Callback context
                          &queue);
    ఒకవేళ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s dspqueue_create failed: 0x%08x\n", ఇది->name.c_str(), (గుర్తులేని) err);
        విసురు std::runtime_error("ggml-hex: failed to create dspqueue (see log for details)");
    }

    ఇది->valid_వరుస = నిజం;

    // Export queue for use on the DSP
    err = dspqueue_export(queue, &ఇది->వరుస_id);
    ఒకవేళ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: dspqueue_export failed: 0x%08x\n", (గుర్తులేని) err);
        విసురు std::runtime_error("ggml-hex: dspqueue export failed (see log for details)");
    }

    ఒకవేళ (opt_etm) {
        err = htp_iface_etm(ఇది->handle, 1);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable ETM tracing: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    // Allocate buffers and state for op batching
    ఇది->op_వరుస = కొత్త ggml_hexagon_opqueue(ఇది, opt_opbatch, opt_opqueue);

    ఇది->fence_buf = కొత్త ggml_hexagon_fence_బఫరు(ఇది, &dev_ctx->fence_బఫరు_type, 64 * 1024);
    ఒకవేళ (ఇది->mdev.count > 1) {
        ఇది->mdev_fence_స్థానం = ఇది->alloc_fence(ఇది->mdev.count);
    }

    ఒకవేళ (!opt_vmem) {
        opt_vmem = ggml_hexagon_measure_max_vmem(ఇది);
        GGML_LOG_INFO("ggml-hex: %s measured max vmem %zu\n", ఇది->c_name(), opt_vmem);
    }
    స్థిరరాశి size_t shm_కొలత = ఇది->op_వరుస->shm_కొలత();
    ఇది->max_vmem = (opt_vmem > shm_కొలత) ? (opt_vmem - shm_కొలత) : opt_vmem;

    ఇది->op_batch = కొత్త ggml_hexagon_opbatch(ఇది, opt_opbatch, ఇది->max_vmem);

    // Start dspqueue/opbatch processing
    err = htp_iface_ప్రారంభం(ఇది->handle, ఇది->session_id, ఇది->వరుస_id, ఇది->n_దారాలు, opt_nhmx, ఇది->max_vmem, (uint32_t) opt_power_బహుళకం);
    ఒకవేళ (err != 0) {
        GGML_LOG_ERROR("ggml-hex: %s failed to start session: 0x%08x\n", ఇది->c_name(), (గుర్తులేని) err);
        విసురు std::runtime_error("ggml-hex: iface start failed (see log for details)");
    }
    ఇది->valid_iface = నిజం;
    GGML_LOG_INFO("ggml-hex: %s power mode: %s\n", ఇది->c_name(), htp_power_బహుళకం_గమ్యం_str(opt_power_బహుళకం));

    ఒకవేళ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        std::copy(opt_pmu_evt.begin(), opt_pmu_evt.end(), pmu_conf.events);

        err = htp_iface_profiler(ఇది->handle, opt_profile, &pmu_conf);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to enable profiling: 0x%08x\n", (గుర్తులేని) err);
        }
    }
}

శూన్య ggml_hexagon_session::release() noexcept(నిజం) {
    GGML_LOG_INFO("ggml-hex: releasing session: %s\n", ఇది->name.c_str());

    ఇది->mdev.sessions.clear();

    పూర్ణాంకం err;

    ఒకవేళ (ఇది->valid_iface) {
        // Stop dspqueue/opbatch processing
        err = htp_iface_stop(ఇది->handle);
        ఒకవేళ (err != 0) {
            GGML_ABORT("ggml-hex: htp_iface_stop failed: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    తొలగించు ఇది->op_batch;
    తొలగించు ఇది->op_వరుస;
    కొరకు (స్వయంచాలక & it : ఇది->cpy_fence_slots) {
        free_fence((శూన్య *) it.second, 1);
    }
    ఇది->cpy_fence_slots.clear();

    ఒకవేళ (ఇది->fence_buf) {
        unclone_బఫరు(ఇది->fence_buf);
        తొలగించు ఇది->fence_buf;
        ఇది->fence_buf = శూన్యసూచిక;
    }
    అయితే (!ఇది->cloned_buffers.ఖాళీలు()) {
        release_బఫరు(ఇది->cloned_buffers.begin()->second.get());
    }

    ఒకవేళ (opt_etm) {
        err = htp_iface_etm(ఇది->handle, 0);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable ETM tracing: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    ఒకవేళ (opt_profile) {
        htp_iface_pmu_conf pmu_conf{};
        err = htp_iface_profiler(ఇది->handle, 0, &pmu_conf);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: warn : failed to disable profiling: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    ఒకవేళ (ఇది->valid_వరుస) {
        err = dspqueue_close(queue);
        ఒకవేళ (err != 0) {
            GGML_ABORT("ggml-hex: dspqueue_close failed: 0x%08x\n", (గుర్తులేని) err);
        }
    }

    ఒకవేళ (ఇది->valid_handle) {
        htp_iface_close(ఇది->handle);
    }
}

ggml_hexagon_session::ggml_hexagon_session(స్థిరరాశి ggml_hexagon_device_config & config, ggml_backend_dev_t dev, uint32_t mdev_idx, uint32_t mdev_లెక్క) noexcept(అబద్ధం) {
    ఇది->dev        = dev;
    ఇది->dev_ctx    = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    ఇది->mdev.idx   = mdev_idx;
    ఇది->mdev.count = mdev_లెక్క > 0 ? mdev_లెక్క : (uint32_t) (1 + config.mdev_సమూహం.size());
    op_batch         = శూన్యసూచిక;
    op_వరుస         = శూన్యసూచిక;
    fence_buf        = శూన్యసూచిక;
    fence_seq        = ((uintptr_t)ఇది) & 0xFFFF;

    ప్రయత్నించు {
        allocate(config);
        ఒకవేళ (ఇది->mdev.idx == 0 && !config.mdev_సమూహం.ఖాళీలు()) {
            కొరకు (size_t i = 0; i < config.mdev_సమూహం.size(); i++) {
                ఇది->mdev.sessions.push_back(std::make_unique<ggml_hexagon_session>(
                    config.mdev_సమూహం[i], ఇది->dev, (uint32_t) (i + 1), ఇది->mdev.count));
            }
        }
    } పట్టుకో (స్థిరరాశి std::exception & exc) {
        release();
        విసురు;
    }
}

ggml_hexagon_session::~ggml_hexagon_session() noexcept(నిజం) {
    release();
}

// ** backend interface

స్థిర తార్కిక ggml_hexagon_flash_attn_is_hmx_eligible(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * q,
    స్థిరరాశి నిర్మాణం ggml_tensor * k,
    స్థిరరాశి నిర్మాణం ggml_tensor * v,
    స్థిరరాశి నిర్మాణం ggml_tensor * sinks
) {
    ఒకవేళ (sess->n_hmx == 0) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (opt_fa_select < 2) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ ((k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_Q8_0) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_Q8_0)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t DK = q->ne[0];
    స్థిరరాశి uint32_t DV = v->ne[0];

    // Head dims that are not multiples of 64 are handled by internally padding to
    // DK_pad/DV_pad = round_up(.,64) and zero-filling the tail lanes.
    ఒకవేళ (DK % 8 != 0 || DV % 8 != 0) {
        తిరిగి అబద్ధం;
    }

    GGML_వాడని(sinks);

    // Explicit force mode
    ఒకవేళ (opt_fa_select > 2) {
        తిరిగి నిజం;
    }

    స్థిరరాశి uint32_t M = q->ne[1];

    // Prefill or batched decode
    ఒకవేళ (M > 1) {
        తిరిగి నిజం;
    }

    స్థిరరాశి uint32_t n_తల = q->ne[2];
    స్థిరరాశి uint32_t n_kv_heads = k->ne[2];
    స్థిరరాశి uint32_t G = n_kv_heads > 0 ? n_తల / n_kv_heads : 1;
    స్థిరరాశి uint32_t S = k->ne[1];
    స్థిరరాశి uint32_t మొత్తం_దారాలు = sess->n_దారాలు * sess->mdev.count;

    // Compute-bound head dim (for DK == 256 decode, HVX is faster when KV heads span all threads)
    ఒకవేళ (DK > 256 || (DK == 256 && (DV > 256 || n_kv_heads < మొత్తం_దారాలు))) {
        తిరిగి నిజం;
    }

    // Tile alignment for 32-row HMX tiles
    స్థిరరాశి తార్కిక is_tile_aligned = (G > 0 && (32 % G == 0));
    ఒకవేళ (!is_tile_aligned) {
        ఒకవేళ (sess->n_దారాలు >= 6) {
            తిరిగి అబద్ధం;
        }
        తిరిగి S >= 1024;
    }

    // Context depth crossover
    uint32_t s_cross = 512;
    ఒకవేళ (DK == 256) {
        s_cross = 4096;
    } లేకపోతే ఒకవేళ (DK <= 64) {
        s_cross = (sess->n_దారాలు >= 8) ? 2048 : ((sess->n_దారాలు >= 6) ? 768 : 512);
    } లేకపోతే {
        s_cross = (sess->n_దారాలు >= 8) ? 1024 : 512;
    }

    తిరిగి S >= s_cross;
}

స్థిర తార్కిక ggml_hexagon_precompute_flash_attn_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_fa_kernel_params * kparams
) {
    ఒకవేళ (opt_fa_select < 1) {
        తిరిగి అబద్ధం;
    }

    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి నిర్మాణం ggml_tensor * q    = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * k    = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * v    = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * ముసుగు = op->src[3];

    స్థిరరాశి uint32_t neq0 = q->ne[0];  // head_dim (DK)
    స్థిరరాశి uint32_t neq1 = q->ne[1];  // n_tokens
    స్థిరరాశి uint32_t neq2 = q->ne[2];  // n_heads

    స్థిరరాశి uint32_t nek1 = k->ne[1];  // kv_len

    స్థిరరాశి uint32_t nev0 = v->ne[0];  // head_dim (DV)

    స్థిరరాశి uint32_t DK = neq0;
    స్థిరరాశి uint32_t DV = nev0;

    స్థిరరాశి uint32_t n_kv_heads = k->ne[2];
    స్థిరరాశి uint32_t G          = neq2 / n_kv_heads;

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&scale,         &op->op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      &op->op_params[1], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, &op->op_params[2], పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale = scale;
    kparams->max_bias = max_bias;
    kparams->logit_softcap = logit_softcap;

    kparams->తల_విభజించు = (opt_fa_తల_విభజించు != 0) ? 1 : 0;
    kparams->flags      = 0;
    kparams->G = G;

    స్థిరరాశి uint32_t n_తల = q->ne[2];
    kparams->n_తల_log2 = 1u << (uint32_t) std::floor(std::log2(n_తల));
    kparams->m0 = std::ఘాతం(2.0f, -(max_bias) / kparams->n_తల_log2);
    kparams->m1 = std::ఘాతం(2.0f, -(max_bias / 2.0f) / kparams->n_తల_log2);

    // Check HMX eligibility
    స్థిరరాశి నిర్మాణం ggml_tensor * sinks = op->src[4];
    ఒకవేళ (ggml_hexagon_flash_attn_is_hmx_eligible(sess, q, k, v, sinks)) {
        // HMX tiles head_dim in units of 64; when DK/DV are not 64-aligned the kernel
        // operates on padded dims with zero-filled tail lanes. VTCM budget and chunk-size
        // are sized for the padded tiles.
        స్థిరరాశి uint32_t DK_pad = hex_విడత_పైకి(DK, 64);
        స్థిరరాశి uint32_t DV_pad = hex_విడత_పైకి(DV, 64);
        size_t Br = 0, Bc = 0;
        పూర్ణాంకం ret = hmx_fa_కనుగొను_chunk_కొలత(&Br, &Bc, G, DK_pad, DV_pad, neq1, nek1, sess->vtcm_కొలత, sess->n_దారాలు, (q->type == GGML_TYPE_F32), sinks != శూన్యసూచిక, n_తల);
        ఒకవేళ (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            ఒకవేళ (logit_softcap == 0.0f) {
                kparams->scale = scale * (దశాంశం) M_LOG2E;
                kparams->logit_softcap = 0.0f;
            } లేకపోతే {
                kparams->scale = scale;
                kparams->logit_softcap = logit_softcap * (దశాంశం) M_LOG2E;
            }
            kparams->Br = Br;
            kparams->Bc = Bc;
            kparams->n_kv_ఖండాలు = (nek1 + Bc - 1) / Bc;
            kparams->n_దారాలు = (kparams->n_kv_ఖండాలు >= 3 && sess->n_దారాలు >= 2) ? sess->n_దారాలు : 1;

            kparams->u.hmx.g_br = hex_align_పైకి(G * Br, 32);
            kparams->u.hmx.pipeline = (kparams->n_kv_ఖండాలు >= 3 && sess->n_దారాలు >= 2) ? 1 : 0;
            kparams->vtcm_కొలత = hmx_fa_compute_vtcm_usage(G, DK_pad, DV_pad, Br, Bc, kparams->n_దారాలు, kparams->u.hmx.pipeline != 0, (q->type == GGML_TYPE_F32), sinks != శూన్యసూచిక, n_తల);

            స్థిరరాశి size_t అడ్డువరుస_vec_బైట్లు = hex_align_పైకి(Bc * పరిమాణం(uint16_t), 256);
            kparams->u.hmx.అడ్డువరుస_buf_stride = అడ్డువరుస_vec_బైట్లు / 128; // HVX vector is 128 bytes

            స్థిరరాశి size_t m_పంక్తి_బైట్లు = hex_align_పైకి(Bc * పరిమాణం(uint16_t), 128);
            kparams->u.hmx.ముసుగు_buf_అడ్డువరుస_stride = m_పంక్తి_బైట్లు / పరిమాణం(uint16_t);
            kparams->u.hmx.ముసుగు_ప్రసారం = (ముసుగు != శూన్యసూచిక && ముసుగు->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_విలువలు(G);
            ఒకవేళ (ముసుగు) {
                kparams->src3_div2 = init_fastdiv_విలువలు(ముసుగు->ne[2]);
                kparams->src3_div3 = init_fastdiv_విలువలు(ముసుగు->ne[3]);
            }

            kparams->qrows = 0;
            kparams->qrows_చొప్పున_thread = 0;
            తిరిగి నిజం;
        }
    }

    // Fallback to HVX
    kparams->kernel_type = HTP_FA_KERNEL_HVX;
    kparams->Br = 1;
    kparams->Bc = 64; // FLASH_ATTN_BLOCK_SIZE
    kparams->n_kv_ఖండాలు = (k->ne[1] + 64 - 1) / 64;
    kparams->n_దారాలు = sess->n_దారాలు;

    స్థిరరాశి size_t కొలత_q_అడ్డువరుస_padded = hex_విడత_పైకి(q->ne[0] * ((q->type == GGML_TYPE_F32) ? 4 : 2), 128);
    స్థిరరాశి size_t కొలత_k_అడ్డువరుస_padded = hex_విడత_పైకి(k->ne[0] * 2, 128);
    స్థిరరాశి size_t కొలత_v_అడ్డువరుస_padded = hex_విడత_పైకి(v->ne[0] * 2, 128);

    kparams->vtcm_కొలత = hvx_fa_compute_vtcm_usage(DK, DV, (q->type == GGML_TYPE_F32), ముసుగు != శూన్యసూచిక, sinks != శూన్యసూచిక, n_తల, sess->n_దారాలు);

    kparams->u.hvx.కొలత_q_అడ్డువరుస_padded = కొలత_q_అడ్డువరుస_padded;
    kparams->u.hvx.కొలత_k_అడ్డువరుస_padded = కొలత_k_అడ్డువరుస_padded;
    kparams->u.hvx.కొలత_v_అడ్డువరుస_padded = కొలత_v_అడ్డువరుస_padded;
    kparams->u.hvx.src0_div21 = init_fastdiv_విలువలు(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1 = init_fastdiv_విలువలు(q->ne[1]);
    kparams->ప్రసారం_rk2 = init_fastdiv_విలువలు(q->ne[2]/k->ne[2]);
    kparams->ప్రసారం_rk3 = init_fastdiv_విలువలు(q->ne[3]/k->ne[3]);
    kparams->ప్రసారం_rv2 = init_fastdiv_విలువలు(q->ne[2]/v->ne[2]);
    kparams->ప్రసారం_rv3 = init_fastdiv_విలువలు(q->ne[3]/v->ne[3]);
    ఒకవేళ (ముసుగు) {
        kparams->src3_div2 = init_fastdiv_విలువలు(ముసుగు->ne[2]);
        kparams->src3_div3 = init_fastdiv_విలువలు(ముసుగు->ne[3]);
    }

    kparams->qrows = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_చొప్పున_thread = (kparams->qrows + sess->n_దారాలు - 1) / sess->n_దారాలు;

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_flash_attn_ext(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * src3 = op->src[3];
    స్థిరరాశి నిర్మాణం ggml_tensor * src4 = op->src[4];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    // Check for F16/Q8_0 support
    ఒకవేళ ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) ||
        (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_Q8_0) ||
        (src2->type != GGML_TYPE_F16 && src2->type != GGML_TYPE_Q8_0)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src3 && src3->type != GGML_TYPE_F16) {  //ముసుగు
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src4 && src4->type != GGML_TYPE_F32) {  // sinks
        తిరిగి అబద్ధం;
    }

    // For now we support F32 or F16 output as htp backend often converts output on the fly if needed,
    // but the op implementation writes to F16 or F32.
    // Let's assume dst can be F32 or F16.
    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_fa_kernel_params kparams;
    ఒకవేళ (!ggml_hexagon_precompute_flash_attn_params(sess, op, &kparams)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
        HEX_VERBOSE("ggml-hex: skip flash_attn_ext because VTCM needed (%d) > budget (%zu)\n",
                    kparams.vtcm_కొలత, sess->vtcm_కొలత);
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_gated_delta_net(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    ఒకవేళ (opt_gdn_select < 1) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి నిర్మాణం ggml_tensor * q     = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * k     = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * v     = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * g     = op->src[3];
    స్థిరరాశి నిర్మాణం ggml_tensor * beta  = op->src[4];
    స్థిరరాశి నిర్మాణం ggml_tensor * state = op->src[5];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst   = op;

    ఒకవేళ (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(q) || !ggml_is_contiguous_అడ్డువరుసలు(k) || !ggml_is_contiguous_అడ్డువరుసలు(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t S_v      = v->ne[0];
    స్థిరరాశి int64_t H        = v->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = v->ne[2];
    స్థిరరాశి int64_t n_seqs   = v->ne[3];
    స్థిరరాశి int64_t K        = ggml_get_op_params_i32(op, 0);

    ఒకవేళ (S_v <= 0 || S_v > 128 || H <= 0 || n_టోకెన్లు <= 0 || n_seqs <= 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_టోకెన్లు || k->ne[2] != n_టోకెన్లు || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        తిరిగి అబద్ధం;
    }
    // state holds s0 only [S_v, S_v, H, n_seqs]; K is op param 0.
    ఒకవేళ (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->ne[0] != S_v * H || dst->ne[1] != n_టోకెన్లు * n_seqs + S_v * n_seqs * K) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = (uint32_t) (H * n_seqs);
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, మొత్తం_అడ్డువరుసలు);

    స్థిరరాశి తార్కిక can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_టోకెన్లు >= HTP_GDN_MIN_టోకెన్లు) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    ఒకవేళ (can_use_hmx) {
        నిర్మాణం htp_gdn_hmx_vtcm_layout layout;
        uint32_t n_heads_batch = 0;
        ఒకవేళ (!htp_gdn_hmx_solve_layout(&layout, (uint32_t) S_v, HTP_GDN_CHUNK_కొలత, మొత్తం_అడ్డువరుసలు, sess->vtcm_కొలత, n_దారాలు, నిజం, &n_heads_batch)) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే {
        నిర్మాణం htp_gdn_vtcm_layout layout;
        htp_gdn_vtcm_layout_build(&layout, (uint32_t) S_v, n_దారాలు);
        ఒకవేళ (layout.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_matmul_is_hmx_eligible(
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    తార్కిక is_matmul_id,
    తార్కిక is_batched
) {
    GGML_వాడని(dst);

    ఒకవేళ (src1->type != GGML_TYPE_F32 && (src1->type != GGML_TYPE_F16 || is_matmul_id)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి పూర్ణాంకం ne00  = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం ne01  = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం ne11  = src1->ne[1];
    స్థిరరాశి పూర్ణాంకం ne12  = src1->ne[2];
    స్థిరరాశి పూర్ణాంకం wtype = src0->type;

    // HMX weight tiles accept non-32-aligned N for non-matmul_id.
    ఒకవేళ (ne01 % HTP_TENSOR_TILE_N_అడ్డువరుసలు != 0 && is_matmul_id) {
        తిరిగి అబద్ధం;
    }

    // HMX supports F16, F32, and repack quantized types.
    ఒకవేళ (!htp_is_hmx_బరువు_type((ggml_type) wtype)) {
        తిరిగి అబద్ధం;
    }

    // HMX paths require K aligned to HMX tile cols (32).
    ఒకవేళ (ne00 % HTP_TENSOR_TILE_N_నిలువుల_సంఖ్య != 0) {
        తిరిగి అబద్ధం;
    }

    // Quantized and tiled HMX kernels only handle flat 2D matmul (or matmul_id wrapping flat 2D matmuls).
    స్థిరరాశి తార్కిక is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    ఒకవేళ (ggml_is_quantized((ggml_type) wtype) && !is_tiled) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!is_matmul_id && is_batched && is_tiled) {
        తిరిగి అబద్ధం;
    }

    // HMX assumes contiguous row-major layout.
    ఒకవేళ (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        తిరిగి అబద్ధం;
    }

    // M alignment: Use HMX when M > HTP_MM_HMX_MIN_NROWS.
    // For MUL_MAT_ID, src1 shape is [K, n_expert_used, n_tokens, 1], so n_tokens is ne12.
    స్థిరరాశి పూర్ణాంకం m = is_matmul_id ? ne12 : ne11;
    ఒకవేళ (m <= HTP_MM_HMX_MIN_NROWS) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(dst);
}

స్థిర తార్కిక ggml_hexagon_precompute_hmx_mm_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * act,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    పూర్ణాంకం wtype,
    పూర్ణాంకం ne00_padded,
    పూర్ణాంకం ne01_padded,
    పూర్ణాంకం ne02,
    పూర్ణాంకం ne11,
    పూర్ణాంకం ne12,
    పూర్ణాంకం ne11_padded,
    తార్కిక is_matmul_id,
    తార్కిక is_batched,
    size_t src2_కొలత,
    size_t vtcm_budget,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    స్థిరరాశి తార్కిక is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    స్థిరరాశి పూర్ణాంకం aligned_tile_కొలత = is_tiled ? htp_tensor_get_tile_padded_కొలత(wtype) : 0;
    స్థిరరాశి పూర్ణాంకం n_దారాలు = (పూర్ణాంకం)sess->n_దారాలు;
    స్థిరరాశి పూర్ణాంకం ne10 = act->ne[0];

    పూర్ణాంకం m_for_solver = ne11;
    పూర్ణాంకం m_for_solver_padded = ne11_padded;
    // matmul_id partitions by expert; regular matmul partitions M rows (ne11) across devices
    ఒకవేళ (!is_matmul_id && sess->mdev.count > 1 && ((uint32_t) ne11 >= sess->mdev.count)) {
        // when dst is null, padded dims are used for estimate which are 128-byte aligned
        స్థిరరాశి తార్కిక dst_అడ్డువరుస_విభజించు = dst ? ggml_hexagon_can_అడ్డువరుస_విభజన(dst) : నిజం;
        స్థిరరాశి తార్కిక act_అడ్డువరుస_విభజించు = ggml_hexagon_can_అడ్డువరుస_విభజన(act);
        ఒకవేళ (dst_అడ్డువరుస_విభజించు && act_అడ్డువరుస_విభజించు) {
            m_for_solver = (ne11 + (పూర్ణాంకం) sess->mdev.count - 1) / (పూర్ణాంకం) sess->mdev.count;
            m_for_solver_padded = hex_విడత_పైకి(std::max(m_for_solver, 32), 32);
        }
    }
    స్థిరరాశి తార్కిక pipeline = is_matmul_id ? అబద్ధం : htp_mm_hmx_pipeline(m_for_solver);

    స్థిరరాశి తార్కిక is_batched_val = is_matmul_id ? అబద్ధం : is_batched;
    స్థిరరాశి పూర్ణాంకం సమూహం_కొలత = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk = 0;
    size_t n_chunk = 0;
    size_t vtcm_కొలత = 0;
    తార్కిక use_grouped = అబద్ధం;
    పూర్ణాంకం act_దారాలు_selected = 0;

    ఒకవేళ (is_batched_val && wtype == GGML_TYPE_F16 && !is_tiled && సమూహం_కొలత > 1) {
        // Try grouped path first
        ఒకవేళ (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, m_for_solver, సమూహం_కొలత, n_దారాలు, pipeline, src2_కొలత, vtcm_budget, &m_chunk, &n_chunk, &act_దారాలు_selected, &vtcm_కొలత)) {
            use_grouped = నిజం;
        }
    }

    ఒకవేళ (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        పూర్ణాంకం m_id_అడ్డువరుసలు = 0;
        ఒకవేళ (dst && is_matmul_id) {
            స్థిరరాశి పూర్ణాంకం n_experts = ne02 > 0 ? ne02 : 1;
            స్థిరరాశి size_t మొత్తం_expert_అడ్డువరుసలు = (size_t) dst->ne[1] * dst->ne[2];
            పూర్ణాంకం m_చొప్పున_expert = (పూర్ణాంకం) ((మొత్తం_expert_అడ్డువరుసలు + n_experts - 1) / n_experts);
            ఒకవేళ (sess->mdev.count > 1 && ggml_hexagon_can_అడ్డువరుస_విభజన(dst)) {
                m_చొప్పున_expert = (m_చొప్పున_expert + (పూర్ణాంకం) sess->mdev.count - 1) / (పూర్ణాంకం) sess->mdev.count;
            }
            m_id_అడ్డువరుసలు = hex_విడత_పైకి(std::max(m_చొప్పున_expert, 32), 32);
        }
        స్థిరరాశి uint32_t ఖర్చు_m = is_matmul_id ? (uint32_t) m_id_అడ్డువరుసలు : (uint32_t) m_for_solver;
        ఒకవేళ (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_అడ్డువరుసలు, ne01_padded, m_for_solver_padded, ఖర్చు_m, n_దారాలు, pipeline, is_matmul_id, aligned_tile_కొలత, src2_కొలత, vtcm_budget, &m_chunk, &n_chunk, &act_దారాలు_selected, &vtcm_కొలత)) {
            తిరిగి అబద్ధం;
        }
    }

    kparams->n_hmx = 1;
    kparams->pipeline = pipeline ? 1 : 0;
    kparams->m_chunk = m_chunk;
    kparams->n_chunk = n_chunk;
    kparams->n_దారాలు = n_దారాలు;
    kparams->n_act_దారాలు = act_దారాలు_selected;
    kparams->tile_కొలత = is_tiled ? htp_tensor_get_tile_కొలత(wtype) : 0;
    kparams->aligned_tile_కొలత = aligned_tile_కొలత;
    kparams->act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
    kparams->vtcm_కొలత = vtcm_కొలత;
    kparams->vtcm_src0_కొలత = 0;
    kparams->div_n_act_దారాలు = init_fastdiv_విలువలు(act_దారాలు_selected);
    kparams->div_ne00_padded   = init_fastdiv_విలువలు(ne00_padded);
    kparams->vtcm_act_కొలత     = 0;
    kparams->vtcm_bias_కొలత    = (int32_t) src2_కొలత;
    kparams->vtcm_dst_కొలత     = 0;

    ఒకవేళ (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_BATCHED;
    } లేకపోతే {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    తిరిగి నిజం;
}

స్థిర శూన్య ggml_hexagon_precompute_hvx_mm_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    పూర్ణాంకం wtype,
    తార్కిక is_tiled,
    పూర్ణాంకం ne02,
    పూర్ణాంకం ne03,
    పూర్ణాంకం ne10,
    పూర్ణాంకం ne11,
    పూర్ణాంకం ne12,
    పూర్ణాంకం ne13,
    తార్కిక is_matmul_id,
    స్థిరరాశి size_t src2_అడ్డువరుస_కొలత,
    size_t vtcm_budget,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    ఒకవేళ (opt_mm_select < 1) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    }

    kparams->n_hmx = 0;
    kparams->n_దారాలు = sess->n_దారాలు;

    GGML_వాడని(ne02);
    GGML_వాడని(ne03);

    స్థిరరాశి పూర్ణాంకం src1_nrows = ne11 * ne12 * ne13;

    ఒకవేళ (is_tiled) {
        // Tiled HVX (quantized weights or tiled F16/F32/BF16)
        kparams->tile_కొలత = htp_tensor_get_tile_కొలత(wtype);
        kparams->aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(wtype);

        స్థిరరాశి తార్కిక k_align = (ne10 % htp_tensor_get_tile_k(wtype) == 0);

        ఒకవేళ (is_matmul_id) {
            kparams->kernel_type  = (src1_nrows < (పూర్ణాంకం) sess->n_దారాలు)
                                  ? HTP_MM_KERNEL_HVX_QUANT_ఖండం : HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస;
            kparams->act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);

            నిర్మాణం htp_mm_hvx_vtcm_layout L;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t ఉత్తమ_n_prefetch = 2;
            కొరకు (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                htp_mm_hvx_vtcm_layout_build(
                    &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_దారాలు,
                    0, src0->nb[1], kparams->act_అడ్డువరుస_కొలత, 0, d, నిజం, అబద్ధం
                );
                ఒకవేళ (L.మొత్తం_బైట్లు <= vtcm_budget) {
                    ఉత్తమ_n_prefetch = d;
                    ఆపు;
                }
            }
            ఒకవేళ (L.మొత్తం_బైట్లు > vtcm_budget) {
                kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
                తిరిగి;
            }
            kparams->n_prefetch     = ఉత్తమ_n_prefetch;
            kparams->vtcm_కొలత      = L.మొత్తం_బైట్లు;
            kparams->vtcm_src0_కొలత = L.src0_బైట్లు;
            kparams->vtcm_act_కొలత  = L.act_బైట్లు;
            kparams->vtcm_bias_కొలత = 0;
            kparams->vtcm_dst_కొలత  = L.dst_బైట్లు;
            తిరిగి;
        } లేకపోతే {
            తార్కిక try_tiled = (k_align && opt_mm_select >= 1);
            ఒకవేళ (try_tiled) {
                kparams->act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
                ఒకవేళ (src1_nrows < (పూర్ణాంకం) sess->n_దారాలు) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ఖండం;
                } లేకపోతే {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస;
                }

                నిర్మాణం htp_mm_hvx_vtcm_layout L;
                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t ఉత్తమ_n_prefetch = 2;
                కొరకు (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    htp_mm_hvx_vtcm_layout_build(
                        &L, kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_దారాలు,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_అడ్డువరుస_కొలత, d, అబద్ధం, అబద్ధం
                    );
                    ఒకవేళ (L.మొత్తం_బైట్లు <= vtcm_budget) {
                        ఉత్తమ_n_prefetch = d;
                        ఆపు;
                    }
                }

                uint32_t m_chunk = 0;
                ఒకవేళ (htp_mm_hvx_solve_vtcm_params(
                        kparams->kernel_type, wtype, ne10, src1_nrows, sess->n_దారాలు,
                        dst->nb[1], src0->nb[1], src1->nb[1], src2_అడ్డువరుస_కొలత, ఉత్తమ_n_prefetch, vtcm_budget,
                        &L, &m_chunk)) {
                    kparams->n_prefetch     = ఉత్తమ_n_prefetch;
                    kparams->m_chunk        = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
                    kparams->vtcm_కొలత      = L.మొత్తం_బైట్లు;
                    kparams->vtcm_src0_కొలత = L.src0_బైట్లు;
                    kparams->vtcm_act_కొలత  = L.act_బైట్లు;
                    kparams->vtcm_bias_కొలత = L.bias_బైట్లు;
                    kparams->vtcm_dst_కొలత  = L.dst_బైట్లు;
                    తిరిగి;
                }
            }

            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            తిరిగి;
        }
    } లేకపోతే ఒకవేళ (is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    } లేకపోతే ఒకవేళ (wtype == GGML_TYPE_F16) {
        // Flat F16 HVX
        నిర్మాణం htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ఒకవేళ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, sess->n_దారాలు,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_అడ్డువరుస_కొలత, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * 2, 128);
            kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;
            kparams->vtcm_src0_కొలత = L.src0_బైట్లు;
            kparams->vtcm_act_కొలత  = L.act_బైట్లు;
            kparams->vtcm_bias_కొలత = L.bias_బైట్లు;
            kparams->vtcm_dst_కొలత = L.dst_బైట్లు;
            kparams->n_prefetch = 16;
            తిరిగి;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    } లేకపోతే ఒకవేళ (wtype == GGML_TYPE_F32) {
        // Flat F32 HVX
        నిర్మాణం htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ఒకవేళ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, sess->n_దారాలు,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_అడ్డువరుస_కొలత, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * 4, 128);
            kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;
            kparams->vtcm_src0_కొలత = L.src0_బైట్లు;
            kparams->vtcm_act_కొలత  = L.act_బైట్లు;
            kparams->vtcm_bias_కొలత = L.bias_బైట్లు;
            kparams->vtcm_dst_కొలత = L.dst_బైట్లు;
            kparams->n_prefetch = 16;
            తిరిగి;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    } లేకపోతే ఒకవేళ (wtype == GGML_TYPE_BF16) {
        // Flat BF16 HVX
        నిర్మాణం htp_mm_hvx_vtcm_layout L;
        uint32_t m_chunk = 0;
        ఒకవేళ (htp_mm_hvx_solve_vtcm_params(
                HTP_MM_KERNEL_HVX_BF16_F32_VTCM, wtype, ne10, src1_nrows, sess->n_దారాలు,
                dst->nb[1], src0->nb[1], src1->nb[1], src2_అడ్డువరుస_కొలత, 16, vtcm_budget,
                &L, &m_chunk)) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_BF16_F32_VTCM;
            kparams->m_chunk = (m_chunk < (uint32_t) src1_nrows) ? m_chunk : 0;
            kparams->act_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * 4, 128);
            kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;
            kparams->vtcm_src0_కొలత = L.src0_బైట్లు;
            kparams->vtcm_act_కొలత  = L.act_బైట్లు;
            kparams->vtcm_bias_కొలత = L.bias_బైట్లు;
            kparams->vtcm_dst_కొలత = L.dst_బైట్లు;
            kparams->n_prefetch = 16;
            తిరిగి;
        }

        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    } లేకపోతే {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    }
}

స్థిర శూన్య ggml_hexagon_precompute_matmul_params_impl(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    స్థిరరాశి size_t src2_అడ్డువరుస_కొలత,
    స్థిరరాశి size_t src2_కొలత,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి పూర్ణాంకం ne00 = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం ne01 = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం ne02 = src0->ne[2];
    స్థిరరాశి పూర్ణాంకం ne03 = src0->ne[3];

    స్థిరరాశి పూర్ణాంకం ne10 = src1->ne[0];
    స్థిరరాశి పూర్ణాంకం ne11 = src1->ne[1];
    స్థిరరాశి పూర్ణాంకం ne12 = src1->ne[2];
    స్థిరరాశి పూర్ణాంకం ne13 = src1->ne[3];

    స్థిరరాశి తార్కిక is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    స్థిరరాశి పూర్ణాంకం wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    స్థిరరాశి పూర్ణాంకం ne00_padded = is_tiled ? hex_విడత_పైకి(ne00, htp_tensor_get_tile_k(wtype)) : ne00;
    స్థిరరాశి పూర్ణాంకం ne11_padded = hex_విడత_పైకి(ne11, 32);
    // VTCM has to hold whole 32-row weight tiles, so size for the rounded-up N
    // even when the tensor itself is ragged.
    స్థిరరాశి పూర్ణాంకం ne01_tiled  = hex_విడత_పైకి(ne01, 32);

    స్థిరరాశి తార్కిక is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    స్థిరరాశి తార్కిక is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    స్థిరరాశి size_t vtcm_budget = sess->vtcm_కొలత;

    // Check HMX eligibility and try precomputing HMX parameters
    తార్కిక hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ఒకవేళ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, is_matmul_id, is_batched)) {
        ఒకవేళ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, src1, dst, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched, src2_కొలత, vtcm_budget, kparams)) {
            వెళ్ళు finalize;
        }
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(sess, src0, src1, dst, wtype, is_tiled, ne02, ne03, ne10, ne11, ne12, ne13, is_matmul_id, src2_అడ్డువరుస_కొలత, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_విలువలు(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_విలువలు(ne11);
    kparams->div_r2       = init_fastdiv_విలువలు(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_విలువలు(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_విలువలు(ne12);
}

// The rows of dims 1..3 can be walked with one stride (size-1 dims skipped); returns that stride
స్థిర తార్కిక ggml_hexagon_అడ్డువరుసలు_stride(స్థిరరాశి int64_t * ne, స్థిరరాశి size_t * nb, size_t * stride) {
    size_t s = 0, తదుపరి = 0;
    కొరకు (పూర్ణాంకం i = 1; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (ne[i] == 1) కొనసాగు;
        ఒకవేళ (s == 0) { s = nb[i]; తదుపరి = s * ne[i]; కొనసాగు; }
        ఒకవేళ (nb[i] != తదుపరి) తిరిగి అబద్ధం;
        తదుపరి *= ne[i];
    }
    *stride = s ? s : nb[1];
    తిరిగి నిజం;
}

// A 2D weight applied to a batched activation whose rows are evenly strided is the same matmul over ne11 * ne12 * ne13 rows
స్థిర తార్కిక ggml_hexagon_matmul_can_collapse(స్థిరరాశి నిర్మాణం ggml_tensor * src0, స్థిరరాశి నిర్మాణం ggml_tensor * src1, స్థిరరాశి నిర్మాణం ggml_tensor * dst) {
    size_t s1, sd;
    తిరిగి (dst->op == GGML_OP_MUL_MAT || dst->op == GGML_OP_కూడు) &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] > 1 &&
           src1->nb[0] == ggml_type_కొలత(src1->type) && ggml_hexagon_అడ్డువరుసలు_stride(src1->ne, src1->nb, &s1) &&
           dst->nb[0] == ggml_type_కొలత(dst->type) && ggml_hexagon_అడ్డువరుసలు_stride(dst->ne, dst->nb, &sd);
}

స్థిర తార్కిక ggml_hexagon_matmul_కూడు_can_collapse(
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * src2,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst
) {
    ఒకవేళ (!ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!src2) {
        తిరిగి నిజం;
    }
    స్థిరరాశి int64_t src2_nrows = src2->ne[1] * src2->ne[2] * src2->ne[3];
    ఒకవేళ (src2_nrows == 1) {
        తిరిగి నిజం;
    }
    size_t s2;
    తిరిగి src2->nb[0] == ggml_type_కొలత(src2->type) &&
           src2->ne[0] == dst->ne[0] &&
           src2->ne[1] == dst->ne[1] &&
           src2->ne[2] == dst->ne[2] &&
           src2->ne[3] == dst->ne[3] &&
           ggml_hexagon_అడ్డువరుసలు_stride(src2->ne, src2->nb, &s2);
}

స్థిర ggml_tensor ggml_hexagon_tensor_collapse_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    size_t stride = 0;
    ggml_hexagon_అడ్డువరుసలు_stride(t->ne, t->nb, &stride);
    ggml_tensor c = *t;
    c.ne[1] = t->ne[1] * t->ne[2] * t->ne[3];
    c.ne[2] = 1;
    c.ne[3] = 1;
    c.nb[1] = stride;
    c.nb[2] = c.nb[1] * c.ne[1];
    c.nb[3] = c.nb[2];
    తిరిగి c;
}

స్థిర శూన్య ggml_hexagon_precompute_matmul_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    ఒకవేళ (ggml_hexagon_matmul_can_collapse(src0, src1, dst)) {
        స్థిరరాశి ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(src1);
        స్థిరరాశి ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(dst);
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, 0, 0, kparams);
        kparams->collapse = 1;
        తిరిగి;
    }
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, 0, 0, kparams);
}

స్థిర శూన్య ggml_hexagon_precompute_fused_matmul_కూడు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * src2,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    ఒకవేళ (ggml_hexagon_matmul_కూడు_can_collapse(src0, src1, src2, dst)) {
        స్థిరరాశి ggml_tensor src1_collapsed = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(src1);
        స్థిరరాశి ggml_tensor dst_collapsed  = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(dst);
        స్థిరరాశి ggml_tensor src2_collapsed = (src2 && (src2->ne[1] * src2->ne[2] * src2->ne[3] > 1))
                                           ? ggml_hexagon_tensor_collapse_అడ్డువరుసలు(src2)
                                           : (src2 ? *src2 : ggml_tensor{});
        స్థిరరాశి నిర్మాణం ggml_tensor * p_src2 = src2 ? &src2_collapsed : శూన్యసూచిక;
        స్థిరరాశి size_t src2_కొలత = p_src2 ? hex_విడత_పైకి(ggml_nbytes(p_src2), 128) : 0;
        ggml_hexagon_precompute_matmul_params_impl(sess, src0, &src1_collapsed, &dst_collapsed, p_src2 ? p_src2->nb[1] : 0, src2_కొలత, kparams);
        kparams->collapse = 1;
        తిరిగి;
    }
    స్థిరరాశి size_t src2_కొలత = src2 ? hex_విడత_పైకి(ggml_nbytes(src2), 128) : 0;
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, src1, dst, src2 ? src2->nb[1] : 0, src2_కొలత, kparams);
}

స్థిర తార్కిక ggml_hexagon_precompute_ద్విమాన_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    uint32_t op,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_ద్విమాన_kernel_params * kparams,
    స్థిరరాశి నిర్మాణం ggml_tensor * src2
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి size_t elem_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->ne[0] * elem_కొలత;
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత = src1->ne[0] * elem_కొలత;
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత  = dst->ne[0]  * elem_కొలత;

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);
    స్థిరరాశి size_t src1_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src1_అడ్డువరుస_కొలత, 128);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత,  128);

    ఒకవేళ (!htp_ద్విమాన_strides_సురక్షిత(src0, src1, dst, elem_కొలత)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక is_కూడు_id = op == HTP_OP_కూడు_ID;
    స్థిరరాశి తార్కిక is_scalar = !is_కూడు_id && src1->ne[0] == 1;
    స్థిరరాశి తార్కిక is_అడ్డువరుస_bcast = !is_కూడు_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        (src0->ne[1] > 1 || src0->ne[2] > 1 || src0->ne[3] > 1) &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
    స్థిరరాశి తార్కిక is_same_shape = !is_కూడు_id && !is_scalar &&
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
        (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1);
    స్థిరరాశి తార్కిక is_complex   = !is_కూడు_id && !is_scalar && !is_same_shape && !is_అడ్డువరుస_bcast && (src1->ne[0] == src0->ne[0]);
    స్థిరరాశి తార్కిక is_contig    = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst);
    స్థిరరాశి తార్కిక is_scalar_ప్రసారం = !is_కూడు_id && (ggml_nelements(src1) == 1);

    ఒకవేళ (!is_కూడు_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ప్రసారం)) {
        స్థిరరాశి uint32_t మొత్తం_elems = (uint32_t) ggml_nelements(src0);
        స్థిరరాశి uint32_t n_దారాలు = sess->n_దారాలు;
        స్థిరరాశి uint32_t max_chunk_elems = 32768 / elem_కొలత;
        స్థిరరాశి uint32_t min_chunk_elems = 256;
        స్థిరరాశి uint32_t లక్ష్యం_chunk_elems = hex_విడత_పైకి((మొత్తం_elems + (2 * n_దారాలు) - 1) / (2 * n_దారాలు), 32);
        స్థిరరాశి uint32_t chunk_కొలత = (std::min)(max_chunk_elems, (std::max)(లక్ష్యం_chunk_elems, min_chunk_elems));
        స్థిరరాశి uint32_t chunk_బైట్లు = hex_విడత_పైకి(chunk_కొలత * elem_కొలత, 128);

        kparams->kernel_type           = HTP_ద్విమాన_KERNEL_CHUNKED;
        kparams->n_దారాలు             = n_దారాలు;
        kparams->అడ్డువరుసలు_చొప్పున_బఫరు       = 1;
        kparams->src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
        kparams->src1_అడ్డువరుస_కొలత_aligned = is_scalar_ప్రసారం ? 0 : src1_అడ్డువరుస_కొలత_aligned;
        kparams->dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;
        kparams->src1_కొలత             = 0;
        kparams->chunk_కొలత            = chunk_కొలత;
        kparams->chunk_బైట్లు           = chunk_బైట్లు;
        kparams->is_scalar             = is_scalar_ప్రసారం ? 1 : 0;

        నిర్మాణం htp_ద్విమాన_vtcm_layout L;
        htp_ద్విమాన_vtcm_layout_build(&L, kparams, sess->vtcm_కొలత);
        ఒకవేళ (L.మొత్తం_బైట్లు == 0 || L.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }

        kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;
        తిరిగి నిజం;
    }

    గణన htp_ద్విమాన_kernel_type kernel_type;
    size_t src1_కొలత = 0;
    size_t src2_కొలత = 0;

    ఒకవేళ (is_కూడు_id) {
        GGML_ASSERT(src2 != శూన్యసూచిక);
        kernel_type = HTP_ద్విమాన_KERNEL_కూడు_ID;
        src1_కొలత = hex_విడత_పైకి(src1->ne[1] * src1_అడ్డువరుస_కొలత_aligned, 128);
        src2_కొలత = hex_విడత_పైకి(ggml_nelements(src2) * పరిమాణం(int32_t), 128);
        ఒకవేళ (src1_కొలత + src2_కొలత > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే ఒకవేళ (is_అడ్డువరుస_bcast) {
        kernel_type = HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST;
        src1_కొలత = src1_అడ్డువరుస_కొలత_aligned;
    } లేకపోతే ఒకవేళ (is_scalar) {
        src1_కొలత = hex_విడత_పైకి(ggml_nelements(src1) * elem_కొలత, 128);
        ఒకవేళ (src1_కొలత > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }
        kernel_type = HTP_ద్విమాన_KERNEL_SCALAR;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } లేకపోతే ఒకవేళ (is_same_shape) {
        kernel_type = HTP_ద్విమాన_KERNEL_SAME_SHAPE;
    } లేకపోతే ఒకవేళ (is_complex) {
        src1_కొలత = hex_విడత_పైకి(ggml_nelements(src1) * elem_కొలత + 128, 128);
        ఒకవేళ (src1_కొలత > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }
        kernel_type = HTP_ద్విమాన_KERNEL_COMPLEX;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    } లేకపోతే {
        src1_కొలత = hex_విడత_పైకి(ggml_nelements(src1) * elem_కొలత + 128, 128);
        ఒకవేళ (src1_కొలత > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }
        kernel_type = HTP_ద్విమాన_KERNEL_REPEAT;
        kparams->src1_strided = !ggml_is_contiguous(src1) ? 1 : 0;
    }

    kparams->kernel_type = kernel_type;
    kparams->n_దారాలు = sess->n_దారాలు;
    kparams->src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
    kparams->src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned;
    kparams->dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;
    kparams->src1_కొలత             = src1_కొలత;
    kparams->src2_కొలత             = src2_కొలత;

    స్థిరరాశి తార్కిక src0_contig_dim1 = (src0->nb[2] == src0->ne[1] * src0->nb[1]);
    స్థిరరాశి తార్కిక src1_contig_dim1 = (src1->nb[2] == src1->ne[1] * src1->nb[1]);
    స్థిరరాశి తార్కిక dst_contig_dim1  = (dst->nb[2]  == src0->ne[1] * dst->nb[1]);

    స్థిరరాశి తార్కిక src0_contig_dim2 = (src0->nb[3] == src0->ne[2] * src0->nb[2]);
    స్థిరరాశి తార్కిక src1_contig_dim2 = (src1->nb[3] == src1->ne[2] * src1->nb[2]);
    స్థిరరాశి తార్కిక dst_contig_dim2  = (dst->nb[3]  == src0->ne[2] * dst->nb[2]);

    తార్కిక విభజించు_at_ne01 = అబద్ధం;
    ఒకవేళ (src0->ne[2] > 1) {
        ఒకవేళ (kernel_type == HTP_ద్విమాన_KERNEL_SAME_SHAPE) {
            విభజించు_at_ne01 = (src1->ne[2] != src0->ne[2]) || !src0_contig_dim1 || !src1_contig_dim1 || !dst_contig_dim1;
        } లేకపోతే ఒకవేళ (kernel_type == HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST) {
            విభజించు_at_ne01 = !src0_contig_dim1 || !dst_contig_dim1;
        } లేకపోతే {
            విభజించు_at_ne01 = నిజం;
        }
    }
    kparams->విభజించు_at_ne01 = విభజించు_at_ne01 ? 1 : 0;

    తార్కిక విభజించు_at_ne02 = అబద్ధం;
    ఒకవేళ (src0->ne[3] > 1) {
        ఒకవేళ (kernel_type == HTP_ద్విమాన_KERNEL_SAME_SHAPE) {
            విభజించు_at_ne02 = (src1->ne[3] != src0->ne[3]) || !src0_contig_dim2 || !src1_contig_dim2 || !dst_contig_dim2;
        } లేకపోతే ఒకవేళ (kernel_type == HTP_ద్విమాన_KERNEL_అడ్డువరుస_BCAST) {
            విభజించు_at_ne02 = !src0_contig_dim2 || !dst_contig_dim2;
        } లేకపోతే {
            విభజించు_at_ne02 = నిజం;
        }
    }
    kparams->విభజించు_at_ne02 = విభజించు_at_ne02 ? 1 : 0;

    నిర్మాణం htp_ద్విమాన_vtcm_layout L;
    htp_ద్విమాన_vtcm_layout_build(&L, kparams, sess->vtcm_కొలత);
    ఒకవేళ (L.అడ్డువరుసలు_చొప్పున_బఫరు == 0 || L.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
        ఒకవేళ (!is_కూడు_id && is_contig && (ggml_are_same_shape(src0, src1) || is_scalar_ప్రసారం)) {
            స్థిరరాశి uint32_t మొత్తం_elems = (uint32_t) ggml_nelements(src0);
            స్థిరరాశి uint32_t n_దారాలు = sess->n_దారాలు;
            స్థిరరాశి uint32_t max_chunk_elems = 32768 / elem_కొలత;
            స్థిరరాశి uint32_t min_chunk_elems = 256;
            స్థిరరాశి uint32_t లక్ష్యం_chunk_elems = hex_విడత_పైకి((మొత్తం_elems + (2 * n_దారాలు) - 1) / (2 * n_దారాలు), 32);
            స్థిరరాశి uint32_t chunk_కొలత = (std::min)(max_chunk_elems, (std::max)(లక్ష్యం_chunk_elems, min_chunk_elems));
            స్థిరరాశి uint32_t chunk_బైట్లు = hex_విడత_పైకి(chunk_కొలత * elem_కొలత, 128);

            kparams->kernel_type           = HTP_ద్విమాన_KERNEL_CHUNKED;
            kparams->n_దారాలు             = n_దారాలు;
            kparams->అడ్డువరుసలు_చొప్పున_బఫరు       = 1;
            kparams->src1_అడ్డువరుస_కొలత_aligned = is_scalar_ప్రసారం ? 0 : src1_అడ్డువరుస_కొలత_aligned;
            kparams->src1_కొలత             = 0;
            kparams->chunk_కొలత            = chunk_కొలత;
            kparams->chunk_బైట్లు           = chunk_బైట్లు;
            kparams->is_scalar             = is_scalar_ప్రసారం ? 1 : 0;

            htp_ద్విమాన_vtcm_layout_build(&L, kparams, sess->vtcm_కొలత);
            ఒకవేళ (L.మొత్తం_బైట్లు == 0 || L.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
                తిరిగి అబద్ధం;
            }

            kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    }

    kparams->అడ్డువరుసలు_చొప్పున_బఫరు = L.అడ్డువరుసలు_చొప్పున_బఫరు;
    kparams->vtcm_కొలత       = L.మొత్తం_బైట్లు;

    తిరిగి నిజం;
}

స్థిర శూన్య ggml_hexagon_precompute_unary_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    uint32_t op,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t)sess->n_దారాలు, src0_nrows);

    kparams->n_దారాలు = n_దారాలు;

    స్థిరరాశి size_t elem_కొలత = ggml_type_కొలత(src0->type);

    స్థిరరాశి size_t src0_దత్తాంశం_అడ్డువరుస_కొలత = src0->ne[0] * elem_కొలత;
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత  = dst->ne[0]  * ggml_type_కొలత(dst->type);

    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_దత్తాంశం_అడ్డువరుస_కొలత, 128);
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_దత్తాంశం_అడ్డువరుస_కొలత,  128);

    kparams->src0_అడ్డువరుస_కొలత_aligned = src0_అడ్డువరుస_కొలత_aligned;
    kparams->dst_అడ్డువరుస_కొలత_aligned  = dst_అడ్డువరుస_కొలత_aligned;

    size_t src1_దత్తాంశం_అడ్డువరుస_కొలత = 0;
    size_t src1_అడ్డువరుస_కొలత_aligned = 0;
    తార్కిక ప్రసారం_బరువు = అబద్ధం;

    ఒకవేళ (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != శూన్యసూచిక);
        src1_దత్తాంశం_అడ్డువరుస_కొలత = src1->ne[0] * ggml_type_కొలత(src1->type);
        src1_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src1_దత్తాంశం_అడ్డువరుస_కొలత, 128);
        ప్రసారం_బరువు = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_అడ్డువరుస_కొలత_aligned = src1_అడ్డువరుస_కొలత_aligned;
    kparams->ప్రసారం_బరువు      = ప్రసారం_బరువు;

    నిర్మాణం htp_unary_vtcm_layout L;
    uint32_t నిలువు_tile = 0;
    uint32_t vtcm_అడ్డువరుస_చొప్పున_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, src0->ne[0], dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? src1->ne[0] : 0,
                                ప్రసారం_బరువు, n_దారాలు, sess->vtcm_కొలత, elem_కొలత,
                                &నిలువు_tile, &vtcm_అడ్డువరుస_చొప్పున_thread);

    kparams->నిలువు_tile = నిలువు_tile;
    kparams->vtcm_అడ్డువరుస_చొప్పున_thread = vtcm_అడ్డువరుస_చొప్పున_thread;
    kparams->vtcm_కొలత = L.మొత్తం_బైట్లు;

    kparams->vtcm_src0_కొలత_చొప్పున_thread = L.src0_బైట్లు;
    kparams->vtcm_src1_కొలత_చొప్పున_thread = L.src1_బైట్లు;
    kparams->vtcm_dst_కొలత_చొప్పున_thread  = L.dst_బైట్లు;

    kparams->vtcm_src0_కొలత = L.src0_బైట్లు * n_దారాలు;
    kparams->vtcm_src1_కొలత = L.src1_బైట్లు * n_దారాలు;
    kparams->vtcm_dst_కొలత  = L.dst_బైట్లు * n_దారాలు;

    kparams->ఖండం = నిలువు_tile ? 0 : ((L.src0_బైట్లు / 2) / src0_అడ్డువరుస_కొలత_aligned);

    స్థిరరాశి uint32_t tiles_చొప్పున_అడ్డువరుస = నిలువు_tile > 0 ? (src0->ne[0] + నిలువు_tile - 1) / నిలువు_tile : 1;
    kparams->div_ne01  = init_fastdiv_విలువలు(src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_విలువలు(src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_విలువలు(src0->ne[1] * src0->ne[2]);
    kparams->div_tpr   = init_fastdiv_విలువలు(tiles_చొప్పున_అడ్డువరుస);
}

స్థిర శూన్య ggml_hexagon_precompute_get_అడ్డువరుసలు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t ne02 = src0->ne[2];
    స్థిరరాశి uint32_t ne03 = src0->ne[3];

    స్థిరరాశి uint32_t ne10 = src1->ne[0];
    స్థిరరాశి uint32_t ne11 = src1->ne[1];
    స్థిరరాశి uint32_t ne12 = src1->ne[2];
    స్థిరరాశి uint32_t nr = ne10 * ne11 * ne12;

    స్థిరరాశి తార్కిక tiled = ggml_hexagon_tensor_is_tiled(src0);
    స్థిరరాశి పూర్ణాంకం wtype = (tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;

    ఒకవేళ (tiled) {
        kparams->kernel_type = HTP_GET_అడ్డువరుసలు_KERNEL_TILED;
    } లేకపోతే ఒకవేళ (src0->type == dst->type) {
        kparams->kernel_type = HTP_GET_అడ్డువరుసలు_KERNEL_SAMETYPE;
    } లేకపోతే {
        kparams->kernel_type = HTP_GET_అడ్డువరుసలు_KERNEL_FLAT;
    }

    స్థిరరాశి uint32_t మొత్తం_tasks = nr;

    kparams->n_దారాలు = (std::min)((uint32_t)sess->n_దారాలు, మొత్తం_tasks);

    నిర్మాణం htp_get_అడ్డువరుసలు_vtcm_layout vtcm_layout = {};
    అయితే (kparams->n_దారాలు > 0) {
        htp_get_అడ్డువరుసలు_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, మొత్తం_tasks, src1->type, kparams->n_దారాలు);
        ఒకవేళ (vtcm_layout.మొత్తం_బైట్లు <= sess->vtcm_కొలత) {
            ఆపు;
        }
        --kparams->n_దారాలు;
    }

    ఒకవేళ (kparams->n_దారాలు == 0 && మొత్తం_tasks > 0) {
        htp_get_అడ్డువరుసలు_vtcm_layout_build(&vtcm_layout, kparams->kernel_type, wtype, ne00, మొత్తం_tasks, src1->type, 1);
    }

    kparams->vtcm_కొలత = (మొత్తం_tasks == 0) ? 0 : vtcm_layout.మొత్తం_బైట్లు;
    kparams->tasks_చొప్పున_thread = kparams->n_దారాలు > 0 ? (మొత్తం_tasks + kparams->n_దారాలు - 1) / kparams->n_దారాలు : 0;
    kparams->మొత్తం_tasks = మొత్తం_tasks;

    kparams->div_ne10 = ne10 > 0 ? init_fastdiv_విలువలు(ne10) : fastdiv_విలువలు{0, 0};
    kparams->div_ne10_ne11 = (ne10 * ne11) > 0 ? init_fastdiv_విలువలు(ne10 * ne11) : fastdiv_విలువలు{0, 0};
    kparams->div_ne02 = ne02 > 0 ? init_fastdiv_విలువలు(ne02) : fastdiv_విలువలు{0, 0};
    kparams->div_ne03 = ne03 > 0 ? init_fastdiv_విలువలు(ne03) : fastdiv_విలువలు{0, 0};
}

స్థిర శూన్య ggml_hexagon_precompute_సమితి_అడ్డువరుసలు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0, //విలువలు
    స్థిరరాశి నిర్మాణం ggml_tensor * src1, // indices
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,  // destination
    నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి uint32_t nr = src0->ne[1];

    kparams->n_దారాలు = (std::min)((uint32_t)sess->n_దారాలు, nr);
    kparams->tasks_చొప్పున_thread = (nr + kparams->n_దారాలు - 1) / kparams->n_దారాలు;
    kparams->మొత్తం_tasks = nr;

    kparams->div_ne11 = init_fastdiv_విలువలు(src1->ne[1]);
    kparams->div_ne12 = init_fastdiv_విలువలు(src1->ne[2]);
    kparams->div_tasks_చొప్పున_thread = init_fastdiv_విలువలు(kparams->tasks_చొప్పున_thread);
    kparams->div_ne02 = init_fastdiv_విలువలు(src0->ne[2]);

    నిర్మాణం htp_సమితి_అడ్డువరుసలు_vtcm_layout vtcm_layout;
    htp_సమితి_అడ్డువరుసలు_vtcm_layout_build(&vtcm_layout, dst->type, src0->ne[0], kparams->n_దారాలు);
    kparams->vtcm_కొలత = vtcm_layout.మొత్తం_బైట్లు;
}

స్థిర శూన్య ggml_hexagon_precompute_softmax_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_softmax_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, src0_nrows);

    దశాంశం scale = 1.0f;
    దశాంశం max_bias = 0.0f;
    జ్ఞాపకం_నకలు(&scale,    &op->op_params[0], పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, &op->op_params[1], పరిమాణం(దశాంశం));

    kparams->scale    = scale;
    kparams->max_bias = max_bias;

    స్థిరరాశి uint32_t n_తల = src0->ne[2];
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));
    kparams->n_తల      = n_తల;
    kparams->n_తల_log2 = n_తల_log2;

    ఒకవేళ (max_bias > 0.0f && n_తల_log2 > 0) {
        kparams->m0 = powf(2.0f, -(max_bias) / n_తల_log2);
        kparams->m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);
    } లేకపోతే {
        kparams->m0 = 1.0f;
        kparams->m1 = 1.0f;
    }

    kparams->use_src1 = (src1 != శూన్యసూచిక) ? 1 : 0;
    kparams->use_f16  = (src1 != శూన్యసూచిక && src1->type == GGML_TYPE_F16) ? 1 : 0;

    స్థిరరాశి uint32_t ne00 = src0->ne[0];
    స్థిరరాశి uint32_t ne10 = src1 ? src1->ne[0] : 1;

    నిర్మాణం htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, ne00, ne10, kparams->use_src1 != 0, kparams->use_f16 != 0, n_దారాలు);

    kparams->n_దారాలు                 = n_దారాలు;
    kparams->src0_nrows                = src0_nrows;
    kparams->src0_nrows_చొప్పున_thread     = (src0_nrows + n_దారాలు - 1) / n_దారాలు;
    kparams->vtcm_కొలత                 = (uint32_t) layout.మొత్తం_బైట్లు;
    kparams->vtcm_src0_కొలత_చొప్పున_thread = (uint32_t) layout.src0_బైట్లు_చొప్పున_thread;
    kparams->vtcm_src1_కొలత_చొప్పున_thread = (uint32_t) layout.src1_బైట్లు_చొప్పున_thread;
    kparams->vtcm_dst_కొలత_చొప్పున_thread  = (uint32_t) layout.dst_బైట్లు_చొప్పున_thread;
    kparams->src0_అడ్డువరుస_కొలత_aligned     = (uint32_t) layout.src0_spad_half_కొలత;
    kparams->src1_అడ్డువరుస_కొలత_aligned     = (uint32_t) layout.src1_spad_half_కొలత;
    kparams->dst_అడ్డువరుస_కొలత_aligned      = (uint32_t) layout.dst_spad_half_కొలత;
    kparams->src0_spad_half_కొలత       = (uint32_t) layout.src0_spad_half_కొలత;
    kparams->src1_spad_half_కొలత       = (uint32_t) layout.src1_spad_half_కొలత;
    kparams->dst_spad_half_కొలత        = (uint32_t) layout.dst_spad_half_కొలత;
    ఒకవేళ (!kparams->use_src1) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_NOMASK;
    } లేకపోతే ఒకవేళ (kparams->use_f16) {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ముసుగు_F16;
    } లేకపోతే {
        kparams->kernel_id = HTP_SOFTMAX_KERNEL_ముసుగు_F32;
    }

    ఒకవేళ (src0->ne[1] > 0) kparams->div_ne01 = init_fastdiv_విలువలు(src0->ne[1]);
    ఒకవేళ (src0->ne[2] > 0) kparams->div_ne02 = init_fastdiv_విలువలు(src0->ne[2]);
    స్థిరరాశి uint32_t ne12 = src1 ? src1->ne[2] : 1;
    స్థిరరాశి uint32_t ne13 = src1 ? src1->ne[3] : 1;
    ఒకవేళ (ne12 > 0) kparams->div_ne12 = init_fastdiv_విలువలు(ne12);
    ఒకవేళ (ne13 > 0) kparams->div_ne13 = init_fastdiv_విలువలు(ne13);
}

స్థిర శూన్య ggml_hexagon_precompute_rope_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_rope_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, src0_nrows);
    స్థిరరాశి uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    నిర్మాణం htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_దారాలు, n_freq_factors);

    kparams->n_దారాలు              = n_దారాలు;
    kparams->src0_nrows             = src0_nrows;
    kparams->src0_nrows_చొప్పున_thread  = (src0_nrows + n_దారాలు - 1) / n_దారాలు;
    kparams->vtcm_కొలత              = (uint32_t) layout.మొత్తం_బైట్లు;
    kparams->spad_చొప్పున_thread        = (uint32_t) layout.బైట్లు_చొప్పున_thread;
    kparams->theta_cache_స్థానభ్రంశం     = (uint32_t) layout.theta_cache_కొలత_aligned;
    kparams->src0_అడ్డువరుస_కొలత_aligned  = (uint32_t) layout.src0_అడ్డువరుస_కొలత_aligned;
    kparams->freq_factors_స్థానభ్రంశం    = (uint32_t) (layout.బైట్లు_చొప్పున_thread * n_దారాలు);
    kparams->freq_factors_కొలత      = (uint32_t) layout.freq_factors_కొలత_aligned;

    ఒకవేళ (src0_nrows > 0) {
        kparams->div_ne2_ne1 = init_fastdiv_విలువలు(dst->ne[2] * dst->ne[1]);
        kparams->div_ne1     = init_fastdiv_విలువలు(dst->ne[1]);
    }
}

స్థిర శూన్య ggml_hexagon_precompute_ssm_conv_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * src1,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_ssm_conv_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి uint32_t d_conv  = (uint32_t) src1->ne[0];
    స్థిరరాశి uint32_t d_inner = (uint32_t) src0->ne[1];
    స్థిరరాశి uint32_t n_t     = (uint32_t) dst->ne[1];
    స్థిరరాశి uint32_t n_s     = (uint32_t) dst->ne[2];
    స్థిరరాశి uint32_t ncs     = (uint32_t) src0->ne[0];

    స్థిరరాశి uint32_t n_దారాలు = (std::min)((uint32_t) sess->n_దారాలు, (d_inner + 31) / 32);

    kparams->n_దారాలు = n_దారాలు;
    kparams->d_conv    = d_conv;
    kparams->d_inner   = d_inner;
    kparams->n_t       = n_t;
    kparams->n_s       = n_s;

    స్థిరరాశి uint32_t raw_rpt = (d_inner + n_దారాలు - 1) / n_దారాలు;
    స్థిరరాశి uint32_t d_inner_చొప్పున_thread = hex_విడత_పైకి(raw_rpt, 32);

    స్థిరరాశి uint32_t src1_raw_బైట్లు = hex_విడత_పైకి(d_inner_చొప్పున_thread * d_conv * పరిమాణం(దశాంశం), 128) + 128;
    స్థిరరాశి uint32_t src1_T_బైట్లు   = hex_విడత_పైకి(d_conv * d_inner_చొప్పున_thread * పరిమాణం(దశాంశం), 128);
    స్థిరరాశి uint32_t vtcm_src1_చొప్పున_thread = src1_raw_బైట్లు + src1_T_బైట్లు;

    uint32_t vtcm_src0_చొప్పున_thread = 0;
    uint32_t vtcm_dst_చొప్పున_thread  = 0;

    ఒకవేళ (n_t == 1) {
        kparams->d_inner_tile = d_inner_చొప్పున_thread;

        స్థిరరాశి uint32_t src0_tile_raw_బైట్లు = hex_విడత_పైకి(d_inner_చొప్పున_thread * d_conv * పరిమాణం(దశాంశం), 128);
        స్థిరరాశి uint32_t src0_T_బైట్లు        = hex_విడత_పైకి(d_conv * d_inner_చొప్పున_thread * పరిమాణం(దశాంశం), 128);
        vtcm_src0_చొప్పున_thread = 2 * src0_tile_raw_బైట్లు + src0_T_బైట్లు;

        స్థిరరాశి uint32_t dst_tile_బైట్లు = hex_విడత_పైకి(d_inner_చొప్పున_thread * పరిమాణం(దశాంశం), 128);
        vtcm_dst_చొప్పున_thread = 2 * dst_tile_బైట్లు;
    } లేకపోతే {
        స్థిరరాశి size_t vtcm_budget = (sess->vtcm_కొలత > 0 ? sess->vtcm_కొలత / n_దారాలు : (1024 * 1024));

        // the kernel double-buffers the raw src0 tile and the dst tile, and transposes
        // one 32-channel block at a time
        స్థిరరాశి uint32_t src0_ఖండం_T    = hex_విడత_పైకి(ncs * 32 * పరిమాణం(దశాంశం), 128);
        స్థిరరాశి size_t   fixed_బైట్లు     = vtcm_src1_చొప్పున_thread + src0_ఖండం_T;
        స్థిరరాశి size_t   avail_for_tiles = vtcm_budget > fixed_బైట్లు ? vtcm_budget - fixed_బైట్లు : (128 * 1024);

        uint32_t లక్ష్యం_max_tile = hex_విడత_పైకి((d_inner_చొప్పున_thread + 3) / 4, 32);
        లక్ష్యం_max_tile = (std::max)(లక్ష్యం_max_tile, 32u);
        లక్ష్యం_max_tile = (std::min)(లక్ష్యం_max_tile, 128u);

        uint32_t d_inner_tile = (uint32_t)(avail_for_tiles / (2 * (ncs + n_t) * పరిమాణం(దశాంశం)));
        d_inner_tile = (d_inner_tile / 32) * 32;
        ఒకవేళ (d_inner_tile == 0) {
            d_inner_tile = 32;
        }
        ఒకవేళ (d_inner_tile > లక్ష్యం_max_tile) {
            d_inner_tile = లక్ష్యం_max_tile;
        }
        ఒకవేళ (d_inner_tile > d_inner_చొప్పున_thread) {
            d_inner_tile = d_inner_చొప్పున_thread;
        }
        kparams->d_inner_tile = d_inner_tile;

        స్థిరరాశి uint32_t src0_tile_raw = hex_విడత_పైకి(d_inner_tile * ncs * పరిమాణం(దశాంశం), 128);
        vtcm_src0_చొప్పున_thread = 2 * src0_tile_raw + src0_ఖండం_T;

        vtcm_dst_చొప్పున_thread = 2 * hex_విడత_పైకి(d_inner_tile * n_t * పరిమాణం(దశాంశం), 128);
    }

    kparams->vtcm_src0_కొలత_చొప్పున_thread = vtcm_src0_చొప్పున_thread;
    kparams->vtcm_src1_కొలత_చొప్పున_thread = vtcm_src1_చొప్పున_thread;
    kparams->vtcm_dst_కొలత_చొప్పున_thread  = vtcm_dst_చొప్పున_thread;

    kparams->vtcm_src0_కొలత = vtcm_src0_చొప్పున_thread * n_దారాలు;
    kparams->vtcm_src1_కొలత = vtcm_src1_చొప్పున_thread * n_దారాలు;
    kparams->vtcm_dst_కొలత  = vtcm_dst_చొప్పున_thread  * n_దారాలు;
    kparams->vtcm_కొలత      = kparams->vtcm_src0_కొలత + kparams->vtcm_src1_కొలత + kparams->vtcm_dst_కొలత;
}

స్థిర శూన్య ggml_hexagon_precompute_gated_delta_net_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_gdn_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి నిర్మాణం ggml_tensor * q     = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * k     = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * v     = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * g     = op->src[3];
    స్థిరరాశి నిర్మాణం ggml_tensor * state = op->src[5];

    స్థిరరాశి uint32_t S_v      = (uint32_t) v->ne[0];
    స్థిరరాశి uint32_t H        = (uint32_t) v->ne[1];
    స్థిరరాశి uint32_t n_టోకెన్లు = (uint32_t) v->ne[2];
    స్థిరరాశి uint32_t n_seqs   = (uint32_t) v->ne[3];
    స్థిరరాశి uint32_t K        = (uint32_t) ggml_get_op_params_i32(op, 0);

    స్థిరరాశి uint32_t rq3 = (uint32_t) (n_seqs / q->ne[3]);
    స్థిరరాశి uint32_t rk3 = (uint32_t) (n_seqs / k->ne[3]);
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = H * n_seqs;
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, మొత్తం_అడ్డువరుసలు);

    స్థిరరాశి తార్కిక can_use_hmx = (opt_gdn_select >= 2) &&
                             (sess->n_hmx > 0) &&
                             (S_v % 64 == 0) &&
                             (n_టోకెన్లు >= HTP_GDN_MIN_టోకెన్లు) &&
                             (g->ne[0] == 1) &&
                             (K == 1);

    నిర్మాణం htp_gdn_hmx_vtcm_layout hmx_layout;
    నిర్మాణం htp_gdn_vtcm_layout hvx_layout;
    uint32_t n_heads_batch = 1;

    ఒకవేళ (can_use_hmx && htp_gdn_hmx_solve_layout(&hmx_layout, S_v, HTP_GDN_CHUNK_కొలత, మొత్తం_అడ్డువరుసలు, sess->vtcm_కొలత, n_దారాలు, నిజం, &n_heads_batch)) {
        kparams->kernel_type     = HTP_GDN_KERNEL_HMX_CHUNKED;
        kparams->pipeline        = hmx_layout.pipeline ? 1 : 0;
        kparams->chunk_కొలత      = HTP_GDN_CHUNK_కొలత;
        kparams->n_chunks        = (n_టోకెన్లు + HTP_GDN_CHUNK_కొలత - 1) / HTP_GDN_CHUNK_కొలత;
        kparams->n_heads_batch   = (uint16_t) n_heads_batch;
        kparams->vtcm_కొలత       = (uint32_t) hmx_layout.మొత్తం_బైట్లు;
        kparams->state_aligned   = (uint32_t) hmx_layout.state_f32_బైట్లు;
        kparams->vtcm_చొప్పున_thread = (uint32_t) (hmx_layout.మొత్తం_బైట్లు / (n_దారాలు > 0 ? n_దారాలు : 1));
    } లేకపోతే {
        htp_gdn_vtcm_layout_build(&hvx_layout, S_v, n_దారాలు);
        kparams->kernel_type     = HTP_GDN_KERNEL_HVX_RECURRENT;
        kparams->pipeline        = 0;
        kparams->n_heads_batch   = 1;
        kparams->state_aligned   = (uint32_t) hvx_layout.state_aligned;
        kparams->vtcm_చొప్పున_thread = (uint32_t) hvx_layout.బైట్లు_చొప్పున_thread;
        kparams->vtcm_కొలత       = (uint32_t) hvx_layout.మొత్తం_బైట్లు;
    }

    kparams->n_దారాలు           = n_దారాలు;
    kparams->S_v                 = S_v;
    kparams->H                   = H;
    kparams->n_టోకెన్లు            = n_టోకెన్లు;
    kparams->n_seqs              = n_seqs;
    kparams->K                   = K;
    kparams->మొత్తం_అడ్డువరుసలు          = మొత్తం_అడ్డువరుసలు;
    kparams->అడ్డువరుసలు_చొప్పున_thread     = (మొత్తం_అడ్డువరుసలు + kparams->n_దారాలు - 1) / kparams->n_దారాలు;
    kparams->kda                 = (g->ne[0] == S_v) ? 1 : 0;
    kparams->state_seq_stride    = (uint32_t) (state->nb[3] / పరిమాణం(దశాంశం));
    kparams->state_కొలత_చొప్పున_snap = S_v * S_v * H * n_seqs;
    kparams->scale               = 1.0f / sqrtf((దశాంశం) S_v);

    ఒకవేళ (H > 0)                  kparams->div_H         = init_fastdiv_విలువలు(H);
    ఒకవేళ (q->ne[1] > 0)           kparams->div_q1        = init_fastdiv_విలువలు((uint32_t) q->ne[1]);
    ఒకవేళ (k->ne[1] > 0)           kparams->div_k1        = init_fastdiv_విలువలు((uint32_t) k->ne[1]);
    ఒకవేళ (rq3 > 0)                kparams->div_rq3       = init_fastdiv_విలువలు(rq3);
    ఒకవేళ (rk3 > 0)                kparams->div_rk3       = init_fastdiv_విలువలు(rk3);
    ఒకవేళ (kparams->n_దారాలు > 0) kparams->div_n_దారాలు = init_fastdiv_విలువలు(kparams->n_దారాలు);
}

స్థిర శూన్య ggml_hexagon_precompute_క్రమీకరించు_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    తార్కిక is_శిఖరం_k,
    నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t ne00       = src0->ne[0];
    స్థిరరాశి uint32_t k          = dst->ne[0];

    int32_t క్రమం = GGML_క్రమీకరించు_క్రమం_DESC;
    ఒకవేళ (!is_శిఖరం_k) {
        క్రమం = ((స్థిరరాశి int32_t *) op->op_params)[0];
    }

    స్థిరరాశి uint32_t n_దారాలు_max = sess->n_దారాలు > 0 ? sess->n_దారాలు : 4;
    స్థిరరాశి size_t vtcm_budget = sess->vtcm_కొలత > 0 ? sess->vtcm_కొలత : (8 * 1024 * 1024);

    నిర్మాణం htp_క్రమీకరించు_vtcm_layout layout;
    తార్కిక ok = htp_క్రమీకరించు_solve_layout(&layout, ne00, మొత్తం_అడ్డువరుసలు, k, n_దారాలు_max, vtcm_budget, is_శిఖరం_k);
    GGML_ASSERT(ok);

    kparams->n_దారాలు         = (int32_t) layout.n_దారాలు;
    kparams->మొత్తం_అడ్డువరుసలు        = (int32_t) మొత్తం_అడ్డువరుసలు;
    kparams->అడ్డువరుస_ప్రారంభం         = 0;
    kparams->అడ్డువరుస_చివర           = (int32_t) మొత్తం_అడ్డువరుసలు;
    kparams->ne00              = (int32_t) ne00;
    kparams->k                 = (int32_t) k;
    kparams->క్రమం             = క్రమం;
    kparams->is_శిఖరం_k          = is_శిఖరం_k ? 1 : 0;
    kparams->use_dma           = 1;
    kparams->chunk_elems       = (int32_t) layout.chunk_elems;
    kparams->n_chunks          = (int32_t) layout.n_chunks;
    kparams->vtcm_కొలత         = (int32_t) layout.మొత్తం_బైట్లు;
    kparams->phase1_స్థానం_కొలత  = (int32_t) layout.phase1_స్థానం_కొలత;
    kparams->విలీనం_విలువలు_off  = (int32_t) layout.విలీనం_విలువలు_off;
    kparams->విలీనం_indices_off = (int32_t) layout.విలీనం_indices_off;
    kparams->విలీనం_elems       = (int32_t) layout.విలీనం_elems;
    kparams->n_slots           = (int32_t) layout.n_slots;
}

స్థిర శూన్య ggml_hexagon_pool_interior_range(
    uint32_t src_x, uint32_t dst_x, uint32_t kernel_x, uint32_t stride_x, int32_t pad_x,
    uint32_t * ox_lo, uint32_t * ox_hi) {
    స్థిరరాశి uint32_t lo_raw = ((uint32_t) pad_x + stride_x - 1) / stride_x;
    *ox_lo = (lo_raw < dst_x) ? lo_raw : dst_x;

    స్థిరరాశి int32_t numer_hi = (int32_t) src_x - (int32_t) kernel_x + pad_x;
    ఒకవేళ (numer_hi < 0) {
        *ox_hi = 0;
    } లేకపోతే {
        స్థిరరాశి uint32_t hi_raw = (uint32_t) numer_hi / stride_x + 1;
        *ox_hi = (hi_raw < dst_x) ? hi_raw : dst_x;
    }
    ఒకవేళ (*ox_hi < *ox_lo) {
        *ox_hi = *ox_lo;
    }
}

స్థిర తార్కిక ggml_hexagon_pool_shape_సరిపోతుంది(
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst
) {
    తిరిగి ggml_nbytes(src0) <= UINT32_MAX - 256 && ggml_nbytes(dst) <= UINT32_MAX - 256;
}

స్థిర శూన్య ggml_hexagon_precompute_pool_2d_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0,
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,
    నిర్మాణం htp_pool_2d_kernel_params * kparams,
    తార్కిక is_pool_1d
) {
    memset(kparams, 0, పరిమాణం(*kparams));

    స్థిరరాశి uint32_t src_x = (uint32_t) src0->ne[0];
    స్థిరరాశి uint32_t src_y = is_pool_1d ? 1 : (uint32_t) src0->ne[1];
    స్థిరరాశి uint32_t dst_x = (uint32_t) dst->ne[0];
    స్థిరరాశి uint32_t dst_y = is_pool_1d ? 1 : (uint32_t) dst->ne[1];
    స్థిరరాశి uint32_t src_plane_బైట్లు = src_x * src_y * పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t dst_plane_బైట్లు = dst_x * dst_y * పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t planes = (uint32_t) (is_pool_1d
        ? (uint64_t) src0->ne[1] * (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]
        : (uint64_t) src0->ne[2] * (uint64_t) src0->ne[3]);
    స్థిరరాశి uint32_t n_దారాలు = (std::min)((uint32_t) sess->n_దారాలు, planes > 0 ? planes : 1);

    నిర్మాణం htp_pool_vtcm_layout layout;
    స్థిరరాశి తార్కిక ok = htp_pool_solve_layout(&layout, src_x, src_y, dst_x, dst_y, n_దారాలు, sess->vtcm_కొలత);
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
    kparams->src_plane_బైట్లు = src_plane_బైట్లు;
    kparams->dst_plane_బైట్లు = dst_plane_బైట్లు;
    kparams->src_plane_బైట్లు_aligned = (uint32_t) layout.src_spad_half_కొలత;
    kparams->dst_plane_బైట్లు_aligned = (uint32_t) layout.dst_spad_half_కొలత;
    kparams->n_దారాలు = n_దారాలు;
    kparams->planes = planes;
    kparams->pool_op = (uint32_t) ggml_get_op_params_i32(dst, 0);
    // Fast HVX path requires exact tiling and a supported kernel width.
    kparams->fast_మార్గం = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->stride_x == kparams->kernel_x && kparams->stride_y == kparams->kernel_y &&
                           (kparams->kernel_x == 1 || kparams->kernel_x == 2)) ? 1 : 0;
    kparams->global_మార్గం = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                            kparams->stride_x == kparams->kernel_x &&
                            kparams->stride_y == kparams->kernel_y &&
                            kparams->kernel_x == kparams->src_x &&
                            kparams->kernel_y == kparams->src_y &&
                            kparams->dst_x == 1 && kparams->dst_y == 1) ? 1 : 0;
    kparams->ఖండం_మార్గం = (kparams->pad_x == 0 && kparams->pad_y == 0 &&
                           kparams->kernel_y == 1 && kparams->stride_y == 1 &&
                           kparams->stride_x == kparams->kernel_x &&
                           kparams->kernel_x >= 32) ? 1 : 0;
    kparams->avg_divide_లెక్క = (is_pool_1d && kparams->pool_op == GGML_OP_POOL_AVG) ? 1 : 0;
    ggml_hexagon_pool_interior_range(kparams->src_x, kparams->dst_x, kparams->kernel_x,
                                     kparams->stride_x, kparams->pad_x,
                                     &kparams->ox_lo, &kparams->ox_hi);

    స్థిరరాశి తార్కిక narrow_ok = (uint64_t) kparams->dst_x * kparams->kernel_x <= 32;
    kparams->narrow_మార్గం = (kparams->fast_మార్గం && narrow_ok) ? 1 : 0;
    kparams->inv_kernel_area = 1.0f / (దశాంశం) (kparams->kernel_x * kparams->kernel_y);
}

స్థిర తార్కిక ggml_hexagon_precompute_concat_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_concat_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->kernel_type = HTP_CONCAT_KERNEL_UNSUPPORTED;

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!src0 || !src1 || !dst) {
        తిరిగి అబద్ధం;
    }

    పూర్ణాంకం dim = ((స్థిరరాశి int32_t *) op->op_params)[0];
    ఒకవేళ (dim < 0 || dim >= GGML_MAX_DIMS) {
        తిరిగి అబద్ధం;
    }
    kparams->dim = dim;

    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->type != dst->type || src1->type != dst->type) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t type_కొలత = ggml_type_కొలత(dst->type);

    కొరకు (పూర్ణాంకం d = 0; d < GGML_MAX_DIMS; d++) {
        స్థిరరాశి int64_t ne_d = (d == dim) ? src0->ne[d] + src1->ne[d] : src0->ne[d];
        ఒకవేళ (dst->ne[d] != ne_d || (d != dim && src1->ne[d] != dst->ne[d])) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి uint32_t max_gather_ne0 = 128 / type_కొలత;
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = (uint32_t) (dst->ne[1] * dst->ne[2] * dst->ne[3]);
    స్థిరరాశి తార్కిక b_dense = (src1->nb[0] == type_కొలత);
    స్థిరరాశి తార్కిక gather_strides_ok =
        src0->nb[1] >= (size_t) src0->ne[0] * type_కొలత &&
        (b_dense ? src1->nb[1] >= (size_t) src1->ne[0] * type_కొలత
                 : src1->nb[0] >= (size_t) src1->ne[1] * type_కొలత);

    ఒకవేళ (dim == 0 &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) &&
        dst->ne[0] <= max_gather_ne0 &&
        dst->nb[0] == type_కొలత && dst->nb[1] == (size_t) dst->ne[0] * type_కొలత &&
        src0->nb[0] == type_కొలత &&
        (src1->nb[0] == type_కొలత || src1->nb[1] == type_కొలత) &&
        gather_strides_ok &&
        htp_gather_అడ్డువరుసలు_tensor_outer_contiguous(dst) &&
        htp_gather_అడ్డువరుసలు_tensor_outer_contiguous(src0) &&
        htp_gather_అడ్డువరుసలు_tensor_outer_contiguous(src1)) {

        ఒకవేళ (htp_gather_అడ్డువరుసలు_solve_layout(&kparams->u.gather,
                                         (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                         (uint32_t) src1->ne[0], (uint32_t) src1->nb[1], b_dense,
                                         type_కొలత, మొత్తం_అడ్డువరుసలు,
                                         sess->n_దారాలు, sess->mdev.count, sess->vtcm_కొలత)) {
            kparams->kernel_type = HTP_CONCAT_KERNEL_GATHER_అడ్డువరుసలు;
            kparams->n_దారాలు   = (uint8_t) sess->n_దారాలు;
            kparams->vtcm_కొలత   = kparams->u.gather.ముక్క_2 * sess->n_దారాలు;
            తిరిగి నిజం;
        }
    }

    స్థిరరాశి తార్కిక dma_strides_ok = (src0->nb[0] == type_కొలత && src1->nb[0] == type_కొలత && dst->nb[0] == type_కొలత);

    ఒకవేళ (dma_strides_ok) {
        kparams->kernel_type = HTP_CONCAT_KERNEL_REGULAR;
        kparams->n_దారాలు   = 1;
        తిరిగి నిజం;
    }

    స్థిరరాశి తార్కిక is_src1_transposed = (src1->nb[0] > src1->nb[1]);
    స్థిరరాశి తార్కిక is_src0_transposed = (src0->nb[0] > src0->nb[1]);
    స్థిరరాశి తార్కిక transposed_అడ్డువరుసలు_ok = (src0->nb[0] == type_కొలత && src1->nb[1] == type_కొలత && dst->nb[0] == type_కొలత);

    ఒకవేళ (dim == 0 && is_src1_transposed && !is_src0_transposed && transposed_అడ్డువరుసలు_ok &&
        (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16)) {

        స్థిరరాశి uint32_t n_దారాలు = sess->n_దారాలు > 0 ? (uint32_t) sess->n_దారాలు : 8;
        నిర్మాణం htp_concat_transposed_vtcm_layout layout;
        htp_concat_transposed_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src1->ne[0], type_కొలత, n_దారాలు);

        ఒకవేళ (sess->vtcm_కొలత > 0 && layout.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
            తిరిగి అబద్ధం;
        }

        kparams->kernel_type                       = HTP_CONCAT_KERNEL_TRANSPOSED;
        kparams->n_దారాలు                         = n_దారాలు;
        kparams->vtcm_కొలత                         = layout.మొత్తం_బైట్లు;
        kparams->u.transposed.spad0_కొలత_చొప్పున_thread = layout.src0_spad_కొలత_చొప్పున_thread;
        kparams->u.transposed.spad1_కొలత_చొప్పున_thread = layout.src1_spad_కొలత_చొప్పున_thread;
        తిరిగి నిజం;
    }

    తిరిగి అబద్ధం;
}

స్థిర తార్కిక ggml_hexagon_precompute_cpy_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_copy_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->kernel_type = HTP_COPY_KERNEL_UNSUPPORTED;

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!src0 || !dst) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t nelem_src = ggml_nelements(src0);
    స్థిరరాశి int64_t nelem_dst = ggml_nelements(dst);
    ఒకవేళ (nelem_src != nelem_dst || nelem_src < 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t src_type_కొలత = ggml_type_కొలత(src0->type);
    స్థిరరాశి uint32_t dst_type_కొలత = ggml_type_కొలత(dst->type);

    kparams->src0_type_కొలత = (uint8_t) src_type_కొలత;
    kparams->dst_type_కొలత  = (uint8_t) dst_type_కొలత;
    kparams->మొత్తం_elems    = (uint32_t) nelem_src;

    ఒకవేళ (nelem_src == 0) {
        kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
        తిరిగి నిజం;
    }

    ఒకవేళ (nelem_src == 1) {
        ఒకవేళ (src0->type == dst->type) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            తిరిగి నిజం;
        }
        ఒకవేళ ((src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
            (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) ||
            (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
            (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32)) {
            kparams->kernel_type = HTP_COPY_KERNEL_SCALAR;
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక sametype = (src0->type == dst->type);

    తార్కిక same_extents = నిజం;
    కొరకు (పూర్ణాంకం d = 0; d < GGML_MAX_DIMS; d++) {
        ఒకవేళ (src0->ne[d] != dst->ne[d]) {
            same_extents = అబద్ధం;
            ఆపు;
        }
    }

    స్థిరరాశి తార్కిక transposed = (src0->nb[0] > src0->nb[1])    || (dst->nb[0] > dst->nb[1]) ||
                            (src0->nb[0] != src_type_కొలత) || (dst->nb[0] != dst_type_కొలత) ||
                            (src0->nb[1] < (size_t) src0->ne[0] * src_type_కొలత) || (dst->nb[1] < (size_t) dst->ne[0] * dst_type_కొలత);
    స్థిరరాశి తార్కిక sameshape  = same_extents && !transposed;

    స్థిరరాశి తార్కిక src_is_contiguous = ggml_is_contiguous(src0);
    స్థిరరాశి తార్కిక dst_is_contiguous = ggml_is_contiguous(dst);

    ఒకవేళ (sametype) {
        ఒకవేళ (src_is_contiguous && dst_is_contiguous) {
            kparams->kernel_type = HTP_COPY_KERNEL_1D_CONTIG;
            తిరిగి నిజం;
        }

        స్థిరరాశి uint32_t max_gather_ne0 = 128 / src_type_కొలత;
        స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
        ఒకవేళ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) &&
            dst_is_contiguous && !src_is_contiguous &&
            src0->nb[0] == src_type_కొలత && src0->ne[0] <= max_gather_ne0 &&
            src0->nb[1] >= (size_t) src0->ne[0] * src_type_కొలత &&
            మొత్తం_అడ్డువరుసలు > 1 &&
            htp_gather_అడ్డువరుసలు_tensor_outer_contiguous(src0)) {

            ఒకవేళ (htp_gather_అడ్డువరుసలు_solve_layout(&kparams->u.gather,
                                             (uint32_t) src0->ne[0], (uint32_t) src0->nb[1],
                                             0, 0, అబద్ధం,
                                             src_type_కొలత, మొత్తం_అడ్డువరుసలు,
                                             sess->n_దారాలు, sess->mdev.count, sess->vtcm_కొలత)) {
                kparams->kernel_type = HTP_COPY_KERNEL_GATHER_అడ్డువరుసలు;
                kparams->n_దారాలు   = (uint8_t) sess->n_దారాలు;
                kparams->vtcm_కొలత   = kparams->u.gather.ముక్క_2 * sess->n_దారాలు;
                తిరిగి నిజం;
            }
        }

        ఒకవేళ (sameshape) {
            kparams->kernel_type = HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE;
            kparams->మొత్తం_అడ్డువరుసలు  = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
            తిరిగి నిజం;
        }

        kparams->kernel_type = HTP_COPY_KERNEL_RESHAPE;
        kparams->n_దారాలు   = sess->n_దారాలు > 0 ? (uint8_t) sess->n_దారాలు : 4;
        kparams->u.reshape.div_ne0            = init_fastdiv_విలువలు((uint32_t) dst->ne[0]);
        kparams->u.reshape.div_ne1_ne0        = init_fastdiv_విలువలు((uint32_t) (dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne2_ne1_ne0    = init_fastdiv_విలువలు((uint32_t) (dst->ne[2] * dst->ne[1] * dst->ne[0]));
        kparams->u.reshape.div_ne00           = init_fastdiv_విలువలు((uint32_t) src0->ne[0]);
        kparams->u.reshape.div_ne01_ne00      = init_fastdiv_విలువలు((uint32_t) (src0->ne[1] * src0->ne[0]));
        kparams->u.reshape.div_ne02_ne01_ne00 = init_fastdiv_విలువలు((uint32_t) (src0->ne[2] * src0->ne[1] * src0->ne[0]));
        తిరిగి నిజం;
    }

    ఒకవేళ (!sameshape) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక valid_conversion = (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) ||
                                  (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) ||
                                  (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) ||
                                  (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    ఒకవేళ (!valid_conversion) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t n_దారాలు = sess->n_దారాలు > 0 ? (uint32_t) sess->n_దారాలు : 4;
    నిర్మాణం htp_copy_convert_vtcm_layout layout;
    htp_copy_convert_vtcm_layout_build(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1], src_type_కొలత, dst_type_కొలత, n_దారాలు);

    ఒకవేళ (sess->vtcm_కొలత > 0 && layout.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
        తిరిగి అబద్ధం;
    }

    kparams->kernel_type             = HTP_COPY_KERNEL_SAMESHAPE_CONVERT;
    kparams->మొత్తం_అడ్డువరుసలు              = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    kparams->n_దారాలు               = (uint8_t) n_దారాలు;
    kparams->vtcm_కొలత               = layout.మొత్తం_బైట్లు;
    kparams->u.convert.src0_buf_కొలత = layout.src0_buf_కొలత;
    kparams->u.convert.dst_buf_కొలత  = layout.dst_buf_కొలత;
    kparams->u.convert.spad0_కొలత_చొప్పున_thread = layout.spad0_కొలత_చొప్పున_thread;
    kparams->u.convert.spad1_కొలత_చొప్పున_thread = layout.spad1_కొలత_చొప్పున_thread;
    kparams->u.convert.src0_అడ్డువరుస_stride = layout.src0_అడ్డువరుస_stride;
    kparams->u.convert.dst_అడ్డువరుస_stride  = layout.dst_అడ్డువరుస_stride;
    kparams->u.convert.blk_అడ్డువరుసలు        = layout.blk_అడ్డువరుసలు;
    kparams->u.convert.div_ne01      = init_fastdiv_విలువలు((uint32_t) src0->ne[1]);
    kparams->u.convert.div_ne02_ne01 = init_fastdiv_విలువలు((uint32_t) (src0->ne[2] * src0->ne[1]));

    తిరిగి నిజం;
}

స్థిర శూన్య ggml_hexagon_precompute_fused_mmnx_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0, // W0
    స్థిరరాశి నిర్మాణం ggml_tensor * act,  // x
    int32_t n_weights,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->n_దారాలు = sess->n_దారాలు;

    స్థిరరాశి పూర్ణాంకం ne00 = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం ne01 = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం ne02 = src0->ne[2];
    స్థిరరాశి పూర్ణాంకం ne03 = src0->ne[3];

    స్థిరరాశి పూర్ణాంకం ne10 = act->ne[0];
    స్థిరరాశి పూర్ణాంకం ne11 = act->ne[1];
    స్థిరరాశి పూర్ణాంకం ne12 = act->ne[2];
    స్థిరరాశి పూర్ణాంకం ne13 = act->ne[3];

    స్థిరరాశి తార్కిక is_tiled = ggml_hexagon_tensor_is_tiled(src0);
    స్థిరరాశి పూర్ణాంకం wtype = (is_tiled && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16)) ? GGML_TYPE_F16 : src0->type;
    స్థిరరాశి పూర్ణాంకం ne00_padded = is_tiled ? hex_విడత_పైకి(ne00, 32) : ne00;
    స్థిరరాశి పూర్ణాంకం ne11_padded = hex_విడత_పైకి(ne11, 32);
    స్థిరరాశి పూర్ణాంకం ne01_tiled  = hex_విడత_పైకి(ne01, 32);

    స్థిరరాశి size_t vtcm_budget = sess->vtcm_కొలత;
    స్థిరరాశి తార్కిక is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    తార్కిక hmx_enabled = (sess->n_hmx > 0) && (opt_mm_select >= 2);
    ఒకవేళ (hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(src0, act, శూన్యసూచిక, అబద్ధం, is_batched)) {
        ఒకవేళ (ggml_hexagon_precompute_hmx_mm_params(sess, src0, act, శూన్యసూచిక, wtype, ne00_padded, ne01_tiled, ne02, ne11, ne12, ne11_padded, అబద్ధం, is_batched, 0, vtcm_budget, kparams)) {
            kparams->n_weights = n_weights;
            వెళ్ళు finalize;
        }
    }

    ఒకవేళ (!is_tiled || ne10 % 32 != 0) {
        kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
        తిరిగి;
    }

    {
        స్థిరరాశి పూర్ణాంకం act_nrows = ne11 * ne12 * ne13;
        స్థిరరాశి size_t act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
        స్థిరరాశి size_t src0_అడ్డువరుస_కొలత = src0->nb[1];

        స్థిరరాశి uint32_t max_prefetch = (act_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        uint32_t ఉత్తమ_n_prefetch = 2;
        కొరకు (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            నిర్మాణం htp_mm_hvx_vtcm_layout L;
            htp_mm_hvx_vtcm_layout_build(
                &L, HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస, wtype, ne10, act_nrows, sess->n_దారాలు,
                0, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, 0, d, అబద్ధం, నిజం
            );
            ఒకవేళ (L.మొత్తం_బైట్లు <= sess->vtcm_కొలత) {
                ఉత్తమ_n_prefetch = d;
                ఆపు;
            }
        }

        నిర్మాణం htp_mm_hvx_vtcm_layout L;
        తార్కిక try_tiled = (opt_mm_select >= 1);

        // Test tiled first
        htp_mm_hvx_vtcm_layout_build(
            &L, HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస, wtype, ne10, act_nrows, sess->n_దారాలు,
            0, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, 0, ఉత్తమ_n_prefetch, అబద్ధం, నిజం
        );

        ఒకవేళ (try_tiled && L.మొత్తం_బైట్లు <= sess->vtcm_కొలత) {
            kparams->kernel_type       = HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస;
            kparams->tile_కొలత         = htp_tensor_get_tile_కొలత(wtype);
            kparams->aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(wtype);
            kparams->act_అడ్డువరుస_కొలత      = act_అడ్డువరుస_కొలత;
            kparams->vtcm_src0_కొలత    = L.src0_బైట్లు;
            kparams->vtcm_act_కొలత     = L.act_బైట్లు;
            kparams->vtcm_bias_కొలత    = 0;
            kparams->vtcm_dst_కొలత     = L.dst_బైట్లు;
            kparams->vtcm_కొలత         = L.మొత్తం_బైట్లు;
            kparams->n_prefetch        = ఉత్తమ_n_prefetch;
            kparams->n_weights         = n_weights;
        } లేకపోతే {
            kparams->kernel_type = HTP_MM_KERNEL_UNSUPPORTED;
            తిరిగి;
        }
    }

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_విలువలు(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_విలువలు(ne11);
    kparams->div_r2       = init_fastdiv_విలువలు(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_విలువలు(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne12     = init_fastdiv_విలువలు(ne12);
}

స్థిర శూన్య ggml_hexagon_precompute_fused_mmidnx_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * src0, // W0
    స్థిరరాశి నిర్మాణం ggml_tensor * act,  // x
    స్థిరరాశి నిర్మాణం ggml_tensor * dst,  // dst0
    int32_t n_weights,
    నిర్మాణం htp_mm_kernel_params * kparams
) {
    ggml_hexagon_precompute_matmul_params_impl(sess, src0, act, dst, 0, 0, kparams);
    kparams->n_weights = n_weights;
}

స్థిర తార్కిక ggml_hexagon_op_mul_mat_use_fwht(స్థిరరాశి నిర్మాణం ggml_tensor * dst) {
    ఒకవేళ (dst->op != GGML_OP_MUL_MAT || ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    స్థిరరాశి int64_t n = src1->ne[0];
    ఒకవేళ (n < 32 || (n & (n - 1)) != 0) {
        తిరిగి అబద్ధం;
    }

    తిరిగి src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_is_contiguous_అడ్డువరుసలు(src1) && ggml_is_contiguous(dst) &&
           ggml_are_same_shape(src1, dst);
}

స్థిర తార్కిక ggml_hexagon_supported_mul_mat(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * dst) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];

    ఒకవేళ (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        ఒకవేళ (!ggml_hexagon_op_mul_mat_use_fwht(dst)) {
            తిరిగి అబద్ధం;
        }
        నిర్మాణం htp_unary_kernel_params ukparams;
        ggml_hexagon_precompute_unary_params(sess, HTP_OP_HADAMARD, src1, శూన్యసూచిక, dst, &ukparams);
        తిరిగి ukparams.vtcm_అడ్డువరుస_చొప్పున_thread > 0 && (size_t) ukparams.vtcm_కొలత <= sess->vtcm_కొలత;
    }

    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q2_K: {
            ఒకవేళ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి int64_t qk = ggml_blck_కొలత(src0->type);
            ఒకవేళ (src0->ne[0] % qk) {
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;
        }

        స్థితి GGML_TYPE_F16:
            ఒకవేళ (src0->nb[1] < src0->nb[0]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;

        స్థితి GGML_TYPE_F32:
            ఒకవేళ (src1->type != GGML_TYPE_F32) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src0->nb[1] < src0->nb[0]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;

        స్థితి GGML_TYPE_BF16:
            ఒకవేళ (src0->nb[1] < src0->nb[0]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;

        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ఒకవేళ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_mul_mat_id(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
        తిరిగి అబద్ధం;
    }

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q2_K:
            ఒకవేళ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0) || ggml_is_view(src0)) {
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (src0->ne[0] % ggml_blck_కొలత(src0->type) != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;

        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_BF16:
            ఒకవేళ (!ggml_is_contiguous(src0) || ggml_is_permuted(src0)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (src0->ne[0] % 32 != 0) {
                తిరిగి అబద్ధం;
            }
            ఆపు;

        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_mm_kernel_params kparams;
    ggml_hexagon_precompute_matmul_params(sess, src0, src1, dst, &kparams);
    ఒకవేళ (kparams.kernel_type == HTP_MM_KERNEL_UNSUPPORTED || (size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
        HEX_VERBOSE("ggml-hex: %s supported MUL_MAT_ID VTCM size needed (%d) > budget (%zu)\n", sess->c_name(), kparams.vtcm_కొలత, sess->vtcm_కొలత);
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_ద్విమాన(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type == GGML_TYPE_F32) {
        ఒకవేళ (src1->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (dst->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
    }
    లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
        ఒకవేళ (src1->type != GGML_TYPE_F16) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (dst->type != GGML_TYPE_F16) {
            తిరిగి అబద్ధం;
        }
    }
    లేకపోతే {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ggml_is_permuted(src0) || ggml_is_permuted(dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_are_same_shape(src0, dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1)) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_ద్విమాన_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_ద్విమాన_params(sess, op_remap_గమ్యం_htp(op), src0, src1, dst, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_కూడు_id(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!src2) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_are_same_shape(src0, dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src2->ne[0] != src0->ne[1] || src2->ne[1] != src0->ne[2]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->nb[0] != పరిమాణం(దశాంశం) || src1->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం) || src2->nb[0] != పరిమాణం(int32_t)) {
        తిరిగి అబద్ధం;
    }

    // REVISIT: add support for non-contiguous tensors
    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src2->nb[1] < src2->ne[0] * పరిమాణం(int32_t)) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_ద్విమాన_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_ద్విమాన_params(sess, HTP_OP_కూడు_ID, src0, src1, dst, &kparams, src2);
}

స్థిర తార్కిక ggml_hexagon_supported_unary(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != src0->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(src0)) {
        తిరిగి అబద్ధం;
    }

    // F16 device kernels only cover this explicit whitelist (must stay in sync with
    // the is_f16 whitelist in execute_op_unary(), unary-ops.c).
    ఒకవేళ (src0->type == GGML_TYPE_F16) {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_NORM:
            స్థితి GGML_OP_RMS_NORM:
            స్థితి GGML_OP_L2_NORM:
            స్థితి GGML_OP_SCALE:
            స్థితి GGML_OP_CLAMP:
            స్థితి GGML_OP_SQR:
            స్థితి GGML_OP_SQRT:
            స్థితి GGML_OP_LOG:
                ఆపు;
            స్థితి GGML_OP_UNARY:
                ఒకవేళ (ggml_get_unary_op(op) != GGML_UNARY_OP_ABS &&
                    ggml_get_unary_op(op) != GGML_UNARY_OP_అడుగు) {
                    తిరిగి అబద్ధం;
                }
                ఆపు;
            అప్రమేయ:
                తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (!ggml_are_same_shape(src0, dst)) {
        తిరిగి అబద్ధం;
    }

    // dst must be contiguous; src0 may be non-contiguous
    ఒకవేళ (!ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_కూడిక(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_కూడిక_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    // TODO: add support for non-contigiuos tensors
    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_argmax(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_pool_2d(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి int32_t * params = op->op_params;

    ఒకవేళ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_hexagon_pool_shape_సరిపోతుంది(src0, op)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t kernel_x = params[1];
    స్థిరరాశి int32_t kernel_y = params[2];
    స్థిరరాశి int32_t stride_x = params[3];
    స్థిరరాశి int32_t stride_y = params[4];
    స్థిరరాశి int32_t pad_x    = params[5];
    స్థిరరాశి int32_t pad_y    = params[6];

    // Keep invalid parameters on CPU, avoiding unsafe unsigned values in HTP.
    ఒకవేళ (kernel_x <= 0 || kernel_y <= 0 || stride_x <= 0 || stride_y <= 0 ||
        pad_x < 0 || pad_y < 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t planes = (uint32_t) (src0->ne[2] * src0->ne[3]);
    స్థిరరాశి uint32_t n_దారాలు = (std::min)((uint32_t) sess->n_దారాలు, planes > 0 ? planes : 1);

    నిర్మాణం htp_pool_vtcm_layout layout;
    తిరిగి htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], (uint32_t) src0->ne[1],
                                 (uint32_t) op->ne[0], (uint32_t) op->ne[1],
                                 n_దారాలు, sess->vtcm_కొలత);
}

స్థిర తార్కిక ggml_hexagon_supported_pool_1d(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి int32_t * params = op->op_params;

    ఒకవేళ (params[0] != GGML_OP_POOL_AVG && params[0] != GGML_OP_POOL_MAX) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_hexagon_pool_shape_సరిపోతుంది(src0, op)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t kernel = params[1];
    స్థిరరాశి int32_t stride = params[2];
    స్థిరరాశి int32_t pad    = params[3];
    ఒకవేళ (kernel <= 0 || stride <= 0 || pad < 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t planes = (uint32_t) (src0->ne[1] * src0->ne[2] * src0->ne[3]);
    స్థిరరాశి uint32_t n_దారాలు = (std::min)((uint32_t) sess->n_దారాలు, planes > 0 ? planes : 1);

    నిర్మాణం htp_pool_vtcm_layout layout;
    తిరిగి htp_pool_solve_layout(&layout, (uint32_t) src0->ne[0], 1, (uint32_t) op->ne[0], 1, n_దారాలు, sess->vtcm_కొలత);
}

స్థిర తార్కిక ggml_hexagon_supported_activations(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != src0->type) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous_1(src0)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1) {
        ఒకవేళ (src1->type != src0->type) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_are_same_shape(src0, src1)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_is_contiguous_1(src1)) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_softmax(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src2) {
        తిరిగి అబద్ధం;  // FIXME: add support for sinks
    }

    ఒకవేళ (src0->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->ne[2] > 512) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1) {
        ఒకవేళ (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src0->ne[0] != src1->ne[0]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src1->ne[1] < src0->ne[1]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src0->ne[2] % src1->ne[2] != 0) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src0->ne[3] % src1->ne[3] != 0) {
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (src1) {
        ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే {
        ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
            తిరిగి అబద్ధం;
        }
    }

    // Reject non-HVX-aligned sizes when ne[0] > HVX_F32_LANES
    // The HVX softmax implementation has issues with tail handling for larger non-aligned sizes
    // Small sizes (ne[0] <= 32) work correctly with tail-only processing
    స్థిరరాశి int64_t ne0 = src0->ne[0];
    ఒకవేళ (ne0 > 32 && (ne0 & (32 - 1)) != 0) {
        తిరిగి అబద్ధం;
    }

    // HVX vector size constraints for softmax
    #నిర్వచించు SOFTMAX_MAX_అడ్డువరుస_కొలత 131072  // 128K elements max for numerical precision

    // Reject very large row sizes to avoid numerical precision issues
    // Softmax accumulation over many elements can lead to precision loss
    ఒకవేళ (ne0 > SOFTMAX_MAX_అడ్డువరుస_కొలత) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు  = (std::min)((uint32_t) sess->n_దారాలు, src0_nrows);
    నిర్మాణం htp_softmax_vtcm_layout layout;
    htp_softmax_vtcm_layout_build(&layout, src0->ne[0], src1 ? src1->ne[0] : 1, src1 != శూన్యసూచిక, src1 && src1->type == GGML_TYPE_F16, n_దారాలు);
    ఒకవేళ (layout.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_సమితి_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0]; //విలువలు
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1]; // indices
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op->src[2] ? op->src[2] : op;

    ఒకవేళ (dst->type == GGML_TYPE_Q8_0 && src0->ne[0] < 32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_Q8_0) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_get_అడ్డువరుసలు(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0]; //విలువలు
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1]; // indices
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32  && src0->type != GGML_TYPE_F16 && src0->type  != GGML_TYPE_BF16 && src0->type != GGML_TYPE_I32  &&
        src0->type != GGML_TYPE_Q1_0 && src0->type != GGML_TYPE_Q2_0 && src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K ) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(src0) || !ggml_is_contiguous_అడ్డువరుసలు(src1) || !ggml_is_contiguous_అడ్డువరుసలు(dst)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->type == dst->type) {
        ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F16) {
            తిరిగి అబద్ధం;
        }
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    } లేకపోతే ఒకవేళ (dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    // Empty recurrent-state gathers are skipped at execution; do not split the graph for them.
    ఒకవేళ (ggml_is_ఖాళీలు(op)) {
        తిరిగి నిజం;
    }

    నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params kparams;
    ggml_hexagon_precompute_get_అడ్డువరుసలు_params(sess, src0, src1, dst, &kparams);
    ఒకవేళ (kparams.n_దారాలు == 0 || (size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_argsort(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0]; //విలువలు
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;         // indices

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు_max = sess->n_దారాలు > 0 ? sess->n_దారాలు : 4;
    స్థిరరాశి size_t vtcm_budget = sess->vtcm_కొలత > 0 ? sess->vtcm_కొలత : (8 * 1024 * 1024);

    నిర్మాణం htp_క్రమీకరించు_vtcm_layout layout;
    ఒకవేళ (!htp_క్రమీకరించు_solve_layout(&layout, src0->ne[0], మొత్తం_అడ్డువరుసలు, dst->ne[0], n_దారాలు_max, vtcm_budget, అబద్ధం)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_శిఖరం_k(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0]; //విలువలు
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;         // indices

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి uint32_t n_దారాలు_max = sess->n_దారాలు > 0 ? sess->n_దారాలు : 4;
    స్థిరరాశి size_t vtcm_budget = sess->vtcm_కొలత > 0 ? sess->vtcm_కొలత : (8 * 1024 * 1024);

    నిర్మాణం htp_క్రమీకరించు_vtcm_layout layout;
    ఒకవేళ (!htp_క్రమీకరించు_solve_layout(&layout, src0->ne[0], మొత్తం_అడ్డువరుసలు, dst->ne[0], n_దారాలు_max, vtcm_budget, నిజం)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_rope(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * src2 = op->src[2];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!ggml_are_same_shape(src0, dst)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->ne[0] <= 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ఒకవేళ (src0_nrows == 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t * op_params = &op->op_params[0];
    స్థిరరాశి పూర్ణాంకం n_dims = op_params[1];
    స్థిరరాశి పూర్ణాంకం mode   = op_params[2];
    స్థిరరాశి పూర్ణాంకం n_offs = op_params[15];

    // llama probes weight placement with a dummy rope where every param is 0 (llama-model-loader.cpp).
    // Rejecting it puts rope_freqs on the CPU, which then splits the graph at every full-attention layer.
    ఒకవేళ (n_dims < 0 || n_dims % 2 != 0) {
        తిరిగి అబద్ధం;
    }

    // ggml_rope_set_offset: HVX kernels need a VLEN-aligned window start (32 f32 elems)
    ఒకవేళ (n_offs < 0 || (n_offs % 32 != 0) || (n_offs + n_dims > src0->ne[0])) {
        తిరిగి అబద్ధం;
    }

    దశాంశం freq_పునాది;
    జ్ఞాపకం_నకలు(&freq_పునాది, op_params + 5, పరిమాణం(దశాంశం));
    ఒకవేళ (freq_పునాది < 0.0f) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (mode != GGML_ROPE_TYPE_సాధారణ &&
        mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_VISION &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి తార్కిక is_mrope = (mode & GGML_ROPE_TYPE_MROPE) != 0;

    // n_dims == ne0/2, so the rotation spans the full row
    ఒకవేళ (mode == GGML_ROPE_TYPE_VISION) {
        ఒకవేళ (n_dims != (పూర్ణాంకం) (src0->ne[0] / 2) || n_offs != 0) {
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (is_mrope) {
        స్థిరరాశి int32_t * sections = op_params + 11;
        ఒకవేళ (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
            తిరిగి అబద్ధం;
        }
    }

    స్థిరరాశి int64_t min_pos_len = (is_mrope || mode == GGML_ROPE_TYPE_VISION) ? src0->ne[2] * 4 : src0->ne[2];
    ఒకవేళ (src1->ne[0] < min_pos_len || !ggml_is_contiguous(src1)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src2) {
        ఒకవేళ (src2->type != GGML_TYPE_F32 || !ggml_is_contiguous(src2)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (src2->ne[0] < (n_dims / 2)) {
            తిరిగి అబద్ధం;
        }
    }

    // src0/dst elements within a row must be contiguous (nb[0] == sizeof(float)).
    // nb[1] may exceed ne[0]*sizeof(float) when the tensor is a strided view of a larger one
    ఒకవేళ (src0->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->nb[1] < src0->ne[0] * పరిమాణం(దశాంశం) || dst->nb[1] < dst->ne[0] * పరిమాణం(దశాంశం)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t n_దారాలు = (std::min)((uint32_t) sess->n_దారాలు, src0_nrows);
    స్థిరరాశి uint32_t n_freq_factors = src2 ? (uint32_t) src2->ne[0] : 0;

    నిర్మాణం htp_rope_vtcm_layout layout;
    htp_rope_vtcm_layout_build(&layout, src0->ne[0], n_దారాలు, n_freq_factors);
    ఒకవేళ (layout.మొత్తం_బైట్లు > sess->vtcm_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_ssm_conv(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    // Only support FP32 for now
    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    // Check IO tensor shapes and dims
    ఒకవేళ (src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) {
        తిరిగి అబద్ధం; // src0 should be effectively 3D
    }

    స్థిరరాశి పూర్ణాంకం d_conv  = src1->ne[0];
    స్థిరరాశి పూర్ణాంకం d_inner = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం n_t = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం n_s = dst->ne[2];

    ఒకవేళ (d_conv == 0 || d_conv > 32 || d_inner == 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->ne[0] != d_conv - 1 + n_t || src0->ne[1] != d_inner || src0->ne[2] != n_s) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src1->ne[0] != d_conv || src1->ne[1] != d_inner) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (dst->ne[0] != d_inner || dst->ne[1] != n_t || dst->ne[2] != n_s) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->nb[0] != పరిమాణం(దశాంశం) || src1->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (src0->nb[1] != src0->ne[0] * పరిమాణం(దశాంశం) || src1->nb[1] != src1->ne[0] * పరిమాణం(దశాంశం)) {
        తిరిగి అబద్ధం;
    }

    నిర్మాణం htp_ssm_conv_kernel_params kparams;
    ggml_hexagon_precompute_ssm_conv_params(sess, src0, src1, dst, &kparams);
    ఒకవేళ ((size_t) kparams.vtcm_కొలత > sess->vtcm_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

స్థిర తార్కిక im2col_3d_calc_output_కొలత(uint32_t ఆదానం, uint32_t kernel, int32_t stride, int32_t padding,
                                       int32_t dilation, uint32_t * output) {
    ఒకవేళ (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t numerator = (int64_t) ఆదానం + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    ఒకవేళ (numerator < 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint64_t size = (uint64_t) (numerator / stride) + 1;
    ఒకవేళ (size == 0 || size > UINT32_MAX) {
        తిరిగి అబద్ధం;
    }

    *output = (uint32_t) size;
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_precompute_im2col_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->kernel_type = HTP_IM2_నిలువు_KERNEL_UNSUPPORTED;

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!src0 || !src1 || !dst) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->nb[0] != పరిమాణం(దశాంశం) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t * op_params = (స్థిరరాశి int32_t *) op->op_params;
    స్థిరరాశి int32_t s0 = op_params[0];
    స్థిరరాశి int32_t s1 = op_params[1];
    స్థిరరాశి int32_t p0 = op_params[2];
    స్థిరరాశి int32_t p1 = op_params[3];
    స్థిరరాశి int32_t d0 = op_params[4];
    స్థిరరాశి int32_t d1 = op_params[5];
    స్థిరరాశి తార్కిక is_2D = (op_params[6] == 1);

    ఒకవేళ (s0 <= 0 || d0 <= 0 || p0 < 0) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (is_2D && (s1 <= 0 || d1 <= 0 || p1 < 0)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t N  = is_2D ? (uint32_t) src1->ne[3] : (uint32_t) src1->ne[2];
    స్థిరరాశి uint32_t IC = is_2D ? (uint32_t) src1->ne[2] : (uint32_t) src1->ne[1];
    స్థిరరాశి uint32_t IH = is_2D ? (uint32_t) src1->ne[1] : 1;
    స్థిరరాశి uint32_t IW = (uint32_t) src1->ne[0];
    స్థిరరాశి uint32_t KH = is_2D ? (uint32_t) src0->ne[1] : 1;
    స్థిరరాశి uint32_t KW = (uint32_t) src0->ne[0];
    స్థిరరాశి uint32_t OH = is_2D ? (uint32_t) dst->ne[2] : 1;
    స్థిరరాశి uint32_t OW = (uint32_t) dst->ne[1];

    ఒకవేళ (KW == 0 || KH == 0 || IW == 0 || IH == 0 || IC == 0 || OW == 0 || OH == 0 || N == 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint64_t patch_stride_64 = (uint64_t) IC * KH * KW;
    స్థిరరాశి uint64_t మొత్తం_అడ్డువరుసలు_64   = (uint64_t) N * OH;
    ఒకవేళ (patch_stride_64 > UINT32_MAX || మొత్తం_అడ్డువరుసలు_64 > UINT32_MAX) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి uint32_t patch_stride = (uint32_t) patch_stride_64;
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు   = (uint32_t) మొత్తం_అడ్డువరుసలు_64;

    ఒకవేళ (dst->ne[0] != patch_stride || dst->ne[1] != OW) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (is_2D && dst->ne[2] != OH) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? పరిమాణం(ggml_fp16_t) : పరిమాణం(దశాంశం);

    తార్కిక is_patchembed = ggml_is_contiguous(src1) && (s0 == (int32_t) KW && p0 == 0 && d0 == 1);
    ఒకవేళ (is_2D) {
        is_patchembed = is_patchembed && (s1 == (int32_t) KH && p1 == 0 && d1 == 1);
    }

    స్థిరరాశి uint32_t max_దారాలు = (sess && sess->n_దారాలు > 0) ? (uint32_t) sess->n_దారాలు : 8;
    స్థిరరాశి uint32_t n_దారాలు   = (మొత్తం_అడ్డువరుసలు < max_దారాలు) ? మొత్తం_అడ్డువరుసలు : max_దారాలు;
    స్థిరరాశి uint32_t vtcm_కొలత   = (sess && sess->vtcm_కొలత > 0) ? sess->vtcm_కొలత : 0;

    ఒకవేళ (is_patchembed) {
        స్థిరరాశి uint64_t src_అడ్డువరుస_బైట్లు_raw = (uint64_t) IC * KH * IW * పరిమాణం(దశాంశం);
        స్థిరరాశి uint64_t dst_అడ్డువరుస_బైట్లు_raw = (uint64_t) OW * patch_stride * dst_elem;
        ఒకవేళ (src_అడ్డువరుస_బైట్లు_raw <= UINT32_MAX && dst_అడ్డువరుస_బైట్లు_raw <= UINT32_MAX) {
            స్థిరరాశి uint32_t src_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) src_అడ్డువరుస_బైట్లు_raw, 256);
            స్థిరరాశి uint32_t dst_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) dst_అడ్డువరుస_బైట్లు_raw, 256);

            నిర్మాణం htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_అడ్డువరుస_బైట్లు, dst_అడ్డువరుస_బైట్లు, n_దారాలు);

            ఒకవేళ (vtcm_కొలత == 0 || layout.మొత్తం_బైట్లు <= vtcm_కొలత) {
                kparams->kernel_type          = HTP_IM2_నిలువు_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = is_2D ? 1 : 0;
                kparams->is_3d                = 0;
                kparams->n_దారాలు            = (uint8_t) n_దారాలు;
                kparams->vtcm_కొలత            = layout.మొత్తం_బైట్లు;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_బైట్లు_చొప్పున_thread = layout.src_బైట్లు_చొప్పున_thread;
                kparams->dst_బైట్లు_చొప్పున_thread = layout.dst_బైట్లు_చొప్పున_thread;
                kparams->src_అడ్డువరుస_బైట్లు        = src_అడ్డువరుస_బైట్లు;
                kparams->dst_అడ్డువరుస_బైట్లు        = dst_అడ్డువరుస_బైట్లు;
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
                kparams->మొత్తం_అడ్డువరుసలు           = మొత్తం_అడ్డువరుసలు;
                తిరిగి నిజం;
            }
        }
    }

    కొరకు (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        స్థిరరాశి uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ఒకవేళ (Wb > UINT32_MAX) {
            ఒకవేళ (owb == 1) ఆపు;
            కొనసాగు;
        }

        స్థిరరాశి uint64_t src_అడ్డువరుస_బైట్లు_raw = (uint64_t) IC * KH * Wb * పరిమాణం(దశాంశం);
        స్థిరరాశి uint64_t dst_అడ్డువరుస_బైట్లు_raw = (uint64_t) owb * patch_stride * dst_elem;
        ఒకవేళ (src_అడ్డువరుస_బైట్లు_raw > UINT32_MAX || dst_అడ్డువరుస_బైట్లు_raw > UINT32_MAX) {
            ఒకవేళ (owb == 1) ఆపు;
            కొనసాగు;
        }

        స్థిరరాశి uint32_t src_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) src_అడ్డువరుస_బైట్లు_raw, 256);
        స్థిరరాశి uint32_t dst_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) dst_అడ్డువరుస_బైట్లు_raw, 256);

        నిర్మాణం htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_అడ్డువరుస_బైట్లు, dst_అడ్డువరుస_బైట్లు, n_దారాలు);

        ఒకవేళ (vtcm_కొలత == 0 || layout.మొత్తం_బైట్లు <= vtcm_కొలత) {
            kparams->kernel_type          = HTP_IM2_నిలువు_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = is_2D ? 1 : 0;
            kparams->is_3d                = 0;
            kparams->n_దారాలు            = (uint8_t) n_దారాలు;
            kparams->vtcm_కొలత            = layout.మొత్తం_బైట్లు;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_బైట్లు_చొప్పున_thread = layout.src_బైట్లు_చొప్పున_thread;
            kparams->dst_బైట్లు_చొప్పున_thread = layout.dst_బైట్లు_చొప్పున_thread;
            kparams->src_అడ్డువరుస_బైట్లు        = src_అడ్డువరుస_బైట్లు;
            kparams->dst_అడ్డువరుస_బైట్లు        = dst_అడ్డువరుస_బైట్లు;
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
            kparams->మొత్తం_అడ్డువరుసలు           = మొత్తం_అడ్డువరుసలు;
            తిరిగి నిజం;
        }
        ఒకవేళ (owb == 1) ఆపు;
    }

    తిరిగి అబద్ధం;
}

స్థిర తార్కిక ggml_hexagon_precompute_im2col_3d_params(
    స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess,
    స్థిరరాశి నిర్మాణం ggml_tensor * op,
    నిర్మాణం htp_im2col_kernel_params * kparams
) {
    memset(kparams, 0, పరిమాణం(*kparams));
    kparams->kernel_type = HTP_IM2_నిలువు_KERNEL_UNSUPPORTED;

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (!src0 || !src1 || !dst) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->nb[0] != పరిమాణం(దశాంశం) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t * params = (స్థిరరాశి int32_t *) op->op_params;
    స్థిరరాశి int32_t s0 = params[0], s1 = params[1], s2 = params[2];
    స్థిరరాశి int32_t p0 = params[3], p1 = params[4], p2 = params[5];
    స్థిరరాశి int32_t d0 = params[6], d1 = params[7], d2 = params[8];
    స్థిరరాశి int32_t ic = params[9];

    ఒకవేళ (s0 <= 0 || s1 <= 0 || s2 <= 0 ||
        p0 < 0  || p1 < 0  || p2 < 0  ||
        d0 <= 0 || d1 <= 0 || d2 <= 0 || ic <= 0) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src1->ne[3] % ic != 0) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t IC = (uint32_t) ic;
    స్థిరరాశి uint32_t N  = (uint32_t) (src1->ne[3] / IC);
    ఒకవేళ (N == 0) {
        తిరిగి అబద్ధం;
    }

    uint32_t OW, OH, OD;
    ఒకవేళ (!im2col_3d_calc_output_కొలత(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_calc_output_కొలత(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_calc_output_కొలత(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t KD = (uint32_t) src0->ne[2];
    స్థిరరాశి uint32_t KH = (uint32_t) src0->ne[1];
    స్థిరరాశి uint32_t KW = (uint32_t) src0->ne[0];
    స్థిరరాశి uint32_t ID = (uint32_t) src1->ne[2];
    స్థిరరాశి uint32_t IH = (uint32_t) src1->ne[1];
    స్థిరరాశి uint32_t IW = (uint32_t) src1->ne[0];

    స్థిరరాశి uint64_t patch_stride_64 = (uint64_t) IC * KD * KH * KW;
    స్థిరరాశి uint64_t మొత్తం_అడ్డువరుసలు_64   = (uint64_t) N * OD * OH;
    స్థిరరాశి uint64_t dst_ne3_64      = (uint64_t) OD * N;

    ఒకవేళ (patch_stride_64 > UINT32_MAX || మొత్తం_అడ్డువరుసలు_64 > UINT32_MAX || dst_ne3_64 > UINT32_MAX) {
        తిరిగి అబద్ధం;
    }
    స్థిరరాశి uint32_t patch_stride = (uint32_t) patch_stride_64;
    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు   = (uint32_t) మొత్తం_అడ్డువరుసలు_64;
    స్థిరరాశి uint32_t dst_ne3      = (uint32_t) dst_ne3_64;

    ఒకవేళ (dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి uint32_t dst_elem = (dst->type == GGML_TYPE_F16) ? పరిమాణం(ggml_fp16_t) : పరిమాణం(దశాంశం);

    స్థిరరాశి తార్కిక is_patchembed = ggml_is_contiguous(src1) &&
                               (s0 == (int32_t) KW && s1 == (int32_t) KH && s2 == (int32_t) KD &&
                                p0 == 0 && p1 == 0 && p2 == 0 &&
                                d0 == 1 && d1 == 1 && d2 == 1);

    స్థిరరాశి uint32_t max_దారాలు = (sess && sess->n_దారాలు > 0) ? (uint32_t) sess->n_దారాలు : 8;
    స్థిరరాశి uint32_t n_దారాలు   = (మొత్తం_అడ్డువరుసలు < max_దారాలు) ? మొత్తం_అడ్డువరుసలు : max_దారాలు;
    స్థిరరాశి uint32_t vtcm_కొలత   = (sess && sess->vtcm_కొలత > 0) ? sess->vtcm_కొలత : 0;

    ఒకవేళ (is_patchembed) {
        స్థిరరాశి uint64_t src_అడ్డువరుస_బైట్లు_raw = (uint64_t) IC * KD * KH * IW * పరిమాణం(దశాంశం);
        స్థిరరాశి uint64_t dst_అడ్డువరుస_బైట్లు_raw = (uint64_t) OW * patch_stride * dst_elem;
        ఒకవేళ (src_అడ్డువరుస_బైట్లు_raw <= UINT32_MAX && dst_అడ్డువరుస_బైట్లు_raw <= UINT32_MAX) {
            స్థిరరాశి uint32_t src_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) src_అడ్డువరుస_బైట్లు_raw, 256);
            స్థిరరాశి uint32_t dst_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) dst_అడ్డువరుస_బైట్లు_raw, 256);

            నిర్మాణం htp_im2col_vtcm_layout layout;
            htp_im2col_vtcm_layout_build(&layout, src_అడ్డువరుస_బైట్లు, dst_అడ్డువరుస_బైట్లు, n_దారాలు);

            ఒకవేళ (vtcm_కొలత == 0 || layout.మొత్తం_బైట్లు <= vtcm_కొలత) {
                kparams->kernel_type          = HTP_IM2_నిలువు_KERNEL_PATCHEMBED_DMA;
                kparams->is_2d                = 0;
                kparams->is_3d                = 1;
                kparams->n_దారాలు            = (uint8_t) n_దారాలు;
                kparams->vtcm_కొలత            = layout.మొత్తం_బైట్లు;
                kparams->off_src              = layout.off_src;
                kparams->off_dst              = layout.off_dst;
                kparams->src_బైట్లు_చొప్పున_thread = layout.src_బైట్లు_చొప్పున_thread;
                kparams->dst_బైట్లు_చొప్పున_thread = layout.dst_బైట్లు_చొప్పున_thread;
                kparams->src_అడ్డువరుస_బైట్లు        = src_అడ్డువరుస_బైట్లు;
                kparams->dst_అడ్డువరుస_బైట్లు        = dst_అడ్డువరుస_బైట్లు;
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
                kparams->మొత్తం_అడ్డువరుసలు           = మొత్తం_అడ్డువరుసలు;
                తిరిగి నిజం;
            }
        }
    }

    కొరకు (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        స్థిరరాశి uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        ఒకవేళ (Wb > UINT32_MAX) {
            ఒకవేళ (owb == 1) ఆపు;
            కొనసాగు;
        }

        స్థిరరాశి uint64_t src_అడ్డువరుస_బైట్లు_raw = (uint64_t) IC * KD * KH * Wb * పరిమాణం(దశాంశం);
        స్థిరరాశి uint64_t dst_అడ్డువరుస_బైట్లు_raw = (uint64_t) owb * patch_stride * dst_elem;
        ఒకవేళ (src_అడ్డువరుస_బైట్లు_raw > UINT32_MAX || dst_అడ్డువరుస_బైట్లు_raw > UINT32_MAX) {
            ఒకవేళ (owb == 1) ఆపు;
            కొనసాగు;
        }

        స్థిరరాశి uint32_t src_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) src_అడ్డువరుస_బైట్లు_raw, 256);
        స్థిరరాశి uint32_t dst_అడ్డువరుస_బైట్లు = hex_విడత_పైకి((uint32_t) dst_అడ్డువరుస_బైట్లు_raw, 256);

        నిర్మాణం htp_im2col_vtcm_layout layout;
        htp_im2col_vtcm_layout_build(&layout, src_అడ్డువరుస_బైట్లు, dst_అడ్డువరుస_బైట్లు, n_దారాలు);

        ఒకవేళ (vtcm_కొలత == 0 || layout.మొత్తం_బైట్లు <= vtcm_కొలత) {
            kparams->kernel_type          = HTP_IM2_నిలువు_KERNEL_BLOCKED_DMA;
            kparams->is_2d                = 0;
            kparams->is_3d                = 1;
            kparams->n_దారాలు            = (uint8_t) n_దారాలు;
            kparams->vtcm_కొలత            = layout.మొత్తం_బైట్లు;
            kparams->off_src              = layout.off_src;
            kparams->off_dst              = layout.off_dst;
            kparams->src_బైట్లు_చొప్పున_thread = layout.src_బైట్లు_చొప్పున_thread;
            kparams->dst_బైట్లు_చొప్పున_thread = layout.dst_బైట్లు_చొప్పున_thread;
            kparams->src_అడ్డువరుస_బైట్లు        = src_అడ్డువరుస_బైట్లు;
            kparams->dst_అడ్డువరుస_బైట్లు        = dst_అడ్డువరుస_బైట్లు;
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
            kparams->మొత్తం_అడ్డువరుసలు           = మొత్తం_అడ్డువరుసలు;
            తిరిగి నిజం;
        }
        ఒకవేళ (owb == 1) ఆపు;
    }

    తిరిగి అబద్ధం;
}

స్థిర తార్కిక ggml_hexagon_supported_im2col(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    నిర్మాణం htp_im2col_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_im2col_params(sess, op, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_im2col_3d(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    నిర్మాణం htp_im2col_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_im2col_3d_params(sess, op, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_pad(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int32_t lp0 = ((స్థిరరాశి int32_t *) op->op_params)[0];
    స్థిరరాశి int32_t rp0 = ((స్థిరరాశి int32_t *) op->op_params)[1];
    స్థిరరాశి int32_t circular = ((స్థిరరాశి int32_t *) op->op_params)[8];

    ఒకవేళ (circular && (lp0 > src0->ne[0] || rp0 > src0->ne[0])) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_cumsum(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_diag(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    // diag only supports F32 currently
    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    // Input must have ne[1] == 1 (vector input)
    ఒకవేళ (src0->ne[1] != 1) {
        తిరిగి అబద్ధం;
    }

    // Output must be square in first two dimensions
    ఒకవేళ (dst->ne[0] != dst->ne[1] || dst->ne[0] != src0->ne[0]) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_solve_tri(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0]; // A
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = op->src[1]; // B
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;         // X

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->ne[0] != src0->ne[1]) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->ne[1] != src1->ne[1]) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_tri(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32) { తిరిగి అబద్ధం; }
    ఒకవేళ (dst->type  != GGML_TYPE_F32) { తిరిగి అబద్ధం; }
    ఒకవేళ (!ggml_are_same_shape(src0, dst)) { తిరిగి అబద్ధం; }
    ఒకవేళ (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) { తిరిగి అబద్ధం; }

    తిరిగి నిజం;

    GGML_వాడని(sess);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_hexagon_name(ggml_backend_t backend) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    తిరిగి sess->c_name();
}

స్థిర శూన్య ggml_backend_hexagon_free(ggml_backend_t backend) {
    // we just need to delete the backend here
    // the sessions are allocated & freed as part of the registry
    తొలగించు backend;
}

స్థిర htp_op_సంకేతం op_remap_గమ్యం_htp(స్థిరరాశి ggml_tensor * t) {
    ఎంపిక (t->op) {
        స్థితి GGML_OP_FLASH_ATTN_EXT:  తిరిగి HTP_OP_FLASH_ATTN_EXT;
        స్థితి GGML_OP_MUL_MAT:
            ఒకవేళ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                తిరిగి HTP_OP_HADAMARD;
            }
            తిరిగి HTP_OP_MUL_MAT;
        స్థితి GGML_OP_MUL_MAT_ID:      తిరిగి HTP_OP_MUL_MAT_ID;
        స్థితి GGML_OP_MUL:             తిరిగి HTP_OP_MUL;
        స్థితి GGML_OP_కూడు:             తిరిగి HTP_OP_కూడు;
        స్థితి GGML_OP_కూడు_ID:          తిరిగి HTP_OP_కూడు_ID;
        స్థితి GGML_OP_SUB:             తిరిగి HTP_OP_SUB;
        స్థితి GGML_OP_DIV:             తిరిగి HTP_OP_DIV;
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_CPY:             తిరిగి HTP_OP_CPY;
        స్థితి GGML_OP_CONT:            తిరిగి HTP_OP_CPY;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:        తిరిగి HTP_OP_GET_అడ్డువరుసలు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:        తిరిగి HTP_OP_సమితి_అడ్డువరుసలు;
        స్థితి GGML_OP_కూడిక:             తిరిగి HTP_OP_కూడిక;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:        తిరిగి HTP_OP_కూడిక_అడ్డువరుసలు;
        స్థితి GGML_OP_POOL_2D:         తిరిగి HTP_OP_POOL_2D;
        స్థితి GGML_OP_POOL_1D:         తిరిగి HTP_OP_POOL_1D;
        స్థితి GGML_OP_ARGSORT:         తిరిగి HTP_OP_ARGSORT;
        స్థితి GGML_OP_శిఖరం_K:           తిరిగి HTP_OP_శిఖరం_K;
        స్థితి GGML_OP_ARGMAX:          తిరిగి HTP_OP_ARGMAX;
        స్థితి GGML_OP_NORM:            తిరిగి HTP_OP_NORM;
        స్థితి GGML_OP_L2_NORM:         తిరిగి HTP_OP_L2_NORM;
        స్థితి GGML_OP_RMS_NORM:        తిరిగి HTP_OP_RMS_NORM;
        స్థితి GGML_OP_CONCAT:          తిరిగి HTP_OP_CONCAT;
        స్థితి GGML_OP_SCALE:           తిరిగి HTP_OP_SCALE;
        స్థితి GGML_OP_CLAMP:           తిరిగి HTP_OP_CLAMP;
        స్థితి GGML_OP_LEAKY_RELU:      తిరిగి HTP_OP_LEAKY_RELU;
        స్థితి GGML_OP_SQR:             తిరిగి HTP_OP_SQR;
        స్థితి GGML_OP_SQRT:            తిరిగి HTP_OP_SQRT;
        స్థితి GGML_OP_LOG:             తిరిగి HTP_OP_UNARY_LOG;
        స్థితి GGML_OP_SOFT_MAX:        తిరిగి HTP_OP_SOFTMAX;
        స్థితి GGML_OP_SSM_CONV:        తిరిగి HTP_OP_SSM_CONV;
        స్థితి GGML_OP_GATED_DELTA_NET: తిరిగి HTP_OP_GATED_DELTA_NET;
        స్థితి GGML_OP_ROPE:            తిరిగి HTP_OP_ROPE;
        స్థితి GGML_OP_REPEAT:          తిరిగి HTP_OP_REPEAT;
        స్థితి GGML_OP_CUMSUM:          తిరిగి HTP_OP_CUMSUM;
        స్థితి GGML_OP_FILL:            తిరిగి HTP_OP_FILL;
        స్థితి GGML_OP_DIAG:            తిరిగి HTP_OP_DIAG;
        స్థితి GGML_OP_SOLVE_TRI:       తిరిగి HTP_OP_SOLVE_TRI;
        స్థితి GGML_OP_TRI:             తిరిగి HTP_OP_TRI;
        స్థితి GGML_OP_PAD:             తిరిగి HTP_OP_PAD;
        స్థితి GGML_OP_IM2_నిలువు:          తిరిగి HTP_OP_IM2_నిలువు;
        స్థితి GGML_OP_IM2_నిలువు_3_D:       తిరిగి HTP_OP_IM2_నిలువు_3_D;
        స్థితి GGML_OP_ROLL:            తిరిగి HTP_OP_ROLL;

        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(t)) {
                స్థితి GGML_UNARY_OP_SILU:       తిరిగి HTP_OP_UNARY_SILU;
                స్థితి GGML_UNARY_OP_GELU:       తిరిగి HTP_OP_UNARY_GELU;
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర: తిరిగి HTP_OP_UNARY_GELU_శీఘ్ర;
                స్థితి GGML_UNARY_OP_GELU_ERF:   తిరిగి HTP_OP_UNARY_GELU_ERF;
                స్థితి GGML_UNARY_OP_SIGMOID:    తిరిగి HTP_OP_UNARY_SIGMOID;
                స్థితి GGML_UNARY_OP_NEG:        తిరిగి HTP_OP_UNARY_NEG;
                స్థితి GGML_UNARY_OP_EXP:        తిరిగి HTP_OP_UNARY_EXP;
                స్థితి GGML_UNARY_OP_SOFTPLUS:   తిరిగి HTP_OP_UNARY_SOFTPLUS;
                స్థితి GGML_UNARY_OP_TANH:       తిరిగి HTP_OP_UNARY_TANH;
                స్థితి GGML_UNARY_OP_ABS:        తిరిగి HTP_OP_UNARY_ABS;
                స్థితి GGML_UNARY_OP_RELU:       తిరిగి HTP_OP_UNARY_RELU;
                స్థితి GGML_UNARY_OP_అడుగు:       తిరిగి HTP_OP_UNARY_అడుగు;
            అప్రమేయ:
                ఆపు;
            }
            ఆపు;

        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(t)) {
                స్థితి GGML_GLU_OP_SWIGLU:       తిరిగి HTP_OP_GLU_SWIGLU;
                స్థితి GGML_GLU_OP_SWIGLU_OAI:   తిరిగి HTP_OP_GLU_SWIGLU_OAI;
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP: తిరిగి HTP_OP_GLU_SWIGLU_CLAMP;
                స్థితి GGML_GLU_OP_GEGLU:        తిరిగి HTP_OP_GLU_GEGLU;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:  తిరిగి HTP_OP_GLU_GEGLU_శీఘ్ర;
                స్థితి GGML_GLU_OP_GEGLU_ERF:    తిరిగి HTP_OP_GLU_GEGLU_ERF;
                అప్రమేయ: ఆపు;
            }
            ఆపు;

        అప్రమేయ:
            GGML_ABORT("\nggml-hex: graph-compute %s is not supported\n", ggml_op_desc(t));
    }
    తిరిగి HTP_OP_INVALID;
}

స్థిర అంతఃవరుస తార్కిక op_is_compute(ggml_tensor *కణుపు)
{
    తిరిగి !ggml_op_is_ఖాళీలు(కణుపు->op) && !ggml_is_ఖాళీలు(కణుపు) && (కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE);
}

స్థిర తార్కిక mm_is_hmx_eligible(స్థిరరాశి ggml_tensor * t) {
    ఒకవేళ (opt_nhmx == 0) { తిరిగి అబద్ధం; }

    స్థిరరాశి ggml_tensor * src0 = t->src[0];
    స్థిరరాశి ggml_tensor * src1 = t->src[1];

    ఒకవేళ (ggml_hexagon_matmul_can_collapse(src0, src1, t)) {
        స్థిరరాశి ggml_tensor src1_c = ggml_hexagon_tensor_collapse_అడ్డువరుసలు(src1);
        తిరిగి ggml_hexagon_matmul_is_hmx_eligible(src0, &src1_c, t, అబద్ధం, అబద్ధం);
    }

    స్థిరరాశి తార్కిక is_matmul_id = (t->op == GGML_OP_MUL_MAT_ID);
    స్థిరరాశి తార్కిక is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);

    తిరిగి ggml_hexagon_matmul_is_hmx_eligible(src0, src1, t, is_matmul_id, is_batched);
}

స్థిర తార్కిక is_supported_mul_mat_nx_kernel(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams) {
    ఒకవేళ (kparams->n_hmx) {
        తిరిగి kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ఒకవేళ (!ggml_hexagon_tensor_is_tiled(src0)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస;
}

స్థిర తార్కిక is_supported_mul_mat_id_nx_kernel(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి నిర్మాణం htp_mm_kernel_params * kparams) {
    ఒకవేళ (kparams->n_hmx) {
        తిరిగి kparams->kernel_type == HTP_MM_KERNEL_HMX_2D;
    }

    ఒకవేళ (!ggml_hexagon_tensor_is_tiled(src0)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస || kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం;
}

స్థిర తార్కిక is_mergeable_mul_mat(స్థిరరాశి ggml_tensor * t) {
    ఒకవేళ (t->op != GGML_OP_MUL_MAT) తిరిగి అబద్ధం;
    ఒకవేళ (ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) తిరిగి అబద్ధం;

    స్థిరరాశి ggml_tensor * src0 = t->src[0];
    స్థిరరాశి ggml_tensor * src1 = t->src[1];
    ఒకవేళ (src0->ne[2] != 1 || src0->ne[3] != 1) తిరిగి అబద్ధం;

    ఒకవేళ (mm_is_hmx_eligible(t)) {
        తిరిగి htp_is_hmx_బరువు_type(src0->type);
    }

    // HVX path requires F32 activations and tiled weights
    ఒకవేళ (src1->type != GGML_TYPE_F32) తిరిగి అబద్ధం;

    తిరిగి ggml_hexagon_tensor_is_tiled(src0);
}

స్థిర తార్కిక is_mergeable_mul_mat_pair(స్థిరరాశి ggml_tensor * n1, స్థిరరాశి ggml_tensor * n2) {
    ఒకవేళ (!is_mergeable_mul_mat(n1) || !is_mergeable_mul_mat(n2)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[1] != n2->src[1]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[0]->type != n2->src[0]->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక is_mergeable_mul_mat_id(స్థిరరాశి ggml_tensor * t) {
    ఒకవేళ (t->op != GGML_OP_MUL_MAT_ID) తిరిగి అబద్ధం;

    స్థిరరాశి ggml_tensor * src0 = t->src[0];
    ఒకవేళ (mm_is_hmx_eligible(t)) {
        తిరిగి htp_is_hmx_బరువు_type(src0->type);
    }

    తిరిగి ggml_hexagon_tensor_is_tiled(src0);
}

స్థిర తార్కిక is_mergeable_mul_mat_id_pair(స్థిరరాశి ggml_tensor * n1, స్థిరరాశి ggml_tensor * n2) {
    ఒకవేళ (!is_mergeable_mul_mat_id(n1) || !is_mergeable_mul_mat_id(n2)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[1] != n2->src[1]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[2] != n2->src[2]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[0]->ne[0] != n2->src[0]->ne[0]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[0]->ne[2] != n2->src[0]->ne[2]) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (n1->src[0]->type != n2->src[0]->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ggml_hexagon_tensor_is_tiled(n1->src[0]) != ggml_hexagon_tensor_is_tiled(n2->src[0])) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (mm_is_hmx_eligible(n1) != mm_is_hmx_eligible(n2)) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;
}

స్థిర ggml_స్థితి ggml_backend_hexagon_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);

    ఒకవేళ (sess->చివరి_error > HTP_స్థితి_OK) {
        తిరిగి GGML_స్థితి_FAILED;
    }

    HEX_VERBOSE("ggml-hex: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    స్థిరరాశి std::vector<htp_opnode> * nodes_ptr = శూన్యసూచిక;
    std::vector<htp_opnode> computed_nodes;

    // Check for cache hit
    తార్కిక cache_తగిలింది = (graph->uid != 0 && sess->cached_uid == graph->uid);
    ఒకవేళ (cache_తగిలింది) {
        nodes_ptr = &sess->cached_nodes;
    } లేకపోతే {
        // Tag fusable tensors in graph
        కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; i++) {
            స్వయంచాలక * extra = (ggml_hexagon_tensor_extra *) graph->nodes[i]->extra;
            ఒకవేళ (!extra) కొనసాగు;

            extra->flags &= ~GGML_HEXAGON_TENSOR_FUSEABLE;

            ఒకవేళ (graph->nodes[i]->op == GGML_OP_RMS_NORM && ggml_can_fuse(graph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
            } లేకపోతే ఒకవేళ ((graph->nodes[i]->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(graph->nodes[i], 1) != GGML_HINT_SRC0_IS_HADAMARD) ||
                       graph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
                ఒకవేళ ((i + 1 < graph->n_nodes && graph->nodes[i + 1]->op == GGML_OP_కూడు && ggml_can_fuse(graph, i, { graph->nodes[i]->op, GGML_OP_కూడు })) ||
                    ggml_కణుపు_has_n_ఉపయోగిస్తుంది(graph, i, 1)) {
                    extra->flags |= GGML_HEXAGON_TENSOR_FUSEABLE;
                }
            }
        }

        computed_nodes.reserve(graph->n_nodes);

        కొరకు (పూర్ణాంకం i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            ఒకవేళ (!op_is_compute(n)) {
                కొనసాగు;
            }

            htp_opnode కణుపు(HTP_OP_INVALID, n);
            కణుపు.opcode = op_remap_గమ్యం_htp(n);
            ఒకవేళ (కణుపు.opcode == HTP_OP_MUL_MAT || కణుపు.opcode == HTP_OP_MUL_MAT_ID || కణుపు.opcode == HTP_OP_GET_అడ్డువరుసలు) {
                ggml_tensor * src0 = కణుపు.కణుపు->src[0];
                ఒకవేళ (ggml_hexagon_needs_repack(src0)) {
                    ఒకవేళ (opt_verbose) {
                        అక్షరం dims_buf[64];
                        అక్షరం strides_buf[64];
                        htp_opformat fmt;
                        fmt.format_tensor_dims(dims_buf, పరిమాణం(dims_buf), src0);
                        fmt.format_tensor_strides(strides_buf, పరిమాణం(strides_buf), src0);
                        GGML_LOG_DEBUG("ggml-hex: %s rt-repack %s|%s|%s|%s|size %zu\n",
                                       sess->c_name(), src0->name, dims_buf, ggml_type_name(src0->type), strides_buf, ggml_nbytes(src0));
                    }
                    std::vector<uint8_t> raw(ggml_nbytes(src0));
                    జ్ఞాపకం_నకలు(raw.దత్తాంశం(), src0->దత్తాంశం, raw.size());
                    htp_repack_tensor_tiled(src0, raw.దత్తాంశం(), raw.size());
                    స్వయంచాలక * extra = (ggml_hexagon_tensor_extra *) src0->extra;
                    extra->flags |= GGML_HEXAGON_TENSOR_TILED;
                }
            }

            ఒకవేళ (కణుపు.opcode == HTP_OP_MUL_MAT || కణుపు.opcode == HTP_OP_MUL_MAT_ID) {
                ggml_hexagon_precompute_matmul_params(sess,
                    కణుపు.కణుపు->src[0], కణుపు.కణుపు->src[1], కణుపు.కణుపు,
                    (నిర్మాణం htp_mm_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_MUL || కణుపు.opcode == HTP_OP_కూడు || కణుపు.opcode == HTP_OP_కూడు_ID || కణుపు.opcode == HTP_OP_SUB || కణుపు.opcode == HTP_OP_DIV) {
                స్థిరరాశి ggml_tensor * src1 = కణుపు.కణుపు->src[1];
                స్థిరరాశి ggml_tensor * src2 = (కణుపు.opcode == HTP_OP_కూడు_ID) ? కణుపు.కణుపు->src[2] : శూన్యసూచిక;
                GGML_ASSERT(ggml_hexagon_precompute_ద్విమాన_params(sess,
                    కణుపు.opcode, కణుపు.కణుపు->src[0], src1, కణుపు.కణుపు,
                    (నిర్మాణం htp_ద్విమాన_kernel_params *) కణుపు.kernel_params, src2));
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_precompute_flash_attn_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_fa_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (htp_op_is_unary(కణుపు.opcode)) {
                స్వయంచాలక inputs = కణుపు.get_inputs();
                స్థిరరాశి నిర్మాణం ggml_tensor * src0 = inputs[0];
                స్థిరరాశి నిర్మాణం ggml_tensor * src1 = inputs.size() > 1 ? inputs[1] : శూన్యసూచిక;
                ggml_hexagon_precompute_unary_params(sess,
                    కణుపు.opcode, src0, src1, కణుపు.dst(),
                    (నిర్మాణం htp_unary_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_GET_అడ్డువరుసలు) {
                ggml_hexagon_precompute_get_అడ్డువరుసలు_params(sess,
                    కణుపు.కణుపు->src[0], కణుపు.కణుపు->src[1], కణుపు.dst(),
                    (నిర్మాణం htp_get_అడ్డువరుసలు_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_సమితి_అడ్డువరుసలు) {
                ggml_hexagon_precompute_సమితి_అడ్డువరుసలు_params(sess,
                    కణుపు.కణుపు->src[0], కణుపు.కణుపు->src[1], కణుపు.dst(),
                    (నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_ROPE) {
                ggml_hexagon_precompute_rope_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_rope_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_SSM_CONV) {
                ggml_hexagon_precompute_ssm_conv_params(sess,
                    కణుపు.కణుపు->src[0], కణుపు.కణుపు->src[1], కణుపు.dst(),
                    (నిర్మాణం htp_ssm_conv_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_SOFTMAX) {
                ggml_hexagon_precompute_softmax_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_softmax_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_GATED_DELTA_NET) {
                ggml_hexagon_precompute_gated_delta_net_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_gdn_kernel_params *)కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_ARGSORT || కణుపు.opcode == HTP_OP_శిఖరం_K) {
                ggml_hexagon_precompute_క్రమీకరించు_params(sess,
                    కణుపు.కణుపు,
                    కణుపు.opcode == HTP_OP_శిఖరం_K,
                    (నిర్మాణం htp_క్రమీకరించు_kernel_params *) కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_POOL_2D || కణుపు.opcode == HTP_OP_POOL_1D) {
                ggml_hexagon_precompute_pool_2d_params(
                    sess, కణుపు.కణుపు->src[0], కణుపు.dst(),
                    (నిర్మాణం htp_pool_2d_kernel_params *)కణుపు.kernel_params,
                    కణుపు.opcode == HTP_OP_POOL_1D);
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_CONCAT) {
                ggml_hexagon_precompute_concat_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_concat_kernel_params *) కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_CPY || కణుపు.opcode == HTP_OP_CPY_FENCE) {
                స్థిరరాశి తార్కిక ok = ggml_hexagon_precompute_cpy_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_copy_kernel_params *) కణుపు.kernel_params
                );
                స్థిరరాశి స్వయంచాలక * kparams = (స్థిరరాశి నిర్మాణం htp_copy_kernel_params *) కణుపు.kernel_params;
                ఒకవేళ (ok && కణుపు.opcode == HTP_OP_CPY && kparams->మొత్తం_elems == 0) {
                    కొనసాగు;
                }
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_IM2_నిలువు) {
                ggml_hexagon_precompute_im2col_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_im2col_kernel_params *) కణుపు.kernel_params
                );
            } లేకపోతే ఒకవేళ (కణుపు.opcode == HTP_OP_IM2_నిలువు_3_D) {
                ggml_hexagon_precompute_im2col_3d_params(sess,
                    కణుపు.కణుపు,
                    (నిర్మాణం htp_im2col_kernel_params *) కణుపు.kernel_params
                );
            }
            computed_nodes.push_back(std::move(కణుపు));
        }

        ఒకవేళ (graph->uid != 0) {
            sess->cached_uid   = graph->uid;
            sess->cached_nodes = std::move(computed_nodes);
            nodes_ptr = &sess->cached_nodes;
        } లేకపోతే {
            nodes_ptr = &computed_nodes;
        }
    }

    // Queue and execute
    కొరకు (స్థిరరాశి స్వయంచాలక & కణుపు : *nodes_ptr) {
        sess->వరుసలో_చేర్చు_op(కణుపు);
    }

    ఒకవేళ (sess->చివరి_error > HTP_స్థితి_OK) {
        తిరిగి GGML_స్థితి_FAILED;
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_hexagon_synchronize(ggml_backend_t backend) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);

    HEX_VERBOSE("ggml-hex: %s synchronize\n", sess->c_name());

    // Wait until all pending ops complete
    sess->flush_sync();
    ఒకవేళ (sess->చివరి_error > HTP_స్థితి_OK) {
        GGML_ABORT("ggml-hex: %s synchronize failed : dsp-error %s\n", sess->c_name(), స్థితి_గమ్యం_str(sess->చివరి_error));
    }
}

గణన ggml_hexagon_mem_range_type {
    HEXAGON_MEM_RANGE_TYPE_SRC,
    HEXAGON_MEM_RANGE_TYPE_DST,
};

నిర్మాణం ggml_hexagon_mem_range {
    uint64_t pb;
    uint64_t p0;
    uint64_t p1;
    ggml_hexagon_mem_range_type pt;
};

నిర్మాణం ggml_hexagon_mem_ranges {
    std::vector<ggml_hexagon_mem_range> ranges;

    శూన్య reset() {
        ranges.clear();
    }

    శూన్య కూడు_2(స్థిరరాశి ggml_hexagon_mem_range & mr) {
        ranges.push_back(mr);
    }

    తార్కిక check(స్థిరరాశి ggml_hexagon_mem_range & mr) స్థిరరాశి {
        కొరకు (స్థిరరాశి స్వయంచాలక & cmp : ranges) {
            ఒకవేళ (mr.pb != cmp.pb) {
                కొనసాగు;
            }
            ఒకవేళ (mr.pt == HEXAGON_MEM_RANGE_TYPE_SRC && cmp.pt == HEXAGON_MEM_RANGE_TYPE_SRC) {
                కొనసాగు;
            }
            ఒకవేళ (mr.p0 < cmp.p1 && mr.p1 > cmp.p0) {
                తిరిగి అబద్ధం;
            }
        }
        తిరిగి నిజం;
    }
};

స్థిర ggml_hexagon_mem_range ggml_hexagon_mem_range_from_tensor(స్థిరరాశి ggml_tensor * tensor, ggml_hexagon_mem_range_type pt) {
    స్థిరరాశి ggml_tensor * పునాది = tensor->view_src ? tensor->view_src : tensor;
    ggml_hexagon_mem_range mr;
    ఒకవేళ (tensor->బఫరు) {
        mr = {
            /*.pb =*/ (uint64_t) tensor->బఫరు,
            /*.p0 =*/ (uint64_t) tensor->దత్తాంశం,
            /*.p1 =*/ (uint64_t) tensor->దత్తాంశం + ggml_backend_buft_get_alloc_కొలత(tensor->బఫరు->buft, tensor),
            /*.pt =*/ pt,
        };
    } లేకపోతే {
        mr = {
            /*.pb =*/ (uint64_t) పునాది,
            /*.p0 =*/ 0,
            /*.p1 =*/ 1024,
            /*.pt =*/ pt,
        };
    }
    తిరిగి mr;
}

స్థిర శూన్య ggml_hexagon_mem_ranges_కూడు_కణుపు(ggml_hexagon_mem_ranges & mrs, స్థిరరాశి htp_opnode & కణుపు) {
    ఒకవేళ (కణుపు.is_ఖాళీలు()) తిరిగి;

    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (కణుపు.కణుపు->src[i]) {
            mrs.కూడు_2(ggml_hexagon_mem_range_from_tensor(కణుపు.కణుపు->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
        }
    }
    కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (fused->src[i]) {
                mrs.కూడు_2(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC));
            }
        }
    }
    mrs.కూడు_2(ggml_hexagon_mem_range_from_tensor(కణుపు.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

స్థిర తార్కిక ggml_hexagon_mem_ranges_check_కణుపు(స్థిరరాశి ggml_hexagon_mem_ranges & mrs, స్థిరరాశి htp_opnode & కణుపు) {
    ఒకవేళ (కణుపు.is_ఖాళీలు()) తిరిగి నిజం;

    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
        ఒకవేళ (కణుపు.కణుపు->src[i]) {
            ఒకవేళ (!mrs.check(ggml_hexagon_mem_range_from_tensor(కణుపు.కణుపు->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                తిరిగి అబద్ధం;
            }
        }
    }
    కొరకు (స్థిరరాశి స్వయంచాలక * fused : కణుపు.fused) {
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC; i++) {
            ఒకవేళ (fused->src[i]) {
                ఒకవేళ (!mrs.check(ggml_hexagon_mem_range_from_tensor(fused->src[i], HEXAGON_MEM_RANGE_TYPE_SRC))) {
                    తిరిగి అబద్ధం;
                }
            }
        }
    }
    తిరిగి mrs.check(ggml_hexagon_mem_range_from_tensor(కణుపు.dst(), HEXAGON_MEM_RANGE_TYPE_DST));
}

స్థిర std::vector<పూర్ణాంకం> ggml_hexagon_లేఖాచిత్రం_optimize_reorder(స్థిరరాశి std::vector<htp_opnode> & nodes) {
    స్థిరరాశి పూర్ణాంకం n = nodes.size();

    std::vector<పూర్ణాంకం> res;
    res.reserve(n);

    std::vector<తార్కిక> వాడిన(n, అబద్ధం);

    ggml_hexagon_mem_ranges mrs;

    // The main goal here is to stack the MUL_MAT ops with the same src1 input.
    // This allows us to reuse dynamically quantized src1 in VTCM.

    కొరకు (పూర్ణాంకం i0 = 0; i0 < n; i0++) {
        ఒకవేళ (వాడిన[i0]) {
            కొనసాగు;
        }

        స్థిరరాశి స్వయంచాలక & node0 = nodes[i0];

        ఒకవేళ (!node0.stackable()) {
            res.push_back(i0);
            వాడిన[i0] = నిజం;
            కొనసాగు;
        }

        // that many nodes forward to search for stackable nodes that can reuse VTCM
        constexpr పూర్ణాంకం N_FORWARD = 16;

        std::vector<పూర్ణాంకం> దొంతర;
        దొంతర.push_back(i0);

        mrs.reset();

        కొరకు (పూర్ణాంకం i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
            ఒకవేళ (వాడిన[i1]) {
                కొనసాగు;
            }

            స్థిరరాశి స్వయంచాలక & node1 = nodes[i1];

            ఒకవేళ (node1.stackable() && node1.same_ఆదానం(node0) && ggml_hexagon_mem_ranges_check_కణుపు(mrs, node1)) {
                దొంతర.push_back(i1);
            } లేకపోతే {
                ggml_hexagon_mem_ranges_కూడు_కణుపు(mrs, node1);
            }
        }

        కొరకు (పూర్ణాంకం idx : దొంతర) {
            res.push_back(idx);
            వాడిన[idx] = నిజం;
        }
    }

    తిరిగి res;
}

స్థిర శూన్య ggml_backend_hexagon_లేఖాచిత్రం_optimize(ggml_backend_t backend, ggml_cgraph * gf, ggml_backend_లేఖాచిత్రం_optimize_params * params) {
    GGML_వాడని(params);

    స్థిరరాశి పూర్ణాంకం n = gf->n_nodes;

    constexpr పూర్ణాంకం MAX_FUSE = 16;

    గణన ggml_op ops[MAX_FUSE];

    std::vector<htp_opnode> nodes;
    nodes.reserve(gf->n_nodes);

    // Pack nodes for reordering
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        htp_opnode కణుపు(HTP_OP_INVALID, gf->nodes[i]);

        // fuse only ops that start with these operations
        // can be expanded when needed
        ఒకవేళ (కణుపు.op() == GGML_OP_కూడు ||
            కణుపు.op() == GGML_OP_NORM ||
            కణుపు.op() == GGML_OP_RMS_NORM) {
            ops[0] = కణుపు.op();

            పూర్ణాంకం f = i + 1;
            అయితే (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                ఒకవేళ (gf->nodes[f]->op != GGML_OP_కూడు &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    ఆపు;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            కొరకు (; f > 1; f--) {
                ఒకవేళ (ggml_can_fuse(gf, i, ops, f)) {
                    ఆపు;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            కొరకు (పూర్ణాంకం k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                కణుపు.కూడు_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(కణుపు));
    }

    స్థిరరాశి స్వయంచాలక క్రమం = ggml_hexagon_లేఖాచిత్రం_optimize_reorder(nodes);

    // unfuse
    {
        పూర్ణాంకం j = 0;
        కొరకు (స్థిరరాశి స్వయంచాలక i : క్రమం) {
            స్థిరరాశి స్వయంచాలక & కణుపు = nodes[i];

            gf->nodes[j++] = కణుపు.కణుపు;

            కొరకు (స్వయంచాలక * fused : కణుపు.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }

    GGML_వాడని(backend);
}

స్థిర uint64_t ggml_hexagon_session_కీలకం(స్థిరరాశి ggml_hexagon_session * sess) {
    తిరిగి ((uint64_t) (uint32_t) sess->phys_idx << 32) | (uint32_t) sess->virt_idx;
}

స్థిర తార్కిక ggml_hexagon_cpy_tensor_async_phys(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    స్వయంచాలక sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    స్వయంచాలక sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    sess_src->mmap_tensor(src);
    స్వయంచాలక sbuf_dst = sess_dst->mmap_tensor(dst);

    ఒకవేళ (!sess_src->clone_బఫరు(sbuf_dst)) { తిరిగి అబద్ధం; }

    స్థిరరాశి uint64_t src_కీలకం = ggml_hexagon_session_కీలకం(sess_src);
    స్వయంచాలక & fence_స్థానం = sess_dst->cpy_fence_slots[src_కీలకం];
    ఒకవేళ (!fence_స్థానం) {
        fence_స్థానం = (అస్థిర uint32_t *) sess_dst->alloc_fence(1);
    }

    ఒకవేళ (!sess_src->clone_బఫరు(sess_dst->fence_buf)) { తిరిగి అబద్ధం; }

    ఒకవేళ (++sess_dst->fence_seq == 0) sess_dst->fence_seq = 1;
    uint32_t fence_seq = sess_dst->fence_seq;

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu : seq 0x%x\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src), fence_seq);

    // dummy fence extra (must be static)
    స్థిర ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };

    ggml_tensor fence_tensor {};
    fence_tensor.బఫరు = &sess_dst->fence_buf->backend_బఫరు;
    fence_tensor.extra  = &fence_extra;
    fence_tensor.దత్తాంశం   = (శూన్య *) fence_స్థానం;
    fence_tensor.type   = GGML_TYPE_I32;
    fence_tensor.ne[0]  = 1;
    fence_tensor.ne[1]  = 1;
    fence_tensor.ne[2]  = 1;
    fence_tensor.ne[3]  = 1;
    fence_tensor.nb[0]  = పరిమాణం(int32_t);
    fence_tensor.nb[1]  = పరిమాణం(int32_t);
    fence_tensor.nb[2]  = పరిమాణం(int32_t);
    fence_tensor.nb[3]  = పరిమాణం(int32_t);
    fence_tensor.op     = GGML_OP_NONE;

    sess_src->వరుసలో_చేర్చు_cpy(src, dst, &fence_tensor, fence_seq);
    sess_dst->వరుసలో_చేర్చు_fence(&fence_tensor, fence_seq, /* wait = */ నిజం);

    sess_dst->కూడు_peer(sess_src);

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_cpy_tensor_async_virt(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    స్వయంచాలక sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    స్వయంచాలక sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    స్వయంచాలక sbuf_src = sess_src->mmap_tensor(src);
    sess_dst->mmap_tensor(dst);

    ఒకవేళ (!sess_dst->clone_బఫరు(sbuf_src)) { తిరిగి అబద్ధం; }

    HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n",
                sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));

    sess_dst->వరుసలో_చేర్చు_cpy(src, dst);
    sess_dst->కూడు_peer(sess_src);

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_backend_hexagon_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, స్థిరరాశి ggml_tensor * src, ggml_tensor * dst) {
    ఒకవేళ (!ggml_backend_is_hexagon(backend_src) || !ggml_backend_is_hexagon(backend_dst)) {
        తిరిగి అబద్ధం;
    }

    // FIXME: ggml-meta needs to call init_tensor on auxiliary tensors
    ఒకవేళ (!dst->extra) {
        ggml_backend_బఫరు_init_tensor(dst->బఫరు, dst);
    }

    స్వయంచాలక * dst_extra = static_cast<ggml_hexagon_tensor_extra *>(dst->extra);
    స్థిరరాశి స్వయంచాలక * src_extra = static_cast<స్థిరరాశి ggml_hexagon_tensor_extra *>(src->extra);
    dst_extra->flags = src_extra->flags & ~GGML_HEXAGON_TENSOR_FUSEABLE;

    స్వయంచాలక sess_src = static_cast<ggml_hexagon_session *>(backend_src->context);
    స్వయంచాలక sess_dst = static_cast<ggml_hexagon_session *>(backend_dst->context);

    ఒకవేళ (sess_src == sess_dst) {
        HEX_VERBOSE("ggml-hex: %s cpy-tensor-async %s -> %s size %zu\n", sess_dst->name.c_str(), src->name, dst->name, ggml_nbytes(src));
        sess_src->వరుసలో_చేర్చు_cpy(src, dst);
        తిరిగి నిజం;
    }

    ఒకవేళ (sess_src->phys_idx != sess_dst->phys_idx)
        తిరిగి ggml_hexagon_cpy_tensor_async_phys(backend_src, backend_dst, src, dst);

    తిరిగి ggml_hexagon_cpy_tensor_async_virt(backend_src, backend_dst, src, dst);
}

స్థిర ggml_backend_event_t ggml_backend_hexagon_device_event_new(ggml_backend_dev_t dev) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    స్వయంచాలక sess    = dev_ctx->session();

    ggml_hexagon_event * hex_event = కొత్త ggml_hexagon_event();
    hex_event->fence_sess = sess;
    hex_event->sess       = sess;
    hex_event->fence_స్థానం = (అస్థిర uint32_t *) sess->alloc_fence(1);

    స్థిర ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    hex_event->fence_tensor.బఫరు = &sess->fence_buf->backend_బఫరు;
    hex_event->fence_tensor.extra  = &fence_extra;
    hex_event->fence_tensor.దత్తాంశం   = (శూన్య *) hex_event->fence_స్థానం;
    hex_event->fence_tensor.type   = GGML_TYPE_I32;
    hex_event->fence_tensor.ne[0]  = 1;
    hex_event->fence_tensor.ne[1]  = 1;
    hex_event->fence_tensor.ne[2]  = 1;
    hex_event->fence_tensor.ne[3]  = 1;
    hex_event->fence_tensor.nb[0]  = పరిమాణం(int32_t);
    hex_event->fence_tensor.nb[1]  = పరిమాణం(int32_t);
    hex_event->fence_tensor.nb[2]  = పరిమాణం(int32_t);
    hex_event->fence_tensor.nb[3]  = పరిమాణం(int32_t);
    hex_event->fence_tensor.op     = GGML_OP_NONE;

    HEX_VERBOSE("ggml-hex: %s event-new : event %p fence %p\n", ggml_backend_dev_name(dev), (శూన్య *)hex_event, (శూన్య *)hex_event->fence_స్థానం);

    తిరిగి కొత్త ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ hex_event,
    };
}

స్థిర శూన్య ggml_hexagon_event_synchronize(ggml_backend_dev_t dev, ggml_hexagon_event * hex_event) {
    ఒకవేళ (hex_event->seq == 0) {
        తిరిగి;
    }

    HEX_VERBOSE("ggml-hex: %s event-synchronize : event %p seq 0x%x fence %p\n",
                ggml_backend_dev_name(dev), (శూన్య *)hex_event, hex_event->seq, (శూన్య *)hex_event->fence_స్థానం);

    స్వయంచాలక * fence = reinterpret_cast<స్థిరరాశి అస్థిర std::atomic<uint32_t> *>(hex_event->fence_స్థానం);

    ఒకవేళ ((int32_t)(fence[0].load(std::memory_order_relaxed) - hex_event->seq) < 0) {
        hex_event->sess->flush_async();
    }

    అయితే (నిజం) {
        ఒకవేళ ((int32_t)(fence[0].load(std::memory_order_acquire) - hex_event->seq) >= 0) {
            uint32_t స్థితి_2 = fence[1].load(std::memory_order_acquire);
            ఒకవేళ (స్థితి_2 > HTP_స్థితి_OK) {
                GGML_ABORT("ggml-hex: %s event-synchronize failed : dsp-error %s\n",
                           hex_event->sess->c_name(), స్థితి_గమ్యం_str(స్థితి_2));
            }
            ఆపు;
        }
        std::this_thread::yield();
    }
}

స్థిర శూన్య ggml_backend_hexagon_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    స్వయంచాలక * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
    HEX_VERBOSE("ggml-hex: %s event-free : event %p\n", ggml_backend_dev_name(dev), (శూన్య *)hex_event);
    hex_event->fence_sess->free_fence((శూన్య *) hex_event->fence_స్థానం, 1);
    తొలగించు hex_event;
    తొలగించు event;
}

స్థిర శూన్య ggml_backend_hexagon_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    స్వయంచాలక * hex_event = static_cast<ggml_hexagon_event *>(event->context);
    ggml_hexagon_event_synchronize(dev, hex_event);
}

స్థిర శూన్య ggml_backend_hexagon_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    స్వయంచాలక hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ఒకవేళ (++sess->fence_seq == 0) sess->fence_seq = 1;
    hex_event->sess = sess;
    hex_event->seq  = sess->fence_seq;

    sess->వరుసలో_చేర్చు_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ అబద్ధం);

    HEX_VERBOSE("ggml-hex: %s event-record : event %p seq 0x%x fence %p\n",
                sess->c_name(), (శూన్య *)hex_event, hex_event->seq, (శూన్య *)hex_event->fence_స్థానం);
}

స్థిర శూన్య ggml_backend_hexagon_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    స్వయంచాలక hex_event = static_cast<ggml_hexagon_event *>(event->context);

    ఒకవేళ (hex_event->seq == 0) {
        తిరిగి;
    }

    HEX_VERBOSE("ggml-hex: %s event-wait : event %p seq 0x%x fence %p\n",
                sess->c_name(), (శూన్య *)hex_event, hex_event->seq, (శూన్య *)hex_event->fence_స్థానం);

    // same physical NPU runs sequentially in FIFO order
    ఒకవేళ (sess->phys_idx == hex_event->sess->phys_idx) {
        ఒకవేళ (sess != hex_event->sess) {
            sess->కూడు_peer(hex_event->sess);
        }
        తిరిగి;
    }

    sess->clone_బఫరు(hex_event->fence_sess->fence_buf);
    sess->కూడు_peer(hex_event->sess);
    sess->వరుసలో_చేర్చు_fence(&hex_event->fence_tensor, hex_event->seq, /* wait = */ నిజం);
}

స్థిర శూన్య ggml_backend_hexagon_సమితి_tensor_async(ggml_backend_t backend, నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, tensor->బఫరు ? (పూర్ణాంకం) tensor->బఫరు->usage : -1);
    ggml_backend_tensor_సమితి(tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_hexagon_get_tensor_async(ggml_backend_t backend, స్థిరరాశి నిర్మాణం ggml_tensor * tensor, శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-async %s : data %p offset %zu size %zu usage %d\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, tensor->బఫరు ? (పూర్ణాంకం) tensor->బఫరు->usage : -1);
    sess->flush_sync();
    ఒకవేళ (sess->చివరి_error > HTP_స్థితి_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-async failed : dsp-error %s\n", sess->c_name(), స్థితి_గమ్యం_str(sess->చివరి_error));
    }
    ggml_backend_tensor_get(tensor, దత్తాంశం, స్థానభ్రంశం, size);
}

స్థిర శూన్య ggml_backend_hexagon_సమితి_tensor_2d_async(ggml_backend_t backend,
                                                     నిర్మాణం ggml_tensor * tensor,
                                                     స్థిరరాశి శూన్య * దత్తాంశం,
                                                     size_t స్థానభ్రంశం,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_దత్తాంశం) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s set-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం, tensor->బఫరు ? (పూర్ణాంకం) tensor->బఫరు->usage : -1);
    ggml_backend_tensor_సమితి_2d(tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

స్థిర శూన్య ggml_backend_hexagon_get_tensor_2d_async(ggml_backend_t backend,
                                                     స్థిరరాశి నిర్మాణం ggml_tensor * tensor,
                                                     శూన్య * దత్తాంశం,
                                                     size_t స్థానభ్రంశం,
                                                     size_t size,
                                                     size_t n_copies,
                                                     size_t stride_tensor,
                                                     size_t stride_దత్తాంశం) {
    స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(backend->context);
    HEX_VERBOSE("ggml-hex: %s get-tensor-2d-async %s : data %p offset %zu size %zu n_copies %zu stride_tensor %zu stride_data %zu usage %d\n",
                sess->c_name(), tensor->name, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం, tensor->బఫరు ? (పూర్ణాంకం) tensor->బఫరు->usage : -1);
    sess->flush_sync();
    ఒకవేళ (sess->చివరి_error > HTP_స్థితి_OK) {
        GGML_ABORT("ggml-hex: %s get-tensor-2d-async failed : dsp-error %s\n", sess->c_name(), స్థితి_గమ్యం_str(sess->చివరి_error));
    }
    ggml_backend_tensor_get_2d(tensor, దత్తాంశం, స్థానభ్రంశం, size, n_copies, stride_tensor, stride_దత్తాంశం);
}

స్థిర నిర్మాణం ggml_backend_i hexagon_backend_i = {
    /* .get_name                = */ ggml_backend_hexagon_name,
    /* .free                    = */ ggml_backend_hexagon_free,
    /* .set_tensor_async        = */ ggml_backend_hexagon_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hexagon_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_hexagon_సమితి_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_hexagon_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_hexagon_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hexagon_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_hexagon_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_hexagon_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_hexagon_event_wait,
    /* .graph_optimize          = */ ggml_backend_hexagon_లేఖాచిత్రం_optimize,
};

స్థిర ggml_guid_t ggml_backend_hexagon_guid() {
    స్థిర ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49,
                              0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
    తిరిగి &guid;
}

తార్కిక ggml_backend_is_hexagon(ggml_backend_t backend) {
    తిరిగి backend && backend->iface.get_name == ggml_backend_hexagon_name;
}

// device interface

స్థిర ggml_backend_t ggml_backend_hexagon_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    స్వయంచాలక sess    = dev_ctx->session();

    తిరిగి కొత్త ggml_backend{
        /* .guid      = */ ggml_backend_hexagon_guid(),
        /* .interface = */ hexagon_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_వాడని(params);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    తిరిగి dev_ctx->c_name();

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    తిరిగి "Hexagon";
    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_hexagon_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    *విడుదల  = opt_mbuf_మొత్తం;
    *మొత్తం = opt_mbuf_మొత్తం;

    GGML_వాడని(dev);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_వాడని(dev);
}

స్థిర శూన్య ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    ggml_backend_hexagon_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->caps = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ నిజం,
        /* .mmap_support          = */ అబద్ధం,
    };
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_hexagon_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    తిరిగి &dev_ctx->బఫరు_type;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_hexagon_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    ఒకవేళ (!opt_hostbuf) {
        తిరిగి ఖాళీ;
    }
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    తిరిగి &dev_ctx->host_బఫరు_type;
}

స్థిర తార్కిక ggml_hexagon_supported_cpy(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    నిర్మాణం htp_copy_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_cont(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    నిర్మాణం htp_copy_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_cpy_params(sess, op, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_repeat(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_వాడని(sess);
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    // Support f32 and f16
    ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) తిరిగి అబద్ధం;

    // src and dst must be the same type
    ఒకవేళ (src0->type != dst->type) తిరిగి అబద్ధం;

    // dst dims must be multiples of src dims
    ఒకవేళ (dst->ne[0] % src0->ne[0] != 0) తిరిగి అబద్ధం;
    ఒకవేళ (dst->ne[1] % src0->ne[1] != 0) తిరిగి అబద్ధం;
    ఒకవేళ (dst->ne[2] % src0->ne[2] != 0) తిరిగి అబద్ధం;
    ఒకవేళ (dst->ne[3] % src0->ne[3] != 0) తిరిగి అబద్ధం;

    // require contiguous tensors (no transposition)
    ఒకవేళ (ggml_is_transposed(src0) || ggml_is_transposed(dst)) తిరిగి అబద్ధం;

    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_hexagon_supported_concat(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    నిర్మాణం htp_concat_kernel_params kparams;
    తిరిగి ggml_hexagon_precompute_concat_params(sess, op, &kparams);
}

స్థిర తార్కిక ggml_hexagon_supported_fill(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్థిరరాశి నిర్మాణం ggml_tensor * dst = op;

    ఒకవేళ (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
    GGML_వాడని(sess);
}

స్థిర తార్కిక ggml_hexagon_supported_roll(స్థిరరాశి నిర్మాణం ggml_hexagon_session * sess, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    GGML_వాడని(sess);

    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = op->src[0];
    స్థిరరాశి నిర్మాణం ggml_tensor * dst  = op;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_are_same_shape(src0, dst)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (src0->nb[0] != ggml_type_కొలత(src0->type) || dst->nb[0] != ggml_type_కొలత(dst->type)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (!ggml_is_contiguous(dst)) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

// IMPORTANT: supports_op runs before buffer allocation and before tensor data is populated.
// Never check buffer usage, buffer properties, or repacking state (is_tiled) here.
// Support must depend only on static tensor properties (shape, type, strides).
స్థిర తార్కిక ggml_backend_hexagon_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    స్వయంచాలక sess    = dev_ctx->session();

    // reject ops that match the filter
    ఒకవేళ (opt_opfilter && std::regex_match(ggml_op_desc(op), *opt_opfilter)) {
        తిరిగి అబద్ధం;
    }

    తార్కిక supp = అబద్ధం;
    ఎంపిక (op->op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
            supp = నిజం;
            ఆపు;

        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_DIV:
            supp = ggml_hexagon_supported_ద్విమాన(sess, op);
            ఆపు;

        స్థితి GGML_OP_MUL_MAT:
            supp = ggml_hexagon_supported_mul_mat(sess, op);
            ఆపు;

        స్థితి GGML_OP_MUL_MAT_ID:
            supp = ggml_hexagon_supported_mul_mat_id(sess, op);
            ఆపు;

        స్థితి GGML_OP_కూడు_ID:
            supp = ggml_hexagon_supported_కూడు_id(sess, op);
            ఆపు;

        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_L2_NORM:
        స్థితి GGML_OP_RMS_NORM:
        స్థితి GGML_OP_SCALE:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_LEAKY_RELU:
            supp = ggml_hexagon_supported_unary(sess, op);
            ఆపు;

        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_LOG:
            supp = ggml_hexagon_supported_unary(sess, op);
            ఆపు;

        స్థితి GGML_OP_కూడిక:
            supp = ggml_hexagon_supported_కూడిక(sess, op);
            ఆపు;

        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            supp = ggml_hexagon_supported_కూడిక_అడ్డువరుసలు(sess, op);
            ఆపు;

        స్థితి GGML_OP_ARGMAX:
            supp = ggml_hexagon_supported_argmax(sess, op);
            ఆపు;

        స్థితి GGML_OP_POOL_2D:
            supp = ggml_hexagon_supported_pool_2d(sess, op);
            ఆపు;

        స్థితి GGML_OP_POOL_1D:
            supp = ggml_hexagon_supported_pool_1d(sess, op);
            ఆపు;

        స్థితి GGML_OP_SOFT_MAX:
            supp = ggml_hexagon_supported_softmax(sess, op);
            ఆపు;

        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(op)) {
                స్థితి GGML_UNARY_OP_NEG:
                స్థితి GGML_UNARY_OP_EXP:
                స్థితి GGML_UNARY_OP_SIGMOID:
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                స్థితి GGML_UNARY_OP_TANH:
                స్థితి GGML_UNARY_OP_ABS:
                స్థితి GGML_UNARY_OP_SILU:
                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                స్థితి GGML_UNARY_OP_RELU:
                స్థితి GGML_UNARY_OP_అడుగు:
                    supp = ggml_hexagon_supported_unary(sess, op);
                    ఆపు;
                అప్రమేయ:
                    supp = అబద్ధం;
                    ఆపు;
            }
            ఆపు;

        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                స్థితి GGML_GLU_OP_GEGLU:
                    supp = ggml_hexagon_supported_activations(sess, op);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                    supp = (op->src[0]->type == GGML_TYPE_F32) && ggml_hexagon_supported_activations(sess, op);
                    ఆపు;
                అప్రమేయ:
                    supp = అబద్ధం;
                    ఆపు;
            }
            ఆపు;

        స్థితి GGML_OP_ROPE:
            supp = ggml_hexagon_supported_rope(sess, op);
            ఆపు;

        స్థితి GGML_OP_FLASH_ATTN_EXT:
            supp = ggml_hexagon_supported_flash_attn_ext(sess, op);
            ఆపు;

        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            supp = ggml_hexagon_supported_సమితి_అడ్డువరుసలు(sess, op);
            ఆపు;

        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            supp = ggml_hexagon_supported_get_అడ్డువరుసలు(sess, op);
            ఆపు;

        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_CPY:
            supp = ggml_hexagon_supported_cpy(sess, op);
            ఆపు;

        స్థితి GGML_OP_CONT:
            supp = ggml_hexagon_supported_cont(sess, op);
            ఆపు;

        స్థితి GGML_OP_REPEAT:
            supp = ggml_hexagon_supported_repeat(sess, op);
            ఆపు;

        స్థితి GGML_OP_ARGSORT:
            supp = ggml_hexagon_supported_argsort(sess, op);
            ఆపు;

        స్థితి GGML_OP_శిఖరం_K:
            supp = ggml_hexagon_supported_శిఖరం_k(sess, op);
            ఆపు;

        స్థితి GGML_OP_SSM_CONV:
            supp = ggml_hexagon_supported_ssm_conv(sess, op);
            ఆపు;

        స్థితి GGML_OP_IM2_నిలువు:
            supp = ggml_hexagon_supported_im2col(sess, op);
            ఆపు;

        స్థితి GGML_OP_IM2_నిలువు_3_D:
            supp = ggml_hexagon_supported_im2col_3d(sess, op);
            ఆపు;

        స్థితి GGML_OP_GATED_DELTA_NET:
            supp = ggml_hexagon_supported_gated_delta_net(sess, op);
            ఆపు;

        స్థితి GGML_OP_CUMSUM:
            supp = ggml_hexagon_supported_cumsum(sess, op);
            ఆపు;

        స్థితి GGML_OP_CONCAT:
            supp = ggml_hexagon_supported_concat(sess, op);
            ఆపు;

        స్థితి GGML_OP_FILL:
            supp = ggml_hexagon_supported_fill(sess, op);
            ఆపు;

        స్థితి GGML_OP_DIAG:
            supp = ggml_hexagon_supported_diag(sess, op);
            ఆపు;

        స్థితి GGML_OP_SOLVE_TRI:
            supp = ggml_hexagon_supported_solve_tri(sess, op);
            ఆపు;

        స్థితి GGML_OP_TRI:
            supp = ggml_hexagon_supported_tri(sess, op);
            ఆపు;

        స్థితి GGML_OP_PAD:
            supp = ggml_hexagon_supported_pad(sess, op);
            ఆపు;

        స్థితి GGML_OP_ROLL:
            supp = ggml_hexagon_supported_roll(sess, op);
            ఆపు;

        అప్రమేయ:
            ఆపు;
    }

    ggml_hexagon_dump_op_supp(sess->name, op, supp);
    తిరిగి supp;
}

స్థిర తార్కిక ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);

    తార్కిక supp = (buft == &dev_ctx->host_బఫరు_type) || (buft == &dev_ctx->బఫరు_type);

    HEX_VERBOSE("ggml-hex: %s device-supports-buft %s %s\n", dev_ctx->c_name(), ggml_backend_buft_name(buft), supp ? "yes" : "no");
    తిరిగి supp;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_hexagon_device_i = {
    /* .get_name             = */ ggml_backend_hexagon_device_get_name,
    /* .get_description      = */ ggml_backend_hexagon_device_get_description,
    /* .get_memory           = */ ggml_backend_hexagon_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_hexagon_device_get_type,
    /* .get_props            = */ ggml_backend_hexagon_device_get_props,
    /* .init_backend         = */ ggml_backend_hexagon_device_init,
    /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ggml_backend_hexagon_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr = */ ఖాళీ,  // ggml_backend_hexagon_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_hexagon_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
    /* .offload_op           = */ ఖాళీ,  // ggml_backend_hexagon_device_offload_op,
    /* .event_new            = */ ggml_backend_hexagon_device_event_new,
    /* .event_free           = */ ggml_backend_hexagon_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hexagon_device_event_synchronize,
};

//** backend registry

ggml_hexagon_registry::ggml_hexagon_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev %zu\n", opt_ndev);

    GGML_LOG_INFO("ggml-hex: Hexagon Arch version v%d, DMA64 %s\n", opt_arch, opt_dma64 ? "enabled" : "disabled");

    // Create devices
    కొరకు (size_t i = 0; i < opt_ndev; i++) {
        స్థిరరాశి స్వయంచాలక & cfg = opt_device_configs[i];
        ఒకవేళ (cfg.mdev_సమూహం.ఖాళీలు()) {
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d)\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id);
        } లేకపోతే {
            std::string peers_str;
            కొరకు (స్థిరరాశి స్వయంచాలక & p : cfg.mdev_సమూహం) {
                ఒకవేళ (!peers_str.ఖాళీలు()) peers_str += ", ";
                peers_str += p.name + " (phys=" + std::to_string(p.physical_idx) + ")";
            }
            GGML_LOG_INFO("ggml-hex: device %zu: %s (phys=%d, virt=%d, domain=%s:%d) [mdev peers: %s]\n",
                          i, cfg.name.c_str(), cfg.physical_idx, cfg.virtual_idx, cfg.domain_name.c_str(), cfg.domain_id, peers_str.c_str());
        }
        devices[i].iface   = ggml_backend_hexagon_device_i;
        devices[i].reg     = reg;
        devices[i].context = కొత్త ggml_backend_hexagon_device_context(i, opt_device_configs[i], &devices[i]);
    }

}

ggml_hexagon_registry::~ggml_hexagon_registry() {
    GGML_LOG_INFO("ggml-hex: releasing registry\n");

    // Release devices
    కొరకు (size_t i = 0; i < opt_ndev; i++) {
        స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(devices[i].context);
        తొలగించు dev_ctx;
    }
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    తిరిగి "HTP";
    GGML_వాడని(reg);
}

స్థిర size_t ggml_backend_hexagon_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    తిరిగి opt_ndev;
    GGML_వాడని(reg);
}

స్థిర ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    స్వయంచాలక hreg = static_cast<ggml_hexagon_registry *>(reg->context);

    ఒకవేళ (సూచీ >= opt_ndev || !hreg->devices[సూచీ].context) {
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి &hreg->devices[సూచీ];
}

// ** communication context for tensor-split allreduce

స్థిర శూన్య * ggml_backend_hexagon_comm_init(ggml_backend_t * backends, size_t n_backends) {
    ఒకవేళ (n_backends < 2 || n_backends > 4) {
        తిరిగి శూన్యసూచిక;
    }

    కొరకు (size_t i = 0; i < n_backends; ++i) {
        ఒకవేళ (!ggml_backend_is_hexagon(backends[i])) {
            తిరిగి శూన్యసూచిక;
        }
    }

    కొరకు (size_t i = 0; i < n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        కొరకు (size_t j = i + 1; j < n_backends; j++) {
            స్వయంచాలక sess_j = static_cast<ggml_hexagon_session *>(backends[j]->context);
            ఒకవేళ (sess_i->phys_idx == sess_j->phys_idx) {
                తిరిగి శూన్యసూచిక;
            }
        }
    }

    స్వయంచాలక * ctx = కొత్త ggml_backend_hexagon_comm_context();
    ctx->backends.కేటాయించు(backends, backends + n_backends);
    ctx->n_backends = n_backends;

    స్థిర ggml_hexagon_tensor_extra fence_extra { {}, 0, GGML_HEXAGON_TENSOR_FENCE };
    కొరకు (size_t i = 0; i < n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(backends[i]->context);
        ctx->fence_slots[i] = (అస్థిర uint32_t *) sess_i->alloc_fence(1);
        ctx->fence_tensors[i] = {};
        ctx->fence_tensors[i].బఫరు = &sess_i->fence_buf->backend_బఫరు;
        ctx->fence_tensors[i].extra  = &fence_extra;
        ctx->fence_tensors[i].దత్తాంశం   = (శూన్య *) ctx->fence_slots[i];
        ctx->fence_tensors[i].type   = GGML_TYPE_I32;
        ctx->fence_tensors[i].ne[0]  = 4;
        ctx->fence_tensors[i].ne[1]  = 1;
        ctx->fence_tensors[i].ne[2]  = 1;
        ctx->fence_tensors[i].ne[3]  = 1;
        ctx->fence_tensors[i].nb[0]  = పరిమాణం(int32_t);
        ctx->fence_tensors[i].nb[1]  = పరిమాణం(int32_t);
        ctx->fence_tensors[i].nb[2]  = పరిమాణం(int32_t);
        ctx->fence_tensors[i].nb[3]  = పరిమాణం(int32_t);
        ctx->fence_tensors[i].op     = GGML_OP_NONE;
    }

    తిరిగి ctx;
}

స్థిర శూన్య ggml_backend_hexagon_comm_free(శూన్య * comm_ctx_v) {
    ఒకవేళ (!comm_ctx_v) తిరిగి;
    స్వయంచాలక * ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    కొరకు (size_t i = 0; i < ctx->n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(ctx->backends[i]->context);
        sess_i->free_fence((శూన్య *) ctx->fence_slots[i], 1);
    }
    తొలగించు ctx;
}

స్థిర తార్కిక ggml_backend_hexagon_comm_allreduce_tensor(శూన్య * comm_ctx_v, నిర్మాణం ggml_tensor ** tensors) {
    ఒకవేళ (opt_ar_select == 0 || !comm_ctx_v) తిరిగి అబద్ధం;
    స్వయంచాలక * comm_ctx = static_cast<ggml_backend_hexagon_comm_context *>(comm_ctx_v);
    స్థిరరాశి size_t n_backends = comm_ctx->n_backends;

    ఒకవేళ (n_backends < 2 || n_backends > 4) తిరిగి అబద్ధం;

    కొరకు (size_t i = 0; i < n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        కొరకు (size_t j = i + 1; j < n_backends; j++) {
            స్వయంచాలక sess_j = static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context);
            ఒకవేళ (sess_i->phys_idx == sess_j->phys_idx) {
                తిరిగి అబద్ధం;
            }
        }
    }

    కొరకు (size_t i = 0; i < n_backends; i++) {
        ఒకవేళ (!tensors[i] || !tensors[i]->బఫరు || !ggml_backend_బఫరు_is_hexagon(tensors[i]->బఫరు)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (tensors[i]->type != tensors[0]->type) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (!ggml_is_contiguous(tensors[i])) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            తిరిగి అబద్ధం;
        }
    }

    ఒకవేళ (tensors[0]->type != GGML_TYPE_F16 && tensors[0]->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }

    కొరకు (size_t r = 0; r < n_backends; r++) {
        స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        నిర్మాణం htp_allreduce_kernel_params kparams;
        ఒకవేళ (!ggml_hexagon_precompute_allreduce_params(sess, tensors[r], (uint32_t) r, (uint32_t) n_backends, అబద్ధం, అబద్ధం, /*is_shard_ok=*/ అబద్ధం, &kparams)) {
            తిరిగి అబద్ధం;
        }
    }

    uint32_t max_seq = static_cast<ggml_hexagon_session *>(comm_ctx->backends[0]->context)->fence_seq;
    కొరకు (size_t i = 1; i < n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        ఒకవేళ ((int32_t)(sess_i->fence_seq - max_seq) > 0) {
            max_seq = sess_i->fence_seq;
        }
    }
    ఒకవేళ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_నమోదు = max_seq;
    ఒకవేళ (++max_seq == 0) max_seq = 1;
    uint32_t fence_seq_exit  = max_seq;

    కొరకు (size_t i = 0; i < n_backends; i++) {
        స్వయంచాలక sess_i = static_cast<ggml_hexagon_session *>(comm_ctx->backends[i]->context);
        sess_i->fence_seq = max_seq;
    }

    std::vector<స్థిరరాశి ggml_tensor *> దత్తాంశం_tensors(n_backends);
    std::vector<స్థిరరాశి ggml_tensor *> sync_tensors(n_backends);
    కొరకు (size_t i = 0; i < n_backends; i++) {
        దత్తాంశం_tensors[i] = tensors[i];
        sync_tensors[i] = &comm_ctx->fence_tensors[i];
    }

    కొరకు (size_t r = 0; r < n_backends; r++) {
        స్వయంచాలక sess = static_cast<ggml_hexagon_session *>(comm_ctx->backends[r]->context);
        sess->వరుసలో_చేర్చు_allreduce(tensors[r], దత్తాంశం_tensors, sync_tensors, (uint32_t) r, (uint32_t) n_backends, fence_seq_నమోదు, fence_seq_exit);
        కొరకు (size_t j = 0; j < n_backends; j++) {
            ఒకవేళ (r != j) {
                sess->కూడు_peer(static_cast<ggml_hexagon_session *>(comm_ctx->backends[j]->context));
            }
        }
    }

    తిరిగి నిజం;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_hexagon_విభజించు_బఫరు_type(పూర్ణాంకం main_device, స్థిరరాశి దశాంశం * tensor_విభజించు) {
    GGML_వాడని(tensor_విభజించు);
    స్వయంచాలక reg = ggml_backend_hexagon_reg();
    స్వయంచాలక dev = ggml_backend_reg_dev_get(reg, main_device);
    ఒకవేళ (!dev) {
        dev = ggml_backend_reg_dev_get(reg, 0);
    }
    ఒకవేళ (!dev) తిరిగి శూన్యసూచిక;
    స్వయంచాలక dev_ctx = static_cast<ggml_backend_hexagon_device_context *>(dev->context);
    తిరిగి &dev_ctx->బఫరు_type;
}

స్థిర శూన్య * ggml_backend_hexagon_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    GGML_వాడని(reg);
    ఒకవేళ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        తిరిగి (శూన్య *) ggml_backend_hexagon_విభజించు_బఫరు_type;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_comm_init") == 0) {
        తిరిగి (శూన్య *) ggml_backend_hexagon_comm_init;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_comm_free") == 0) {
        తిరిగి (శూన్య *) ggml_backend_hexagon_comm_free;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        తిరిగి (శూన్య *) ggml_backend_hexagon_comm_allreduce_tensor;
    }
    తిరిగి ఖాళీ;
}

మూస<రకం_పేరు T> std::vector<T> str_గమ్యం_vec(స్థిరరాశి అక్షరం* str) {
    std::stringstream ss(str);
    std::vector<T> v;
    std::string    t;

    అయితే (std::getline(ss, t, ',')) {
        v.push_back(std::stoul(t, శూన్యసూచిక, 0));
    }

    తిరిగి v;
}

మూస<రకం_పేరు T, పూర్ణాంకం పునాది_2=10> std::string vec_గమ్యం_str(std::vector<T> v) {
    std::stringstream ss;
    ss << std::setbase(పునాది_2) << std::showbase;
    కొరకు (స్వయంచాలక i : v) { ss << i << ','; }
    స్వయంచాలక str = ss.str(); str.pop_back(); // drop last comma
    తిరిగి str;
}

స్థిర శూన్య ggml_hexagon_resolve_device_domain(ggml_hexagon_device_config & cfg, తార్కిక discovery_supported, స్థిరరాశి std::unordered_map<పూర్ణాంకం, fastrpc_domain> & cdsp_map) {
    ఒకవేళ (discovery_supported) {
        స్వయంచాలక it = cdsp_map.find(cfg.physical_idx);
        ఒకవేళ (it != cdsp_map.end()) {
            cfg.domain_id   = it->second.id;
            cfg.domain_name = it->second.name;
        } లేకపోతే {
            GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not found on device (%zu CDSP core(s) available)\n",
                           cfg.physical_idx, cdsp_map.size());
            cfg.domain_id   = -1;
            cfg.domain_name = "";
        }
    } లేకపోతే {
        ఎంపిక (cfg.physical_idx) {
            స్థితి 0:
                cfg.domain_id   = 3;
                cfg.domain_name = CDSP_DOMAIN_NAME;
                ఆపు;
            స్థితి 1:
                cfg.domain_id   = 4;
                cfg.domain_name = "cdsp1";
                ఆపు;
            అప్రమేయ:
                GGML_LOG_ERROR("ggml-hex: physical CDSP core %d not supported without dynamic discovery\n",
                               cfg.physical_idx);
                cfg.domain_id   = -1;
                cfg.domain_name = "";
                ఆపు;
        }
    }
    కొరకు (స్వయంచాలక & sub_cfg : cfg.mdev_సమూహం) {
        ggml_hexagon_resolve_device_domain(sub_cfg, discovery_supported, cdsp_map);
    }
}

// Enumerate NPU (aka CDSP) domains via FASTRPC_GET_DOMAINS if supported,
// and populate domain_id and domain_name for all configured devices.
స్థిర శూన్య ggml_hexagon_discover_devices() {
    std::unordered_map<పూర్ణాంకం, fastrpc_domain> cdsp_map;
    తార్కిక discovery_supported = అబద్ధం;

    system_req_payload domain_info = {};
    domain_info.id              = FASTRPC_GET_DOMAINS;
    domain_info.sys.domains     = శూన్యసూచిక;
    domain_info.sys.max_domains = 0;
    domain_info.sys.flags       = DOMAINS_LIST_FLAGS_సమితి_TYPE(0, FASTRPC_NSP);

    పూర్ణాంకం err = remote_system_request(&domain_info);
    ఒకవేళ (err == AEE_SUCCESS && domain_info.sys.అంకె_domains > 0) {
        std::vector<fastrpc_domain> domains(domain_info.sys.అంకె_domains);
        domain_info.sys.domains     = domains.దత్తాంశం();
        domain_info.sys.max_domains = (పూర్ణాంకం) domains.size();

        err = remote_system_request(&domain_info);
        ఒకవేళ (err == AEE_SUCCESS) {
            discovery_supported = నిజం;
            స్థిరరాశి పూర్ణాంకం n_domains = std::min(domain_info.sys.అంకె_domains, (పూర్ణాంకం) domains.size());
            కొరకు (పూర్ణాంకం i = 0; i < n_domains; i++) {
                GGML_LOG_INFO("ggml-hex: FASTRPC_GET_DOMAINS[%d]: type %d id %d name '%s' status %d instance-id %d\n",
                              i, (పూర్ణాంకం) domains[i].type, domains[i].id, domains[i].name, domains[i].స్థితి_2, domains[i].instance_id);
                ఒకవేళ (domains[i].type != FASTRPC_NSP) {
                    GGML_LOG_DEBUG("ggml-hex:   skipping non-CDSP domain (type=%d)\n", (పూర్ణాంకం) domains[i].type);
                    కొనసాగు;
                }
                ఒకవేళ (!domains[i].స్థితి_2) {
                    GGML_LOG_WARN("ggml-hex:   skipping CDSP domain id=%d (status=down)\n", domains[i].id);
                    కొనసాగు;
                }
                cdsp_map[domains[i].instance_id] = domains[i];
                GGML_LOG_INFO("ggml-hex: using CDSP domain: instance-id %d id %d name '%s'\n",
                              domains[i].instance_id, domains[i].id, domains[i].name);
            }
        } లేకపోతే {
            GGML_LOG_WARN("ggml-hex: FASTRPC_GET_DOMAINS fetch failed (0x%x), using static CDSP domains\n", (గుర్తులేని) err);
        }
    } లేకపోతే ఒకవేళ (err != AEE_SUCCESS) {
        GGML_LOG_DEBUG("ggml-hex: FASTRPC_GET_DOMAINS query failed (0x%x), using static CDSP domains\n", (గుర్తులేని) err);
    }

    // Populate domain IDs and names for all configured devices
    కొరకు (size_t i = 0; i < opt_ndev; i++) {
        ggml_hexagon_resolve_device_domain(opt_device_configs[i], discovery_supported, cdsp_map);
    }
}

స్థిర శూన్య ggml_hexagon_init(ggml_backend_reg * reg) {
    // Basic sanity checks to make sure definitions match
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q4_0 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q4_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q4_1 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q4_1,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q8_0 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q8_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_MXFP4 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_MXFP4,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_IQ4_NL == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_IQ4_NL,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q4_K == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q4_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q5_K == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q5_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q6_K == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q6_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q3_K == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q3_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q2_K == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q2_K,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q1_0 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q1_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_Q2_0 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_Q2_0,
                  "please update hexagon_type to match ggml_type");
    static_assert((గుర్తులేని పూర్ణాంకం) HTP_TYPE_BF16 == (గుర్తులేని పూర్ణాంకం) GGML_TYPE_BF16,
                  "please update hexagon_type to match ggml_type");

    స్థిరరాశి అక్షరం * str_verbose  = getenv("GGML_HEXAGON_VERBOSE");
    స్థిరరాశి అక్షరం * str_opbatch  = getenv("GGML_HEXAGON_OPBATCH");
    స్థిరరాశి అక్షరం * str_opqueue  = getenv("GGML_HEXAGON_OPQUEUE");
    స్థిరరాశి అక్షరం * str_oppoll   = getenv("GGML_HEXAGON_OPPOLL");
    స్థిరరాశి అక్షరం * str_opfusion = getenv("GGML_HEXAGON_OPFUSION");
    స్థిరరాశి అక్షరం * str_rt_repack = getenv("GGML_HEXAGON_RT_REPACK");
    స్థిరరాశి అక్షరం * str_opfilter = getenv("GGML_HEXAGON_OPFILTER");
    స్థిరరాశి అక్షరం * str_profile  = getenv("GGML_HEXAGON_PROFILE");
    స్థిరరాశి అక్షరం * str_etm      = getenv("GGML_HEXAGON_ETM");
    స్థిరరాశి అక్షరం * str_nhvx     = getenv("GGML_HEXAGON_NHVX");
    స్థిరరాశి అక్షరం * str_nhmx     = getenv("GGML_HEXAGON_NHMX");
    స్థిరరాశి అక్షరం * str_mm_select = getenv("GGML_HEXAGON_MM_SELECT");
    స్థిరరాశి అక్షరం * str_fa_select = getenv("GGML_HEXAGON_FA_SELECT");
    స్థిరరాశి అక్షరం * str_fa_తల_విభజించు = getenv("GGML_HEXAGON_FA_HEAD_SPLIT");
    స్థిరరాశి అక్షరం * str_gdn_select = getenv("GGML_HEXAGON_GDN_SELECT");
    స్థిరరాశి అక్షరం * str_ar_select = getenv("GGML_HEXAGON_AR_SELECT");
    స్థిరరాశి అక్షరం * str_ar_scatter = getenv("GGML_HEXAGON_AR_SCATTER");
    స్థిరరాశి అక్షరం * str_ndev     = getenv("GGML_HEXAGON_NDEV");
    స్థిరరాశి అక్షరం * str_arch     = getenv("GGML_HEXAGON_ARCH");
    స్థిరరాశి అక్షరం * str_vmem     = getenv("GGML_HEXAGON_VMEM");
    స్థిరరాశి అక్షరం * str_mbuf     = getenv("GGML_HEXAGON_MBUF");
    స్థిరరాశి అక్షరం * str_optrace  = getenv("GGML_HEXAGON_OPTRACE");
    స్థిరరాశి అక్షరం * str_hostbuf  = getenv("GGML_HEXAGON_HOSTBUF");
    స్థిరరాశి అక్షరం * str_dma64    = getenv("GGML_HEXAGON_DMA64");
    స్థిరరాశి అక్షరం * str_power_బహుళకం = getenv("GGML_HEXAGON_POWER_MODE");

    // Init Arch first since it affects other defaults
    ఒకవేళ (!str_arch) {
        పూర్ణాంకం err = htpdrv_get_arch(CDSP_DOMAIN_ID, &opt_arch);
        ఒకవేళ (err != 0) {
            GGML_LOG_ERROR("ggml-hex: failed to query HTP version (err %d) defaulting to v73\n", err);
            opt_arch = 73;
        } లేకపోతే {
            ఒకవేళ (opt_arch < 73) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is under supported range, capping at v73\n", opt_arch);
                opt_arch = 73;
            } లేకపోతే ఒకవేళ (opt_arch > 85) {
                GGML_LOG_WARN("ggml-hex: Hexagon arch v%d is over supported range, capping at v85\n", opt_arch);
                opt_arch = 85;
            }
        }
    } లేకపోతే {
        ఒకవేళ (str_arch[0] == 'v' || str_arch[0] == 'V') {
            str_arch++;
        }
        opt_arch = strtoul(str_arch, ఖాళీ, 0);
    }

    size_t MiB = 1024 * 1024;

    // Update vmem default
    opt_vmem  = opt_arch >= 75 ? HTP_OP_MAX_VMEM_DEFAULT : 3000 * MiB;
    opt_dma64 = opt_arch > 79 && (!str_dma64 || atoi(str_dma64) != 0);

    స్వయంచాలక RE_ICASE = std::regex_constants::icase;

    opt_opfilter  = str_opfilter ? కొత్త std::regex(str_opfilter, RE_ICASE) : ఖాళీ;
    opt_verbose   = str_verbose  ? atoi(str_verbose)                      : 0;
    opt_opbatch   = str_opbatch  ? strtoul(str_opbatch, ఖాళీ, 0)          : opt_opbatch;
    opt_opqueue   = str_opqueue  ? strtoul(str_opqueue, ఖాళీ, 0)          : opt_opqueue;
    opt_optrace   = str_optrace  ? strtoul(str_optrace, ఖాళీ, 0)          : (opt_opbatch * 256);
    opt_oppoll    = str_oppoll   ? strtoul(str_oppoll,  ఖాళీ, 0)          : opt_oppoll;
    opt_opfusion  = str_opfusion ? atoi(str_opfusion)                     : opt_opfusion;
    opt_rt_repack = str_rt_repack ? atoi(str_rt_repack)                   : opt_rt_repack;
    opt_profile   = str_profile  ? atoi(str_profile)                      : 0;
    opt_etm       = str_etm      ? atoi(str_etm)                          : 0;
    ఒకవేళ (str_power_బహుళకం) {
        పూర్ణాంకం m = htp_power_బహుళకం_from_str(str_power_బహుళకం);
        ఒకవేళ (m < 0) {
            GGML_LOG_WARN("ggml-hex: unknown GGML_HEXAGON_POWER_MODE '%s', falling back to 'burst'\n", str_power_బహుళకం);
        } లేకపోతే {
            opt_power_బహుళకం = m;
        }
    }
    opt_nhvx      = str_nhvx     ? strtoul(str_nhvx, ఖాళీ, 0)             : opt_nhvx;
    opt_nhmx      = str_nhmx     ? atoi(str_nhmx)                         : opt_nhmx;
    opt_mm_select = str_mm_select ? atoi(str_mm_select)                   : opt_mm_select;
    opt_fa_select = str_fa_select ? atoi(str_fa_select)                   : opt_fa_select;
    opt_fa_తల_విభజించు = str_fa_తల_విభజించు ? atoi(str_fa_తల_విభజించు)       : opt_fa_తల_విభజించు;
    opt_gdn_select    = str_gdn_select    ? atoi(str_gdn_select)          : opt_gdn_select;
    opt_ar_select     = str_ar_select     ? atoi(str_ar_select)           : opt_ar_select;
    opt_ar_scatter    = str_ar_scatter    ? atoi(str_ar_scatter)          : opt_ar_scatter;

    ఒకవేళ (str_mbuf) {
        స్థిరరాశి అక్షరం * p = str_mbuf;
        కొరకు (పూర్ణాంకం idx = 0; idx < 3 && p && *p; idx++) {
            అయితే (*p == ' ') p++;
            స్థిరరాశి అక్షరం * comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : అక్షరశ్రేణి_పొడవు(p);
            అయితే (len > 0 && p[len - 1] == ' ') len--;
            ఒకవేళ (len > 0) {
                std::string token(p, len);
                ఒకవేళ (idx == 0) opt_mbuf_dyn    = parse_కొలత(token.c_str());
                ఒకవేళ (idx == 1) opt_mbuf_static = parse_కొలత(token.c_str());
                ఒకవేళ (idx == 2) opt_mbuf_మొత్తం  = parse_కొలత(token.c_str());
            }
            ఒకవేళ (!comma) ఆపు;
            p = comma + 1;
        }
    }
    opt_vmem    = str_vmem    ? parse_కొలత(str_vmem)   : opt_vmem;
    opt_hostbuf = str_hostbuf ? atoi(str_hostbuf) != 0 : opt_hostbuf;

    // Parse device configuration
    స్థిరరాశి అక్షరం * str_devices  = getenv("GGML_HEXAGON_DEVICES");
    ఒకవేళ (!str_devices && str_ndev && str_ndev[0] != '\0') {
        GGML_LOG_WARN("DEPRECATED: GGML_HEXAGON_NDEV is deprecated. use GGML_HEXAGON_DEVICES instead\n");
        str_devices = str_ndev;
    }

    ఒకవేళ (str_devices && str_devices[0] != '\0') {
        తార్కిక is_single_సంఖ్య = నిజం;
        కొరకు (పూర్ణాంకం i = 0; str_devices[i] != '\0'; i++) {
            ఒకవేళ (!అంకెనా((గుర్తులేని అక్షరం)str_devices[i])) {
                is_single_సంఖ్య = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (is_single_సంఖ్య) {
            పూర్ణాంకం n = atoi(str_devices);
            ఒకవేళ (n < 1) n = 1;
            ఒకవేళ (n > GGML_HEXAGON_MAX_SESSIONS) n = GGML_HEXAGON_MAX_SESSIONS;
            opt_ndev = n;
            కొరకు (size_t i = 0; i < opt_ndev; i++) {
                opt_device_configs[i].physical_idx = 0;
                opt_device_configs[i].virtual_idx  = (పూర్ణాంకం)i;
                opt_device_configs[i].name         = "HTP" + std::to_string(i);
                opt_device_configs[i].mdev_సమూహం.clear();
            }
        } లేకపోతే {
            std::string s_devices(str_devices);
            std::vector<std::string> వస్తువులు;
            std::string curr_అంశం;
            పూర్ణాంకం bracket_లోతు = 0;
            కొరకు (అక్షరం ch : s_devices) {
                ఒకవేళ (ch == '[') {
                    bracket_లోతు++;
                    curr_అంశం += ch;
                } లేకపోతే ఒకవేళ (ch == ']') {
                    ఒకవేళ (bracket_లోతు > 0) bracket_లోతు--;
                    curr_అంశం += ch;
                } లేకపోతే ఒకవేళ (ch == ',' && bracket_లోతు == 0) {
                    size_t s = curr_అంశం.find_first_not_of(" \t\r\n");
                    size_t e = curr_అంశం.find_last_not_of(" \t\r\n");
                    ఒకవేళ (s != std::string::npos) {
                        వస్తువులు.push_back(curr_అంశం.substr(s, e - s + 1));
                    }
                    curr_అంశం.clear();
                } లేకపోతే {
                    curr_అంశం += ch;
                }
            }
            size_t s = curr_అంశం.find_first_not_of(" \t\r\n");
            size_t e = curr_అంశం.find_last_not_of(" \t\r\n");
            ఒకవేళ (s != std::string::npos) {
                వస్తువులు.push_back(curr_అంశం.substr(s, e - s + 1));
            }

            opt_ndev = 0;
            కొరకు (స్థిరరాశి స్వయంచాలక & అంశం : వస్తువులు) {
                size_t b_open  = అంశం.find('[');
                size_t b_close = అంశం.rfind(']');

                ఒకవేళ (b_open != std::string::npos && b_close != std::string::npos && b_close > b_open) {
                    // Grouped / composite syntax: Name[phys_spec:virt] or Name[phys_spec]
                    std::string dev_name = అంశం.substr(0, b_open);
                    std::string content  = అంశం.substr(b_open + 1, b_close - b_open - 1);

                    పూర్ణాంకం virt = 0;
                    std::string phys_spec = content;
                    size_t colon_pos = content.find(':');
                    ఒకవేళ (colon_pos != std::string::npos) {
                        phys_spec = content.substr(0, colon_pos);
                        ప్రయత్నించు {
                            virt = std::stoi(content.substr(colon_pos + 1));
                        } పట్టుకో (...) {
                            virt = 0;
                        }
                    } లేకపోతే {
                        size_t dev_colon = dev_name.find(':');
                        ఒకవేళ (dev_colon != std::string::npos) {
                            ప్రయత్నించు {
                                virt = std::stoi(dev_name.substr(dev_colon + 1));
                            } పట్టుకో (...) {
                                virt = 0;
                            }
                        }
                    }

                    // Parse physical indices from phys_spec (e.g. 0-1, 0,1, 0-3, etc.)
                    std::vector<పూర్ణాంకం> phys_list;
                    std::stringstream pss(phys_spec);
                    std::string p_part;
                    అయితే (std::getline(pss, p_part, ',')) {
                        size_t ps = p_part.find_first_not_of(" \t\r\n");
                        size_t pe = p_part.find_last_not_of(" \t\r\n");
                        ఒకవేళ (ps == std::string::npos) కొనసాగు;
                        p_part = p_part.substr(ps, pe - ps + 1);

                        size_t dash_pos = p_part.find('-');
                        ఒకవేళ (dash_pos != std::string::npos) {
                            ప్రయత్నించు {
                                పూర్ణాంకం p_ప్రారంభం = std::stoi(p_part.substr(0, dash_pos));
                                పూర్ణాంకం p_చివర   = std::stoi(p_part.substr(dash_pos + 1));
                                కొరకు (పూర్ణాంకం p = p_ప్రారంభం; p <= p_చివర; p++) {
                                    ఒకవేళ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                        phys_list.push_back(p);
                                    }
                                }
                            } పట్టుకో (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical range in '%s'\n", p_part.c_str());
                            }
                        } లేకపోతే {
                            ప్రయత్నించు {
                                పూర్ణాంకం p = std::stoi(p_part);
                                ఒకవేళ (std::find(phys_list.begin(), phys_list.end(), p) == phys_list.end()) {
                                    phys_list.push_back(p);
                                }
                            } పట్టుకో (...) {
                                GGML_LOG_WARN("ggml-hex: failed to parse physical index in '%s'\n", p_part.c_str());
                            }
                        }
                    }

                    ఒకవేళ (phys_list.ఖాళీలు()) {
                        phys_list.push_back(0);
                    }

                    ఒకవేళ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        స్వయంచాలక & cfg = opt_device_configs[opt_ndev];
                        cfg.name         = dev_name;
                        cfg.physical_idx = phys_list[0];
                        cfg.virtual_idx  = virt;
                        cfg.mdev_సమూహం.clear();

                        కొరకు (size_t k = 1; k < phys_list.size(); k++) {
                            ggml_hexagon_device_config sub_cfg;
                            sub_cfg.physical_idx = phys_list[k];
                            sub_cfg.virtual_idx  = virt;
                            sub_cfg.name         = "HTP" + std::to_string(phys_list[k]) + ":" + std::to_string(virt);
                            cfg.mdev_సమూహం.push_back(sub_cfg);
                        }
                        opt_ndev++;
                    } లేకపోతే {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, అంశం.c_str());
                    }
                } లేకపోతే ఒకవేళ (అంశం.rfind("HTP", 0) == 0) {
                    std::string rest = అంశం.substr(3);
                    size_t colon_pos = rest.find(':');
                    పూర్ణాంకం phys = 0;
                    పూర్ణాంకం virt = 0;
                    ప్రయత్నించు {
                        ఒకవేళ (colon_pos == std::string::npos) {
                            phys = std::stoi(rest);
                            virt = 0;
                        } లేకపోతే {
                            phys = std::stoi(rest.substr(0, colon_pos));
                            virt = std::stoi(rest.substr(colon_pos + 1));
                        }
                    } పట్టుకో (...) {
                        GGML_LOG_WARN("ggml-hex: failed to parse device index in '%s'\n", అంశం.c_str());
                        కొనసాగు;
                    }

                    ఒకవేళ (opt_ndev < GGML_HEXAGON_MAX_SESSIONS) {
                        opt_device_configs[opt_ndev].physical_idx = phys;
                        opt_device_configs[opt_ndev].virtual_idx  = virt;
                        opt_device_configs[opt_ndev].name         = colon_pos == std::string::npos
                            ? "HTP" + std::to_string(phys)
                            : "HTP" + std::to_string(phys) + ":" + std::to_string(virt);
                        opt_device_configs[opt_ndev].mdev_సమూహం.clear();
                        opt_ndev++;
                    } లేకపోతే {
                        GGML_LOG_WARN("ggml-hex: max sessions limit reached (%d), ignoring device %s\n", GGML_HEXAGON_MAX_SESSIONS, అంశం.c_str());
                    }
                } లేకపోతే {
                    GGML_LOG_WARN("ggml-hex: invalid device name format '%s', must start with HTP\n", అంశం.c_str());
                }
            }
        }
    } లేకపోతే {
        opt_ndev = 1;
        opt_device_configs[0].physical_idx = 0;
        opt_device_configs[0].virtual_idx  = 0;
        opt_device_configs[0].name         = "HTP0";
        opt_device_configs[0].mdev_సమూహం.clear();
    }

#ఒకవేళ defined(__ANDROID__)
    ఒకవేళ (opt_arch < 75) {
        opt_ndev = 1;
        GGML_LOG_WARN("ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.\n");
    }
#అయితే_ముగింపు

    // Resolve domain info for all configured devices
    ggml_hexagon_discover_devices();

    ఒకవేళ (str_profile) {
        opt_pmu_evt = [&]() -> std::vector<uint32_t> {
            స్వయంచాలక v  = str_గమ్యం_vec<uint32_t>(str_profile);
            ఎంపిక (v.size()) {
                స్థితి 1:  opt_profile = v[0]; తిరిగి opt_pmu_evt; // mode with default pmu events
                స్థితి 8:  opt_profile = 2;    తిరిగి v;           // mode with custom  pmu events
                అప్రమేయ: opt_profile = 0;    తిరిగి {};          // garbage input
            }}();
        ఒకవేళ (opt_profile == 1) opt_pmu_evt = {};
        GGML_LOG_INFO("ggml-hex: Profiling mode %u : pmu-evt [ %s ]\n", opt_profile,
                vec_గమ్యం_str<uint32_t, 16>(opt_pmu_evt).c_str());
    }

    reg->context = కొత్త ggml_hexagon_registry(reg);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_hexagon_reg_i = {
    /* .get_name         = */ ggml_backend_hexagon_reg_get_name,
    /* .get_device_count = */ ggml_backend_hexagon_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_hexagon_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hexagon_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_hexagon_reg(శూన్య) {
    స్థిర తార్కిక initialized = అబద్ధం;

    స్థిర ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
                                    /* .iface       = */ ggml_backend_hexagon_reg_i,
                                    /* .context     = */ ఖాళీ };

    {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);
        ఒకవేళ (!initialized) {
            స్వయంచాలక nErr = htpdrv_init();
            ఒకవేళ (nErr != AEE_SUCCESS) {
                తిరిగి ఖాళీ;
            }

            ggml_hexagon_init(&reg);
        }

        initialized = నిజం;
    }

    తిరిగి &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
