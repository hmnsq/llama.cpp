#సూచన once

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "openvino/decoder.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstring>
#అన్తర్భావ <map>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/partial_shape.hpp>
#అన్తర్భావ <optional>
#అన్తర్భావ <set>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

నిర్మాణం ModelParams {
    పూర్ణాంకం ctx = -1;
    పూర్ణాంకం ctx_చొప్పున_seq = -1;
    పూర్ణాంకం ctx_చొప్పున_seq_swa = -1;
    పూర్ణాంకం n_seq = 1;
    పూర్ణాంకం n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<పూర్ణాంకం, పూర్ణాంకం> n_heads_kv_చొప్పున_layer;
    పూర్ణాంకం తల_కొలత = -1;
    పూర్ణాంకం state_కొలత = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    పూర్ణాంకం n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::సదిశ<పూర్ణాంకం> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    స్థిరరాశి ggml_tensor * swa_ముసుగు = nullptr;

    std::సదిశ<std::string> kv_పేర్లు;
    size_t kv_బఫరు_ctx_id = 0;

    bool same_rope_params(స్థిరరాశి ModelParams & other) స్థిరరాశి {
        తిరిగి mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, పరిమాణం(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(స్థిరరాశి ModelParams & other) స్థిరరాశి {
        తిరిగి same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(స్థిరరాశి ModelParams & other) స్థిరరాశి {
        తిరిగి same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_బఫరు_మారింది(స్థిరరాశి ModelParams & other) స్థిరరాశి { తిరిగి kv_బఫరు_ctx_id != other.kv_బఫరు_ctx_id; }
};

నిర్మాణం ComputeParams {
    పూర్ణాంకం n_seq_చురుకైన = 1;
    పూర్ణాంకం seq_చురుకైన_ప్రారంభం = 0;
    పూర్ణాంకం attention_కొలత = -1;
    పూర్ణాంకం attention_కొలత_swa = -1;
    పూర్ణాంకం attention_కొలత_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    పూర్ణాంకం swa_కిటికీ = -1;
    పూర్ణాంకం ఆదానం_len = -1;
    పూర్ణాంకం token_len_చొప్పున_seq = -1;
    పూర్ణాంకం past_kv_len = -1;
    పూర్ణాంకం output_len = 1;

    పూర్ణాంకం cache_rs_reset_idx = -1;
    పూర్ణాంకం cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    పూర్ణాంకం s_copy_చురుకైన_స్థానం_len = -1;
    // SSM/DeltaNet models otionally reorder slots of state cache, to make the active slots contiguous
    // leaf_5 is the inp->s_copy in llama-graph.cpp, eg if there are 8 slots in total and slot 3 and 7
    // are active in the current batch, leaf_5 will be [3, 7, 5, 6, 4]
    //  6: [     2,     1,     1,     1] VIEW                  (view)
    //      [     2,     1,     1,     1]            0: NONE        leaf_5
    //  7: [ 18432,     2,     1,     1] GET_ROWS             conv_states-0
    //      [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    //      [     2,     1,     1,     1]            1: VIEW         (view)
    //  8: [     0,     1,     1,     1] VIEW                  (view)
    //      [     2,     1,     1,     1]            0: NONE        leaf_5
    //  9: [ 18432,     0,     1,     1] GET_ROWS             node_9
    //      [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    //      [     0,     1,     1,     1]            1: VIEW         (view)
    // 10: [ 18432,     0,     1,     1] VIEW                 cache_r_l0 (view)
    //      [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 11: [ 18432,     0,     1,     1] CPY                  cache_r_l0 (view) (copy of )
    //      [ 18432,     0,     1,     1]            0: GET_ROWS    node_9
    //      [ 18432,     0,     1,     1]            1: VIEW        cache_r_l0 (view)

    నిర్మాణం RsWriteback {
        పూర్ణాంకం స్థానం_begin = 0;  // first cache slot written by the CPY
        పూర్ణాంకం src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<పూర్ణాంకం> వెలికితీయి_layer_from_name(స్థిరరాశి std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_కూడిక_కూడు(స్థిరరాశి ggml_tensor * కణుపు);

class GgmlOvDecoder : బహిరంగ ov::frontend::ggml::GgmlDecoder {
బహిరంగ:
    స్థిర std::string get_tensor_name(స్థిరరాశి ggml_cgraph * cgraph, స్థిరరాశి ggml_tensor * tensor);
    నిర్మాణం కణుపు_Info {
        ggml_tensor * కణుపు;
        std::string కణుపు_name;
        std::string కణుపు_op_type;
        std::map<std::string, ggml_tensor *> కణుపు_inputs;
        std::map<std::string, std::సదిశ<std::pair<std::string, ggml_tensor *>>> కణుపు_inputs_views;
        std::సదిశ<std::string> కణుపు_inputs_పేర్లు;
        పూర్ణాంకం కణుపు_op_case = 0;
        శూన్య * దత్తాంశం_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::కణుపు_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  పూర్ణాంకం prefill_chunk_కొలత = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::కణుపు_2>> & model_weights);

    మాయా ov::Any get_attribute(స్థిరరాశి std::string & name) స్థిరరాశి override {
        తిరిగి nullptr;
        GGML_వాడని(name);
    }

    మాయా ov::PartialShape get_ఆదానం_shape(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి override;

    మాయా std::సదిశ<size_t> get_ఆదానం_stride(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి override;

    మాయా size_t get_view_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి override;

    మాయా size_t get_view_ఆదానం_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి override;

    మాయా size_t get_view_ఆదానం_src_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి override;

    మాయా std::సదిశ<size_t> get_view_ఆదానం_stride(పూర్ణాంకం కణుపు_idx,
                                                      స్థిరరాశి std::string & name,
                                                      size_t view_సూచీ) స్థిరరాశి override;

    మాయా std::సదిశ<size_t> get_view_ఆదానం_src_stride(పూర్ణాంకం కణుపు_idx,
                                                          స్థిరరాశి std::string & name,
                                                          size_t view_సూచీ) స్థిరరాశి override;

    మాయా ov::Shape get_view_ఆదానం_ggml_shape(పూర్ణాంకం కణుపు_idx,
                                                స్థిరరాశి std::string & name,
                                                size_t view_సూచీ) స్థిరరాశి override;

    మాయా ov::Shape get_view_ఆదానం_src_ggml_shape(పూర్ణాంకం కణుపు_idx,
                                                    స్థిరరాశి std::string & name,
                                                    size_t view_సూచీ) స్థిరరాశి override;

    మాయా ov::PartialShape get_view_ఆదానం_ov_shape(పూర్ణాంకం కణుపు_idx,
                                                     స్థిరరాశి std::string & name,
                                                     size_t view_సూచీ) స్థిరరాశి override;

    మాయా ov::PartialShape get_view_ఆదానం_src_ov_shape(పూర్ణాంకం కణుపు_idx,
                                                         స్థిరరాశి std::string & name,
                                                         size_t view_సూచీ) స్థిరరాశి override;

    మాయా std::string get_view_ఆదానం_name(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name, size_t view_సూచీ) స్థిరరాశి override;

    మాయా std::string get_view_ఆదానం_src_name(పూర్ణాంకం కణుపు_idx,
                                                స్థిరరాశి std::string & name,
                                                size_t view_సూచీ) స్థిరరాశి override;

    మాయా ov::element::Type get_ఆదానం_type(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి override;

    మాయా size_t get_ఆదానం_కొలత() స్థిరరాశి override;

    మాయా size_t get_ఆదానం_కొలత(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా శూన్య get_ఆదానం_కణుపు(size_t ఆదానం_పోర్టు_idx,
                                std::string & ఉత్పాదకుడు_name,
                                std::string & ఉత్పాదకుడు_output_పోర్టు_name,
                                size_t & ఉత్పాదకుడు_output_పోర్టు_సూచీ) స్థిరరాశి override {
        GGML_వాడని(ఆదానం_పోర్టు_idx);
        GGML_వాడని(ఉత్పాదకుడు_name);
        GGML_వాడని(ఉత్పాదకుడు_output_పోర్టు_name);
        GGML_వాడని(ఉత్పాదకుడు_output_పోర్టు_సూచీ);
    }

    మాయా std::సదిశ<std::string> get_ఆదానం_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా ov::PartialShape get_output_shape(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా ov::element::Type get_output_type(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా std::సదిశ<size_t> get_output_stride(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా int32_t * get_ఆదానం_op_params(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & name) స్థిరరాశి override;

    మాయా int32_t * get_output_op_params(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా size_t get_output_op_స్థానభ్రంశం(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా std::సదిశ<std::string> get_output_పేర్లు(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా std::string get_inplace_op_src(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా bool is_view_like_alias_యొక్క(పూర్ణాంకం కణుపు_idx, స్థిరరాశి std::string & view_src_name) స్థిరరాశి override;

    మాయా స్థిరరాశి std::string & get_op_type() స్థిరరాశి override;

    మాయా స్థిరరాశి std::string & get_op_type(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా స్థిరరాశి std::string & get_op_name() స్థిరరాశి override;

    మాయా స్థిరరాశి std::string & get_op_name(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా int32_t get_op_dynamic_dim(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override;

    మాయా శూన్య సందర్శించు_subgraph(
        std::function<శూన్య(std::shared_ptr<GgmlDecoder>, పూర్ణాంకం కణుపు_idx)> కణుపు_visitor) స్థిరరాశి override;

    ggml_tensor * get_ఆదానం_ggml_tensor(స్థిరరాశి std::string & name) స్థిరరాశి { తిరిగి m_inputs.at(name); }

    మాయా పూర్ణాంకం get_op_case(పూర్ణాంకం కణుపు_idx) స్థిరరాశి override { తిరిగి m_కణుపు_info_list[కణుపు_idx].కణుపు_op_case; }

    మాయా స్థిరరాశి std::map<std::string, ov::frontend::ggml::Model_ఆదానం_Info> & get_model_inputs() స్థిరరాశి override {
        తిరిగి m_model_inputs;
    }

    మాయా స్థిరరాశి std::map<std::string, ov::frontend::ggml::Model_Extra_ఆదానం_Info> & get_model_extra_inputs() స్థిరరాశి override {
        తిరిగి m_model_extra_inputs;
    }

    మాయా స్థిరరాశి std::map<std::string, std::shared_ptr<ov::కణుపు_2>> & get_model_weights() స్థిరరాశి override {
        తిరిగి m_model_weights;
    }

    మాయా std::సమితి<std::string> get_model_output_పేర్లు() స్థిరరాశి override { తిరిగి m_model_output_పేర్లు; }

    స్థిరరాశి std::map<std::string, ggml_tensor *> & get_model_outputs() స్థిరరాశి { తిరిగి m_model_outputs; }

    మాయా పూర్ణాంకం get_ctx_కొలత() స్థిరరాశి { తిరిగి m_model_params.ctx; }

    మాయా పూర్ణాంకం get_ctx_చొప్పున_seq() స్థిరరాశి { తిరిగి m_model_params.ctx_చొప్పున_seq; }

    మాయా పూర్ణాంకం get_ctx_చొప్పున_seq_swa() స్థిరరాశి { తిరిగి m_model_params.ctx_చొప్పున_seq_swa; }

    మాయా పూర్ణాంకం get_n_seq() స్థిరరాశి { తిరిగి m_model_params.n_seq; }

    మాయా పూర్ణాంకం is_swa_layer(పూర్ణాంకం layer) స్థిరరాశి override {
        తిరిగి std::కనుగొను(m_model_params.swa_layers.begin(), m_model_params.swa_layers.చివర(), layer) !=
               m_model_params.swa_layers.చివర();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    పూర్ణాంకం get_n_heads_kv_for_layer(పూర్ణాంకం layer) స్థిరరాశి {
        స్వయంచాలక it = m_model_params.n_heads_kv_చొప్పున_layer.కనుగొను(layer);
        తిరిగి it != m_model_params.n_heads_kv_చొప్పున_layer.చివర() ? it->రెండవ : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    పూర్ణాంకం get_n_heads_kv_for_tensor(స్థిరరాశి ggml_tensor * kv_tensor) స్థిరరాశి {
        ఒకవేళ (స్వయంచాలక layer = వెలికితీయి_layer_from_name(std::string(kv_tensor->name)); layer.has_విలువ()) {
            తిరిగి get_n_heads_kv_for_layer(layer.విలువ());
        }
        తిరిగి m_model_params.n_heads_kv;
    }

    పూర్ణాంకం get_past_kv_len() స్థిరరాశి { తిరిగి m_compute_params.past_kv_len; }

    పూర్ణాంకం get_ఆదానం_len() స్థిరరాశి { తిరిగి m_compute_params.ఆదానం_len; }

    మాయా int32_t * get_rope_params() స్థిరరాశి override { తిరిగి const_cast<int32_t *>(m_model_params.rope_params); }

    మాయా bool has_mixed_rope_params() స్థిరరాశి override { తిరిగి m_model_params.mixed_rope_params; }

    మాయా పూర్ణాంకం get_ssm_state_కొలత() స్థిరరాశి override { తిరిగి m_model_params.state_కొలత; }

    మాయా std::map<std::string, std::string> get_kv_param_res_పేర్లు() స్థిరరాశి override;

    మాయా bool is_static() స్థిరరాశి override { తిరిగి m_is_static; }

    మాయా bool is_stateful() స్థిరరాశి override { తిరిగి m_is_stateful; }

    పూర్ణాంకం get_static_n_టోకెన్లు() స్థిరరాశి { తిరిగి m_is_prefill ? m_prefill_chunk_కొలత : 1; }

    మాయా bool is_splited_model() స్థిరరాశి override { తిరిగి m_model_is_splitted; }

    ov::PartialShape get_లేఖాచిత్రం_ఆదానం_shape(స్థిరరాశి ggml_tensor * op,
                                           స్థిరరాశి ggml_tensor * ఆదానం,
                                           పూర్ణాంకం dynamic_dim_సూచీ = -1) స్థిరరాశి;

    స్థిర శూన్య dump_cgraph(స్థిరరాశి ggml_cgraph * cgraph, std::string & filename);

    స్థిర std::shared_ptr<ov::కణుపు_2> create_బరువు_కణుపు(ggml_tensor * tensor, bool naive = false);

    స్థిర std::map<std::string, std::shared_ptr<ov::కణుపు_2>> create_బరువు_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    స్థిర std::సమితి<std::string> collect_బరువు_పేర్లు(ggml_cgraph * cgraph);

    స్థిరరాశి ggml_tensor * get_tensor_వాడిన_op(స్థిరరాశి ggml_tensor * tensor) స్థిరరాశి;

    స్థిరరాశి ggml_tensor * get_tensor_from_name(స్థిరరాశి std::string & name) స్థిరరాశి;

    శూన్య clear_model_weights() { m_model_weights.clear(); }

    స్థిర std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() స్థిరరాశి { తిరిగి m_model_params; }

    ComputeParams get_compute_params() స్థిరరాశి { తిరిగి m_compute_params; }

    శూన్య సమితి_model_params(స్థిరరాశి ModelParams & model_params) { m_model_params = model_params; }

    శూన్య సమితి_compute_params(స్థిరరాశి ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    పూర్ణాంకం m_prefill_chunk_కొలత = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    స్థిర ov::Shape get_shape(స్థిరరాశి ggml_tensor * tensor);
    స్థిర std::సదిశ<size_t> get_stride(స్థిరరాశి ggml_tensor * tensor);
    స్థిర ov::element::Type get_ov_type(స్థిరరాశి ggml_tensor * tensor);
    స్థిర std::string compute_op_type(స్థిరరాశి ggml_tensor * కణుపు);
    శూన్య కూడు_extra_inputs();

    శూన్య update_io(ggml_cgraph * cgraph);
    bool is_bound_గమ్యం(స్థిరరాశి ggml_cgraph * cgraph) స్థిరరాశి;

    స్థిర bool is_inp_tok(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_GET_అడ్డువరుసలు && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    స్థిర bool is_inp_pos(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    స్థిర పూర్ణాంకం get_inp_pos_n_planes(స్థిరరాశి ggml_tensor * op) {
        స్థిరరాశి పూర్ణాంకం బహుళకం = op->op_params[2];
        తిరిగి (బహుళకం == GGML_ROPE_TYPE_IMROPE || బహుళకం == GGML_ROPE_TYPE_VISION || (బహుళకం & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    స్థిర bool is_inp_emb(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి tensor->op == GGML_OP_GET_అడ్డువరుసలు && op->op == GGML_OP_RMS_NORM;
    }

    స్థిర bool is_inp_ముసుగు(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    స్థిర bool is_inp_సగటు(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_ఆదానం) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    స్థిర bool is_inp_scale_అడ్డువరుసలు(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    స్థిర bool is_rope_freqs_బరువు(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    అంతఃవరుస స్థిర bool is_recurrent_cache(స్థిరరాశి ggml_tensor * tensor) {
        తిరిగి tensor != nullptr && (అక్షరశ్రేణి_పోలిక(tensor->name, "cache_r_l", అక్షరశ్రేణి_పొడవు("cache_r_l")) == 0 ||
                                     అక్షరశ్రేణి_పోలిక(tensor->name, "cache_s_l", అక్షరశ్రేణి_పొడవు("cache_s_l")) == 0 ||
                                     అక్షరశ్రేణి_పోలిక(tensor->name, "cache_ple_r_l", అక్షరశ్రేణి_పొడవు("cache_ple_r_l")) == 0);
    }

    అంతఃవరుస స్థిర bool is_cache(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    అంతఃవరుస స్థిర bool is_kvcache(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        ఒకవేళ (tensor == nullptr || is_recurrent_cache(tensor)) {
            తిరిగి false;
        }
        తిరిగి (tensor->బఫరు != nullptr && tensor->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_సమితి_అడ్డువరుసలు && op->src[2] == tensor);
    }

    అంతఃవరుస స్థిర bool is_conv_state_writeback(స్థిరరాశి ggml_tensor * కణుపు) {
        తిరిగి కణుపు->op == GGML_OP_CPY && కణుపు->view_src != nullptr && is_recurrent_cache(కణుపు->view_src) &&
               కణుపు->src[0] != nullptr && కణుపు->src[0]->op == GGML_OP_VIEW && కణుపు->src[0]->src[0] != nullptr &&
               కణుపు->src[0]->src[0]->op == GGML_OP_CONCAT && కణుపు->src[1] != nullptr &&
               కణుపు->src[1]->op == GGML_OP_VIEW && కణుపు->src[1]->view_src == కణుపు->view_src;
    }

    స్థిర bool is_kv_idx(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_సమితి_అడ్డువరుసలు && op->src[1] == tensor;
    }

    bool is_swa_ముసుగు(స్థిరరాశి ggml_tensor * tensor) స్థిరరాశి {
        తిరిగి m_model_params.swa_ముసుగు != nullptr && tensor == m_model_params.swa_ముసుగు;
    }

    స్థిర bool is_output_idx(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_GET_అడ్డువరుసలు && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    స్థిర bool is_inp_s_copy(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) {
        తిరిగి op->op == GGML_OP_GET_అడ్డువరుసలు && tensor == op->src[1] &&
               op->src[0]->బఫరు->usage == GGML_BACKEND_బఫరు_USAGE_ANY;
    }

    std::string get_లేఖాచిత్రం_ఆదానం_ov_name(స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి ggml_tensor * op) స్థిరరాశి {
        ఒకవేళ (is_inp_pos(tensor, op)) {
            తిరిగి "inp_pos";
        }
        ఒకవేళ (is_inp_emb(tensor, op)) {
            తిరిగి "embd";
        }
        ఒకవేళ (is_inp_ముసుగు(tensor, op)) {
            // Give the two attention masks distinct OV parameter names.
            //
            // An interleaved-SWA model builds one full-attention mask and one sliding-window mask,
            // but build_attn_inp_kq_mask() names them identically, so keying a parameter off
            // tensor->name alone makes the second mask OVERWRITE the first in m_model_inputs: both
            // attention types then read a single parameter, and the windowed layers silently run
            // against an unbanded mask. Disambiguate using the SWA layer set computed in
            // compute_llm_params(), which classifies by mask tensor identity rather than by name.
            //
            // When no SWA layer was found there is only one mask in play, so the plain name is
            // correct and no _swa parameter is created.
            ఒకవేళ (m_model_params.swa_layers.ఖాళీలు()) {
                తిరిగి "self_kq_mask";
            }
            తిరిగి is_swa_ముసుగు(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        తిరిగి tensor->name;
    }

అంతరంగ:
    శూన్య సమితి_ఆదానం_output();
    పూర్ణాంకం compute_op_case(స్థిరరాశి ggml_tensor * కణుపు) స్థిరరాశి;
    bool కణుపు_is_వాడిన_as_src(స్థిరరాశి పూర్ణాంకం కణుపు_idx);
    శూన్య compute_model_inputs();
    శూన్య compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_ఆకు(స్థిరరాశి ggml_tensor * tensor) స్థిరరాశి;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    శూన్య compute_కణుపు_dynamic_dims();

    శూన్య validate_cgraph() స్థిరరాశి;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_ఆదానం_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_ఆదానం_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::కణుపు_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::సమితి<std::string> m_model_output_పేర్లు;
    std::సదిశ<కణుపు_Info> m_కణుపు_info_list;
    std::map<ggml_tensor *, పూర్ణాంకం> m_కణుపు_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

శూన్య print_tensor_చిరునామా_map(స్థిరరాశి ggml_cgraph * cgraph);
