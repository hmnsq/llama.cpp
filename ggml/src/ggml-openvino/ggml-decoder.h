#pragma once

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "openvino/decoder.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <map>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <openvino/core/partial_shape.hpp>
#অন্তর্ভাব <optional>
#অন্তর্ভাব <set>
#অন্তর্ভাব <string>
#অন্তর্ভাব <vector>

কাঠামো ModelParams {
    পূর্ণসংখ্যা ctx = -1;
    পূর্ণসংখ্যা ctx_per_seq = -1;
    পূর্ণসংখ্যা ctx_per_seq_swa = -1;
    পূর্ণসংখ্যা n_seq = 1;
    পূর্ণসংখ্যা n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<পূর্ণসংখ্যা, পূর্ণসংখ্যা> n_heads_kv_per_layer;
    পূর্ণসংখ্যা শীর্ষ_আকার = -1;
    পূর্ণসংখ্যা state_আকার = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    পূর্ণসংখ্যা n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<পূর্ণসংখ্যা> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ধ্রুবক ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_বাফার_ctx_id = 0;

    bool same_rope_params(ধ্রুবক ModelParams & other) ধ্রুবক {
        ফেরত mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, আকার(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ধ্রুবক ModelParams & other) ধ্রুবক {
        ফেরত same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ধ্রুবক ModelParams & other) ধ্রুবক {
        ফেরত same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_বাফার_changed(ধ্রুবক ModelParams & other) ধ্রুবক { ফেরত kv_বাফার_ctx_id != other.kv_বাফার_ctx_id; }
};

কাঠামো ComputeParams {
    পূর্ণসংখ্যা n_seq_active = 1;
    পূর্ণসংখ্যা seq_active_start = 0;
    পূর্ণসংখ্যা attention_আকার = -1;
    পূর্ণসংখ্যা attention_আকার_swa = -1;
    পূর্ণসংখ্যা attention_আকার_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    পূর্ণসংখ্যা swa_window = -1;
    পূর্ণসংখ্যা প্রবেশ_len = -1;
    পূর্ণসংখ্যা token_len_per_seq = -1;
    পূর্ণসংখ্যা past_kv_len = -1;
    পূর্ণসংখ্যা output_len = 1;

    পূর্ণসংখ্যা cache_rs_reset_idx = -1;
    পূর্ণসংখ্যা cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    পূর্ণসংখ্যা s_copy_active_slot_len = -1;
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

    কাঠামো RsWriteback {
        পূর্ণসংখ্যা slot_begin = 0;  // first cache slot written by the CPY
        পূর্ণসংখ্যা src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<পূর্ণসংখ্যা> extract_layer_from_name(ধ্রুবক std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_যোগ_add(ধ্রুবক ggml_tensor * গ্রন্থি);

class GgmlOvDecoder : সর্বজনীন ov::frontend::ggml::GgmlDecoder {
সর্বজনীন:
    স্থির std::string get_tensor_name(ধ্রুবক ggml_cgraph * cgraph, ধ্রুবক ggml_tensor * tensor);
    কাঠামো গ্রন্থি_Info {
        ggml_tensor * গ্রন্থি;
        std::string গ্রন্থি_name;
        std::string গ্রন্থি_op_type;
        std::map<std::string, ggml_tensor *> গ্রন্থি_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> গ্রন্থি_inputs_views;
        std::vector<std::string> গ্রন্থি_inputs_names;
        পূর্ণসংখ্যা গ্রন্থি_op_case = 0;
        শূন্য * তথ্য_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  পূর্ণসংখ্যা prefill_chunk_আকার = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> & model_weights);

    আভাসী ov::Any get_attribute(ধ্রুবক std::string & name) ধ্রুবক override {
        ফেরত nullptr;
        GGML_UNUSED(name);
    }

    আভাসী ov::PartialShape get_প্রবেশ_shape(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক override;

    আভাসী std::vector<size_t> get_প্রবেশ_stride(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক override;

    আভাসী size_t get_view_প্রবেশ_আকার(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক override;

    আভাসী size_t get_view_প্রবেশ_offset(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক override;

    আভাসী size_t get_view_প্রবেশ_src_offset(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক override;

    আভাসী std::vector<size_t> get_view_প্রবেশ_stride(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                      ধ্রুবক std::string & name,
                                                      size_t view_সূচক) ধ্রুবক override;

    আভাসী std::vector<size_t> get_view_প্রবেশ_src_stride(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                          ধ্রুবক std::string & name,
                                                          size_t view_সূচক) ধ্রুবক override;

    আভাসী ov::Shape get_view_প্রবেশ_ggml_shape(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                ধ্রুবক std::string & name,
                                                size_t view_সূচক) ধ্রুবক override;

    আভাসী ov::Shape get_view_প্রবেশ_src_ggml_shape(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                    ধ্রুবক std::string & name,
                                                    size_t view_সূচক) ধ্রুবক override;

    আভাসী ov::PartialShape get_view_প্রবেশ_ov_shape(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                     ধ্রুবক std::string & name,
                                                     size_t view_সূচক) ধ্রুবক override;

    আভাসী ov::PartialShape get_view_প্রবেশ_src_ov_shape(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                         ধ্রুবক std::string & name,
                                                         size_t view_সূচক) ধ্রুবক override;

    আভাসী std::string get_view_প্রবেশ_name(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name, size_t view_সূচক) ধ্রুবক override;

    আভাসী std::string get_view_প্রবেশ_src_name(পূর্ণসংখ্যা গ্রন্থি_idx,
                                                ধ্রুবক std::string & name,
                                                size_t view_সূচক) ধ্রুবক override;

    আভাসী ov::element::Type get_প্রবেশ_type(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক override;

    আভাসী size_t get_প্রবেশ_আকার() ধ্রুবক override;

    আভাসী size_t get_প্রবেশ_আকার(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী শূন্য get_প্রবেশ_গ্রন্থি(size_t প্রবেশ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_সূচক) ধ্রুবক override {
        GGML_UNUSED(প্রবেশ_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_সূচক);
    }

    আভাসী std::vector<std::string> get_প্রবেশ_names(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী ov::PartialShape get_output_shape(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী ov::element::Type get_output_type(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী std::vector<size_t> get_output_stride(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী int32_t * get_প্রবেশ_op_params(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & name) ধ্রুবক override;

    আভাসী int32_t * get_output_op_params(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী size_t get_output_op_offset(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী std::vector<std::string> get_output_names(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী std::string get_inplace_op_src(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী bool is_view_like_alias_of(পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক std::string & view_src_name) ধ্রুবক override;

    আভাসী ধ্রুবক std::string & get_op_type() ধ্রুবক override;

    আভাসী ধ্রুবক std::string & get_op_type(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী ধ্রুবক std::string & get_op_name() ধ্রুবক override;

    আভাসী ধ্রুবক std::string & get_op_name(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী int32_t get_op_dynamic_dim(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override;

    আভাসী শূন্য visit_subgraph(
        std::function<শূন্য(std::shared_ptr<GgmlDecoder>, পূর্ণসংখ্যা গ্রন্থি_idx)> গ্রন্থি_visitor) ধ্রুবক override;

    ggml_tensor * get_প্রবেশ_ggml_tensor(ধ্রুবক std::string & name) ধ্রুবক { ফেরত m_inputs.at(name); }

    আভাসী পূর্ণসংখ্যা get_op_case(পূর্ণসংখ্যা গ্রন্থি_idx) ধ্রুবক override { ফেরত m_গ্রন্থি_info_list[গ্রন্থি_idx].গ্রন্থি_op_case; }

    আভাসী ধ্রুবক std::map<std::string, ov::frontend::ggml::Model_প্রবেশ_Info> & get_model_inputs() ধ্রুবক override {
        ফেরত m_model_inputs;
    }

    আভাসী ধ্রুবক std::map<std::string, ov::frontend::ggml::Model_Extra_প্রবেশ_Info> & get_model_extra_inputs() ধ্রুবক override {
        ফেরত m_model_extra_inputs;
    }

    আভাসী ধ্রুবক std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> & get_model_weights() ধ্রুবক override {
        ফেরত m_model_weights;
    }

    আভাসী std::set<std::string> get_model_output_names() ধ্রুবক override { ফেরত m_model_output_names; }

    ধ্রুবক std::map<std::string, ggml_tensor *> & get_model_outputs() ধ্রুবক { ফেরত m_model_outputs; }

    আভাসী পূর্ণসংখ্যা get_ctx_আকার() ধ্রুবক { ফেরত m_model_params.ctx; }

    আভাসী পূর্ণসংখ্যা get_ctx_per_seq() ধ্রুবক { ফেরত m_model_params.ctx_per_seq; }

    আভাসী পূর্ণসংখ্যা get_ctx_per_seq_swa() ধ্রুবক { ফেরত m_model_params.ctx_per_seq_swa; }

    আভাসী পূর্ণসংখ্যা get_n_seq() ধ্রুবক { ফেরত m_model_params.n_seq; }

    আভাসী পূর্ণসংখ্যা is_swa_layer(পূর্ণসংখ্যা layer) ধ্রুবক override {
        ফেরত std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    পূর্ণসংখ্যা get_n_heads_kv_for_layer(পূর্ণসংখ্যা layer) ধ্রুবক {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        ফেরত it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    পূর্ণসংখ্যা get_n_heads_kv_for_tensor(ধ্রুবক ggml_tensor * kv_tensor) ধ্রুবক {
        যদি (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_মান()) {
            ফেরত get_n_heads_kv_for_layer(layer.মান());
        }
        ফেরত m_model_params.n_heads_kv;
    }

    পূর্ণসংখ্যা get_past_kv_len() ধ্রুবক { ফেরত m_compute_params.past_kv_len; }

    পূর্ণসংখ্যা get_প্রবেশ_len() ধ্রুবক { ফেরত m_compute_params.প্রবেশ_len; }

    আভাসী int32_t * get_rope_params() ধ্রুবক override { ফেরত const_cast<int32_t *>(m_model_params.rope_params); }

    আভাসী bool has_mixed_rope_params() ধ্রুবক override { ফেরত m_model_params.mixed_rope_params; }

    আভাসী পূর্ণসংখ্যা get_ssm_state_আকার() ধ্রুবক override { ফেরত m_model_params.state_আকার; }

    আভাসী std::map<std::string, std::string> get_kv_param_res_names() ধ্রুবক override;

    আভাসী bool is_static() ধ্রুবক override { ফেরত m_is_static; }

    আভাসী bool is_stateful() ধ্রুবক override { ফেরত m_is_stateful; }

    পূর্ণসংখ্যা get_static_n_tokens() ধ্রুবক { ফেরত m_is_prefill ? m_prefill_chunk_আকার : 1; }

    আভাসী bool is_splited_model() ধ্রুবক override { ফেরত m_model_is_splitted; }

    ov::PartialShape get_graph_প্রবেশ_shape(ধ্রুবক ggml_tensor * op,
                                           ধ্রুবক ggml_tensor * প্রবেশ,
                                           পূর্ণসংখ্যা dynamic_dim_সূচক = -1) ধ্রুবক;

    স্থির শূন্য dump_cgraph(ধ্রুবক ggml_cgraph * cgraph, std::string & filename);

    স্থির std::shared_ptr<ov::গ্রন্থি_2> create_weight_গ্রন্থি(ggml_tensor * tensor, bool naive = false);

    স্থির std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    স্থির std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    ধ্রুবক ggml_tensor * get_tensor_used_op(ধ্রুবক ggml_tensor * tensor) ধ্রুবক;

    ধ্রুবক ggml_tensor * get_tensor_from_name(ধ্রুবক std::string & name) ধ্রুবক;

    শূন্য clear_model_weights() { m_model_weights.clear(); }

    স্থির std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ধ্রুবক { ফেরত m_model_params; }

    ComputeParams get_compute_params() ধ্রুবক { ফেরত m_compute_params; }

    শূন্য set_model_params(ধ্রুবক ModelParams & model_params) { m_model_params = model_params; }

    শূন্য set_compute_params(ধ্রুবক ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    পূর্ণসংখ্যা m_prefill_chunk_আকার = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    স্থির ov::Shape get_shape(ধ্রুবক ggml_tensor * tensor);
    স্থির std::vector<size_t> get_stride(ধ্রুবক ggml_tensor * tensor);
    স্থির ov::element::Type get_ov_type(ধ্রুবক ggml_tensor * tensor);
    স্থির std::string compute_op_type(ধ্রুবক ggml_tensor * গ্রন্থি);
    শূন্য add_extra_inputs();

    শূন্য update_io(ggml_cgraph * cgraph);
    bool is_bound_to(ধ্রুবক ggml_cgraph * cgraph) ধ্রুবক;

    স্থির bool is_inp_tok(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    স্থির bool is_inp_pos(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    স্থির পূর্ণসংখ্যা get_inp_pos_n_planes(ধ্রুবক ggml_tensor * op) {
        ধ্রুবক পূর্ণসংখ্যা mode = op->op_params[2];
        ফেরত (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    স্থির bool is_inp_emb(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    স্থির bool is_inp_mask(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    স্থির bool is_inp_mean(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_প্রবেশ) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    স্থির bool is_inp_scale_rows(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    স্থির bool is_rope_freqs_weight(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    অন্তঃরেখ স্থির bool is_recurrent_cache(ধ্রুবক ggml_tensor * tensor) {
        ফেরত tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    অন্তঃরেখ স্থির bool is_cache(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    অন্তঃরেখ স্থির bool is_kvcache(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        যদি (tensor == nullptr || is_recurrent_cache(tensor)) {
            ফেরত false;
        }
        ফেরত (tensor->বাফার != nullptr && tensor->বাফার->usage == GGML_BACKEND_বাফার_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    অন্তঃরেখ স্থির bool is_conv_state_writeback(ধ্রুবক ggml_tensor * গ্রন্থি) {
        ফেরত গ্রন্থি->op == GGML_OP_CPY && গ্রন্থি->view_src != nullptr && is_recurrent_cache(গ্রন্থি->view_src) &&
               গ্রন্থি->src[0] != nullptr && গ্রন্থি->src[0]->op == GGML_OP_VIEW && গ্রন্থি->src[0]->src[0] != nullptr &&
               গ্রন্থি->src[0]->src[0]->op == GGML_OP_CONCAT && গ্রন্থি->src[1] != nullptr &&
               গ্রন্থি->src[1]->op == GGML_OP_VIEW && গ্রন্থি->src[1]->view_src == গ্রন্থি->view_src;
    }

    স্থির bool is_kv_idx(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(ধ্রুবক ggml_tensor * tensor) ধ্রুবক {
        ফেরত m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    স্থির bool is_output_idx(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    স্থির bool is_inp_s_copy(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) {
        ফেরত op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->বাফার->usage == GGML_BACKEND_বাফার_USAGE_ANY;
    }

    std::string get_graph_প্রবেশ_ov_name(ধ্রুবক ggml_tensor * tensor, ধ্রুবক ggml_tensor * op) ধ্রুবক {
        যদি (is_inp_pos(tensor, op)) {
            ফেরত "inp_pos";
        }
        যদি (is_inp_emb(tensor, op)) {
            ফেরত "embd";
        }
        যদি (is_inp_mask(tensor, op)) {
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
            যদি (m_model_params.swa_layers.empty()) {
                ফেরত "self_kq_mask";
            }
            ফেরত is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ফেরত tensor->name;
    }

ব্যক্তিগত:
    শূন্য set_প্রবেশ_output();
    পূর্ণসংখ্যা compute_op_case(ধ্রুবক ggml_tensor * গ্রন্থি) ধ্রুবক;
    bool গ্রন্থি_is_used_as_src(ধ্রুবক পূর্ণসংখ্যা গ্রন্থি_idx);
    শূন্য compute_model_inputs();
    শূন্য compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(ধ্রুবক ggml_tensor * tensor) ধ্রুবক;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    শূন্য compute_গ্রন্থি_dynamic_dims();

    শূন্য validate_cgraph() ধ্রুবক;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_প্রবেশ_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_প্রবেশ_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::গ্রন্থি_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<গ্রন্থি_Info> m_গ্রন্থি_info_list;
    std::map<ggml_tensor *, পূর্ণসংখ্যা> m_গ্রন্থি_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

শূন্য print_tensor_address_map(ধ্রুবক ggml_cgraph * cgraph);
