#pragma once

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "openvino/decoder.h"

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <map>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/partial_shape.hpp>
#અન્તર્ભાવ <optional>
#અન્તર્ભાવ <set>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <vector>

સંરચના ModelParams {
    પૂર્ણાંક ctx = -1;
    પૂર્ણાંક ctx_per_seq = -1;
    પૂર્ણાંક ctx_per_seq_swa = -1;
    પૂર્ણાંક n_seq = 1;
    પૂર્ણાંક n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<પૂર્ણાંક, પૂર્ણાંક> n_heads_kv_per_layer;
    પૂર્ણાંક head_size = -1;
    પૂર્ણાંક state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    પૂર્ણાંક n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<પૂર્ણાંક> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    અચલ ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(અચલ ModelParams & other) અચલ {
        પાછા mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, કદ(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(અચલ ModelParams & other) અચલ {
        પાછા same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(અચલ ModelParams & other) અચલ {
        પાછા same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(અચલ ModelParams & other) અચલ { પાછા kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

સંરચના ComputeParams {
    પૂર્ણાંક n_seq_active = 1;
    પૂર્ણાંક seq_active_start = 0;
    પૂર્ણાંક attention_size = -1;
    પૂર્ણાંક attention_size_swa = -1;
    પૂર્ણાંક attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    પૂર્ણાંક swa_window = -1;
    પૂર્ણાંક આગત_len = -1;
    પૂર્ણાંક token_len_per_seq = -1;
    પૂર્ણાંક past_kv_len = -1;
    પૂર્ણાંક output_len = 1;

    પૂર્ણાંક cache_rs_reset_idx = -1;
    પૂર્ણાંક cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    પૂર્ણાંક s_copy_active_slot_len = -1;
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

    સંરચના RsWriteback {
        પૂર્ણાંક slot_begin = 0;  // first cache slot written by the CPY
        પૂર્ણાંક src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<પૂર્ણાંક> extract_layer_from_name(અચલ std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(અચલ ggml_tensor * ગાંઠ);

class GgmlOvDecoder : જાહેર ov::frontend::ggml::GgmlDecoder {
જાહેર:
    સ્થિર std::string get_tensor_name(અચલ ggml_cgraph * cgraph, અચલ ggml_tensor * tensor);
    સંરચના ગાંઠ_Info {
        ggml_tensor * ગાંઠ;
        std::string ગાંઠ_name;
        std::string ગાંઠ_op_type;
        std::map<std::string, ggml_tensor *> ગાંઠ_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> ગાંઠ_inputs_views;
        std::vector<std::string> ગાંઠ_inputs_names;
        પૂર્ણાંક ગાંઠ_op_case = 0;
        શૂન્ય * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  પૂર્ણાંક prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> & model_weights);

    આભાસી ov::Any get_attribute(અચલ std::string & name) અચલ override {
        પાછા nullptr;
        GGML_UNUSED(name);
    }

    આભાસી ov::PartialShape get_આગત_shape(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ override;

    આભાસી std::vector<size_t> get_આગત_stride(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ override;

    આભાસી size_t get_view_આગત_size(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ override;

    આભાસી size_t get_view_આગત_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ override;

    આભાસી size_t get_view_આગત_src_offset(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ override;

    આભાસી std::vector<size_t> get_view_આગત_stride(પૂર્ણાંક ગાંઠ_idx,
                                                      અચલ std::string & name,
                                                      size_t view_index) અચલ override;

    આભાસી std::vector<size_t> get_view_આગત_src_stride(પૂર્ણાંક ગાંઠ_idx,
                                                          અચલ std::string & name,
                                                          size_t view_index) અચલ override;

    આભાસી ov::Shape get_view_આગત_ggml_shape(પૂર્ણાંક ગાંઠ_idx,
                                                અચલ std::string & name,
                                                size_t view_index) અચલ override;

    આભાસી ov::Shape get_view_આગત_src_ggml_shape(પૂર્ણાંક ગાંઠ_idx,
                                                    અચલ std::string & name,
                                                    size_t view_index) અચલ override;

    આભાસી ov::PartialShape get_view_આગત_ov_shape(પૂર્ણાંક ગાંઠ_idx,
                                                     અચલ std::string & name,
                                                     size_t view_index) અચલ override;

    આભાસી ov::PartialShape get_view_આગત_src_ov_shape(પૂર્ણાંક ગાંઠ_idx,
                                                         અચલ std::string & name,
                                                         size_t view_index) અચલ override;

    આભાસી std::string get_view_આગત_name(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name, size_t view_index) અચલ override;

    આભાસી std::string get_view_આગત_src_name(પૂર્ણાંક ગાંઠ_idx,
                                                અચલ std::string & name,
                                                size_t view_index) અચલ override;

    આભાસી ov::element::Type get_આગત_type(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ override;

    આભાસી size_t get_આગત_size() અચલ override;

    આભાસી size_t get_આગત_size(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી શૂન્ય get_આગત_ગાંઠ(size_t આગત_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) અચલ override {
        GGML_UNUSED(આગત_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    આભાસી std::vector<std::string> get_આગત_names(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી ov::PartialShape get_output_shape(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી ov::element::Type get_output_type(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી std::vector<size_t> get_output_stride(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી int32_t * get_આગત_op_params(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & name) અચલ override;

    આભાસી int32_t * get_output_op_params(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી size_t get_output_op_offset(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી std::vector<std::string> get_output_names(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી std::string get_inplace_op_src(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી bool is_view_like_alias_of(પૂર્ણાંક ગાંઠ_idx, અચલ std::string & view_src_name) અચલ override;

    આભાસી અચલ std::string & get_op_type() અચલ override;

    આભાસી અચલ std::string & get_op_type(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી અચલ std::string & get_op_name() અચલ override;

    આભાસી અચલ std::string & get_op_name(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી int32_t get_op_dynamic_dim(પૂર્ણાંક ગાંઠ_idx) અચલ override;

    આભાસી શૂન્ય visit_subgraph(
        std::function<શૂન્ય(std::shared_ptr<GgmlDecoder>, પૂર્ણાંક ગાંઠ_idx)> ગાંઠ_visitor) અચલ override;

    ggml_tensor * get_આગત_ggml_tensor(અચલ std::string & name) અચલ { પાછા m_inputs.at(name); }

    આભાસી પૂર્ણાંક get_op_case(પૂર્ણાંક ગાંઠ_idx) અચલ override { પાછા m_ગાંઠ_info_list[ગાંઠ_idx].ગાંઠ_op_case; }

    આભાસી અચલ std::map<std::string, ov::frontend::ggml::Model_આગત_Info> & get_model_inputs() અચલ override {
        પાછા m_model_inputs;
    }

    આભાસી અચલ std::map<std::string, ov::frontend::ggml::Model_Extra_આગત_Info> & get_model_extra_inputs() અચલ override {
        પાછા m_model_extra_inputs;
    }

    આભાસી અચલ std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> & get_model_weights() અચલ override {
        પાછા m_model_weights;
    }

    આભાસી std::set<std::string> get_model_output_names() અચલ override { પાછા m_model_output_names; }

    અચલ std::map<std::string, ggml_tensor *> & get_model_outputs() અચલ { પાછા m_model_outputs; }

    આભાસી પૂર્ણાંક get_ctx_size() અચલ { પાછા m_model_params.ctx; }

    આભાસી પૂર્ણાંક get_ctx_per_seq() અચલ { પાછા m_model_params.ctx_per_seq; }

    આભાસી પૂર્ણાંક get_ctx_per_seq_swa() અચલ { પાછા m_model_params.ctx_per_seq_swa; }

    આભાસી પૂર્ણાંક get_n_seq() અચલ { પાછા m_model_params.n_seq; }

    આભાસી પૂર્ણાંક is_swa_layer(પૂર્ણાંક layer) અચલ override {
        પાછા std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    પૂર્ણાંક get_n_heads_kv_for_layer(પૂર્ણાંક layer) અચલ {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        પાછા it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    પૂર્ણાંક get_n_heads_kv_for_tensor(અચલ ggml_tensor * kv_tensor) અચલ {
        જો (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_કિંમત()) {
            પાછા get_n_heads_kv_for_layer(layer.કિંમત());
        }
        પાછા m_model_params.n_heads_kv;
    }

    પૂર્ણાંક get_past_kv_len() અચલ { પાછા m_compute_params.past_kv_len; }

    પૂર્ણાંક get_આગત_len() અચલ { પાછા m_compute_params.આગત_len; }

    આભાસી int32_t * get_rope_params() અચલ override { પાછા const_cast<int32_t *>(m_model_params.rope_params); }

    આભાસી bool has_mixed_rope_params() અચલ override { પાછા m_model_params.mixed_rope_params; }

    આભાસી પૂર્ણાંક get_ssm_state_size() અચલ override { પાછા m_model_params.state_size; }

    આભાસી std::map<std::string, std::string> get_kv_param_res_names() અચલ override;

    આભાસી bool is_static() અચલ override { પાછા m_is_static; }

    આભાસી bool is_stateful() અચલ override { પાછા m_is_stateful; }

    પૂર્ણાંક get_static_n_tokens() અચલ { પાછા m_is_prefill ? m_prefill_chunk_size : 1; }

    આભાસી bool is_splited_model() અચલ override { પાછા m_model_is_splitted; }

    ov::PartialShape get_graph_આગત_shape(અચલ ggml_tensor * op,
                                           અચલ ggml_tensor * આગત,
                                           પૂર્ણાંક dynamic_dim_index = -1) અચલ;

    સ્થિર શૂન્ય dump_cgraph(અચલ ggml_cgraph * cgraph, std::string & filename);

    સ્થિર std::shared_ptr<ov::ગાંઠ_2> create_weight_ગાંઠ(ggml_tensor * tensor, bool naive = false);

    સ્થિર std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    સ્થિર std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    અચલ ggml_tensor * get_tensor_used_op(અચલ ggml_tensor * tensor) અચલ;

    અચલ ggml_tensor * get_tensor_from_name(અચલ std::string & name) અચલ;

    શૂન્ય clear_model_weights() { m_model_weights.clear(); }

    સ્થિર std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() અચલ { પાછા m_model_params; }

    ComputeParams get_compute_params() અચલ { પાછા m_compute_params; }

    શૂન્ય set_model_params(અચલ ModelParams & model_params) { m_model_params = model_params; }

    શૂન્ય set_compute_params(અચલ ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    પૂર્ણાંક m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    સ્થિર ov::Shape get_shape(અચલ ggml_tensor * tensor);
    સ્થિર std::vector<size_t> get_stride(અચલ ggml_tensor * tensor);
    સ્થિર ov::element::Type get_ov_type(અચલ ggml_tensor * tensor);
    સ્થિર std::string compute_op_type(અચલ ggml_tensor * ગાંઠ);
    શૂન્ય add_extra_inputs();

    શૂન્ય update_io(ggml_cgraph * cgraph);
    bool is_bound_to(અચલ ggml_cgraph * cgraph) અચલ;

    સ્થિર bool is_inp_tok(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    સ્થિર bool is_inp_pos(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    સ્થિર પૂર્ણાંક get_inp_pos_n_planes(અચલ ggml_tensor * op) {
        અચલ પૂર્ણાંક mode = op->op_params[2];
        પાછા (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    સ્થિર bool is_inp_emb(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    સ્થિર bool is_inp_mask(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    સ્થિર bool is_inp_mean(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_આગત) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    સ્થિર bool is_inp_scale_rows(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    સ્થિર bool is_rope_freqs_weight(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    અંતઃરેખ સ્થિર bool is_recurrent_cache(અચલ ggml_tensor * tensor) {
        પાછા tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    અંતઃરેખ સ્થિર bool is_cache(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    અંતઃરેખ સ્થિર bool is_kvcache(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        જો (tensor == nullptr || is_recurrent_cache(tensor)) {
            પાછા false;
        }
        પાછા (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    અંતઃરેખ સ્થિર bool is_conv_state_writeback(અચલ ggml_tensor * ગાંઠ) {
        પાછા ગાંઠ->op == GGML_OP_CPY && ગાંઠ->view_src != nullptr && is_recurrent_cache(ગાંઠ->view_src) &&
               ગાંઠ->src[0] != nullptr && ગાંઠ->src[0]->op == GGML_OP_VIEW && ગાંઠ->src[0]->src[0] != nullptr &&
               ગાંઠ->src[0]->src[0]->op == GGML_OP_CONCAT && ગાંઠ->src[1] != nullptr &&
               ગાંઠ->src[1]->op == GGML_OP_VIEW && ગાંઠ->src[1]->view_src == ગાંઠ->view_src;
    }

    સ્થિર bool is_kv_idx(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(અચલ ggml_tensor * tensor) અચલ {
        પાછા m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    સ્થિર bool is_output_idx(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    સ્થિર bool is_inp_s_copy(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) {
        પાછા op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_આગત_ov_name(અચલ ggml_tensor * tensor, અચલ ggml_tensor * op) અચલ {
        જો (is_inp_pos(tensor, op)) {
            પાછા "inp_pos";
        }
        જો (is_inp_emb(tensor, op)) {
            પાછા "embd";
        }
        જો (is_inp_mask(tensor, op)) {
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
            જો (m_model_params.swa_layers.empty()) {
                પાછા "self_kq_mask";
            }
            પાછા is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        પાછા tensor->name;
    }

ખાનગી:
    શૂન્ય set_આગત_output();
    પૂર્ણાંક compute_op_case(અચલ ggml_tensor * ગાંઠ) અચલ;
    bool ગાંઠ_is_used_as_src(અચલ પૂર્ણાંક ગાંઠ_idx);
    શૂન્ય compute_model_inputs();
    શૂન્ય compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(અચલ ggml_tensor * tensor) અચલ;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    શૂન્ય compute_ગાંઠ_dynamic_dims();

    શૂન્ય validate_cgraph() અચલ;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_આગત_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_આગત_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::ગાંઠ_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<ગાંઠ_Info> m_ગાંઠ_info_list;
    std::map<ggml_tensor *, પૂર્ણાંક> m_ગાંઠ_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

શૂન્ય print_tensor_address_map(અચલ ggml_cgraph * cgraph);
