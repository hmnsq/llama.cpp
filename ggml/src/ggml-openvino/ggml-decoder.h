#pragma once

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "openvino/decoder.h"

#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstring>
#अन्तर्भाव <map>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/partial_shape.hpp>
#अन्तर्भाव <optional>
#अन्तर्भाव <set>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

संरचना ModelParams {
    पूर्णांक ctx = -1;
    पूर्णांक ctx_per_seq = -1;
    पूर्णांक ctx_per_seq_swa = -1;
    पूर्णांक n_seq = 1;
    पूर्णांक n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<पूर्णांक, पूर्णांक> n_heads_kv_per_layer;
    पूर्णांक head_size = -1;
    पूर्णांक state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    पूर्णांक n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<पूर्णांक> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    स्थिरांक ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    माप_प्रकार kv_buffer_ctx_id = 0;

    bool same_rope_params(स्थिरांक ModelParams & other) स्थिरांक {
        परत mixed_rope_params == other.mixed_rope_params &&
               स_भेद(rope_params, other.rope_params, आकार(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(स्थिरांक ModelParams & other) स्थिरांक {
        परत same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(स्थिरांक ModelParams & other) स्थिरांक {
        परत same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(स्थिरांक ModelParams & other) स्थिरांक { परत kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

संरचना ComputeParams {
    पूर्णांक n_seq_active = 1;
    पूर्णांक seq_active_start = 0;
    पूर्णांक attention_size = -1;
    पूर्णांक attention_size_swa = -1;
    पूर्णांक attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    पूर्णांक swa_window = -1;
    पूर्णांक आदान_len = -1;
    पूर्णांक token_len_per_seq = -1;
    पूर्णांक past_kv_len = -1;
    पूर्णांक output_len = 1;

    पूर्णांक cache_rs_reset_idx = -1;
    पूर्णांक cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    पूर्णांक s_copy_active_slot_len = -1;
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

    संरचना RsWriteback {
        पूर्णांक slot_begin = 0;  // first cache slot written by the CPY
        पूर्णांक src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<पूर्णांक> extract_layer_from_name(स्थिरांक std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(स्थिरांक ggml_tensor * गाठ);

class GgmlOvDecoder : सार्वजनिक ov::frontend::ggml::GgmlDecoder {
सार्वजनिक:
    स्थिर std::string get_tensor_name(स्थिरांक ggml_cgraph * cgraph, स्थिरांक ggml_tensor * tensor);
    संरचना गाठ_Info {
        ggml_tensor * गाठ;
        std::string गाठ_name;
        std::string गाठ_op_type;
        std::map<std::string, ggml_tensor *> गाठ_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> गाठ_inputs_views;
        std::vector<std::string> गाठ_inputs_names;
        पूर्णांक गाठ_op_case = 0;
        शून्य * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::गाठ_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  पूर्णांक prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::गाठ_2>> & model_weights);

    आभासी ov::Any get_attribute(स्थिरांक std::string & name) स्थिरांक override {
        परत nullptr;
        GGML_UNUSED(name);
    }

    आभासी ov::PartialShape get_आदान_shape(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक override;

    आभासी std::vector<माप_प्रकार> get_आदान_stride(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक override;

    आभासी माप_प्रकार get_view_आदान_size(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक override;

    आभासी माप_प्रकार get_view_आदान_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक override;

    आभासी माप_प्रकार get_view_आदान_src_offset(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक override;

    आभासी std::vector<माप_प्रकार> get_view_आदान_stride(पूर्णांक गाठ_idx,
                                                      स्थिरांक std::string & name,
                                                      माप_प्रकार view_index) स्थिरांक override;

    आभासी std::vector<माप_प्रकार> get_view_आदान_src_stride(पूर्णांक गाठ_idx,
                                                          स्थिरांक std::string & name,
                                                          माप_प्रकार view_index) स्थिरांक override;

    आभासी ov::Shape get_view_आदान_ggml_shape(पूर्णांक गाठ_idx,
                                                स्थिरांक std::string & name,
                                                माप_प्रकार view_index) स्थिरांक override;

    आभासी ov::Shape get_view_आदान_src_ggml_shape(पूर्णांक गाठ_idx,
                                                    स्थिरांक std::string & name,
                                                    माप_प्रकार view_index) स्थिरांक override;

    आभासी ov::PartialShape get_view_आदान_ov_shape(पूर्णांक गाठ_idx,
                                                     स्थिरांक std::string & name,
                                                     माप_प्रकार view_index) स्थिरांक override;

    आभासी ov::PartialShape get_view_आदान_src_ov_shape(पूर्णांक गाठ_idx,
                                                         स्थिरांक std::string & name,
                                                         माप_प्रकार view_index) स्थिरांक override;

    आभासी std::string get_view_आदान_name(पूर्णांक गाठ_idx, स्थिरांक std::string & name, माप_प्रकार view_index) स्थिरांक override;

    आभासी std::string get_view_आदान_src_name(पूर्णांक गाठ_idx,
                                                स्थिरांक std::string & name,
                                                माप_प्रकार view_index) स्थिरांक override;

    आभासी ov::element::Type get_आदान_type(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक override;

    आभासी माप_प्रकार get_आदान_size() स्थिरांक override;

    आभासी माप_प्रकार get_आदान_size(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी शून्य get_आदान_गाठ(माप_प्रकार आदान_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                माप_प्रकार & producer_output_port_index) स्थिरांक override {
        GGML_UNUSED(आदान_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    आभासी std::vector<std::string> get_आदान_names(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी ov::PartialShape get_output_shape(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी ov::element::Type get_output_type(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी std::vector<माप_प्रकार> get_output_stride(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी int32_t * get_आदान_op_params(पूर्णांक गाठ_idx, स्थिरांक std::string & name) स्थिरांक override;

    आभासी int32_t * get_output_op_params(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी माप_प्रकार get_output_op_offset(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी std::vector<std::string> get_output_names(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी std::string get_inplace_op_src(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी bool is_view_like_alias_of(पूर्णांक गाठ_idx, स्थिरांक std::string & view_src_name) स्थिरांक override;

    आभासी स्थिरांक std::string & get_op_type() स्थिरांक override;

    आभासी स्थिरांक std::string & get_op_type(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी स्थिरांक std::string & get_op_name() स्थिरांक override;

    आभासी स्थिरांक std::string & get_op_name(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी int32_t get_op_dynamic_dim(पूर्णांक गाठ_idx) स्थिरांक override;

    आभासी शून्य visit_subgraph(
        std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णांक गाठ_idx)> गाठ_visitor) स्थिरांक override;

    ggml_tensor * get_आदान_ggml_tensor(स्थिरांक std::string & name) स्थिरांक { परत m_inputs.at(name); }

    आभासी पूर्णांक get_op_case(पूर्णांक गाठ_idx) स्थिरांक override { परत m_गाठ_info_list[गाठ_idx].गाठ_op_case; }

    आभासी स्थिरांक std::map<std::string, ov::frontend::ggml::Model_आदान_Info> & get_model_inputs() स्थिरांक override {
        परत m_model_inputs;
    }

    आभासी स्थिरांक std::map<std::string, ov::frontend::ggml::Model_Extra_आदान_Info> & get_model_extra_inputs() स्थिरांक override {
        परत m_model_extra_inputs;
    }

    आभासी स्थिरांक std::map<std::string, std::shared_ptr<ov::गाठ_2>> & get_model_weights() स्थिरांक override {
        परत m_model_weights;
    }

    आभासी std::set<std::string> get_model_output_names() स्थिरांक override { परत m_model_output_names; }

    स्थिरांक std::map<std::string, ggml_tensor *> & get_model_outputs() स्थिरांक { परत m_model_outputs; }

    आभासी पूर्णांक get_ctx_size() स्थिरांक { परत m_model_params.ctx; }

    आभासी पूर्णांक get_ctx_per_seq() स्थिरांक { परत m_model_params.ctx_per_seq; }

    आभासी पूर्णांक get_ctx_per_seq_swa() स्थिरांक { परत m_model_params.ctx_per_seq_swa; }

    आभासी पूर्णांक get_n_seq() स्थिरांक { परत m_model_params.n_seq; }

    आभासी पूर्णांक is_swa_layer(पूर्णांक layer) स्थिरांक override {
        परत std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    पूर्णांक get_n_heads_kv_for_layer(पूर्णांक layer) स्थिरांक {
        स्वतः it = m_model_params.n_heads_kv_per_layer.find(layer);
        परत it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    पूर्णांक get_n_heads_kv_for_tensor(स्थिरांक ggml_tensor * kv_tensor) स्थिरांक {
        जर (स्वतः layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_मूल्य()) {
            परत get_n_heads_kv_for_layer(layer.मूल्य());
        }
        परत m_model_params.n_heads_kv;
    }

    पूर्णांक get_past_kv_len() स्थिरांक { परत m_compute_params.past_kv_len; }

    पूर्णांक get_आदान_len() स्थिरांक { परत m_compute_params.आदान_len; }

    आभासी int32_t * get_rope_params() स्थिरांक override { परत const_cast<int32_t *>(m_model_params.rope_params); }

    आभासी bool has_mixed_rope_params() स्थिरांक override { परत m_model_params.mixed_rope_params; }

    आभासी पूर्णांक get_ssm_state_size() स्थिरांक override { परत m_model_params.state_size; }

    आभासी std::map<std::string, std::string> get_kv_param_res_names() स्थिरांक override;

    आभासी bool is_static() स्थिरांक override { परत m_is_static; }

    आभासी bool is_stateful() स्थिरांक override { परत m_is_stateful; }

    पूर्णांक get_static_n_tokens() स्थिरांक { परत m_is_prefill ? m_prefill_chunk_size : 1; }

    आभासी bool is_splited_model() स्थिरांक override { परत m_model_is_splitted; }

    ov::PartialShape get_graph_आदान_shape(स्थिरांक ggml_tensor * op,
                                           स्थिरांक ggml_tensor * आदान,
                                           पूर्णांक dynamic_dim_index = -1) स्थिरांक;

    स्थिर शून्य dump_cgraph(स्थिरांक ggml_cgraph * cgraph, std::string & filename);

    स्थिर std::shared_ptr<ov::गाठ_2> create_weight_गाठ(ggml_tensor * tensor, bool naive = false);

    स्थिर std::map<std::string, std::shared_ptr<ov::गाठ_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    स्थिर std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    स्थिरांक ggml_tensor * get_tensor_used_op(स्थिरांक ggml_tensor * tensor) स्थिरांक;

    स्थिरांक ggml_tensor * get_tensor_from_name(स्थिरांक std::string & name) स्थिरांक;

    शून्य clear_model_weights() { m_model_weights.clear(); }

    स्थिर std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() स्थिरांक { परत m_model_params; }

    ComputeParams get_compute_params() स्थिरांक { परत m_compute_params; }

    शून्य set_model_params(स्थिरांक ModelParams & model_params) { m_model_params = model_params; }

    शून्य set_compute_params(स्थिरांक ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    पूर्णांक m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    स्थिर ov::Shape get_shape(स्थिरांक ggml_tensor * tensor);
    स्थिर std::vector<माप_प्रकार> get_stride(स्थिरांक ggml_tensor * tensor);
    स्थिर ov::element::Type get_ov_type(स्थिरांक ggml_tensor * tensor);
    स्थिर std::string compute_op_type(स्थिरांक ggml_tensor * गाठ);
    शून्य add_extra_inputs();

    शून्य update_io(ggml_cgraph * cgraph);
    bool is_bound_to(स्थिरांक ggml_cgraph * cgraph) स्थिरांक;

    स्थिर bool is_inp_tok(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    स्थिर bool is_inp_pos(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    स्थिर पूर्णांक get_inp_pos_n_planes(स्थिरांक ggml_tensor * op) {
        स्थिरांक पूर्णांक mode = op->op_params[2];
        परत (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    स्थिर bool is_inp_emb(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    स्थिर bool is_inp_mask(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    स्थिर bool is_inp_mean(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_आदान) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    स्थिर bool is_inp_scale_rows(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_MUL && tensor == op->src[1] && म_भेद(tensor->name, "inp_scale_rows") == 0;
    }

    स्थिर bool is_rope_freqs_weight(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    ओळीत स्थिर bool is_recurrent_cache(स्थिरांक ggml_tensor * tensor) {
        परत tensor != nullptr && (म_भेदन(tensor->name, "cache_r_l", म_माप("cache_r_l")) == 0 ||
                                     म_भेदन(tensor->name, "cache_s_l", म_माप("cache_s_l")) == 0 ||
                                     म_भेदन(tensor->name, "cache_ple_r_l", म_माप("cache_ple_r_l")) == 0);
    }

    ओळीत स्थिर bool is_cache(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    ओळीत स्थिर bool is_kvcache(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        जर (tensor == nullptr || is_recurrent_cache(tensor)) {
            परत false;
        }
        परत (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    ओळीत स्थिर bool is_conv_state_writeback(स्थिरांक ggml_tensor * गाठ) {
        परत गाठ->op == GGML_OP_CPY && गाठ->view_src != nullptr && is_recurrent_cache(गाठ->view_src) &&
               गाठ->src[0] != nullptr && गाठ->src[0]->op == GGML_OP_VIEW && गाठ->src[0]->src[0] != nullptr &&
               गाठ->src[0]->src[0]->op == GGML_OP_CONCAT && गाठ->src[1] != nullptr &&
               गाठ->src[1]->op == GGML_OP_VIEW && गाठ->src[1]->view_src == गाठ->view_src;
    }

    स्थिर bool is_kv_idx(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(स्थिरांक ggml_tensor * tensor) स्थिरांक {
        परत m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    स्थिर bool is_output_idx(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    स्थिर bool is_inp_s_copy(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) {
        परत op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_आदान_ov_name(स्थिरांक ggml_tensor * tensor, स्थिरांक ggml_tensor * op) स्थिरांक {
        जर (is_inp_pos(tensor, op)) {
            परत "inp_pos";
        }
        जर (is_inp_emb(tensor, op)) {
            परत "embd";
        }
        जर (is_inp_mask(tensor, op)) {
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
            जर (m_model_params.swa_layers.empty()) {
                परत "self_kq_mask";
            }
            परत is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        परत tensor->name;
    }

खाजगी:
    शून्य set_आदान_output();
    पूर्णांक compute_op_case(स्थिरांक ggml_tensor * गाठ) स्थिरांक;
    bool गाठ_is_used_as_src(स्थिरांक पूर्णांक गाठ_idx);
    शून्य compute_model_inputs();
    शून्य compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(स्थिरांक ggml_tensor * tensor) स्थिरांक;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    शून्य compute_गाठ_dynamic_dims();

    शून्य validate_cgraph() स्थिरांक;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_आदान_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_आदान_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::गाठ_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<गाठ_Info> m_गाठ_info_list;
    std::map<ggml_tensor *, पूर्णांक> m_गाठ_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

शून्य print_tensor_address_map(स्थिरांक ggml_cgraph * cgraph);
