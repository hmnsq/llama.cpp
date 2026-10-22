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
    पूर्णाङ्क ctx = -1;
    पूर्णाङ्क ctx_per_seq = -1;
    पूर्णाङ्क ctx_per_seq_swa = -1;
    पूर्णाङ्क n_seq = 1;
    पूर्णाङ्क n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<पूर्णाङ्क, पूर्णाङ्क> n_heads_kv_per_layer;
    पूर्णाङ्क head_size = -1;
    पूर्णाङ्क state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    पूर्णाङ्क n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<पूर्णाङ्क> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ध्रुव ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(ध्रुव ModelParams & other) ध्रुव {
        प्रत्यागम mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, परिमाण(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ध्रुव ModelParams & other) ध्रुव {
        प्रत्यागम same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ध्रुव ModelParams & other) ध्रुव {
        प्रत्यागम same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(ध्रुव ModelParams & other) ध्रुव { प्रत्यागम kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

संरचना ComputeParams {
    पूर्णाङ्क n_seq_active = 1;
    पूर्णाङ्क seq_active_start = 0;
    पूर्णाङ्क attention_size = -1;
    पूर्णाङ्क attention_size_swa = -1;
    पूर्णाङ्क attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    पूर्णाङ्क swa_window = -1;
    पूर्णाङ्क निवेश_len = -1;
    पूर्णाङ्क token_len_per_seq = -1;
    पूर्णाङ्क past_kv_len = -1;
    पूर्णाङ्क output_len = 1;

    पूर्णाङ्क cache_rs_reset_idx = -1;
    पूर्णाङ्क cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    पूर्णाङ्क s_copy_active_slot_len = -1;
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
        पूर्णाङ्क slot_begin = 0;  // first cache slot written by the CPY
        पूर्णाङ्क src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<पूर्णाङ्क> extract_layer_from_name(ध्रुव std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(ध्रुव ggml_tensor * गाँठ);

class GgmlOvDecoder : सार्वजनिक ov::frontend::ggml::GgmlDecoder {
सार्वजनिक:
    स्थिर std::string get_tensor_name(ध्रुव ggml_cgraph * cgraph, ध्रुव ggml_tensor * tensor);
    संरचना गाँठ_Info {
        ggml_tensor * गाँठ;
        std::string गाँठ_name;
        std::string गाँठ_op_type;
        std::map<std::string, ggml_tensor *> गाँठ_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> गाँठ_inputs_views;
        std::vector<std::string> गाँठ_inputs_names;
        पूर्णाङ्क गाँठ_op_case = 0;
        शून्य * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  पूर्णाङ्क prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & model_weights);

    आभास ov::Any get_attribute(ध्रुव std::string & name) ध्रुव override {
        प्रत्यागम nullptr;
        GGML_UNUSED(name);
    }

    आभास ov::PartialShape get_निवेश_shape(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव override;

    आभास std::vector<size_t> get_निवेश_stride(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव override;

    आभास size_t get_view_निवेश_size(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव override;

    आभास size_t get_view_निवेश_offset(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव override;

    आभास size_t get_view_निवेश_src_offset(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव override;

    आभास std::vector<size_t> get_view_निवेश_stride(पूर्णाङ्क गाँठ_idx,
                                                      ध्रुव std::string & name,
                                                      size_t view_index) ध्रुव override;

    आभास std::vector<size_t> get_view_निवेश_src_stride(पूर्णाङ्क गाँठ_idx,
                                                          ध्रुव std::string & name,
                                                          size_t view_index) ध्रुव override;

    आभास ov::Shape get_view_निवेश_ggml_shape(पूर्णाङ्क गाँठ_idx,
                                                ध्रुव std::string & name,
                                                size_t view_index) ध्रुव override;

    आभास ov::Shape get_view_निवेश_src_ggml_shape(पूर्णाङ्क गाँठ_idx,
                                                    ध्रुव std::string & name,
                                                    size_t view_index) ध्रुव override;

    आभास ov::PartialShape get_view_निवेश_ov_shape(पूर्णाङ्क गाँठ_idx,
                                                     ध्रुव std::string & name,
                                                     size_t view_index) ध्रुव override;

    आभास ov::PartialShape get_view_निवेश_src_ov_shape(पूर्णाङ्क गाँठ_idx,
                                                         ध्रुव std::string & name,
                                                         size_t view_index) ध्रुव override;

    आभास std::string get_view_निवेश_name(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name, size_t view_index) ध्रुव override;

    आभास std::string get_view_निवेश_src_name(पूर्णाङ्क गाँठ_idx,
                                                ध्रुव std::string & name,
                                                size_t view_index) ध्रुव override;

    आभास ov::element::Type get_निवेश_type(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव override;

    आभास size_t get_निवेश_size() ध्रुव override;

    आभास size_t get_निवेश_size(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास शून्य get_निवेश_गाँठ(size_t निवेश_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ध्रुव override {
        GGML_UNUSED(निवेश_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    आभास std::vector<std::string> get_निवेश_names(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास ov::PartialShape get_output_shape(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास ov::element::Type get_output_type(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास std::vector<size_t> get_output_stride(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास int32_t * get_निवेश_op_params(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & name) ध्रुव override;

    आभास int32_t * get_output_op_params(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास size_t get_output_op_offset(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास std::vector<std::string> get_output_names(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास std::string get_inplace_op_src(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास bool is_view_like_alias_of(पूर्णाङ्क गाँठ_idx, ध्रुव std::string & view_src_name) ध्रुव override;

    आभास ध्रुव std::string & get_op_type() ध्रुव override;

    आभास ध्रुव std::string & get_op_type(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास ध्रुव std::string & get_op_name() ध्रुव override;

    आभास ध्रुव std::string & get_op_name(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास int32_t get_op_dynamic_dim(पूर्णाङ्क गाँठ_idx) ध्रुव override;

    आभास शून्य visit_subgraph(
        std::function<शून्य(std::shared_ptr<GgmlDecoder>, पूर्णाङ्क गाँठ_idx)> गाँठ_visitor) ध्रुव override;

    ggml_tensor * get_निवेश_ggml_tensor(ध्रुव std::string & name) ध्रुव { प्रत्यागम m_inputs.at(name); }

    आभास पूर्णाङ्क get_op_case(पूर्णाङ्क गाँठ_idx) ध्रुव override { प्रत्यागम m_गाँठ_info_list[गाँठ_idx].गाँठ_op_case; }

    आभास ध्रुव std::map<std::string, ov::frontend::ggml::Model_निवेश_Info> & get_model_inputs() ध्रुव override {
        प्रत्यागम m_model_inputs;
    }

    आभास ध्रुव std::map<std::string, ov::frontend::ggml::Model_Extra_निवेश_Info> & get_model_extra_inputs() ध्रुव override {
        प्रत्यागम m_model_extra_inputs;
    }

    आभास ध्रुव std::map<std::string, std::shared_ptr<ov::गाँठ_2>> & get_model_weights() ध्रुव override {
        प्रत्यागम m_model_weights;
    }

    आभास std::set<std::string> get_model_output_names() ध्रुव override { प्रत्यागम m_model_output_names; }

    ध्रुव std::map<std::string, ggml_tensor *> & get_model_outputs() ध्रुव { प्रत्यागम m_model_outputs; }

    आभास पूर्णाङ्क get_ctx_size() ध्रुव { प्रत्यागम m_model_params.ctx; }

    आभास पूर्णाङ्क get_ctx_per_seq() ध्रुव { प्रत्यागम m_model_params.ctx_per_seq; }

    आभास पूर्णाङ्क get_ctx_per_seq_swa() ध्रुव { प्रत्यागम m_model_params.ctx_per_seq_swa; }

    आभास पूर्णाङ्क get_n_seq() ध्रुव { प्रत्यागम m_model_params.n_seq; }

    आभास पूर्णाङ्क is_swa_layer(पूर्णाङ्क layer) ध्रुव override {
        प्रत्यागम std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    पूर्णाङ्क get_n_heads_kv_for_layer(पूर्णाङ्क layer) ध्रुव {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        प्रत्यागम it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    पूर्णाङ्क get_n_heads_kv_for_tensor(ध्रुव ggml_tensor * kv_tensor) ध्रुव {
        यदि (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_मूल्य()) {
            प्रत्यागम get_n_heads_kv_for_layer(layer.मूल्य());
        }
        प्रत्यागम m_model_params.n_heads_kv;
    }

    पूर्णाङ्क get_past_kv_len() ध्रुव { प्रत्यागम m_compute_params.past_kv_len; }

    पूर्णाङ्क get_निवेश_len() ध्रुव { प्रत्यागम m_compute_params.निवेश_len; }

    आभास int32_t * get_rope_params() ध्रुव override { प्रत्यागम const_cast<int32_t *>(m_model_params.rope_params); }

    आभास bool has_mixed_rope_params() ध्रुव override { प्रत्यागम m_model_params.mixed_rope_params; }

    आभास पूर्णाङ्क get_ssm_state_size() ध्रुव override { प्रत्यागम m_model_params.state_size; }

    आभास std::map<std::string, std::string> get_kv_param_res_names() ध्रुव override;

    आभास bool is_static() ध्रुव override { प्रत्यागम m_is_static; }

    आभास bool is_stateful() ध्रुव override { प्रत्यागम m_is_stateful; }

    पूर्णाङ्क get_static_n_tokens() ध्रुव { प्रत्यागम m_is_prefill ? m_prefill_chunk_size : 1; }

    आभास bool is_splited_model() ध्रुव override { प्रत्यागम m_model_is_splitted; }

    ov::PartialShape get_graph_निवेश_shape(ध्रुव ggml_tensor * op,
                                           ध्रुव ggml_tensor * निवेश,
                                           पूर्णाङ्क dynamic_dim_index = -1) ध्रुव;

    स्थिर शून्य dump_cgraph(ध्रुव ggml_cgraph * cgraph, std::string & filename);

    स्थिर std::shared_ptr<ov::गाँठ_2> create_weight_गाँठ(ggml_tensor * tensor, bool naive = false);

    स्थिर std::map<std::string, std::shared_ptr<ov::गाँठ_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    स्थिर std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    ध्रुव ggml_tensor * get_tensor_used_op(ध्रुव ggml_tensor * tensor) ध्रुव;

    ध्रुव ggml_tensor * get_tensor_from_name(ध्रुव std::string & name) ध्रुव;

    शून्य clear_model_weights() { m_model_weights.clear(); }

    स्थिर std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ध्रुव { प्रत्यागम m_model_params; }

    ComputeParams get_compute_params() ध्रुव { प्रत्यागम m_compute_params; }

    शून्य set_model_params(ध्रुव ModelParams & model_params) { m_model_params = model_params; }

    शून्य set_compute_params(ध्रुव ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    पूर्णाङ्क m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    स्थिर ov::Shape get_shape(ध्रुव ggml_tensor * tensor);
    स्थिर std::vector<size_t> get_stride(ध्रुव ggml_tensor * tensor);
    स्थिर ov::element::Type get_ov_type(ध्रुव ggml_tensor * tensor);
    स्थिर std::string compute_op_type(ध्रुव ggml_tensor * गाँठ);
    शून्य add_extra_inputs();

    शून्य update_io(ggml_cgraph * cgraph);
    bool is_bound_to(ध्रुव ggml_cgraph * cgraph) ध्रुव;

    स्थिर bool is_inp_tok(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    स्थिर bool is_inp_pos(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    स्थिर पूर्णाङ्क get_inp_pos_n_planes(ध्रुव ggml_tensor * op) {
        ध्रुव पूर्णाङ्क mode = op->op_params[2];
        प्रत्यागम (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    स्थिर bool is_inp_emb(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    स्थिर bool is_inp_mask(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    स्थिर bool is_inp_mean(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_निवेश) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    स्थिर bool is_inp_scale_rows(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    स्थिर bool is_rope_freqs_weight(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    अन्तःस्थ स्थिर bool is_recurrent_cache(ध्रुव ggml_tensor * tensor) {
        प्रत्यागम tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    अन्तःस्थ स्थिर bool is_cache(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    अन्तःस्थ स्थिर bool is_kvcache(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        यदि (tensor == nullptr || is_recurrent_cache(tensor)) {
            प्रत्यागम false;
        }
        प्रत्यागम (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    अन्तःस्थ स्थिर bool is_conv_state_writeback(ध्रुव ggml_tensor * गाँठ) {
        प्रत्यागम गाँठ->op == GGML_OP_CPY && गाँठ->view_src != nullptr && is_recurrent_cache(गाँठ->view_src) &&
               गाँठ->src[0] != nullptr && गाँठ->src[0]->op == GGML_OP_VIEW && गाँठ->src[0]->src[0] != nullptr &&
               गाँठ->src[0]->src[0]->op == GGML_OP_CONCAT && गाँठ->src[1] != nullptr &&
               गाँठ->src[1]->op == GGML_OP_VIEW && गाँठ->src[1]->view_src == गाँठ->view_src;
    }

    स्थिर bool is_kv_idx(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(ध्रुव ggml_tensor * tensor) ध्रुव {
        प्रत्यागम m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    स्थिर bool is_output_idx(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    स्थिर bool is_inp_s_copy(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) {
        प्रत्यागम op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_निवेश_ov_name(ध्रुव ggml_tensor * tensor, ध्रुव ggml_tensor * op) ध्रुव {
        यदि (is_inp_pos(tensor, op)) {
            प्रत्यागम "inp_pos";
        }
        यदि (is_inp_emb(tensor, op)) {
            प्रत्यागम "embd";
        }
        यदि (is_inp_mask(tensor, op)) {
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
            यदि (m_model_params.swa_layers.empty()) {
                प्रत्यागम "self_kq_mask";
            }
            प्रत्यागम is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        प्रत्यागम tensor->name;
    }

गुप्त:
    शून्य set_निवेश_output();
    पूर्णाङ्क compute_op_case(ध्रुव ggml_tensor * गाँठ) ध्रुव;
    bool गाँठ_is_used_as_src(ध्रुव पूर्णाङ्क गाँठ_idx);
    शून्य compute_model_inputs();
    शून्य compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(ध्रुव ggml_tensor * tensor) ध्रुव;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    शून्य compute_गाँठ_dynamic_dims();

    शून्य validate_cgraph() ध्रुव;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_निवेश_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_निवेश_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::गाँठ_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<गाँठ_Info> m_गाँठ_info_list;
    std::map<ggml_tensor *, पूर्णाङ्क> m_गाँठ_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

शून्य print_tensor_address_map(ध्रुव ggml_cgraph * cgraph);
