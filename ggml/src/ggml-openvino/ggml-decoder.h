#pragma once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml.h"
#ضمن "openvino/decoder.h"

#ضمن <cstdint>
#ضمن <cstring>
#ضمن <map>
#ضمن <memory>
#ضمن <openvino/core/partial_shape.hpp>
#ضمن <optional>
#ضمن <set>
#ضمن <string>
#ضمن <vector>

بنية ModelParams {
    صحيح ctx = -1;
    صحيح ctx_per_seq = -1;
    صحيح ctx_per_seq_swa = -1;
    صحيح n_seq = 1;
    صحيح n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<صحيح, صحيح> n_heads_kv_per_layer;
    صحيح head_size = -1;
    صحيح state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    صحيح n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<صحيح> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ثابت ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(ثابت ModelParams & other) ثابت {
        أرجع mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, حجم(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ثابت ModelParams & other) ثابت {
        أرجع same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ثابت ModelParams & other) ثابت {
        أرجع same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(ثابت ModelParams & other) ثابت { أرجع kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

بنية ComputeParams {
    صحيح n_seq_active = 1;
    صحيح seq_active_start = 0;
    صحيح attention_size = -1;
    صحيح attention_size_swa = -1;
    صحيح attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    صحيح swa_window = -1;
    صحيح input_len = -1;
    صحيح token_len_per_seq = -1;
    صحيح past_kv_len = -1;
    صحيح output_len = 1;

    صحيح cache_rs_reset_idx = -1;
    صحيح cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    صحيح s_copy_active_slot_len = -1;
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

    بنية RsWriteback {
        صحيح slot_begin = 0;  // first cache slot written by the CPY
        صحيح src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<صحيح> extract_layer_from_name(ثابت std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(ثابت ggml_tensor * node);

class GgmlOvDecoder : عام ov::frontend::ggml::GgmlDecoder {
عام:
    ساكن std::string get_tensor_name(ثابت ggml_cgraph * cgraph, ثابت ggml_tensor * tensor);
    بنية NodeInfo {
        ggml_tensor * node;
        std::string node_name;
        std::string node_op_type;
        std::map<std::string, ggml_tensor *> node_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> node_inputs_views;
        std::vector<std::string> node_inputs_names;
        صحيح node_op_case = 0;
        فراغ * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  صحيح prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    virtual ov::Any get_attribute(ثابت std::string & name) ثابت override {
        أرجع nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(صحيح node_idx, ثابت std::string & name) ثابت override;

    virtual std::vector<size_t> get_input_stride(صحيح node_idx, ثابت std::string & name) ثابت override;

    virtual size_t get_view_input_size(صحيح node_idx, ثابت std::string & name) ثابت override;

    virtual size_t get_view_input_offset(صحيح node_idx, ثابت std::string & name, size_t view_index) ثابت override;

    virtual size_t get_view_input_src_offset(صحيح node_idx, ثابت std::string & name, size_t view_index) ثابت override;

    virtual std::vector<size_t> get_view_input_stride(صحيح node_idx,
                                                      ثابت std::string & name,
                                                      size_t view_index) ثابت override;

    virtual std::vector<size_t> get_view_input_src_stride(صحيح node_idx,
                                                          ثابت std::string & name,
                                                          size_t view_index) ثابت override;

    virtual ov::Shape get_view_input_ggml_shape(صحيح node_idx,
                                                ثابت std::string & name,
                                                size_t view_index) ثابت override;

    virtual ov::Shape get_view_input_src_ggml_shape(صحيح node_idx,
                                                    ثابت std::string & name,
                                                    size_t view_index) ثابت override;

    virtual ov::PartialShape get_view_input_ov_shape(صحيح node_idx,
                                                     ثابت std::string & name,
                                                     size_t view_index) ثابت override;

    virtual ov::PartialShape get_view_input_src_ov_shape(صحيح node_idx,
                                                         ثابت std::string & name,
                                                         size_t view_index) ثابت override;

    virtual std::string get_view_input_name(صحيح node_idx, ثابت std::string & name, size_t view_index) ثابت override;

    virtual std::string get_view_input_src_name(صحيح node_idx,
                                                ثابت std::string & name,
                                                size_t view_index) ثابت override;

    virtual ov::element::Type get_input_type(صحيح node_idx, ثابت std::string & name) ثابت override;

    virtual size_t get_input_size() ثابت override;

    virtual size_t get_input_size(صحيح node_idx) ثابت override;

    virtual فراغ get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ثابت override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::vector<std::string> get_input_names(صحيح node_idx) ثابت override;

    virtual ov::PartialShape get_output_shape(صحيح node_idx) ثابت override;

    virtual ov::element::Type get_output_type(صحيح node_idx) ثابت override;

    virtual std::vector<size_t> get_output_stride(صحيح node_idx) ثابت override;

    virtual int32_t * get_input_op_params(صحيح node_idx, ثابت std::string & name) ثابت override;

    virtual int32_t * get_output_op_params(صحيح node_idx) ثابت override;

    virtual size_t get_output_op_offset(صحيح node_idx) ثابت override;

    virtual std::vector<std::string> get_output_names(صحيح node_idx) ثابت override;

    virtual std::string get_inplace_op_src(صحيح node_idx) ثابت override;

    virtual bool is_view_like_alias_of(صحيح node_idx, ثابت std::string & view_src_name) ثابت override;

    virtual ثابت std::string & get_op_type() ثابت override;

    virtual ثابت std::string & get_op_type(صحيح node_idx) ثابت override;

    virtual ثابت std::string & get_op_name() ثابت override;

    virtual ثابت std::string & get_op_name(صحيح node_idx) ثابت override;

    virtual int32_t get_op_dynamic_dim(صحيح node_idx) ثابت override;

    virtual فراغ visit_subgraph(
        std::function<فراغ(std::shared_ptr<GgmlDecoder>, صحيح node_idx)> node_visitor) ثابت override;

    ggml_tensor * get_input_ggml_tensor(ثابت std::string & name) ثابت { أرجع m_inputs.at(name); }

    virtual صحيح get_op_case(صحيح node_idx) ثابت override { أرجع m_node_info_list[node_idx].node_op_case; }

    virtual ثابت std::map<std::string, ov::frontend::ggml::ModelInputInfo> & get_model_inputs() ثابت override {
        أرجع m_model_inputs;
    }

    virtual ثابت std::map<std::string, ov::frontend::ggml::ModelExtraInputInfo> & get_model_extra_inputs() ثابت override {
        أرجع m_model_extra_inputs;
    }

    virtual ثابت std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() ثابت override {
        أرجع m_model_weights;
    }

    virtual std::set<std::string> get_model_output_names() ثابت override { أرجع m_model_output_names; }

    ثابت std::map<std::string, ggml_tensor *> & get_model_outputs() ثابت { أرجع m_model_outputs; }

    virtual صحيح get_ctx_size() ثابت { أرجع m_model_params.ctx; }

    virtual صحيح get_ctx_per_seq() ثابت { أرجع m_model_params.ctx_per_seq; }

    virtual صحيح get_ctx_per_seq_swa() ثابت { أرجع m_model_params.ctx_per_seq_swa; }

    virtual صحيح get_n_seq() ثابت { أرجع m_model_params.n_seq; }

    virtual صحيح is_swa_layer(صحيح layer) ثابت override {
        أرجع std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    صحيح get_n_heads_kv_for_layer(صحيح layer) ثابت {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        أرجع it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    صحيح get_n_heads_kv_for_tensor(ثابت ggml_tensor * kv_tensor) ثابت {
        𐤀𐤌 (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_value()) {
            أرجع get_n_heads_kv_for_layer(layer.value());
        }
        أرجع m_model_params.n_heads_kv;
    }

    صحيح get_past_kv_len() ثابت { أرجع m_compute_params.past_kv_len; }

    صحيح get_input_len() ثابت { أرجع m_compute_params.input_len; }

    virtual int32_t * get_rope_params() ثابت override { أرجع const_cast<int32_t *>(m_model_params.rope_params); }

    virtual bool has_mixed_rope_params() ثابت override { أرجع m_model_params.mixed_rope_params; }

    virtual صحيح get_ssm_state_size() ثابت override { أرجع m_model_params.state_size; }

    virtual std::map<std::string, std::string> get_kv_param_res_names() ثابت override;

    virtual bool is_static() ثابت override { أرجع m_is_static; }

    virtual bool is_stateful() ثابت override { أرجع m_is_stateful; }

    صحيح get_static_n_tokens() ثابت { أرجع m_is_prefill ? m_prefill_chunk_size : 1; }

    virtual bool is_splited_model() ثابت override { أرجع m_model_is_splitted; }

    ov::PartialShape get_graph_input_shape(ثابت ggml_tensor * op,
                                           ثابت ggml_tensor * input,
                                           صحيح dynamic_dim_index = -1) ثابت;

    ساكن فراغ dump_cgraph(ثابت ggml_cgraph * cgraph, std::string & filename);

    ساكن std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor, bool naive = false);

    ساكن std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ساكن std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    ثابت ggml_tensor * get_tensor_used_op(ثابت ggml_tensor * tensor) ثابت;

    ثابت ggml_tensor * get_tensor_from_name(ثابت std::string & name) ثابت;

    فراغ clear_model_weights() { m_model_weights.clear(); }

    ساكن std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ثابت { أرجع m_model_params; }

    ComputeParams get_compute_params() ثابت { أرجع m_compute_params; }

    فراغ set_model_params(ثابت ModelParams & model_params) { m_model_params = model_params; }

    فراغ set_compute_params(ثابت ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    صحيح m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ساكن ov::Shape get_shape(ثابت ggml_tensor * tensor);
    ساكن std::vector<size_t> get_stride(ثابت ggml_tensor * tensor);
    ساكن ov::element::Type get_ov_type(ثابت ggml_tensor * tensor);
    ساكن std::string compute_op_type(ثابت ggml_tensor * node);
    فراغ add_extra_inputs();

    فراغ update_io(ggml_cgraph * cgraph);
    bool is_bound_to(ثابت ggml_cgraph * cgraph) ثابت;

    ساكن bool is_inp_tok(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ساكن bool is_inp_pos(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ساكن صحيح get_inp_pos_n_planes(ثابت ggml_tensor * op) {
        ثابت صحيح mode = op->op_params[2];
        أرجع (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ساكن bool is_inp_emb(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    ساكن bool is_inp_mask(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ساكن bool is_inp_mean(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_INPUT) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ساكن bool is_inp_scale_rows(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ساكن bool is_rope_freqs_weight(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    مضمن ساكن bool is_recurrent_cache(ثابت ggml_tensor * tensor) {
        أرجع tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    مضمن ساكن bool is_cache(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    مضمن ساكن bool is_kvcache(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        𐤀𐤌 (tensor == nullptr || is_recurrent_cache(tensor)) {
            أرجع false;
        }
        أرجع (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    مضمن ساكن bool is_conv_state_writeback(ثابت ggml_tensor * node) {
        أرجع node->op == GGML_OP_CPY && node->view_src != nullptr && is_recurrent_cache(node->view_src) &&
               node->src[0] != nullptr && node->src[0]->op == GGML_OP_VIEW && node->src[0]->src[0] != nullptr &&
               node->src[0]->src[0]->op == GGML_OP_CONCAT && node->src[1] != nullptr &&
               node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src;
    }

    ساكن bool is_kv_idx(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(ثابت ggml_tensor * tensor) ثابت {
        أرجع m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    ساكن bool is_output_idx(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ساكن bool is_inp_s_copy(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) {
        أرجع op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_input_ov_name(ثابت ggml_tensor * tensor, ثابت ggml_tensor * op) ثابت {
        𐤀𐤌 (is_inp_pos(tensor, op)) {
            أرجع "inp_pos";
        }
        𐤀𐤌 (is_inp_emb(tensor, op)) {
            أرجع "embd";
        }
        𐤀𐤌 (is_inp_mask(tensor, op)) {
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
            𐤀𐤌 (m_model_params.swa_layers.empty()) {
                أرجع "self_kq_mask";
            }
            أرجع is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        أرجع tensor->name;
    }

خاص:
    فراغ set_input_output();
    صحيح compute_op_case(ثابت ggml_tensor * node) ثابت;
    bool node_is_used_as_src(ثابت صحيح node_idx);
    فراغ compute_model_inputs();
    فراغ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(ثابت ggml_tensor * tensor) ثابت;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    فراغ compute_node_dynamic_dims();

    فراغ validate_cgraph() ثابت;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::ModelInputInfo> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::ModelExtraInputInfo> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<NodeInfo> m_node_info_list;
    std::map<ggml_tensor *, صحيح> m_node_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

فراغ print_tensor_address_map(ثابت ggml_cgraph * cgraph);
