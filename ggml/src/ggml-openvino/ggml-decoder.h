#pragma once

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "openvino/decoder.h"

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <map>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <openvino/core/partial_shape.hpp>
#அந்தர்பாவ <optional>
#அந்தர்பாவ <set>
#அந்தர்பாவ <string>
#அந்தர்பாவ <vector>

கட்டமைப்பு ModelParams {
    முழு ctx = -1;
    முழு ctx_per_seq = -1;
    முழு ctx_per_seq_swa = -1;
    முழு n_seq = 1;
    முழு n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<முழு, முழு> n_heads_kv_per_layer;
    முழு head_size = -1;
    முழு state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    முழு n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<முழு> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    மாறிலி ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(மாறிலி ModelParams & other) மாறிலி {
        திரும்பு mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, அளவு(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(மாறிலி ModelParams & other) மாறிலி {
        திரும்பு same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(மாறிலி ModelParams & other) மாறிலி {
        திரும்பு same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(மாறிலி ModelParams & other) மாறிலி { திரும்பு kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

கட்டமைப்பு ComputeParams {
    முழு n_seq_active = 1;
    முழு seq_active_start = 0;
    முழு attention_size = -1;
    முழு attention_size_swa = -1;
    முழு attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    முழு swa_window = -1;
    முழு உள்ளீடு_len = -1;
    முழு token_len_per_seq = -1;
    முழு past_kv_len = -1;
    முழு output_len = 1;

    முழு cache_rs_reset_idx = -1;
    முழு cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    முழு s_copy_active_slot_len = -1;
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

    கட்டமைப்பு RsWriteback {
        முழு slot_begin = 0;  // first cache slot written by the CPY
        முழு src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<முழு> extract_layer_from_name(மாறிலி std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(மாறிலி ggml_tensor * முனை);

class GgmlOvDecoder : பொது ov::frontend::ggml::GgmlDecoder {
பொது:
    நிலையான std::string get_tensor_name(மாறிலி ggml_cgraph * cgraph, மாறிலி ggml_tensor * tensor);
    கட்டமைப்பு முனை_Info {
        ggml_tensor * முனை;
        std::string முனை_name;
        std::string முனை_op_type;
        std::map<std::string, ggml_tensor *> முனை_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> முனை_inputs_views;
        std::vector<std::string> முனை_inputs_names;
        முழு முனை_op_case = 0;
        வெறுமை * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::முனை_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  முழு prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::முனை_2>> & model_weights);

    மெய்நிகர் ov::Any get_attribute(மாறிலி std::string & name) மாறிலி override {
        திரும்பு nullptr;
        GGML_UNUSED(name);
    }

    மெய்நிகர் ov::PartialShape get_உள்ளீடு_shape(முழு முனை_idx, மாறிலி std::string & name) மாறிலி override;

    மெய்நிகர் std::vector<size_t> get_உள்ளீடு_stride(முழு முனை_idx, மாறிலி std::string & name) மாறிலி override;

    மெய்நிகர் size_t get_view_உள்ளீடு_size(முழு முனை_idx, மாறிலி std::string & name) மாறிலி override;

    மெய்நிகர் size_t get_view_உள்ளீடு_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி override;

    மெய்நிகர் size_t get_view_உள்ளீடு_src_offset(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி override;

    மெய்நிகர் std::vector<size_t> get_view_உள்ளீடு_stride(முழு முனை_idx,
                                                      மாறிலி std::string & name,
                                                      size_t view_index) மாறிலி override;

    மெய்நிகர் std::vector<size_t> get_view_உள்ளீடு_src_stride(முழு முனை_idx,
                                                          மாறிலி std::string & name,
                                                          size_t view_index) மாறிலி override;

    மெய்நிகர் ov::Shape get_view_உள்ளீடு_ggml_shape(முழு முனை_idx,
                                                மாறிலி std::string & name,
                                                size_t view_index) மாறிலி override;

    மெய்நிகர் ov::Shape get_view_உள்ளீடு_src_ggml_shape(முழு முனை_idx,
                                                    மாறிலி std::string & name,
                                                    size_t view_index) மாறிலி override;

    மெய்நிகர் ov::PartialShape get_view_உள்ளீடு_ov_shape(முழு முனை_idx,
                                                     மாறிலி std::string & name,
                                                     size_t view_index) மாறிலி override;

    மெய்நிகர் ov::PartialShape get_view_உள்ளீடு_src_ov_shape(முழு முனை_idx,
                                                         மாறிலி std::string & name,
                                                         size_t view_index) மாறிலி override;

    மெய்நிகர் std::string get_view_உள்ளீடு_name(முழு முனை_idx, மாறிலி std::string & name, size_t view_index) மாறிலி override;

    மெய்நிகர் std::string get_view_உள்ளீடு_src_name(முழு முனை_idx,
                                                மாறிலி std::string & name,
                                                size_t view_index) மாறிலி override;

    மெய்நிகர் ov::element::Type get_உள்ளீடு_type(முழு முனை_idx, மாறிலி std::string & name) மாறிலி override;

    மெய்நிகர் size_t get_உள்ளீடு_size() மாறிலி override;

    மெய்நிகர் size_t get_உள்ளீடு_size(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் வெறுமை get_உள்ளீடு_முனை(size_t உள்ளீடு_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) மாறிலி override {
        GGML_UNUSED(உள்ளீடு_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    மெய்நிகர் std::vector<std::string> get_உள்ளீடு_names(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் ov::PartialShape get_output_shape(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் ov::element::Type get_output_type(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் std::vector<size_t> get_output_stride(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் int32_t * get_உள்ளீடு_op_params(முழு முனை_idx, மாறிலி std::string & name) மாறிலி override;

    மெய்நிகர் int32_t * get_output_op_params(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் size_t get_output_op_offset(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் std::vector<std::string> get_output_names(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் std::string get_inplace_op_src(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் bool is_view_like_alias_of(முழு முனை_idx, மாறிலி std::string & view_src_name) மாறிலி override;

    மெய்நிகர் மாறிலி std::string & get_op_type() மாறிலி override;

    மெய்நிகர் மாறிலி std::string & get_op_type(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் மாறிலி std::string & get_op_name() மாறிலி override;

    மெய்நிகர் மாறிலி std::string & get_op_name(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் int32_t get_op_dynamic_dim(முழு முனை_idx) மாறிலி override;

    மெய்நிகர் வெறுமை visit_subgraph(
        std::function<வெறுமை(std::shared_ptr<GgmlDecoder>, முழு முனை_idx)> முனை_visitor) மாறிலி override;

    ggml_tensor * get_உள்ளீடு_ggml_tensor(மாறிலி std::string & name) மாறிலி { திரும்பு m_inputs.at(name); }

    மெய்நிகர் முழு get_op_case(முழு முனை_idx) மாறிலி override { திரும்பு m_முனை_info_list[முனை_idx].முனை_op_case; }

    மெய்நிகர் மாறிலி std::map<std::string, ov::frontend::ggml::Model_உள்ளீடு_Info> & get_model_inputs() மாறிலி override {
        திரும்பு m_model_inputs;
    }

    மெய்நிகர் மாறிலி std::map<std::string, ov::frontend::ggml::Model_Extra_உள்ளீடு_Info> & get_model_extra_inputs() மாறிலி override {
        திரும்பு m_model_extra_inputs;
    }

    மெய்நிகர் மாறிலி std::map<std::string, std::shared_ptr<ov::முனை_2>> & get_model_weights() மாறிலி override {
        திரும்பு m_model_weights;
    }

    மெய்நிகர் std::set<std::string> get_model_output_names() மாறிலி override { திரும்பு m_model_output_names; }

    மாறிலி std::map<std::string, ggml_tensor *> & get_model_outputs() மாறிலி { திரும்பு m_model_outputs; }

    மெய்நிகர் முழு get_ctx_size() மாறிலி { திரும்பு m_model_params.ctx; }

    மெய்நிகர் முழு get_ctx_per_seq() மாறிலி { திரும்பு m_model_params.ctx_per_seq; }

    மெய்நிகர் முழு get_ctx_per_seq_swa() மாறிலி { திரும்பு m_model_params.ctx_per_seq_swa; }

    மெய்நிகர் முழு get_n_seq() மாறிலி { திரும்பு m_model_params.n_seq; }

    மெய்நிகர் முழு is_swa_layer(முழு layer) மாறிலி override {
        திரும்பு std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    முழு get_n_heads_kv_for_layer(முழு layer) மாறிலி {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        திரும்பு it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    முழு get_n_heads_kv_for_tensor(மாறிலி ggml_tensor * kv_tensor) மாறிலி {
        எனில் (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_மதிப்பு()) {
            திரும்பு get_n_heads_kv_for_layer(layer.மதிப்பு());
        }
        திரும்பு m_model_params.n_heads_kv;
    }

    முழு get_past_kv_len() மாறிலி { திரும்பு m_compute_params.past_kv_len; }

    முழு get_உள்ளீடு_len() மாறிலி { திரும்பு m_compute_params.உள்ளீடு_len; }

    மெய்நிகர் int32_t * get_rope_params() மாறிலி override { திரும்பு const_cast<int32_t *>(m_model_params.rope_params); }

    மெய்நிகர் bool has_mixed_rope_params() மாறிலி override { திரும்பு m_model_params.mixed_rope_params; }

    மெய்நிகர் முழு get_ssm_state_size() மாறிலி override { திரும்பு m_model_params.state_size; }

    மெய்நிகர் std::map<std::string, std::string> get_kv_param_res_names() மாறிலி override;

    மெய்நிகர் bool is_static() மாறிலி override { திரும்பு m_is_static; }

    மெய்நிகர் bool is_stateful() மாறிலி override { திரும்பு m_is_stateful; }

    முழு get_static_n_tokens() மாறிலி { திரும்பு m_is_prefill ? m_prefill_chunk_size : 1; }

    மெய்நிகர் bool is_splited_model() மாறிலி override { திரும்பு m_model_is_splitted; }

    ov::PartialShape get_graph_உள்ளீடு_shape(மாறிலி ggml_tensor * op,
                                           மாறிலி ggml_tensor * உள்ளீடு,
                                           முழு dynamic_dim_index = -1) மாறிலி;

    நிலையான வெறுமை dump_cgraph(மாறிலி ggml_cgraph * cgraph, std::string & filename);

    நிலையான std::shared_ptr<ov::முனை_2> create_weight_முனை(ggml_tensor * tensor, bool naive = false);

    நிலையான std::map<std::string, std::shared_ptr<ov::முனை_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    நிலையான std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    மாறிலி ggml_tensor * get_tensor_used_op(மாறிலி ggml_tensor * tensor) மாறிலி;

    மாறிலி ggml_tensor * get_tensor_from_name(மாறிலி std::string & name) மாறிலி;

    வெறுமை clear_model_weights() { m_model_weights.clear(); }

    நிலையான std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() மாறிலி { திரும்பு m_model_params; }

    ComputeParams get_compute_params() மாறிலி { திரும்பு m_compute_params; }

    வெறுமை set_model_params(மாறிலி ModelParams & model_params) { m_model_params = model_params; }

    வெறுமை set_compute_params(மாறிலி ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    முழு m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    நிலையான ov::Shape get_shape(மாறிலி ggml_tensor * tensor);
    நிலையான std::vector<size_t> get_stride(மாறிலி ggml_tensor * tensor);
    நிலையான ov::element::Type get_ov_type(மாறிலி ggml_tensor * tensor);
    நிலையான std::string compute_op_type(மாறிலி ggml_tensor * முனை);
    வெறுமை add_extra_inputs();

    வெறுமை update_io(ggml_cgraph * cgraph);
    bool is_bound_to(மாறிலி ggml_cgraph * cgraph) மாறிலி;

    நிலையான bool is_inp_tok(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    நிலையான bool is_inp_pos(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    நிலையான முழு get_inp_pos_n_planes(மாறிலி ggml_tensor * op) {
        மாறிலி முழு mode = op->op_params[2];
        திரும்பு (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    நிலையான bool is_inp_emb(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    நிலையான bool is_inp_mask(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    நிலையான bool is_inp_mean(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_உள்ளீடு) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    நிலையான bool is_inp_scale_rows(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    நிலையான bool is_rope_freqs_weight(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    உள்வரி நிலையான bool is_recurrent_cache(மாறிலி ggml_tensor * tensor) {
        திரும்பு tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    உள்வரி நிலையான bool is_cache(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    உள்வரி நிலையான bool is_kvcache(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        எனில் (tensor == nullptr || is_recurrent_cache(tensor)) {
            திரும்பு false;
        }
        திரும்பு (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    உள்வரி நிலையான bool is_conv_state_writeback(மாறிலி ggml_tensor * முனை) {
        திரும்பு முனை->op == GGML_OP_CPY && முனை->view_src != nullptr && is_recurrent_cache(முனை->view_src) &&
               முனை->src[0] != nullptr && முனை->src[0]->op == GGML_OP_VIEW && முனை->src[0]->src[0] != nullptr &&
               முனை->src[0]->src[0]->op == GGML_OP_CONCAT && முனை->src[1] != nullptr &&
               முனை->src[1]->op == GGML_OP_VIEW && முனை->src[1]->view_src == முனை->view_src;
    }

    நிலையான bool is_kv_idx(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(மாறிலி ggml_tensor * tensor) மாறிலி {
        திரும்பு m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    நிலையான bool is_output_idx(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    நிலையான bool is_inp_s_copy(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) {
        திரும்பு op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_உள்ளீடு_ov_name(மாறிலி ggml_tensor * tensor, மாறிலி ggml_tensor * op) மாறிலி {
        எனில் (is_inp_pos(tensor, op)) {
            திரும்பு "inp_pos";
        }
        எனில் (is_inp_emb(tensor, op)) {
            திரும்பு "embd";
        }
        எனில் (is_inp_mask(tensor, op)) {
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
            எனில் (m_model_params.swa_layers.empty()) {
                திரும்பு "self_kq_mask";
            }
            திரும்பு is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        திரும்பு tensor->name;
    }

தனிப்பட்ட:
    வெறுமை set_உள்ளீடு_output();
    முழு compute_op_case(மாறிலி ggml_tensor * முனை) மாறிலி;
    bool முனை_is_used_as_src(மாறிலி முழு முனை_idx);
    வெறுமை compute_model_inputs();
    வெறுமை compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(மாறிலி ggml_tensor * tensor) மாறிலி;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    வெறுமை compute_முனை_dynamic_dims();

    வெறுமை validate_cgraph() மாறிலி;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_உள்ளீடு_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_உள்ளீடு_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::முனை_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<முனை_Info> m_முனை_info_list;
    std::map<ggml_tensor *, முழு> m_முனை_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

வெறுமை print_tensor_address_map(மாறிலி ggml_cgraph * cgraph);
