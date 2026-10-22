#pragma once

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "openvino/decoder.h"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <map>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <openvino/core/partial_shape.hpp>
#അന്തര്ഭാവ <optional>
#അന്തര്ഭാവ <set>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <vector>

ഘടന ModelParams {
    പൂർണ്ണസംഖ്യ ctx = -1;
    പൂർണ്ണസംഖ്യ ctx_per_seq = -1;
    പൂർണ്ണസംഖ്യ ctx_per_seq_swa = -1;
    പൂർണ്ണസംഖ്യ n_seq = 1;
    പൂർണ്ണസംഖ്യ n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ> n_heads_kv_per_layer;
    പൂർണ്ണസംഖ്യ head_size = -1;
    പൂർണ്ണസംഖ്യ state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    പൂർണ്ണസംഖ്യ n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<പൂർണ്ണസംഖ്യ> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    സ്ഥിരം ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(സ്ഥിരം ModelParams & other) സ്ഥിരം {
        തിരികെ mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, വലുപ്പം(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(സ്ഥിരം ModelParams & other) സ്ഥിരം {
        തിരികെ same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(സ്ഥിരം ModelParams & other) സ്ഥിരം {
        തിരികെ same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(സ്ഥിരം ModelParams & other) സ്ഥിരം { തിരികെ kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

ഘടന ComputeParams {
    പൂർണ്ണസംഖ്യ n_seq_active = 1;
    പൂർണ്ണസംഖ്യ seq_active_start = 0;
    പൂർണ്ണസംഖ്യ attention_size = -1;
    പൂർണ്ണസംഖ്യ attention_size_swa = -1;
    പൂർണ്ണസംഖ്യ attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    പൂർണ്ണസംഖ്യ swa_window = -1;
    പൂർണ്ണസംഖ്യ നിവേശം_len = -1;
    പൂർണ്ണസംഖ്യ token_len_per_seq = -1;
    പൂർണ്ണസംഖ്യ past_kv_len = -1;
    പൂർണ്ണസംഖ്യ output_len = 1;

    പൂർണ്ണസംഖ്യ cache_rs_reset_idx = -1;
    പൂർണ്ണസംഖ്യ cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    പൂർണ്ണസംഖ്യ s_copy_active_slot_len = -1;
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

    ഘടന RsWriteback {
        പൂർണ്ണസംഖ്യ slot_begin = 0;  // first cache slot written by the CPY
        പൂർണ്ണസംഖ്യ src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<പൂർണ്ണസംഖ്യ> extract_layer_from_name(സ്ഥിരം std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(സ്ഥിരം ggml_tensor * കണ്ണി);

class GgmlOvDecoder : പൊതു ov::frontend::ggml::GgmlDecoder {
പൊതു:
    സ്ഥിതം std::string get_tensor_name(സ്ഥിരം ggml_cgraph * cgraph, സ്ഥിരം ggml_tensor * tensor);
    ഘടന കണ്ണി_Info {
        ggml_tensor * കണ്ണി;
        std::string കണ്ണി_name;
        std::string കണ്ണി_op_type;
        std::map<std::string, ggml_tensor *> കണ്ണി_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> കണ്ണി_inputs_views;
        std::vector<std::string> കണ്ണി_inputs_names;
        പൂർണ്ണസംഖ്യ കണ്ണി_op_case = 0;
        ശൂന്യം * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  പൂർണ്ണസംഖ്യ prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> & model_weights);

    മായാ ov::Any get_attribute(സ്ഥിരം std::string & name) സ്ഥിരം override {
        തിരികെ nullptr;
        GGML_UNUSED(name);
    }

    മായാ ov::PartialShape get_നിവേശം_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം override;

    മായാ std::vector<size_t> get_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം override;

    മായാ size_t get_view_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം override;

    മായാ size_t get_view_നിവേശം_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം override;

    മായാ size_t get_view_നിവേശം_src_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം override;

    മായാ std::vector<size_t> get_view_നിവേശം_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                      സ്ഥിരം std::string & name,
                                                      size_t view_index) സ്ഥിരം override;

    മായാ std::vector<size_t> get_view_നിവേശം_src_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                          സ്ഥിരം std::string & name,
                                                          size_t view_index) സ്ഥിരം override;

    മായാ ov::Shape get_view_നിവേശം_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                സ്ഥിരം std::string & name,
                                                size_t view_index) സ്ഥിരം override;

    മായാ ov::Shape get_view_നിവേശം_src_ggml_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                    സ്ഥിരം std::string & name,
                                                    size_t view_index) സ്ഥിരം override;

    മായാ ov::PartialShape get_view_നിവേശം_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                     സ്ഥിരം std::string & name,
                                                     size_t view_index) സ്ഥിരം override;

    മായാ ov::PartialShape get_view_നിവേശം_src_ov_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                         സ്ഥിരം std::string & name,
                                                         size_t view_index) സ്ഥിരം override;

    മായാ std::string get_view_നിവേശം_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name, size_t view_index) സ്ഥിരം override;

    മായാ std::string get_view_നിവേശം_src_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                                സ്ഥിരം std::string & name,
                                                size_t view_index) സ്ഥിരം override;

    മായാ ov::element::Type get_നിവേശം_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം override;

    മായാ size_t get_നിവേശം_size() സ്ഥിരം override;

    മായാ size_t get_നിവേശം_size(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ ശൂന്യം get_നിവേശം_കണ്ണി(size_t നിവേശം_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) സ്ഥിരം override {
        GGML_UNUSED(നിവേശം_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    മായാ std::vector<std::string> get_നിവേശം_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ ov::PartialShape get_output_shape(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ ov::element::Type get_output_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ std::vector<size_t> get_output_stride(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ int32_t * get_നിവേശം_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & name) സ്ഥിരം override;

    മായാ int32_t * get_output_op_params(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ size_t get_output_op_offset(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ std::vector<std::string> get_output_names(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ std::string get_inplace_op_src(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ bool is_view_like_alias_of(പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം std::string & view_src_name) സ്ഥിരം override;

    മായാ സ്ഥിരം std::string & get_op_type() സ്ഥിരം override;

    മായാ സ്ഥിരം std::string & get_op_type(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ സ്ഥിരം std::string & get_op_name() സ്ഥിരം override;

    മായാ സ്ഥിരം std::string & get_op_name(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ int32_t get_op_dynamic_dim(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override;

    മായാ ശൂന്യം visit_subgraph(
        std::function<ശൂന്യം(std::shared_ptr<GgmlDecoder>, പൂർണ്ണസംഖ്യ കണ്ണി_idx)> കണ്ണി_visitor) സ്ഥിരം override;

    ggml_tensor * get_നിവേശം_ggml_tensor(സ്ഥിരം std::string & name) സ്ഥിരം { തിരികെ m_inputs.at(name); }

    മായാ പൂർണ്ണസംഖ്യ get_op_case(പൂർണ്ണസംഖ്യ കണ്ണി_idx) സ്ഥിരം override { തിരികെ m_കണ്ണി_info_list[കണ്ണി_idx].കണ്ണി_op_case; }

    മായാ സ്ഥിരം std::map<std::string, ov::frontend::ggml::Model_നിവേശം_Info> & get_model_inputs() സ്ഥിരം override {
        തിരികെ m_model_inputs;
    }

    മായാ സ്ഥിരം std::map<std::string, ov::frontend::ggml::Model_Extra_നിവേശം_Info> & get_model_extra_inputs() സ്ഥിരം override {
        തിരികെ m_model_extra_inputs;
    }

    മായാ സ്ഥിരം std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> & get_model_weights() സ്ഥിരം override {
        തിരികെ m_model_weights;
    }

    മായാ std::set<std::string> get_model_output_names() സ്ഥിരം override { തിരികെ m_model_output_names; }

    സ്ഥിരം std::map<std::string, ggml_tensor *> & get_model_outputs() സ്ഥിരം { തിരികെ m_model_outputs; }

    മായാ പൂർണ്ണസംഖ്യ get_ctx_size() സ്ഥിരം { തിരികെ m_model_params.ctx; }

    മായാ പൂർണ്ണസംഖ്യ get_ctx_per_seq() സ്ഥിരം { തിരികെ m_model_params.ctx_per_seq; }

    മായാ പൂർണ്ണസംഖ്യ get_ctx_per_seq_swa() സ്ഥിരം { തിരികെ m_model_params.ctx_per_seq_swa; }

    മായാ പൂർണ്ണസംഖ്യ get_n_seq() സ്ഥിരം { തിരികെ m_model_params.n_seq; }

    മായാ പൂർണ്ണസംഖ്യ is_swa_layer(പൂർണ്ണസംഖ്യ layer) സ്ഥിരം override {
        തിരികെ std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    പൂർണ്ണസംഖ്യ get_n_heads_kv_for_layer(പൂർണ്ണസംഖ്യ layer) സ്ഥിരം {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        തിരികെ it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    പൂർണ്ണസംഖ്യ get_n_heads_kv_for_tensor(സ്ഥിരം ggml_tensor * kv_tensor) സ്ഥിരം {
        എങ്കിൽ (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_മൂല്യം()) {
            തിരികെ get_n_heads_kv_for_layer(layer.മൂല്യം());
        }
        തിരികെ m_model_params.n_heads_kv;
    }

    പൂർണ്ണസംഖ്യ get_past_kv_len() സ്ഥിരം { തിരികെ m_compute_params.past_kv_len; }

    പൂർണ്ണസംഖ്യ get_നിവേശം_len() സ്ഥിരം { തിരികെ m_compute_params.നിവേശം_len; }

    മായാ int32_t * get_rope_params() സ്ഥിരം override { തിരികെ const_cast<int32_t *>(m_model_params.rope_params); }

    മായാ bool has_mixed_rope_params() സ്ഥിരം override { തിരികെ m_model_params.mixed_rope_params; }

    മായാ പൂർണ്ണസംഖ്യ get_ssm_state_size() സ്ഥിരം override { തിരികെ m_model_params.state_size; }

    മായാ std::map<std::string, std::string> get_kv_param_res_names() സ്ഥിരം override;

    മായാ bool is_static() സ്ഥിരം override { തിരികെ m_is_static; }

    മായാ bool is_stateful() സ്ഥിരം override { തിരികെ m_is_stateful; }

    പൂർണ്ണസംഖ്യ get_static_n_tokens() സ്ഥിരം { തിരികെ m_is_prefill ? m_prefill_chunk_size : 1; }

    മായാ bool is_splited_model() സ്ഥിരം override { തിരികെ m_model_is_splitted; }

    ov::PartialShape get_graph_നിവേശം_shape(സ്ഥിരം ggml_tensor * op,
                                           സ്ഥിരം ggml_tensor * നിവേശം,
                                           പൂർണ്ണസംഖ്യ dynamic_dim_index = -1) സ്ഥിരം;

    സ്ഥിതം ശൂന്യം dump_cgraph(സ്ഥിരം ggml_cgraph * cgraph, std::string & filename);

    സ്ഥിതം std::shared_ptr<ov::കണ്ണി_2> create_weight_കണ്ണി(ggml_tensor * tensor, bool naive = false);

    സ്ഥിതം std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    സ്ഥിതം std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    സ്ഥിരം ggml_tensor * get_tensor_used_op(സ്ഥിരം ggml_tensor * tensor) സ്ഥിരം;

    സ്ഥിരം ggml_tensor * get_tensor_from_name(സ്ഥിരം std::string & name) സ്ഥിരം;

    ശൂന്യം clear_model_weights() { m_model_weights.clear(); }

    സ്ഥിതം std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() സ്ഥിരം { തിരികെ m_model_params; }

    ComputeParams get_compute_params() സ്ഥിരം { തിരികെ m_compute_params; }

    ശൂന്യം set_model_params(സ്ഥിരം ModelParams & model_params) { m_model_params = model_params; }

    ശൂന്യം set_compute_params(സ്ഥിരം ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    പൂർണ്ണസംഖ്യ m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    സ്ഥിതം ov::Shape get_shape(സ്ഥിരം ggml_tensor * tensor);
    സ്ഥിതം std::vector<size_t> get_stride(സ്ഥിരം ggml_tensor * tensor);
    സ്ഥിതം ov::element::Type get_ov_type(സ്ഥിരം ggml_tensor * tensor);
    സ്ഥിതം std::string compute_op_type(സ്ഥിരം ggml_tensor * കണ്ണി);
    ശൂന്യം add_extra_inputs();

    ശൂന്യം update_io(ggml_cgraph * cgraph);
    bool is_bound_to(സ്ഥിരം ggml_cgraph * cgraph) സ്ഥിരം;

    സ്ഥിതം bool is_inp_tok(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    സ്ഥിതം bool is_inp_pos(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    സ്ഥിതം പൂർണ്ണസംഖ്യ get_inp_pos_n_planes(സ്ഥിരം ggml_tensor * op) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ mode = op->op_params[2];
        തിരികെ (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    സ്ഥിതം bool is_inp_emb(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    സ്ഥിതം bool is_inp_mask(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    സ്ഥിതം bool is_inp_mean(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_നിവേശം) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    സ്ഥിതം bool is_inp_scale_rows(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    സ്ഥിതം bool is_rope_freqs_weight(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    വരിയിൽ സ്ഥിതം bool is_recurrent_cache(സ്ഥിരം ggml_tensor * tensor) {
        തിരികെ tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    വരിയിൽ സ്ഥിതം bool is_cache(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    വരിയിൽ സ്ഥിതം bool is_kvcache(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        എങ്കിൽ (tensor == nullptr || is_recurrent_cache(tensor)) {
            തിരികെ false;
        }
        തിരികെ (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    വരിയിൽ സ്ഥിതം bool is_conv_state_writeback(സ്ഥിരം ggml_tensor * കണ്ണി) {
        തിരികെ കണ്ണി->op == GGML_OP_CPY && കണ്ണി->view_src != nullptr && is_recurrent_cache(കണ്ണി->view_src) &&
               കണ്ണി->src[0] != nullptr && കണ്ണി->src[0]->op == GGML_OP_VIEW && കണ്ണി->src[0]->src[0] != nullptr &&
               കണ്ണി->src[0]->src[0]->op == GGML_OP_CONCAT && കണ്ണി->src[1] != nullptr &&
               കണ്ണി->src[1]->op == GGML_OP_VIEW && കണ്ണി->src[1]->view_src == കണ്ണി->view_src;
    }

    സ്ഥിതം bool is_kv_idx(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(സ്ഥിരം ggml_tensor * tensor) സ്ഥിരം {
        തിരികെ m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    സ്ഥിതം bool is_output_idx(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    സ്ഥിതം bool is_inp_s_copy(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) {
        തിരികെ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_നിവേശം_ov_name(സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം ggml_tensor * op) സ്ഥിരം {
        എങ്കിൽ (is_inp_pos(tensor, op)) {
            തിരികെ "inp_pos";
        }
        എങ്കിൽ (is_inp_emb(tensor, op)) {
            തിരികെ "embd";
        }
        എങ്കിൽ (is_inp_mask(tensor, op)) {
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
            എങ്കിൽ (m_model_params.swa_layers.empty()) {
                തിരികെ "self_kq_mask";
            }
            തിരികെ is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        തിരികെ tensor->name;
    }

സ്വകാര്യ:
    ശൂന്യം set_നിവേശം_output();
    പൂർണ്ണസംഖ്യ compute_op_case(സ്ഥിരം ggml_tensor * കണ്ണി) സ്ഥിരം;
    bool കണ്ണി_is_used_as_src(സ്ഥിരം പൂർണ്ണസംഖ്യ കണ്ണി_idx);
    ശൂന്യം compute_model_inputs();
    ശൂന്യം compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(സ്ഥിരം ggml_tensor * tensor) സ്ഥിരം;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    ശൂന്യം compute_കണ്ണി_dynamic_dims();

    ശൂന്യം validate_cgraph() സ്ഥിരം;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_നിവേശം_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_നിവേശം_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::കണ്ണി_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<കണ്ണി_Info> m_കണ്ണി_info_list;
    std::map<ggml_tensor *, പൂർണ്ണസംഖ്യ> m_കണ്ണി_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

ശൂന്യം print_tensor_address_map(സ്ഥിരം ggml_cgraph * cgraph);
