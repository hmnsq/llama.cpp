#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "openvino/decoder.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/partial_shape.hpp>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <set>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

ಸಂರಚನಾ ModelParams {
    ಪೂರ್ಣಾಙ್ಕ ctx = -1;
    ಪೂರ್ಣಾಙ್ಕ ctx_per_seq = -1;
    ಪೂರ್ಣಾಙ್ಕ ctx_per_seq_swa = -1;
    ಪೂರ್ಣಾಙ್ಕ n_seq = 1;
    ಪೂರ್ಣಾಙ್ಕ n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> n_heads_kv_per_layer;
    ಪೂರ್ಣಾಙ್ಕ head_size = -1;
    ಪೂರ್ಣಾಙ್ಕ state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    ಪೂರ್ಣಾಙ್ಕ n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<ಪೂರ್ಣಾಙ್ಕ> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ಧ್ರುವ ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(ಧ್ರುವ ModelParams & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, ಪರಿಮಾಣ(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ಧ್ರುವ ModelParams & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ಧ್ರುವ ModelParams & other) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(ಧ್ರುವ ModelParams & other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

ಸಂರಚನಾ ComputeParams {
    ಪೂರ್ಣಾಙ್ಕ n_seq_active = 1;
    ಪೂರ್ಣಾಙ್ಕ seq_active_start = 0;
    ಪೂರ್ಣಾಙ್ಕ attention_size = -1;
    ಪೂರ್ಣಾಙ್ಕ attention_size_swa = -1;
    ಪೂರ್ಣಾಙ್ಕ attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    ಪೂರ್ಣಾಙ್ಕ swa_window = -1;
    ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_len = -1;
    ಪೂರ್ಣಾಙ್ಕ token_len_per_seq = -1;
    ಪೂರ್ಣಾಙ್ಕ past_kv_len = -1;
    ಪೂರ್ಣಾಙ್ಕ output_len = 1;

    ಪೂರ್ಣಾಙ್ಕ cache_rs_reset_idx = -1;
    ಪೂರ್ಣಾಙ್ಕ cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    ಪೂರ್ಣಾಙ್ಕ s_copy_active_slot_len = -1;
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

    ಸಂರಚನಾ RsWriteback {
        ಪೂರ್ಣಾಙ್ಕ slot_begin = 0;  // first cache slot written by the CPY
        ಪೂರ್ಣಾಙ್ಕ src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<ಪೂರ್ಣಾಙ್ಕ> extract_layer_from_name(ಧ್ರುವ std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(ಧ್ರುವ ggml_tensor * ಗಂಟು);

class GgmlOvDecoder : ಸಾರ್ವಜನಿಕ ov::frontend::ggml::GgmlDecoder {
ಸಾರ್ವಜನಿಕ:
    ಸ್ಥಿರ std::string get_tensor_name(ಧ್ರುವ ggml_cgraph * cgraph, ಧ್ರುವ ggml_tensor * tensor);
    ಸಂರಚನಾ ಗಂಟು_Info {
        ggml_tensor * ಗಂಟು;
        std::string ಗಂಟು_name;
        std::string ಗಂಟು_op_type;
        std::map<std::string, ggml_tensor *> ಗಂಟು_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> ಗಂಟು_inputs_views;
        std::vector<std::string> ಗಂಟು_inputs_names;
        ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_op_case = 0;
        ಶೂನ್ಯ * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  ಪೂರ್ಣಾಙ್ಕ prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & model_weights);

    ಆಭಾಸ ov::Any get_attribute(ಧ್ರುವ std::string & name) ಧ್ರುವ override {
        ಪ್ರತ್ಯಾಗಮ nullptr;
        GGML_UNUSED(name);
    }

    ಆಭಾಸ ov::PartialShape get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ override;

    ಆಭಾಸ std::vector<size_t> get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ override;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ override;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ size_t get_view_ಒಳಹರಿವು_src_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ std::vector<size_t> get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                      ಧ್ರುವ std::string & name,
                                                      size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ std::vector<size_t> get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                          ಧ್ರುವ std::string & name,
                                                          size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ ov::Shape get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                ಧ್ರುವ std::string & name,
                                                size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ ov::Shape get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                    ಧ್ರುವ std::string & name,
                                                    size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ ov::PartialShape get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                     ಧ್ರುವ std::string & name,
                                                     size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ ov::PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                         ಧ್ರುವ std::string & name,
                                                         size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ std::string get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name, size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ std::string get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx,
                                                ಧ್ರುವ std::string & name,
                                                size_t view_index) ಧ್ರುವ override;

    ಆಭಾಸ ov::element::Type get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ override;

    ಆಭಾಸ size_t get_ಒಳಹರಿವು_size() ಧ್ರುವ override;

    ಆಭಾಸ size_t get_ಒಳಹರಿವು_size(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ ಶೂನ್ಯ get_ಒಳಹರಿವು_ಗಂಟು(size_t ಒಳಹರಿವು_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ಧ್ರುವ override {
        GGML_UNUSED(ಒಳಹರಿವು_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    ಆಭಾಸ std::vector<std::string> get_ಒಳಹರಿವು_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ ov::PartialShape get_output_shape(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ ov::element::Type get_output_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ std::vector<size_t> get_output_stride(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ int32_t * get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & name) ಧ್ರುವ override;

    ಆಭಾಸ int32_t * get_output_op_params(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ size_t get_output_op_offset(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ std::vector<std::string> get_output_names(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ std::string get_inplace_op_src(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ bool is_view_like_alias_of(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ std::string & view_src_name) ಧ್ರುವ override;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_type() ಧ್ರುವ override;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_type(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_name() ಧ್ರುವ override;

    ಆಭಾಸ ಧ್ರುವ std::string & get_op_name(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ int32_t get_op_dynamic_dim(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override;

    ಆಭಾಸ ಶೂನ್ಯ visit_subgraph(
        std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಧ್ರುವ override;

    ggml_tensor * get_ಒಳಹರಿವು_ggml_tensor(ಧ್ರುವ std::string & name) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_inputs.at(name); }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_op_case(ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_case; }

    ಆಭಾಸ ಧ್ರುವ std::map<std::string, ov::frontend::ggml::Model_ಒಳಹರಿವು_Info> & get_model_inputs() ಧ್ರುವ override {
        ಪ್ರತ್ಯಾಗಮ m_model_inputs;
    }

    ಆಭಾಸ ಧ್ರುವ std::map<std::string, ov::frontend::ggml::Model_Extra_ಒಳಹರಿವು_Info> & get_model_extra_inputs() ಧ್ರುವ override {
        ಪ್ರತ್ಯಾಗಮ m_model_extra_inputs;
    }

    ಆಭಾಸ ಧ್ರುವ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & get_model_weights() ಧ್ರುವ override {
        ಪ್ರತ್ಯಾಗಮ m_model_weights;
    }

    ಆಭಾಸ std::set<std::string> get_model_output_names() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_model_output_names; }

    ಧ್ರುವ std::map<std::string, ggml_tensor *> & get_model_outputs() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_outputs; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_ctx_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_params.ctx; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_ctx_per_seq() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_params.ctx_per_seq; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_ctx_per_seq_swa() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_params.ctx_per_seq_swa; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_n_seq() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_params.n_seq; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ is_swa_layer(ಪೂರ್ಣಾಙ್ಕ layer) ಧ್ರುವ override {
        ಪ್ರತ್ಯಾಗಮ std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    ಪೂರ್ಣಾಙ್ಕ get_n_heads_kv_for_layer(ಪೂರ್ಣಾಙ್ಕ layer) ಧ್ರುವ {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        ಪ್ರತ್ಯಾಗಮ it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    ಪೂರ್ಣಾಙ್ಕ get_n_heads_kv_for_tensor(ಧ್ರುವ ggml_tensor * kv_tensor) ಧ್ರುವ {
        ಯದಿ (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_ಮೌಲ್ಯ()) {
            ಪ್ರತ್ಯಾಗಮ get_n_heads_kv_for_layer(layer.ಮೌಲ್ಯ());
        }
        ಪ್ರತ್ಯಾಗಮ m_model_params.n_heads_kv;
    }

    ಪೂರ್ಣಾಙ್ಕ get_past_kv_len() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_compute_params.past_kv_len; }

    ಪೂರ್ಣಾಙ್ಕ get_ಒಳಹರಿವು_len() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_compute_params.ಒಳಹರಿವು_len; }

    ಆಭಾಸ int32_t * get_rope_params() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ const_cast<int32_t *>(m_model_params.rope_params); }

    ಆಭಾಸ bool has_mixed_rope_params() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_model_params.mixed_rope_params; }

    ಆಭಾಸ ಪೂರ್ಣಾಙ್ಕ get_ssm_state_size() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_model_params.state_size; }

    ಆಭಾಸ std::map<std::string, std::string> get_kv_param_res_names() ಧ್ರುವ override;

    ಆಭಾಸ bool is_static() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_is_static; }

    ಆಭಾಸ bool is_stateful() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_is_stateful; }

    ಪೂರ್ಣಾಙ್ಕ get_static_n_tokens() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_is_prefill ? m_prefill_chunk_size : 1; }

    ಆಭಾಸ bool is_splited_model() ಧ್ರುವ override { ಪ್ರತ್ಯಾಗಮ m_model_is_splitted; }

    ov::PartialShape get_graph_ಒಳಹರಿವು_shape(ಧ್ರುವ ggml_tensor * op,
                                           ಧ್ರುವ ggml_tensor * ಒಳಹರಿವು,
                                           ಪೂರ್ಣಾಙ್ಕ dynamic_dim_index = -1) ಧ್ರುವ;

    ಸ್ಥಿರ ಶೂನ್ಯ dump_cgraph(ಧ್ರುವ ggml_cgraph * cgraph, std::string & filename);

    ಸ್ಥಿರ std::shared_ptr<ov::ಗಂಟು_2> create_weight_ಗಂಟು(ggml_tensor * tensor, bool naive = false);

    ಸ್ಥಿರ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ಸ್ಥಿರ std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    ಧ್ರುವ ggml_tensor * get_tensor_used_op(ಧ್ರುವ ggml_tensor * tensor) ಧ್ರುವ;

    ಧ್ರುವ ggml_tensor * get_tensor_from_name(ಧ್ರುವ std::string & name) ಧ್ರುವ;

    ಶೂನ್ಯ clear_model_weights() { m_model_weights.clear(); }

    ಸ್ಥಿರ std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_model_params; }

    ComputeParams get_compute_params() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ m_compute_params; }

    ಶೂನ್ಯ set_model_params(ಧ್ರುವ ModelParams & model_params) { m_model_params = model_params; }

    ಶೂನ್ಯ set_compute_params(ಧ್ರುವ ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    ಪೂರ್ಣಾಙ್ಕ m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ಸ್ಥಿರ ov::Shape get_shape(ಧ್ರುವ ggml_tensor * tensor);
    ಸ್ಥಿರ std::vector<size_t> get_stride(ಧ್ರುವ ggml_tensor * tensor);
    ಸ್ಥಿರ ov::element::Type get_ov_type(ಧ್ರುವ ggml_tensor * tensor);
    ಸ್ಥಿರ std::string compute_op_type(ಧ್ರುವ ggml_tensor * ಗಂಟು);
    ಶೂನ್ಯ add_extra_inputs();

    ಶೂನ್ಯ update_io(ggml_cgraph * cgraph);
    bool is_bound_to(ಧ್ರುವ ggml_cgraph * cgraph) ಧ್ರುವ;

    ಸ್ಥಿರ bool is_inp_tok(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ಸ್ಥಿರ bool is_inp_pos(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ get_inp_pos_n_planes(ಧ್ರುವ ggml_tensor * op) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ mode = op->op_params[2];
        ಪ್ರತ್ಯಾಗಮ (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ಸ್ಥಿರ bool is_inp_emb(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    ಸ್ಥಿರ bool is_inp_mask(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ಸ್ಥಿರ bool is_inp_mean(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ಸ್ಥಿರ bool is_inp_scale_rows(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ಸ್ಥಿರ bool is_rope_freqs_weight(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ bool is_recurrent_cache(ಧ್ರುವ ggml_tensor * tensor) {
        ಪ್ರತ್ಯಾಗಮ tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ bool is_cache(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ bool is_kvcache(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಯದಿ (tensor == nullptr || is_recurrent_cache(tensor)) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಪ್ರತ್ಯಾಗಮ (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    ಅನ್ತಃಸ್ಥ ಸ್ಥಿರ bool is_conv_state_writeback(ಧ್ರುವ ggml_tensor * ಗಂಟು) {
        ಪ್ರತ್ಯಾಗಮ ಗಂಟು->op == GGML_OP_CPY && ಗಂಟು->view_src != nullptr && is_recurrent_cache(ಗಂಟು->view_src) &&
               ಗಂಟು->src[0] != nullptr && ಗಂಟು->src[0]->op == GGML_OP_VIEW && ಗಂಟು->src[0]->src[0] != nullptr &&
               ಗಂಟು->src[0]->src[0]->op == GGML_OP_CONCAT && ಗಂಟು->src[1] != nullptr &&
               ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src;
    }

    ಸ್ಥಿರ bool is_kv_idx(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(ಧ್ರುವ ggml_tensor * tensor) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    ಸ್ಥಿರ bool is_output_idx(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ಸ್ಥಿರ bool is_inp_s_copy(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) {
        ಪ್ರತ್ಯಾಗಮ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_ಒಳಹರಿವು_ov_name(ಧ್ರುವ ggml_tensor * tensor, ಧ್ರುವ ggml_tensor * op) ಧ್ರುವ {
        ಯದಿ (is_inp_pos(tensor, op)) {
            ಪ್ರತ್ಯಾಗಮ "inp_pos";
        }
        ಯದಿ (is_inp_emb(tensor, op)) {
            ಪ್ರತ್ಯಾಗಮ "embd";
        }
        ಯದಿ (is_inp_mask(tensor, op)) {
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
            ಯದಿ (m_model_params.swa_layers.empty()) {
                ಪ್ರತ್ಯಾಗಮ "self_kq_mask";
            }
            ಪ್ರತ್ಯಾಗಮ is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ಪ್ರತ್ಯಾಗಮ tensor->name;
    }

ಗುಪ್ತ:
    ಶೂನ್ಯ set_ಒಳಹರಿವು_output();
    ಪೂರ್ಣಾಙ್ಕ compute_op_case(ಧ್ರುವ ggml_tensor * ಗಂಟು) ಧ್ರುವ;
    bool ಗಂಟು_is_used_as_src(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx);
    ಶೂನ್ಯ compute_model_inputs();
    ಶೂನ್ಯ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(ಧ್ರುವ ggml_tensor * tensor) ಧ್ರುವ;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    ಶೂನ್ಯ compute_ಗಂಟು_dynamic_dims();

    ಶೂನ್ಯ validate_cgraph() ಧ್ರುವ;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_ಒಳಹರಿವು_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_ಒಳಹರಿವು_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<ಗಂಟು_Info> m_ಗಂಟು_info_list;
    std::map<ggml_tensor *, ಪೂರ್ಣಾಙ್ಕ> m_ಗಂಟು_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

ಶೂನ್ಯ print_tensor_address_map(ಧ್ರುವ ggml_cgraph * cgraph);
