#pragma once

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "openvino/decoder.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <map>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <openvino/core/partial_shape.hpp>
#ଅନ୍ତର୍ଭାଵ <optional>
#ଅନ୍ତର୍ଭାଵ <set>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <vector>

ସଂରଚନା ModelParams {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ctx = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ctx_per_seq = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ctx_per_seq_swa = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seq = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> n_heads_kv_per_layer;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା head_size = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା state_size = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::vector<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ଅଚଳ ggml_tensor * swa_mask = nullptr;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(ଅଚଳ ModelParams & other) ଅଚଳ {
        ଫେରନ୍ତୁ mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, ଆକାର(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ଅଚଳ ModelParams & other) ଅଚଳ {
        ଫେରନ୍ତୁ same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ଅଚଳ ModelParams & other) ଅଚଳ {
        ଫେରନ୍ତୁ same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_buffer_changed(ଅଚଳ ModelParams & other) ଅଚଳ { ଫେରନ୍ତୁ kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

ସଂରଚନା ComputeParams {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_seq_active = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା seq_active_start = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା attention_size = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା attention_size_swa = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା attention_size_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା swa_window = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_len = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା token_len_per_seq = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା past_kv_len = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_len = 1;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cache_rs_reset_idx = -1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା s_copy_active_slot_len = -1;
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

    ସଂରଚନା RsWriteback {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା slot_begin = 0;  // first cache slot written by the CPY
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> extract_layer_from_name(ଅଚଳ std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_sum_add(ଅଚଳ ggml_tensor * ଗଣ୍ଠି);

class GgmlOvDecoder : ସର୍ବସାଧାରଣ ov::frontend::ggml::GgmlDecoder {
ସର୍ବସାଧାରଣ:
    ସ୍ଥିର std::string get_tensor_name(ଅଚଳ ggml_cgraph * cgraph, ଅଚଳ ggml_tensor * tensor);
    ସଂରଚନା ଗଣ୍ଠି_Info {
        ggml_tensor * ଗଣ୍ଠି;
        std::string ଗଣ୍ଠି_name;
        std::string ଗଣ୍ଠି_op_type;
        std::map<std::string, ggml_tensor *> ଗଣ୍ଠି_inputs;
        std::map<std::string, std::vector<std::pair<std::string, ggml_tensor *>>> ଗଣ୍ଠି_inputs_views;
        std::vector<std::string> ଗଣ୍ଠି_inputs_names;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_op_case = 0;
        ଶୂନ୍ୟ * data_addr;
    };

    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool model_is_splitted = false,
                  bool is_prefill = false,
                  ପୂର୍ଣ୍ଣସଂଖ୍ୟା prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> & model_weights);

    ଆଭାସୀ ov::Any get_attribute(ଅଚଳ std::string & name) ଅଚଳ override {
        ଫେରନ୍ତୁ nullptr;
        GGML_UNUSED(name);
    }

    ଆଭାସୀ ov::PartialShape get_ନିବେଶ_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ override;

    ଆଭାସୀ std::vector<size_t> get_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ override;

    ଆଭାସୀ size_t get_view_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ override;

    ଆଭାସୀ size_t get_view_ନିବେଶ_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ override;

    ଆଭାସୀ size_t get_view_ନିବେଶ_src_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ override;

    ଆଭାସୀ std::vector<size_t> get_view_ନିବେଶ_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                      ଅଚଳ std::string & name,
                                                      size_t view_index) ଅଚଳ override;

    ଆଭାସୀ std::vector<size_t> get_view_ନିବେଶ_src_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                          ଅଚଳ std::string & name,
                                                          size_t view_index) ଅଚଳ override;

    ଆଭାସୀ ov::Shape get_view_ନିବେଶ_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                ଅଚଳ std::string & name,
                                                size_t view_index) ଅଚଳ override;

    ଆଭାସୀ ov::Shape get_view_ନିବେଶ_src_ggml_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                    ଅଚଳ std::string & name,
                                                    size_t view_index) ଅଚଳ override;

    ଆଭାସୀ ov::PartialShape get_view_ନିବେଶ_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                     ଅଚଳ std::string & name,
                                                     size_t view_index) ଅଚଳ override;

    ଆଭାସୀ ov::PartialShape get_view_ନିବେଶ_src_ov_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                         ଅଚଳ std::string & name,
                                                         size_t view_index) ଅଚଳ override;

    ଆଭାସୀ std::string get_view_ନିବେଶ_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name, size_t view_index) ଅଚଳ override;

    ଆଭାସୀ std::string get_view_ନିବେଶ_src_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                                ଅଚଳ std::string & name,
                                                size_t view_index) ଅଚଳ override;

    ଆଭାସୀ ov::element::Type get_ନିବେଶ_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ override;

    ଆଭାସୀ size_t get_ନିବେଶ_size() ଅଚଳ override;

    ଆଭାସୀ size_t get_ନିବେଶ_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ ଶୂନ୍ୟ get_ନିବେଶ_ଗଣ୍ଠି(size_t ନିବେଶ_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) ଅଚଳ override {
        GGML_UNUSED(ନିବେଶ_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    ଆଭାସୀ std::vector<std::string> get_ନିବେଶ_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ ov::PartialShape get_output_shape(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ ov::element::Type get_output_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ std::vector<size_t> get_output_stride(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ int32_t * get_ନିବେଶ_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & name) ଅଚଳ override;

    ଆଭାସୀ int32_t * get_output_op_params(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ size_t get_output_op_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ std::vector<std::string> get_output_names(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ std::string get_inplace_op_src(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ bool is_view_like_alias_of(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ std::string & view_src_name) ଅଚଳ override;

    ଆଭାସୀ ଅଚଳ std::string & get_op_type() ଅଚଳ override;

    ଆଭାସୀ ଅଚଳ std::string & get_op_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ ଅଚଳ std::string & get_op_name() ଅଚଳ override;

    ଆଭାସୀ ଅଚଳ std::string & get_op_name(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ int32_t get_op_dynamic_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override;

    ଆଭାସୀ ଶୂନ୍ୟ visit_subgraph(
        std::function<ଶୂନ୍ୟ(std::shared_ptr<GgmlDecoder>, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx)> ଗଣ୍ଠି_visitor) ଅଚଳ override;

    ggml_tensor * get_ନିବେଶ_ggml_tensor(ଅଚଳ std::string & name) ଅଚଳ { ଫେରନ୍ତୁ m_inputs.at(name); }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_op_case(ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) ଅଚଳ override { ଫେରନ୍ତୁ m_ଗଣ୍ଠି_info_list[ଗଣ୍ଠି_idx].ଗଣ୍ଠି_op_case; }

    ଆଭାସୀ ଅଚଳ std::map<std::string, ov::frontend::ggml::Model_ନିବେଶ_Info> & get_model_inputs() ଅଚଳ override {
        ଫେରନ୍ତୁ m_model_inputs;
    }

    ଆଭାସୀ ଅଚଳ std::map<std::string, ov::frontend::ggml::Model_Extra_ନିବେଶ_Info> & get_model_extra_inputs() ଅଚଳ override {
        ଫେରନ୍ତୁ m_model_extra_inputs;
    }

    ଆଭାସୀ ଅଚଳ std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> & get_model_weights() ଅଚଳ override {
        ଫେରନ୍ତୁ m_model_weights;
    }

    ଆଭାସୀ std::set<std::string> get_model_output_names() ଅଚଳ override { ଫେରନ୍ତୁ m_model_output_names; }

    ଅଚଳ std::map<std::string, ggml_tensor *> & get_model_outputs() ଅଚଳ { ଫେରନ୍ତୁ m_model_outputs; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ctx_size() ଅଚଳ { ଫେରନ୍ତୁ m_model_params.ctx; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ctx_per_seq() ଅଚଳ { ଫେରନ୍ତୁ m_model_params.ctx_per_seq; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ctx_per_seq_swa() ଅଚଳ { ଫେରନ୍ତୁ m_model_params.ctx_per_seq_swa; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_n_seq() ଅଚଳ { ଫେରନ୍ତୁ m_model_params.n_seq; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is_swa_layer(ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer) ଅଚଳ override {
        ଫେରନ୍ତୁ std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_n_heads_kv_for_layer(ପୂର୍ଣ୍ଣସଂଖ୍ୟା layer) ଅଚଳ {
        auto it = m_model_params.n_heads_kv_per_layer.find(layer);
        ଫେରନ୍ତୁ it != m_model_params.n_heads_kv_per_layer.end() ? it->second : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_n_heads_kv_for_tensor(ଅଚଳ ggml_tensor * kv_tensor) ଅଚଳ {
        ଯଦି (auto layer = extract_layer_from_name(std::string(kv_tensor->name)); layer.has_ମୂଲ୍ୟ()) {
            ଫେରନ୍ତୁ get_n_heads_kv_for_layer(layer.ମୂଲ୍ୟ());
        }
        ଫେରନ୍ତୁ m_model_params.n_heads_kv;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_past_kv_len() ଅଚଳ { ଫେରନ୍ତୁ m_compute_params.past_kv_len; }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ନିବେଶ_len() ଅଚଳ { ଫେରନ୍ତୁ m_compute_params.ନିବେଶ_len; }

    ଆଭାସୀ int32_t * get_rope_params() ଅଚଳ override { ଫେରନ୍ତୁ const_cast<int32_t *>(m_model_params.rope_params); }

    ଆଭାସୀ bool has_mixed_rope_params() ଅଚଳ override { ଫେରନ୍ତୁ m_model_params.mixed_rope_params; }

    ଆଭାସୀ ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ssm_state_size() ଅଚଳ override { ଫେରନ୍ତୁ m_model_params.state_size; }

    ଆଭାସୀ std::map<std::string, std::string> get_kv_param_res_names() ଅଚଳ override;

    ଆଭାସୀ bool is_static() ଅଚଳ override { ଫେରନ୍ତୁ m_is_static; }

    ଆଭାସୀ bool is_stateful() ଅଚଳ override { ଫେରନ୍ତୁ m_is_stateful; }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_static_n_tokens() ଅଚଳ { ଫେରନ୍ତୁ m_is_prefill ? m_prefill_chunk_size : 1; }

    ଆଭାସୀ bool is_splited_model() ଅଚଳ override { ଫେରନ୍ତୁ m_model_is_splitted; }

    ov::PartialShape get_graph_ନିବେଶ_shape(ଅଚଳ ggml_tensor * op,
                                           ଅଚଳ ggml_tensor * ନିବେଶ,
                                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା dynamic_dim_index = -1) ଅଚଳ;

    ସ୍ଥିର ଶୂନ୍ୟ dump_cgraph(ଅଚଳ ggml_cgraph * cgraph, std::string & filename);

    ସ୍ଥିର std::shared_ptr<ov::ଗଣ୍ଠି_2> create_weight_ଗଣ୍ଠି(ggml_tensor * tensor, bool naive = false);

    ସ୍ଥିର std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ସ୍ଥିର std::set<std::string> collect_weight_names(ggml_cgraph * cgraph);

    ଅଚଳ ggml_tensor * get_tensor_used_op(ଅଚଳ ggml_tensor * tensor) ଅଚଳ;

    ଅଚଳ ggml_tensor * get_tensor_from_name(ଅଚଳ std::string & name) ଅଚଳ;

    ଶୂନ୍ୟ clear_model_weights() { m_model_weights.clear(); }

    ସ୍ଥିର std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ଅଚଳ { ଫେରନ୍ତୁ m_model_params; }

    ComputeParams get_compute_params() ଅଚଳ { ଫେରନ୍ତୁ m_compute_params; }

    ଶୂନ୍ୟ set_model_params(ଅଚଳ ModelParams & model_params) { m_model_params = model_params; }

    ଶୂନ୍ୟ set_compute_params(ଅଚଳ ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_prefill_chunk_size = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ସ୍ଥିର ov::Shape get_shape(ଅଚଳ ggml_tensor * tensor);
    ସ୍ଥିର std::vector<size_t> get_stride(ଅଚଳ ggml_tensor * tensor);
    ସ୍ଥିର ov::element::Type get_ov_type(ଅଚଳ ggml_tensor * tensor);
    ସ୍ଥିର std::string compute_op_type(ଅଚଳ ggml_tensor * ଗଣ୍ଠି);
    ଶୂନ୍ୟ add_extra_inputs();

    ଶୂନ୍ୟ update_io(ggml_cgraph * cgraph);
    bool is_bound_to(ଅଚଳ ggml_cgraph * cgraph) ଅଚଳ;

    ସ୍ଥିର bool is_inp_tok(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ସ୍ଥିର bool is_inp_pos(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_inp_pos_n_planes(ଅଚଳ ggml_tensor * op) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = op->op_params[2];
        ଫେରନ୍ତୁ (mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION || (mode & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ସ୍ଥିର bool is_inp_emb(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    ସ୍ଥିର bool is_inp_mask(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ସ୍ଥିର bool is_inp_mean(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_ନିବେଶ) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ସ୍ଥିର bool is_inp_scale_rows(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ସ୍ଥିର bool is_rope_freqs_weight(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    ଧାଡ଼ିଭିତରେ ସ୍ଥିର bool is_recurrent_cache(ଅଚଳ ggml_tensor * tensor) {
        ଫେରନ୍ତୁ tensor != nullptr && (strncmp(tensor->name, "cache_r_l", strlen("cache_r_l")) == 0 ||
                                     strncmp(tensor->name, "cache_s_l", strlen("cache_s_l")) == 0 ||
                                     strncmp(tensor->name, "cache_ple_r_l", strlen("cache_ple_r_l")) == 0);
    }

    ଧାଡ଼ିଭିତରେ ସ୍ଥିର bool is_cache(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    ଧାଡ଼ିଭିତରେ ସ୍ଥିର bool is_kvcache(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଯଦି (tensor == nullptr || is_recurrent_cache(tensor)) {
            ଫେରନ୍ତୁ false;
        }
        ଫେରନ୍ତୁ (tensor->buffer != nullptr && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_SET_ROWS && op->src[2] == tensor);
    }

    ଧାଡ଼ିଭିତରେ ସ୍ଥିର bool is_conv_state_writeback(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) {
        ଫେରନ୍ତୁ ଗଣ୍ଠି->op == GGML_OP_CPY && ଗଣ୍ଠି->view_src != nullptr && is_recurrent_cache(ଗଣ୍ଠି->view_src) &&
               ଗଣ୍ଠି->src[0] != nullptr && ଗଣ୍ଠି->src[0]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[0]->src[0] != nullptr &&
               ଗଣ୍ଠି->src[0]->src[0]->op == GGML_OP_CONCAT && ଗଣ୍ଠି->src[1] != nullptr &&
               ଗଣ୍ଠି->src[1]->op == GGML_OP_VIEW && ଗଣ୍ଠି->src[1]->view_src == ଗଣ୍ଠି->view_src;
    }

    ସ୍ଥିର bool is_kv_idx(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    bool is_swa_mask(ଅଚଳ ggml_tensor * tensor) ଅଚଳ {
        ଫେରନ୍ତୁ m_model_params.swa_mask != nullptr && tensor == m_model_params.swa_mask;
    }

    ସ୍ଥିର bool is_output_idx(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ସ୍ଥିର bool is_inp_s_copy(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) {
        ଫେରନ୍ତୁ op->op == GGML_OP_GET_ROWS && tensor == op->src[1] &&
               op->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY;
    }

    std::string get_graph_ନିବେଶ_ov_name(ଅଚଳ ggml_tensor * tensor, ଅଚଳ ggml_tensor * op) ଅଚଳ {
        ଯଦି (is_inp_pos(tensor, op)) {
            ଫେରନ୍ତୁ "inp_pos";
        }
        ଯଦି (is_inp_emb(tensor, op)) {
            ଫେରନ୍ତୁ "embd";
        }
        ଯଦି (is_inp_mask(tensor, op)) {
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
            ଯଦି (m_model_params.swa_layers.empty()) {
                ଫେରନ୍ତୁ "self_kq_mask";
            }
            ଫେରନ୍ତୁ is_swa_mask(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ଫେରନ୍ତୁ tensor->name;
    }

ବ୍ୟକ୍ତିଗତ:
    ଶୂନ୍ୟ set_ନିବେଶ_output();
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା compute_op_case(ଅଚଳ ggml_tensor * ଗଣ୍ଠି) ଅଚଳ;
    bool ଗଣ୍ଠି_is_used_as_src(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx);
    ଶୂନ୍ୟ compute_model_inputs();
    ଶୂନ୍ୟ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_leaf(ଅଚଳ ggml_tensor * tensor) ଅଚଳ;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    ଶୂନ୍ୟ compute_ଗଣ୍ଠି_dynamic_dims();

    ଶୂନ୍ୟ validate_cgraph() ଅଚଳ;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_ନିବେଶ_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_ନିବେଶ_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::ଗଣ୍ଠି_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::set<std::string> m_model_output_names;
    std::vector<ଗଣ୍ଠି_Info> m_ଗଣ୍ଠି_info_list;
    std::map<ggml_tensor *, ପୂର୍ଣ୍ଣସଂଖ୍ୟା> m_ଗଣ୍ଠି_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

ଶୂନ୍ୟ print_tensor_address_map(ଅଚଳ ggml_cgraph * cgraph);
