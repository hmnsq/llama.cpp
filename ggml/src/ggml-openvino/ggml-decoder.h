#ಸೂಚನೆ once

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

ರಚನೆ ModelParams {
    ಪೂರ್ಣಾಂಕ ctx = -1;
    ಪೂರ್ಣಾಂಕ ctx_ಪ್ರತಿ_seq = -1;
    ಪೂರ್ಣಾಂಕ ctx_ಪ್ರತಿ_seq_swa = -1;
    ಪೂರ್ಣಾಂಕ n_seq = 1;
    ಪೂರ್ಣಾಂಕ n_heads_kv = -1;
    // Per-layer KV head count. gemma-4 12B interleaves 8 x 256 sliding layers with 1 x 512
    // full-attention layers, so no single scalar describes every layer. Keyed by layer, not by
    // layer TYPE, because the SWA classification depends on the context size (extents tie at a
    // small -c) while the head count does not.
    std::map<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> n_heads_kv_ಪ್ರತಿ_layer;
    ಪೂರ್ಣಾಂಕ ತಲೆ_ಗಾತ್ರ = -1;
    ಪೂರ್ಣಾಂಕ state_ಗಾತ್ರ = -1;  // for SSM molels, eg qwen35
    int32_t rope_params[16] = {};
    ಪೂರ್ಣಾಂಕ n_rs_slots = -1;
    bool has_rs_rollback = false;
    bool mixed_rope_params = false;
    bool is_cacheless_attn = false;
    std::ಸದಿಶ<ಪೂರ್ಣಾಂಕ> swa_layers;
    // The sliding-window mask tensor, identified in compute_llm_params() by grouping attention
    // layers on the mask they consume. Only used to tell the two masks apart when naming OV
    // parameters -- both carry the same tensor name. Null when the graph has a single mask.
    ಸ್ಥಿರಾಂಕ ggml_tensor * swa_ಮುಖವಾಡ = nullptr;

    std::ಸದಿಶ<std::string> kv_ಹೆಸರುಗಳು;
    size_t kv_ಬಫರ್_ctx_id = 0;

    bool same_rope_params(ಸ್ಥಿರಾಂಕ ModelParams & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ mixed_rope_params == other.mixed_rope_params &&
               memcmp(rope_params, other.rope_params, ಗಾತ್ರ(int32_t) * 16) == 0;
    }

    bool can_reuse_dynamically(ಸ್ಥಿರಾಂಕ ModelParams & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ same_rope_params(other) && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool can_reuse_statically(ಸ್ಥಿರಾಂಕ ModelParams & other) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ same_rope_params(other) && ctx == other.ctx && n_rs_slots == other.n_rs_slots &&
               has_rs_rollback == other.has_rs_rollback;
    }

    bool kv_ಬಫರ್_ಬದಲಾಯಿತು(ಸ್ಥಿರಾಂಕ ModelParams & other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ kv_ಬಫರ್_ctx_id != other.kv_ಬಫರ್_ctx_id; }
};

ರಚನೆ ComputeParams {
    ಪೂರ್ಣಾಂಕ n_seq_ಸಕ್ರಿಯ = 1;
    ಪೂರ್ಣಾಂಕ seq_ಸಕ್ರಿಯ_ಆರಂಭ = 0;
    ಪೂರ್ಣಾಂಕ attention_ಗಾತ್ರ = -1;
    ಪೂರ್ಣಾಂಕ attention_ಗಾತ್ರ_swa = -1;
    ಪೂರ್ಣಾಂಕ attention_ಗಾತ್ರ_static = -1;  // encoder/cross-attn KV fill level (whisper)
    // Sliding window width, read back from the band of ggml's own SWA mask. ggml never passes
    // n_swa down to a backend, but fill_mask() bakes it into the mask contents, so the widest
    // unmasked row recovers it. Shorter than n_swa while the sequence is still short, which is
    // harmless: every causal pair is inside the window then anyway.
    ಪೂರ್ಣಾಂಕ swa_ಕಿಟಕಿ = -1;
    ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_len = -1;
    ಪೂರ್ಣಾಂಕ token_len_ಪ್ರತಿ_seq = -1;
    ಪೂರ್ಣಾಂಕ past_kv_len = -1;
    ಪೂರ್ಣಾಂಕ output_len = 1;

    ಪೂರ್ಣಾಂಕ cache_rs_reset_idx = -1;
    ಪೂರ್ಣಾಂಕ cache_rs_reset_len = -1;
    // SSM/DeltaNet models otionally clear cache_r and cache_s of certain slots in the cgraph
    // 3: [ 18432,     4,     1,     1] RESHAPE              cache_r_l0 (reshaped)
    //    [ 18432,     4,     1,     1]            0: NONE        cache_r_l0
    // 4: [ 18432,     1,     1,     1] VIEW                 cache_r_l0 (reshaped) (view)
    //    [ 18432,     4,     1,     1]            0: RESHAPE     cache_r_l0 (reshaped)
    // 5: [ 18432,     1,     1,     1] SCALE                cache_r_l0 (reshaped) (view) (view)
    //    [ 18432,     1,     1,     1]            0: VIEW        cache_r_l0 (reshaped) (view)

    ಪೂರ್ಣಾಂಕ s_copy_ಸಕ್ರಿಯ_ಸ್ಥಾನ_len = -1;
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

    ರಚನೆ RsWriteback {
        ಪೂರ್ಣಾಂಕ ಸ್ಥಾನ_begin = 0;  // first cache slot written by the CPY
        ಪೂರ್ಣಾಂಕ src_begin = -1;  // first source column copied by a conv-state CPY
    };

    std::map<std::string, RsWriteback> rs_writebacks;
    // Destination slot offset of each state cache writeback CPY node, keyed by node name. It
    // changes with the batch (kv head, active sequence count) and, with rollback enabled
    // (cparams.n_rs_seq > 0), the conv state is written back once per snapshot slot. Passed to the
    // cached model as a runtime input. Dynamic models also receive the source-side offset; static
    // models use a fixed end-anchored offset in the translator.
};

// defined below; declared here because GgmlOvDecoder uses it inline
std::optional<ಪೂರ್ಣಾಂಕ> ಹೊರತೆಗೆ_layer_from_name(ಸ್ಥಿರಾಂಕ std::string & name);

// detects the MoE expert-plane-sum ADD chain (see definition); used by supports_op too
bool is_moe_expert_ಮೊತ್ತ_ಕೂಡಿಸು(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು);

class GgmlOvDecoder : ಸಾರ್ವಜನಿಕ ov::frontend::ggml::GgmlDecoder {
ಸಾರ್ವಜನಿಕ:
    ಸ್ಥಿರ std::string get_tensor_name(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
    ರಚನೆ ಗಂಟು_Info {
        ggml_tensor * ಗಂಟು;
        std::string ಗಂಟು_name;
        std::string ಗಂಟು_op_type;
        std::map<std::string, ggml_tensor *> ಗಂಟು_inputs;
        std::map<std::string, std::ಸದಿಶ<std::pair<std::string, ggml_tensor *>>> ಗಂಟು_inputs_views;
        std::ಸದಿಶ<std::string> ಗಂಟು_inputs_ಹೆಸರುಗಳು;
        ಪೂರ್ಣಾಂಕ ಗಂಟು_op_case = 0;
        ಶೂನ್ಯ * ದತ್ತಾಂಶ_addr;
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
                  ಪೂರ್ಣಾಂಕ prefill_chunk_ಗಾತ್ರ = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & model_weights);

    ವಾಸ್ತವಿಕ ov::Any get_attribute(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override {
        ಹಿಂತಿರುಗಿ nullptr;
        GGML_ಬಳಸದ(name);
    }

    ವಾಸ್ತವಿಕ ov::PartialShape get_ಒಳಹರಿವು_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_view_ಒಳಹರಿವು_src_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                      ಸ್ಥಿರಾಂಕ std::string & name,
                                                      size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_view_ಒಳಹರಿವು_src_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                          ಸ್ಥಿರಾಂಕ std::string & name,
                                                          size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::Shape get_view_ಒಳಹರಿವು_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                ಸ್ಥಿರಾಂಕ std::string & name,
                                                size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::Shape get_view_ಒಳಹರಿವು_src_ggml_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                    ಸ್ಥಿರಾಂಕ std::string & name,
                                                    size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::PartialShape get_view_ಒಳಹರಿವು_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                     ಸ್ಥಿರಾಂಕ std::string & name,
                                                     size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::PartialShape get_view_ಒಳಹರಿವು_src_ov_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                         ಸ್ಥಿರಾಂಕ std::string & name,
                                                         size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::string get_view_ಒಳಹರಿವು_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name, size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::string get_view_ಒಳಹರಿವು_src_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                                ಸ್ಥಿರಾಂಕ std::string & name,
                                                size_t view_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::element::Type get_ಒಳಹರಿವು_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_ಒಳಹರಿವು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_ಒಳಹರಿವು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಶೂನ್ಯ get_ಒಳಹರಿವು_ಗಂಟು(size_t ಒಳಹರಿವು_ಪೋರ್ಟ್_idx,
                                std::string & ಉತ್ಪಾದಕ_name,
                                std::string & ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_name,
                                size_t & ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ override {
        GGML_ಬಳಸದ(ಒಳಹರಿವು_ಪೋರ್ಟ್_idx);
        GGML_ಬಳಸದ(ಉತ್ಪಾದಕ_name);
        GGML_ಬಳಸದ(ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_name);
        GGML_ಬಳಸದ(ಉತ್ಪಾದಕ_output_ಪೋರ್ಟ್_ಸೂಚ್ಯಂಕ);
    }

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<std::string> get_ಒಳಹರಿವು_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::PartialShape get_output_shape(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ov::element::Type get_output_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<size_t> get_output_stride(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ int32_t * get_ಒಳಹರಿವು_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ int32_t * get_output_op_params(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ size_t get_output_op_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::ಸದಿಶ<std::string> get_output_ಹೆಸರುಗಳು(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ std::string get_inplace_op_src(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ bool is_view_like_alias_ದ(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ std::string & view_src_name) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_type() ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_type(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_name() ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::string & get_op_name(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ int32_t get_op_dynamic_dim(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಭೇಟಿ_subgraph(
        std::function<ಶೂನ್ಯ(std::shared_ptr<GgmlDecoder>, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx)> ಗಂಟು_visitor) ಸ್ಥಿರಾಂಕ override;

    ggml_tensor * get_ಒಳಹರಿವು_ggml_tensor(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_inputs.at(name); }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_op_case(ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_ಗಂಟು_info_list[ಗಂಟು_idx].ಗಂಟು_op_case; }

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, ov::frontend::ggml::Model_ಒಳಹರಿವು_Info> & get_model_inputs() ಸ್ಥಿರಾಂಕ override {
        ಹಿಂತಿರುಗಿ m_model_inputs;
    }

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, ov::frontend::ggml::Model_Extra_ಒಳಹರಿವು_Info> & get_model_extra_inputs() ಸ್ಥಿರಾಂಕ override {
        ಹಿಂತಿರುಗಿ m_model_extra_inputs;
    }

    ವಾಸ್ತವಿಕ ಸ್ಥಿರಾಂಕ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> & get_model_weights() ಸ್ಥಿರಾಂಕ override {
        ಹಿಂತಿರುಗಿ m_model_weights;
    }

    ವಾಸ್ತವಿಕ std::ಗಣ<std::string> get_model_output_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_model_output_ಹೆಸರುಗಳು; }

    ಸ್ಥಿರಾಂಕ std::map<std::string, ggml_tensor *> & get_model_outputs() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_outputs; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_ctx_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_params.ctx; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_ctx_ಪ್ರತಿ_seq() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_params.ctx_ಪ್ರತಿ_seq; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_ctx_ಪ್ರತಿ_seq_swa() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_params.ctx_ಪ್ರತಿ_seq_swa; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_n_seq() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_params.n_seq; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ is_swa_layer(ಪೂರ್ಣಾಂಕ layer) ಸ್ಥಿರಾಂಕ override {
        ಹಿಂತಿರುಗಿ std::ಕಂಡುಹಿಡಿ(m_model_params.swa_layers.begin(), m_model_params.swa_layers.ಕೊನೆ(), layer) !=
               m_model_params.swa_layers.ಕೊನೆ();
    }

    // KV head count for one layer. Sliding and full layers can differ (gemma-4 12B), so callers
    // that reinterpret a KV buffer must use this and not the model-level n_heads_kv.
    ಪೂರ್ಣಾಂಕ get_n_heads_kv_for_layer(ಪೂರ್ಣಾಂಕ layer) ಸ್ಥಿರಾಂಕ {
        ಸ್ವಯಂಚಾಲಿತ it = m_model_params.n_heads_kv_ಪ್ರತಿ_layer.ಕಂಡುಹಿಡಿ(layer);
        ಹಿಂತಿರುಗಿ it != m_model_params.n_heads_kv_ಪ್ರತಿ_layer.ಕೊನೆ() ? it->ಎರಡನೇ : m_model_params.n_heads_kv;
    }

    // Same, for a KV cache tensor: its layer comes from the leaf name (cache_k_l<N>).
    ಪೂರ್ಣಾಂಕ get_n_heads_kv_for_tensor(ಸ್ಥಿರಾಂಕ ggml_tensor * kv_tensor) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ layer = ಹೊರತೆಗೆ_layer_from_name(std::string(kv_tensor->name)); layer.has_ಮೌಲ್ಯ()) {
            ಹಿಂತಿರುಗಿ get_n_heads_kv_for_layer(layer.ಮೌಲ್ಯ());
        }
        ಹಿಂತಿರುಗಿ m_model_params.n_heads_kv;
    }

    ಪೂರ್ಣಾಂಕ get_past_kv_len() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_compute_params.past_kv_len; }

    ಪೂರ್ಣಾಂಕ get_ಒಳಹರಿವು_len() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_compute_params.ಒಳಹರಿವು_len; }

    ವಾಸ್ತವಿಕ int32_t * get_rope_params() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ const_cast<int32_t *>(m_model_params.rope_params); }

    ವಾಸ್ತವಿಕ bool has_mixed_rope_params() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_model_params.mixed_rope_params; }

    ವಾಸ್ತವಿಕ ಪೂರ್ಣಾಂಕ get_ssm_state_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_model_params.state_ಗಾತ್ರ; }

    ವಾಸ್ತವಿಕ std::map<std::string, std::string> get_kv_param_res_ಹೆಸರುಗಳು() ಸ್ಥಿರಾಂಕ override;

    ವಾಸ್ತವಿಕ bool is_static() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_is_static; }

    ವಾಸ್ತವಿಕ bool is_stateful() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_is_stateful; }

    ಪೂರ್ಣಾಂಕ get_static_n_ಟೋಕನುಗಳು() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_is_prefill ? m_prefill_chunk_ಗಾತ್ರ : 1; }

    ವಾಸ್ತವಿಕ bool is_splited_model() ಸ್ಥಿರಾಂಕ override { ಹಿಂತಿರುಗಿ m_model_is_splitted; }

    ov::PartialShape get_ನಕ್ಷೆ_ಒಳಹರಿವು_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * op,
                                           ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳಹರಿವು,
                                           ಪೂರ್ಣಾಂಕ dynamic_dim_ಸೂಚ್ಯಂಕ = -1) ಸ್ಥಿರಾಂಕ;

    ಸ್ಥಿರ ಶೂನ್ಯ dump_cgraph(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, std::string & filename);

    ಸ್ಥಿರ std::shared_ptr<ov::ಗಂಟು_2> create_ತೂಕ_ಗಂಟು(ggml_tensor * tensor, bool naive = false);

    ಸ್ಥಿರ std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> create_ತೂಕ_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    // Collect just the set of weight-tensor names referenced by the graph, without
    // building (or requantizing) any OV weight nodes. Used by topology checks like
    // is_model_splitted that only need name membership.
    ಸ್ಥಿರ std::ಗಣ<std::string> collect_ತೂಕ_ಹೆಸರುಗಳು(ggml_cgraph * cgraph);

    ಸ್ಥಿರಾಂಕ ggml_tensor * get_tensor_ಬಳಸಿದ_op(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) ಸ್ಥಿರಾಂಕ;

    ಸ್ಥಿರಾಂಕ ggml_tensor * get_tensor_from_name(ಸ್ಥಿರಾಂಕ std::string & name) ಸ್ಥಿರಾಂಕ;

    ಶೂನ್ಯ clear_model_weights() { m_model_weights.clear(); }

    ಸ್ಥಿರ std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_model_params; }

    ComputeParams get_compute_params() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ m_compute_params; }

    ಶೂನ್ಯ ಗಣ_model_params(ಸ್ಥಿರಾಂಕ ModelParams & model_params) { m_model_params = model_params; }

    ಶೂನ್ಯ ಗಣ_compute_params(ಸ್ಥಿರಾಂಕ ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    ಪೂರ್ಣಾಂಕ m_prefill_chunk_ಗಾತ್ರ = 0;
    bool m_model_is_splitted = false;  // label the cgraph is splited or not

    ಸ್ಥಿರ ov::Shape get_shape(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
    ಸ್ಥಿರ std::ಸದಿಶ<size_t> get_stride(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
    ಸ್ಥಿರ ov::element::Type get_ov_type(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);
    ಸ್ಥಿರ std::string compute_op_type(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು);
    ಶೂನ್ಯ ಕೂಡಿಸು_extra_inputs();

    ಶೂನ್ಯ update_io(ggml_cgraph * cgraph);
    bool is_bound_ಗಮ್ಯ(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph) ಸ್ಥಿರಾಂಕ;

    ಸ್ಥಿರ bool is_inp_tok(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    ಸ್ಥಿರ bool is_inp_pos(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    // IMROPE and VISION pack 4 stacked position planes (t/h/w/e) into inp_pos, each of length
    // n_tokens; other modes carry a single position per token.
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_inp_pos_n_planes(ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಹುಲಕ = op->op_params[2];
        ಹಿಂತಿರುಗಿ (ಬಹುಲಕ == GGML_ROPE_TYPE_IMROPE || ಬಹುಲಕ == GGML_ROPE_TYPE_VISION || (ಬಹುಲಕ & GGML_ROPE_TYPE_MROPE)) ? 4 : 1;
    }

    ಸ್ಥಿರ bool is_inp_emb(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ tensor->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && op->op == GGML_OP_RMS_NORM;
    }

    ಸ್ಥಿರ bool is_inp_ಮುಖವಾಡ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]) ||
               (op->op == GGML_OP_SOFT_MAX && tensor == op->src[1]);
    }

    ಸ್ಥಿರ bool is_inp_ಸರಾಸರಿ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_MUL_MAT && tensor == op->src[1] && tensor->op == GGML_OP_NONE &&
               (tensor->flags & GGML_TENSOR_FLAG_ಒಳಹರಿವು) && tensor->type == GGML_TYPE_F32 &&
               op->src[0] != nullptr && op->src[0]->op != GGML_OP_NONE;
    }

    // per-token embedding scale [1, n_tokens] (inp->scale_rows in llama-graph.cpp)
    ಸ್ಥಿರ bool is_inp_scale_ಅಡ್ಡಸಾಲುಗಳು(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_MUL && tensor == op->src[1] && strcmp(tensor->name, "inp_scale_rows") == 0;
    }

    ಸ್ಥಿರ bool is_rope_freqs_ತೂಕ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    ಸಾಲೊಳಗೆ ಸ್ಥಿರ bool is_recurrent_cache(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
        ಹಿಂತಿರುಗಿ tensor != nullptr && (ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "cache_r_l", ಅಕ್ಷರಸರಣಿ_ಉದ್ದ("cache_r_l")) == 0 ||
                                     ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "cache_s_l", ಅಕ್ಷರಸರಣಿ_ಉದ್ದ("cache_s_l")) == 0 ||
                                     ಅಕ್ಷರಸರಣಿ_ಹೋಲಿಕೆ(tensor->name, "cache_ple_r_l", ಅಕ್ಷರಸರಣಿ_ಉದ್ದ("cache_ple_r_l")) == 0);
    }

    ಸಾಲೊಳಗೆ ಸ್ಥಿರ bool is_cache(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ is_recurrent_cache(tensor) || is_kvcache(tensor, op);
    }

    ಸಾಲೊಳಗೆ ಸ್ಥಿರ bool is_kvcache(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಆದರೆ (tensor == nullptr || is_recurrent_cache(tensor)) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಹಿಂತಿರುಗಿ (tensor->ಬಫರ್ != nullptr && tensor->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_ANY) ||
               (op != nullptr && op->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು && op->src[2] == tensor);
    }

    ಸಾಲೊಳಗೆ ಸ್ಥಿರ bool is_conv_state_writeback(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) {
        ಹಿಂತಿರುಗಿ ಗಂಟು->op == GGML_OP_CPY && ಗಂಟು->view_src != nullptr && is_recurrent_cache(ಗಂಟು->view_src) &&
               ಗಂಟು->src[0] != nullptr && ಗಂಟು->src[0]->op == GGML_OP_VIEW && ಗಂಟು->src[0]->src[0] != nullptr &&
               ಗಂಟು->src[0]->src[0]->op == GGML_OP_CONCAT && ಗಂಟು->src[1] != nullptr &&
               ಗಂಟು->src[1]->op == GGML_OP_VIEW && ಗಂಟು->src[1]->view_src == ಗಂಟು->view_src;
    }

    ಸ್ಥಿರ bool is_kv_idx(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು && op->src[1] == tensor;
    }

    bool is_swa_ಮುಖವಾಡ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ m_model_params.swa_ಮುಖವಾಡ != nullptr && tensor == m_model_params.swa_ಮುಖವಾಡ;
    }

    ಸ್ಥಿರ bool is_output_idx(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE &&
               op->src[1]->op == GGML_OP_NONE;
    }

    // the state permutation index input used in SSM/DeltaNet models (inp->s_copy in llama-graph.cpp)
    ಸ್ಥಿರ bool is_inp_s_copy(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
        ಹಿಂತಿರುಗಿ op->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು && tensor == op->src[1] &&
               op->src[0]->ಬಫರ್->usage == GGML_BACKEND_ಬಫರ್_USAGE_ANY;
    }

    std::string get_ನಕ್ಷೆ_ಒಳಹರಿವು_ov_name(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ggml_tensor * op) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (is_inp_pos(tensor, op)) {
            ಹಿಂತಿರುಗಿ "inp_pos";
        }
        ಆದರೆ (is_inp_emb(tensor, op)) {
            ಹಿಂತಿರುಗಿ "embd";
        }
        ಆದರೆ (is_inp_ಮುಖವಾಡ(tensor, op)) {
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
            ಆದರೆ (m_model_params.swa_layers.ಖಾಲಿಗಳು()) {
                ಹಿಂತಿರುಗಿ "self_kq_mask";
            }
            ಹಿಂತಿರುಗಿ is_swa_ಮುಖವಾಡ(tensor) ? "self_kq_mask_swa" : "self_kq_mask";
        }
        ಹಿಂತಿರುಗಿ tensor->name;
    }

ಖಾಸಗಿ:
    ಶೂನ್ಯ ಗಣ_ಒಳಹರಿವು_output();
    ಪೂರ್ಣಾಂಕ compute_op_case(ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು) ಸ್ಥಿರಾಂಕ;
    bool ಗಂಟು_is_ಬಳಸಿದ_as_src(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಂಟು_idx);
    ಶೂನ್ಯ compute_model_inputs();
    ಶೂನ್ಯ compute_model_outputs();

    // True if tensor is the inp->s_copy index leaf gathered by a recurrent state cache GET_ROWS
    // (possibly through a VIEW), so it gets a dynamic [1,1,1,-1] graph-input shape.
    bool is_s_copy_ಎಲೆ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) ಸ್ಥಿರಾಂಕ;

    // Infer and propagate dynamic-dimension indices for all tensors in the GGML graph.
    ಶೂನ್ಯ compute_ಗಂಟು_dynamic_dims();

    ಶೂನ್ಯ validate_cgraph() ಸ್ಥಿರಾಂಕ;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, ov::frontend::ggml::Model_ಒಳಹರಿವು_Info> m_model_inputs;
    std::map<std::string, ov::frontend::ggml::Model_Extra_ಒಳಹರಿವು_Info> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::ಗಂಟು_2>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::ಗಣ<std::string> m_model_output_ಹೆಸರುಗಳು;
    std::ಸದಿಶ<ಗಂಟು_Info> m_ಗಂಟು_info_list;
    std::map<ggml_tensor *, ಪೂರ್ಣಾಂಕ> m_ಗಂಟು_dynamic_dims;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

ಶೂನ್ಯ print_tensor_ವಿಳಾಸ_map(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph);
