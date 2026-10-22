#pragma once

#অন্তর্ভাৱ "decoder.h"

#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <openvino/frontend/node_context.hpp>
#অন্তর্ভাৱ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

প্রকারনির্দেশ std::map<std::string, Output<গ্ৰন্থি_2>> TensorMap;

class গ্ৰন্থি_Context : সার্ৱজনিক frontend::গ্ৰন্থি_Context {
সার্ৱজনিক:
    গ্ৰন্থি_Context(ধ্রুৱ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                পূর্ণাঙ্ক গ্ৰন্থি_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::গ্ৰন্থি_Context(decoder->get_op_type(গ্ৰন্থি_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_গ্ৰন্থি_idx(গ্ৰন্থি_idx),
        m_translate_session(translate_session) {
        m_প্ৰবেশ_names = decoder->get_প্ৰবেশ_names(m_গ্ৰন্থি_idx);
        m_output_names = decoder->get_output_names(m_গ্ৰন্থি_idx);
    }

    TranslateSession * get_translate_session() ধ্রুৱ { ঘূৰাই m_translate_session; }

    ধ্রুৱ std::vector<std::string> & get_প্ৰবেশ_names() ধ্রুৱ { ঘূৰাই m_প্ৰবেশ_names; }

    ধ্রুৱ std::vector<std::string> & get_output_names() ধ্রুৱ { ঘূৰাই m_output_names; }

    size_t get_প্ৰবেশ_size() ধ্রুৱ override { ঘূৰাই m_decoder->get_প্ৰবেশ_size(m_গ্ৰন্থি_idx); }

    ov::element::Type get_প্ৰবেশ_type(size_t index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_প্ৰবেশ_type(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index]);
    }

    PartialShape get_প্ৰবেশ_shape(size_t প্ৰবেশ_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_প্ৰবেশ_shape(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[প্ৰবেশ_index]);
    }

    std::vector<size_t> get_প্ৰবেশ_stride(size_t index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_প্ৰবেশ_stride(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index]);
    }

    std::string get_output_name() ধ্রুৱ { ঘূৰাই m_output_names[0]; }

    PartialShape get_output_shape() ধ্রুৱ { ঘূৰাই m_decoder->get_output_shape(m_গ্ৰন্থি_idx); }

    int32_t * get_প্ৰবেশ_op_params(size_t index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_প্ৰবেশ_op_params(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index]);
    }

    size_t get_view_প্ৰবেশ_size(size_t index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_size(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index]);
    }

    size_t get_view_প্ৰবেশ_offset(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_offset(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    size_t get_view_প্ৰবেশ_src_offset(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_src_offset(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    std::vector<size_t> get_view_প্ৰবেশ_stride(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_stride(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    std::vector<size_t> get_view_প্ৰবেশ_src_stride(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_src_stride(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    ov::Shape get_view_প্ৰবেশ_ggml_shape(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_ggml_shape(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    ov::Shape get_view_প্ৰবেশ_src_ggml_shape(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_src_ggml_shape(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    ov::PartialShape get_view_প্ৰবেশ_ov_shape(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_ov_shape(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    ov::PartialShape get_view_প্ৰবেশ_src_ov_shape(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_src_ov_shape(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    std::string get_view_প্ৰবেশ_name(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_name(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    std::string get_view_প্ৰবেশ_src_name(size_t index, size_t view_index) ধ্রুৱ {
        ঘূৰাই m_decoder->get_view_প্ৰবেশ_src_name(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ধ্রুৱ { ঘূৰাই m_decoder->get_op_dynamic_dim(m_গ্ৰন্থি_idx); }

    int32_t * get_output_op_params() ধ্রুৱ { ঘূৰাই m_decoder->get_output_op_params(m_গ্ৰন্থি_idx); }

    size_t get_output_op_offset() ধ্রুৱ { ঘূৰাই m_decoder->get_output_op_offset(m_গ্ৰন্থি_idx); }

    ov::element::Type get_output_type() ধ্রুৱ { ঘূৰাই m_decoder->get_output_type(m_গ্ৰন্থি_idx); }

    std::vector<size_t> get_output_stride() ধ্রুৱ { ঘূৰাই m_decoder->get_output_stride(m_গ্ৰন্থি_idx); }

    Output<গ্ৰন্থি_2> get_প্ৰবেশ(পূর্ণাঙ্ক idx) ধ্রুৱ override {
        // Check if this input is a VIEW
        size_t view_প্ৰবেশ_size = m_decoder->get_view_প্ৰবেশ_size(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[idx]);
        যদি (view_প্ৰবেশ_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_প্ৰবেশ_src_name(m_গ্ৰন্থি_idx, m_প্ৰবেশ_names[idx], view_প্ৰবেশ_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_প্ৰবেশ_names[idx]);
            যদি (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                যদি (base_it == m_tensor_map->end() ||
                    view_it->second.get_গ্ৰন্থি_shared_ptr() != base_it->second.get_গ্ৰন্থি_shared_ptr()) {
                    ঘূৰাই view_it->second;
                }
                ঘূৰাই base_it->second;
            }
            যদি (!base_name.empty()) {
                ঘূৰাই m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ঘূৰাই m_tensor_map->at(m_প্ৰবেশ_names[idx]);
    }

    Output<গ্ৰন্থি_2> get_প্ৰবেশ(ধ্রুৱ std::string & name) ধ্রুৱ override {
        যদি (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ঘূৰাই m_tensor_map->at(name);
    }

    bool has_প্ৰবেশ(ধ্রুৱ std::string & name) ধ্রুৱ { ঘূৰাই m_tensor_map->find(name) != m_tensor_map->end(); }

    শূন্য put_shared(ধ্রুৱ std::string & name, ধ্রুৱ Output<গ্ৰন্থি_2> & মান) ধ্রুৱ {
        m_tensor_map->ঢোকাও({name, মান});
    }

    ধ্রুৱ std::string & get_name() ধ্রুৱ override { ঘূৰাই m_decoder->get_op_name(m_গ্ৰন্থি_idx); }

    ov::Any get_attribute_as_any(ধ্রুৱ std::string & name) ধ্রুৱ override { ঘূৰাই m_decoder->get_attribute(name); }

    পূর্ণাঙ্ক get_op_case() ধ্রুৱ { ঘূৰাই m_decoder->get_op_case(m_গ্ৰন্থি_idx); }

    bool is_static() ধ্রুৱ { ঘূৰাই m_decoder->is_static(); }

    bool is_stateful() ধ্রুৱ { ঘূৰাই m_decoder->is_stateful(); }

    পূর্ণাঙ্ক get_ssm_state_size() ধ্রুৱ { ঘূৰাই m_decoder->get_ssm_state_size(); }

গুপ্ত:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    পূর্ণাঙ্ক m_গ্ৰন্থি_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_প্ৰবেশ_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ধ্রুৱ ov::frontend::ggml::গ্ৰন্থি_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
