#pragma once

#অন্তর্ভাব "decoder.h"

#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <openvino/frontend/node_context.hpp>
#অন্তর্ভাব <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

প্রকারলেখ std::map<std::string, Output<গ্রন্থি_2>> TensorMap;

class গ্রন্থি_Context : সর্বজনীন frontend::গ্রন্থি_Context {
সর্বজনীন:
    গ্রন্থি_Context(ধ্রুবক std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                পূর্ণসংখ্যা গ্রন্থি_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::গ্রন্থি_Context(decoder->get_op_type(গ্রন্থি_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_গ্রন্থি_idx(গ্রন্থি_idx),
        m_translate_session(translate_session) {
        m_প্রবেশ_names = decoder->get_প্রবেশ_names(m_গ্রন্থি_idx);
        m_output_names = decoder->get_output_names(m_গ্রন্থি_idx);
    }

    TranslateSession * get_translate_session() ধ্রুবক { ফেরত m_translate_session; }

    ধ্রুবক std::vector<std::string> & get_প্রবেশ_names() ধ্রুবক { ফেরত m_প্রবেশ_names; }

    ধ্রুবক std::vector<std::string> & get_output_names() ধ্রুবক { ফেরত m_output_names; }

    size_t get_প্রবেশ_আকার() ধ্রুবক override { ফেরত m_decoder->get_প্রবেশ_আকার(m_গ্রন্থি_idx); }

    ov::element::Type get_প্রবেশ_type(size_t সূচক) ধ্রুবক {
        ফেরত m_decoder->get_প্রবেশ_type(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক]);
    }

    PartialShape get_প্রবেশ_shape(size_t প্রবেশ_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_প্রবেশ_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[প্রবেশ_সূচক]);
    }

    std::vector<size_t> get_প্রবেশ_stride(size_t সূচক) ধ্রুবক {
        ফেরত m_decoder->get_প্রবেশ_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক]);
    }

    std::string get_output_name() ধ্রুবক { ফেরত m_output_names[0]; }

    PartialShape get_output_shape() ধ্রুবক { ফেরত m_decoder->get_output_shape(m_গ্রন্থি_idx); }

    int32_t * get_প্রবেশ_op_params(size_t সূচক) ধ্রুবক {
        ফেরত m_decoder->get_প্রবেশ_op_params(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক]);
    }

    size_t get_view_প্রবেশ_আকার(size_t সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_আকার(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক]);
    }

    size_t get_view_প্রবেশ_offset(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_offset(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    size_t get_view_প্রবেশ_src_offset(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_src_offset(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    std::vector<size_t> get_view_প্রবেশ_stride(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    std::vector<size_t> get_view_প্রবেশ_src_stride(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_src_stride(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    ov::Shape get_view_প্রবেশ_ggml_shape(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_ggml_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    ov::Shape get_view_প্রবেশ_src_ggml_shape(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_src_ggml_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    ov::PartialShape get_view_প্রবেশ_ov_shape(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_ov_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    ov::PartialShape get_view_প্রবেশ_src_ov_shape(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_src_ov_shape(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    std::string get_view_প্রবেশ_name(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_name(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    std::string get_view_প্রবেশ_src_name(size_t সূচক, size_t view_সূচক) ধ্রুবক {
        ফেরত m_decoder->get_view_প্রবেশ_src_name(m_গ্রন্থি_idx, m_প্রবেশ_names[সূচক], view_সূচক);
    }

    int32_t get_op_dynamic_dim() ধ্রুবক { ফেরত m_decoder->get_op_dynamic_dim(m_গ্রন্থি_idx); }

    int32_t * get_output_op_params() ধ্রুবক { ফেরত m_decoder->get_output_op_params(m_গ্রন্থি_idx); }

    size_t get_output_op_offset() ধ্রুবক { ফেরত m_decoder->get_output_op_offset(m_গ্রন্থি_idx); }

    ov::element::Type get_output_type() ধ্রুবক { ফেরত m_decoder->get_output_type(m_গ্রন্থি_idx); }

    std::vector<size_t> get_output_stride() ধ্রুবক { ফেরত m_decoder->get_output_stride(m_গ্রন্থি_idx); }

    Output<গ্রন্থি_2> get_প্রবেশ(পূর্ণসংখ্যা idx) ধ্রুবক override {
        // Check if this input is a VIEW
        size_t view_প্রবেশ_আকার = m_decoder->get_view_প্রবেশ_আকার(m_গ্রন্থি_idx, m_প্রবেশ_names[idx]);
        যদি (view_প্রবেশ_আকার > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_প্রবেশ_src_name(m_গ্রন্থি_idx, m_প্রবেশ_names[idx], view_প্রবেশ_আকার - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_প্রবেশ_names[idx]);
            যদি (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                যদি (base_it == m_tensor_map->end() ||
                    view_it->second.get_গ্রন্থি_shared_ptr() != base_it->second.get_গ্রন্থি_shared_ptr()) {
                    ফেরত view_it->second;
                }
                ফেরত base_it->second;
            }
            যদি (!base_name.empty()) {
                ফেরত m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ফেরত m_tensor_map->at(m_প্রবেশ_names[idx]);
    }

    Output<গ্রন্থি_2> get_প্রবেশ(ধ্রুবক std::string & name) ধ্রুবক override {
        যদি (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ফেরত m_tensor_map->at(name);
    }

    bool has_প্রবেশ(ধ্রুবক std::string & name) ধ্রুবক { ফেরত m_tensor_map->find(name) != m_tensor_map->end(); }

    শূন্য put_shared(ধ্রুবক std::string & name, ধ্রুবক Output<গ্রন্থি_2> & মান) ধ্রুবক {
        m_tensor_map->ঢোকাও({name, মান});
    }

    ধ্রুবক std::string & get_name() ধ্রুবক override { ফেরত m_decoder->get_op_name(m_গ্রন্থি_idx); }

    ov::Any get_attribute_as_any(ধ্রুবক std::string & name) ধ্রুবক override { ফেরত m_decoder->get_attribute(name); }

    পূর্ণসংখ্যা get_op_case() ধ্রুবক { ফেরত m_decoder->get_op_case(m_গ্রন্থি_idx); }

    bool is_static() ধ্রুবক { ফেরত m_decoder->is_static(); }

    bool is_stateful() ধ্রুবক { ফেরত m_decoder->is_stateful(); }

    পূর্ণসংখ্যা get_ssm_state_আকার() ধ্রুবক { ফেরত m_decoder->get_ssm_state_আকার(); }

ব্যক্তিগত:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    পূর্ণসংখ্যা m_গ্রন্থি_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_প্রবেশ_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ধ্রুবক ov::frontend::ggml::গ্রন্থি_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
