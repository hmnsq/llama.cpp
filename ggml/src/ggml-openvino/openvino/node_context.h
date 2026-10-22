#pragma once

#شامل "decoder.h"

#شامل <cstdint>
#شامل <openvino/frontend/node_context.hpp>
#شامل <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعریفنوع std::map<std::string, Output<گره_2>> TensorMap;

class گره_Context : عمومی frontend::گره_Context {
عمومی:
    گره_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحیح گره_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::گره_Context(decoder->get_op_type(گره_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_گره_idx(گره_idx),
        m_translate_session(translate_session) {
        m_درون_داد_names = decoder->get_درون_داد_names(m_گره_idx);
        m_output_names = decoder->get_output_names(m_گره_idx);
    }

    TranslateSession * get_translate_session() ثابت { بازگشت m_translate_session; }

    ثابت std::vector<std::string> & get_درون_داد_names() ثابت { بازگشت m_درون_داد_names; }

    ثابت std::vector<std::string> & get_output_names() ثابت { بازگشت m_output_names; }

    size_t get_درون_داد_size() ثابت override { بازگشت m_decoder->get_درون_داد_size(m_گره_idx); }

    ov::element::Type get_درون_داد_type(size_t index) ثابت {
        بازگشت m_decoder->get_درون_داد_type(m_گره_idx, m_درون_داد_names[index]);
    }

    PartialShape get_درون_داد_shape(size_t درون_داد_index) ثابت {
        بازگشت m_decoder->get_درون_داد_shape(m_گره_idx, m_درون_داد_names[درون_داد_index]);
    }

    std::vector<size_t> get_درون_داد_stride(size_t index) ثابت {
        بازگشت m_decoder->get_درون_داد_stride(m_گره_idx, m_درون_داد_names[index]);
    }

    std::string get_output_name() ثابت { بازگشت m_output_names[0]; }

    PartialShape get_output_shape() ثابت { بازگشت m_decoder->get_output_shape(m_گره_idx); }

    int32_t * get_درون_داد_op_params(size_t index) ثابت {
        بازگشت m_decoder->get_درون_داد_op_params(m_گره_idx, m_درون_داد_names[index]);
    }

    size_t get_view_درون_داد_size(size_t index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_size(m_گره_idx, m_درون_داد_names[index]);
    }

    size_t get_view_درون_داد_offset(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_offset(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    size_t get_view_درون_داد_src_offset(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_src_offset(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    std::vector<size_t> get_view_درون_داد_stride(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_stride(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    std::vector<size_t> get_view_درون_داد_src_stride(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_src_stride(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    ov::Shape get_view_درون_داد_ggml_shape(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_ggml_shape(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    ov::Shape get_view_درون_داد_src_ggml_shape(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_src_ggml_shape(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    ov::PartialShape get_view_درون_داد_ov_shape(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_ov_shape(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    ov::PartialShape get_view_درون_داد_src_ov_shape(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_src_ov_shape(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    std::string get_view_درون_داد_name(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_name(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    std::string get_view_درون_داد_src_name(size_t index, size_t view_index) ثابت {
        بازگشت m_decoder->get_view_درون_داد_src_name(m_گره_idx, m_درون_داد_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ثابت { بازگشت m_decoder->get_op_dynamic_dim(m_گره_idx); }

    int32_t * get_output_op_params() ثابت { بازگشت m_decoder->get_output_op_params(m_گره_idx); }

    size_t get_output_op_offset() ثابت { بازگشت m_decoder->get_output_op_offset(m_گره_idx); }

    ov::element::Type get_output_type() ثابت { بازگشت m_decoder->get_output_type(m_گره_idx); }

    std::vector<size_t> get_output_stride() ثابت { بازگشت m_decoder->get_output_stride(m_گره_idx); }

    Output<گره_2> get_درون_داد(صحیح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_درون_داد_size = m_decoder->get_view_درون_داد_size(m_گره_idx, m_درون_داد_names[idx]);
        اگر (view_درون_داد_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_درون_داد_src_name(m_گره_idx, m_درون_داد_names[idx], view_درون_داد_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_درون_داد_names[idx]);
            اگر (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                اگر (base_it == m_tensor_map->end() ||
                    view_it->second.get_گره_shared_ptr() != base_it->second.get_گره_shared_ptr()) {
                    بازگشت view_it->second;
                }
                بازگشت base_it->second;
            }
            اگر (!base_name.empty()) {
                بازگشت m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        بازگشت m_tensor_map->at(m_درون_داد_names[idx]);
    }

    Output<گره_2> get_درون_داد(ثابت std::string & name) ثابت override {
        اگر (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        بازگشت m_tensor_map->at(name);
    }

    bool has_درون_داد(ثابت std::string & name) ثابت { بازگشت m_tensor_map->find(name) != m_tensor_map->end(); }

    تهی put_shared(ثابت std::string & name, ثابت Output<گره_2> & مقدار) ثابت {
        m_tensor_map->درج({name, مقدار});
    }

    ثابت std::string & get_name() ثابت override { بازگشت m_decoder->get_op_name(m_گره_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { بازگشت m_decoder->get_attribute(name); }

    صحیح get_op_case() ثابت { بازگشت m_decoder->get_op_case(m_گره_idx); }

    bool is_static() ثابت { بازگشت m_decoder->is_static(); }

    bool is_stateful() ثابت { بازگشت m_decoder->is_stateful(); }

    صحیح get_ssm_state_size() ثابت { بازگشت m_decoder->get_ssm_state_size(); }

خصوصی:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحیح m_گره_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_درون_داد_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ثابت ov::frontend::ggml::گره_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
