#pragma once

#അന്തര്ഭാവ "decoder.h"

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <openvino/frontend/node_context.hpp>
#അന്തര്ഭാവ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

തരനിർവചനം std::map<std::string, Output<കണ്ണി_2>> TensorMap;

class കണ്ണി_Context : പൊതു frontend::കണ്ണി_Context {
പൊതു:
    കണ്ണി_Context(സ്ഥിരം std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::കണ്ണി_Context(decoder->get_op_type(കണ്ണി_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_കണ്ണി_idx(കണ്ണി_idx),
        m_translate_session(translate_session) {
        m_നിവേശം_names = decoder->get_നിവേശം_names(m_കണ്ണി_idx);
        m_output_names = decoder->get_output_names(m_കണ്ണി_idx);
    }

    TranslateSession * get_translate_session() സ്ഥിരം { തിരികെ m_translate_session; }

    സ്ഥിരം std::vector<std::string> & get_നിവേശം_names() സ്ഥിരം { തിരികെ m_നിവേശം_names; }

    സ്ഥിരം std::vector<std::string> & get_output_names() സ്ഥിരം { തിരികെ m_output_names; }

    size_t get_നിവേശം_size() സ്ഥിരം override { തിരികെ m_decoder->get_നിവേശം_size(m_കണ്ണി_idx); }

    ov::element::Type get_നിവേശം_type(size_t index) സ്ഥിരം {
        തിരികെ m_decoder->get_നിവേശം_type(m_കണ്ണി_idx, m_നിവേശം_names[index]);
    }

    PartialShape get_നിവേശം_shape(size_t നിവേശം_index) സ്ഥിരം {
        തിരികെ m_decoder->get_നിവേശം_shape(m_കണ്ണി_idx, m_നിവേശം_names[നിവേശം_index]);
    }

    std::vector<size_t> get_നിവേശം_stride(size_t index) സ്ഥിരം {
        തിരികെ m_decoder->get_നിവേശം_stride(m_കണ്ണി_idx, m_നിവേശം_names[index]);
    }

    std::string get_output_name() സ്ഥിരം { തിരികെ m_output_names[0]; }

    PartialShape get_output_shape() സ്ഥിരം { തിരികെ m_decoder->get_output_shape(m_കണ്ണി_idx); }

    int32_t * get_നിവേശം_op_params(size_t index) സ്ഥിരം {
        തിരികെ m_decoder->get_നിവേശം_op_params(m_കണ്ണി_idx, m_നിവേശം_names[index]);
    }

    size_t get_view_നിവേശം_size(size_t index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_size(m_കണ്ണി_idx, m_നിവേശം_names[index]);
    }

    size_t get_view_നിവേശം_offset(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_offset(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    size_t get_view_നിവേശം_src_offset(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_src_offset(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    std::vector<size_t> get_view_നിവേശം_stride(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_stride(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    std::vector<size_t> get_view_നിവേശം_src_stride(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_src_stride(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    ov::Shape get_view_നിവേശം_ggml_shape(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_ggml_shape(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    ov::Shape get_view_നിവേശം_src_ggml_shape(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_src_ggml_shape(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    ov::PartialShape get_view_നിവേശം_ov_shape(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_ov_shape(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    ov::PartialShape get_view_നിവേശം_src_ov_shape(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_src_ov_shape(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    std::string get_view_നിവേശം_name(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_name(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    std::string get_view_നിവേശം_src_name(size_t index, size_t view_index) സ്ഥിരം {
        തിരികെ m_decoder->get_view_നിവേശം_src_name(m_കണ്ണി_idx, m_നിവേശം_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() സ്ഥിരം { തിരികെ m_decoder->get_op_dynamic_dim(m_കണ്ണി_idx); }

    int32_t * get_output_op_params() സ്ഥിരം { തിരികെ m_decoder->get_output_op_params(m_കണ്ണി_idx); }

    size_t get_output_op_offset() സ്ഥിരം { തിരികെ m_decoder->get_output_op_offset(m_കണ്ണി_idx); }

    ov::element::Type get_output_type() സ്ഥിരം { തിരികെ m_decoder->get_output_type(m_കണ്ണി_idx); }

    std::vector<size_t> get_output_stride() സ്ഥിരം { തിരികെ m_decoder->get_output_stride(m_കണ്ണി_idx); }

    Output<കണ്ണി_2> get_നിവേശം(പൂർണ്ണസംഖ്യ idx) സ്ഥിരം override {
        // Check if this input is a VIEW
        size_t view_നിവേശം_size = m_decoder->get_view_നിവേശം_size(m_കണ്ണി_idx, m_നിവേശം_names[idx]);
        എങ്കിൽ (view_നിവേശം_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_നിവേശം_src_name(m_കണ്ണി_idx, m_നിവേശം_names[idx], view_നിവേശം_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_നിവേശം_names[idx]);
            എങ്കിൽ (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                എങ്കിൽ (base_it == m_tensor_map->end() ||
                    view_it->second.get_കണ്ണി_shared_ptr() != base_it->second.get_കണ്ണി_shared_ptr()) {
                    തിരികെ view_it->second;
                }
                തിരികെ base_it->second;
            }
            എങ്കിൽ (!base_name.empty()) {
                തിരികെ m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        തിരികെ m_tensor_map->at(m_നിവേശം_names[idx]);
    }

    Output<കണ്ണി_2> get_നിവേശം(സ്ഥിരം std::string & name) സ്ഥിരം override {
        എങ്കിൽ (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        തിരികെ m_tensor_map->at(name);
    }

    bool has_നിവേശം(സ്ഥിരം std::string & name) സ്ഥിരം { തിരികെ m_tensor_map->find(name) != m_tensor_map->end(); }

    ശൂന്യം put_shared(സ്ഥിരം std::string & name, സ്ഥിരം Output<കണ്ണി_2> & മൂല്യം) സ്ഥിരം {
        m_tensor_map->തിരുകുക({name, മൂല്യം});
    }

    സ്ഥിരം std::string & get_name() സ്ഥിരം override { തിരികെ m_decoder->get_op_name(m_കണ്ണി_idx); }

    ov::Any get_attribute_as_any(സ്ഥിരം std::string & name) സ്ഥിരം override { തിരികെ m_decoder->get_attribute(name); }

    പൂർണ്ണസംഖ്യ get_op_case() സ്ഥിരം { തിരികെ m_decoder->get_op_case(m_കണ്ണി_idx); }

    bool is_static() സ്ഥിരം { തിരികെ m_decoder->is_static(); }

    bool is_stateful() സ്ഥിരം { തിരികെ m_decoder->is_stateful(); }

    പൂർണ്ണസംഖ്യ get_ssm_state_size() സ്ഥിരം { തിരികെ m_decoder->get_ssm_state_size(); }

സ്വകാര്യ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    പൂർണ്ണസംഖ്യ m_കണ്ണി_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_നിവേശം_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(സ്ഥിരം ov::frontend::ggml::കണ്ണി_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
