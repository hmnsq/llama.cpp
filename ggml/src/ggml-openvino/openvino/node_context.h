#pragma once

#ଅନ୍ତର୍ଭାଵ "decoder.h"

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <openvino/frontend/node_context.hpp>
#ଅନ୍ତର୍ଭାଵ <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

ପ୍ରକାରବ୍ୟାଖ୍ୟା std::map<std::string, Output<ଗଣ୍ଠି_2>> TensorMap;

class ଗଣ୍ଠି_Context : ସର୍ବସାଧାରଣ frontend::ଗଣ୍ଠି_Context {
ସର୍ବସାଧାରଣ:
    ଗଣ୍ଠି_Context(ଅଚଳ std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ଗଣ୍ଠି_Context(decoder->get_op_type(ଗଣ୍ଠି_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ଗଣ୍ଠି_idx(ଗଣ୍ଠି_idx),
        m_translate_session(translate_session) {
        m_ନିବେଶ_names = decoder->get_ନିବେଶ_names(m_ଗଣ୍ଠି_idx);
        m_output_names = decoder->get_output_names(m_ଗଣ୍ଠି_idx);
    }

    TranslateSession * get_translate_session() ଅଚଳ { ଫେରନ୍ତୁ m_translate_session; }

    ଅଚଳ std::vector<std::string> & get_ନିବେଶ_names() ଅଚଳ { ଫେରନ୍ତୁ m_ନିବେଶ_names; }

    ଅଚଳ std::vector<std::string> & get_output_names() ଅଚଳ { ଫେରନ୍ତୁ m_output_names; }

    size_t get_ନିବେଶ_size() ଅଚଳ override { ଫେରନ୍ତୁ m_decoder->get_ନିବେଶ_size(m_ଗଣ୍ଠି_idx); }

    ov::element::Type get_ନିବେଶ_type(size_t index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_ନିବେଶ_type(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index]);
    }

    PartialShape get_ନିବେଶ_shape(size_t ନିବେଶ_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_ନିବେଶ_shape(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[ନିବେଶ_index]);
    }

    std::vector<size_t> get_ନିବେଶ_stride(size_t index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_ନିବେଶ_stride(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index]);
    }

    std::string get_output_name() ଅଚଳ { ଫେରନ୍ତୁ m_output_names[0]; }

    PartialShape get_output_shape() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_output_shape(m_ଗଣ୍ଠି_idx); }

    int32_t * get_ନିବେଶ_op_params(size_t index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_ନିବେଶ_op_params(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index]);
    }

    size_t get_view_ନିବେଶ_size(size_t index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_size(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index]);
    }

    size_t get_view_ନିବେଶ_offset(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_offset(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    size_t get_view_ନିବେଶ_src_offset(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_src_offset(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    std::vector<size_t> get_view_ନିବେଶ_stride(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_stride(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    std::vector<size_t> get_view_ନିବେଶ_src_stride(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_src_stride(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    ov::Shape get_view_ନିବେଶ_ggml_shape(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_ggml_shape(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    ov::Shape get_view_ନିବେଶ_src_ggml_shape(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_src_ggml_shape(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    ov::PartialShape get_view_ନିବେଶ_ov_shape(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_ov_shape(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    ov::PartialShape get_view_ନିବେଶ_src_ov_shape(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_src_ov_shape(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    std::string get_view_ନିବେଶ_name(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_name(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    std::string get_view_ନିବେଶ_src_name(size_t index, size_t view_index) ଅଚଳ {
        ଫେରନ୍ତୁ m_decoder->get_view_ନିବେଶ_src_name(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[index], view_index);
    }

    int32_t get_op_dynamic_dim() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_op_dynamic_dim(m_ଗଣ୍ଠି_idx); }

    int32_t * get_output_op_params() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_output_op_params(m_ଗଣ୍ଠି_idx); }

    size_t get_output_op_offset() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_output_op_offset(m_ଗଣ୍ଠି_idx); }

    ov::element::Type get_output_type() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_output_type(m_ଗଣ୍ଠି_idx); }

    std::vector<size_t> get_output_stride() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_output_stride(m_ଗଣ୍ଠି_idx); }

    Output<ଗଣ୍ଠି_2> get_ନିବେଶ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx) ଅଚଳ override {
        // Check if this input is a VIEW
        size_t view_ନିବେଶ_size = m_decoder->get_view_ନିବେଶ_size(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[idx]);
        ଯଦି (view_ନିବେଶ_size > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string base_name =
                m_decoder->get_view_ନିବେଶ_src_name(m_ଗଣ୍ଠି_idx, m_ନିବେଶ_names[idx], view_ନିବେଶ_size - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            auto view_it = m_tensor_map->find(m_ନିବେଶ_names[idx]);
            ଯଦି (!base_name.empty() && view_it != m_tensor_map->end()) {
                auto base_it = m_tensor_map->find(base_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                ଯଦି (base_it == m_tensor_map->end() ||
                    view_it->second.get_ଗଣ୍ଠି_shared_ptr() != base_it->second.get_ଗଣ୍ଠି_shared_ptr()) {
                    ଫେରନ୍ତୁ view_it->second;
                }
                ଫେରନ୍ତୁ base_it->second;
            }
            ଯଦି (!base_name.empty()) {
                ଫେରନ୍ତୁ m_tensor_map->at(base_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        ଫେରନ୍ତୁ m_tensor_map->at(m_ନିବେଶ_names[idx]);
    }

    Output<ଗଣ୍ଠି_2> get_ନିବେଶ(ଅଚଳ std::string & name) ଅଚଳ override {
        ଯଦି (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        ଫେରନ୍ତୁ m_tensor_map->at(name);
    }

    bool has_ନିବେଶ(ଅଚଳ std::string & name) ଅଚଳ { ଫେରନ୍ତୁ m_tensor_map->find(name) != m_tensor_map->end(); }

    ଶୂନ୍ୟ put_shared(ଅଚଳ std::string & name, ଅଚଳ Output<ଗଣ୍ଠି_2> & ମୂଲ୍ୟ) ଅଚଳ {
        m_tensor_map->ଭର୍ତ୍ତି({name, ମୂଲ୍ୟ});
    }

    ଅଚଳ std::string & get_name() ଅଚଳ override { ଫେରନ୍ତୁ m_decoder->get_op_name(m_ଗଣ୍ଠି_idx); }

    ov::Any get_attribute_as_any(ଅଚଳ std::string & name) ଅଚଳ override { ଫେରନ୍ତୁ m_decoder->get_attribute(name); }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_op_case() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_op_case(m_ଗଣ୍ଠି_idx); }

    bool is_static() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->is_static(); }

    bool is_stateful() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->is_stateful(); }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_ssm_state_size() ଅଚଳ { ଫେରନ୍ତୁ m_decoder->get_ssm_state_size(); }

ବ୍ୟକ୍ତିଗତ:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_ଗଣ୍ଠି_idx;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_ନିବେଶ_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(ଅଚଳ ov::frontend::ggml::ଗଣ୍ଠି_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
