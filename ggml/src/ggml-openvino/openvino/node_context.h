#هدايت once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<ڳنڍ_2>> TensorMap;

class ڳنڍ_Context : عام frontend::ڳنڍ_Context {
عام:
    ڳنڍ_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح ڳنڍ_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::ڳنڍ_Context(decoder->get_op_type(ڳنڍ_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_ڳنڍ_idx(ڳنڍ_idx),
        m_translate_session(translate_session) {
        m_داخلا_نالا = decoder->get_داخلا_نالا(m_ڳنڍ_idx);
        m_output_نالا = decoder->get_output_نالا(m_ڳنڍ_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::ويڪٽر<std::string> & get_داخلا_نالا() ثابت { أرجع m_داخلا_نالا; }

    ثابت std::ويڪٽر<std::string> & get_output_نالا() ثابت { أرجع m_output_نالا; }

    size_t get_داخلا_سائيز() ثابت override { أرجع m_decoder->get_داخلا_سائيز(m_ڳنڍ_idx); }

    ov::element::Type get_داخلا_type(size_t اشاريو) ثابت {
        أرجع m_decoder->get_داخلا_type(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو]);
    }

    PartialShape get_داخلا_shape(size_t داخلا_اشاريو) ثابت {
        أرجع m_decoder->get_داخلا_shape(m_ڳنڍ_idx, m_داخلا_نالا[داخلا_اشاريو]);
    }

    std::ويڪٽر<size_t> get_داخلا_stride(size_t اشاريو) ثابت {
        أرجع m_decoder->get_داخلا_stride(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو]);
    }

    std::string get_output_name() ثابت { أرجع m_output_نالا[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_ڳنڍ_idx); }

    int32_t * get_داخلا_op_params(size_t اشاريو) ثابت {
        أرجع m_decoder->get_داخلا_op_params(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو]);
    }

    size_t get_view_داخلا_سائيز(size_t اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_سائيز(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو]);
    }

    size_t get_view_داخلا_وٿي(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_وٿي(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    size_t get_view_داخلا_src_وٿي(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_src_وٿي(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    std::ويڪٽر<size_t> get_view_داخلا_stride(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_stride(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    std::ويڪٽر<size_t> get_view_داخلا_src_stride(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_src_stride(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    ov::Shape get_view_داخلا_ggml_shape(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_ggml_shape(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    ov::Shape get_view_داخلا_src_ggml_shape(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_src_ggml_shape(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    ov::PartialShape get_view_داخلا_ov_shape(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_ov_shape(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    ov::PartialShape get_view_داخلا_src_ov_shape(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_src_ov_shape(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    std::string get_view_داخلا_name(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_name(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    std::string get_view_داخلا_src_name(size_t اشاريو, size_t view_اشاريو) ثابت {
        أرجع m_decoder->get_view_داخلا_src_name(m_ڳنڍ_idx, m_داخلا_نالا[اشاريو], view_اشاريو);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_ڳنڍ_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_ڳنڍ_idx); }

    size_t get_output_op_وٿي() ثابت { أرجع m_decoder->get_output_op_وٿي(m_ڳنڍ_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_ڳنڍ_idx); }

    std::ويڪٽر<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_ڳنڍ_idx); }

    Output<ڳنڍ_2> get_داخلا(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_داخلا_سائيز = m_decoder->get_view_داخلا_سائيز(m_ڳنڍ_idx, m_داخلا_نالا[idx]);
        إذا (view_داخلا_سائيز > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string بنياد_name =
                m_decoder->get_view_داخلا_src_name(m_ڳنڍ_idx, m_داخلا_نالا[idx], view_داخلا_سائيز - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            خودڪار view_it = m_tensor_map->ڳوليو(m_داخلا_نالا[idx]);
            إذا (!بنياد_name.خالي_جايون() && view_it != m_tensor_map->پڇاڙي()) {
                خودڪار بنياد_it = m_tensor_map->ڳوليو(بنياد_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (بنياد_it == m_tensor_map->پڇاڙي() ||
                    view_it->ٻيو.get_ڳنڍ_shared_ptr() != بنياد_it->ٻيو.get_ڳنڍ_shared_ptr()) {
                    أرجع view_it->ٻيو;
                }
                أرجع بنياد_it->ٻيو;
            }
            إذا (!بنياد_name.خالي_جايون()) {
                أرجع m_tensor_map->at(بنياد_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_داخلا_نالا[idx]);
    }

    Output<ڳنڍ_2> get_داخلا(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->ڳوليو(name) == m_tensor_map->پڇاڙي()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_داخلا(ثابت std::string & name) ثابت { أرجع m_tensor_map->ڳوليو(name) != m_tensor_map->پڇاڙي(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<ڳنڍ_2> & قدر) ثابت {
        m_tensor_map->وجهو({name, قدر});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_ڳنڍ_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_ڳنڍ_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_سائيز() ثابت { أرجع m_decoder->get_ssm_state_سائيز(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_ڳنڍ_idx;
    TranslateSession * m_translate_session;
    std::ويڪٽر<std::string> m_داخلا_نالا;
    std::ويڪٽر<std::string> m_output_نالا;
};

using CreatorFunction = std::function<ov::Output_ويڪٽر(ثابت ov::frontend::ggml::ڳنڍ_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
