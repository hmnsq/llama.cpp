#لارښوونه once

#ضمن "decoder.h"

#ضمن <cstdint>
#ضمن <openvino/frontend/node_context.hpp>
#ضمن <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

تعريفنوع std::map<std::string, Output<غوټه_2>> TensorMap;

class غوټه_Context : عام frontend::غوټه_Context {
عام:
    غوټه_Context(ثابت std::shared_ptr<GgmlDecoder> & decoder,
                std::shared_ptr<TensorMap> & tensor_map,
                صحيح غوټه_idx,
                TranslateSession * translate_session = nullptr) :
        ov::frontend::غوټه_Context(decoder->get_op_type(غوټه_idx)),
        m_decoder(decoder),
        m_tensor_map(tensor_map),
        m_غوټه_idx(غوټه_idx),
        m_translate_session(translate_session) {
        m_ننوت_نومونه = decoder->get_ننوت_نومونه(m_غوټه_idx);
        m_output_نومونه = decoder->get_output_نومونه(m_غوټه_idx);
    }

    TranslateSession * get_translate_session() ثابت { أرجع m_translate_session; }

    ثابت std::ویکټور<std::string> & get_ننوت_نومونه() ثابت { أرجع m_ننوت_نومونه; }

    ثابت std::ویکټور<std::string> & get_output_نومونه() ثابت { أرجع m_output_نومونه; }

    size_t get_ننوت_اندازه() ثابت override { أرجع m_decoder->get_ننوت_اندازه(m_غوټه_idx); }

    ov::element::Type get_ننوت_type(size_t شاخص) ثابت {
        أرجع m_decoder->get_ننوت_type(m_غوټه_idx, m_ننوت_نومونه[شاخص]);
    }

    PartialShape get_ننوت_shape(size_t ننوت_شاخص) ثابت {
        أرجع m_decoder->get_ننوت_shape(m_غوټه_idx, m_ننوت_نومونه[ننوت_شاخص]);
    }

    std::ویکټور<size_t> get_ننوت_stride(size_t شاخص) ثابت {
        أرجع m_decoder->get_ننوت_stride(m_غوټه_idx, m_ننوت_نومونه[شاخص]);
    }

    std::string get_output_name() ثابت { أرجع m_output_نومونه[0]; }

    PartialShape get_output_shape() ثابت { أرجع m_decoder->get_output_shape(m_غوټه_idx); }

    int32_t * get_ننوت_op_params(size_t شاخص) ثابت {
        أرجع m_decoder->get_ننوت_op_params(m_غوټه_idx, m_ننوت_نومونه[شاخص]);
    }

    size_t get_view_ننوت_اندازه(size_t شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_اندازه(m_غوټه_idx, m_ننوت_نومونه[شاخص]);
    }

    size_t get_view_ننوت_دننه_واټن(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_دننه_واټن(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    size_t get_view_ننوت_src_دننه_واټن(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_src_دننه_واټن(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    std::ویکټور<size_t> get_view_ننوت_stride(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_stride(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    std::ویکټور<size_t> get_view_ننوت_src_stride(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_src_stride(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    ov::Shape get_view_ننوت_ggml_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_ggml_shape(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    ov::Shape get_view_ننوت_src_ggml_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_src_ggml_shape(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    ov::PartialShape get_view_ننوت_ov_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_ov_shape(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    ov::PartialShape get_view_ننوت_src_ov_shape(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_src_ov_shape(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    std::string get_view_ننوت_name(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_name(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    std::string get_view_ننوت_src_name(size_t شاخص, size_t view_شاخص) ثابت {
        أرجع m_decoder->get_view_ننوت_src_name(m_غوټه_idx, m_ننوت_نومونه[شاخص], view_شاخص);
    }

    int32_t get_op_dynamic_dim() ثابت { أرجع m_decoder->get_op_dynamic_dim(m_غوټه_idx); }

    int32_t * get_output_op_params() ثابت { أرجع m_decoder->get_output_op_params(m_غوټه_idx); }

    size_t get_output_op_دننه_واټن() ثابت { أرجع m_decoder->get_output_op_دننه_واټن(m_غوټه_idx); }

    ov::element::Type get_output_type() ثابت { أرجع m_decoder->get_output_type(m_غوټه_idx); }

    std::ویکټور<size_t> get_output_stride() ثابت { أرجع m_decoder->get_output_stride(m_غوټه_idx); }

    Output<غوټه_2> get_ننوت(صحيح idx) ثابت override {
        // Check if this input is a VIEW
        size_t view_ننوت_اندازه = m_decoder->get_view_ننوت_اندازه(m_غوټه_idx, m_ننوت_نومونه[idx]);
        إذا (view_ننوت_اندازه > 0) {
            // This is a VIEW input, get the base tensor name (last element in the chain)
            std::string بنسټ_name =
                m_decoder->get_view_ننوت_src_name(m_غوټه_idx, m_ننوت_نومونه[idx], view_ننوت_اندازه - 1);
            // Check if the VIEW has been resolved (translate_view produced a Slice)
            خپلکاری view_it = m_tensor_map->پیدا_کړه(m_ننوت_نومونه[idx]);
            إذا (!بنسټ_name.تشه() && view_it != m_tensor_map->پای_ټکی()) {
                خپلکاری بنسټ_it = m_tensor_map->پیدا_کړه(بنسټ_name);
                // A multi-output translator can publish a VIEW directly without materializing
                // its packed parent (GatedDeltaNet attention/state). In that case the VIEW is the
                // authoritative value. The node comparison retains the existing resolved-VIEW path.
                إذا (بنسټ_it == m_tensor_map->پای_ټکی() ||
                    view_it->دوهم.get_غوټه_shared_ptr() != بنسټ_it->دوهم.get_غوټه_shared_ptr()) {
                    أرجع view_it->دوهم;
                }
                أرجع بنسټ_it->دوهم;
            }
            إذا (!بنسټ_name.تشه()) {
                أرجع m_tensor_map->at(بنسټ_name);
            }
        }
        // Not a VIEW or failed to get base name, use the original logic
        أرجع m_tensor_map->at(m_ننوت_نومونه[idx]);
    }

    Output<غوټه_2> get_ننوت(ثابت std::string & name) ثابت override {
        إذا (m_tensor_map->پیدا_کړه(name) == m_tensor_map->پای_ټکی()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        أرجع m_tensor_map->at(name);
    }

    bool has_ننوت(ثابت std::string & name) ثابت { أرجع m_tensor_map->پیدا_کړه(name) != m_tensor_map->پای_ټکی(); }

    فراغ put_shared(ثابت std::string & name, ثابت Output<غوټه_2> & ارزښت) ثابت {
        m_tensor_map->ورننباسه({name, ارزښت});
    }

    ثابت std::string & get_name() ثابت override { أرجع m_decoder->get_op_name(m_غوټه_idx); }

    ov::Any get_attribute_as_any(ثابت std::string & name) ثابت override { أرجع m_decoder->get_attribute(name); }

    صحيح get_op_case() ثابت { أرجع m_decoder->get_op_case(m_غوټه_idx); }

    bool is_static() ثابت { أرجع m_decoder->is_static(); }

    bool is_stateful() ثابت { أرجع m_decoder->is_stateful(); }

    صحيح get_ssm_state_اندازه() ثابت { أرجع m_decoder->get_ssm_state_اندازه(); }

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> & m_tensor_map;
    صحيح m_غوټه_idx;
    TranslateSession * m_translate_session;
    std::ویکټور<std::string> m_ننوت_نومونه;
    std::ویکټور<std::string> m_output_نومونه;
};

using CreatorFunction = std::function<ov::Output_ویکټور(ثابت ov::frontend::ggml::غوټه_Context &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
