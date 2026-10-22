#pragma once

#ضمن "decoder.h"

#ضمن <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class مدخل_Model : عام ov::frontend::مدخل_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

عام:
    explicit مدخل_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خاص:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
