#pragma once

#شامل "decoder.h"

#شامل <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class درون_داد_Model : عمومی ov::frontend::درون_داد_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

عمومی:
    explicit درون_داد_Model(ثابت std::shared_ptr<GgmlDecoder> & gdecoder);

    ثابت std::shared_ptr<GgmlDecoder> & get_model_decoder() ثابت;

خصوصی:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
