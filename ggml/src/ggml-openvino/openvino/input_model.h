#pragma once

#അന്തര്ഭാവ "decoder.h"

#അന്തര്ഭാവ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class നിവേശം_Model : പൊതു ov::frontend::നിവേശം_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

പൊതു:
    explicit നിവേശം_Model(സ്ഥിരം std::shared_ptr<GgmlDecoder> & gdecoder);

    സ്ഥിരം std::shared_ptr<GgmlDecoder> & get_model_decoder() സ്ഥിരം;

സ്വകാര്യ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
