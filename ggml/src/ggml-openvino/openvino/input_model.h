#pragma once

#அந்தர்பாவ "decoder.h"

#அந்தர்பாவ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class உள்ளீடு_Model : பொது ov::frontend::உள்ளீடு_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

பொது:
    explicit உள்ளீடு_Model(மாறிலி std::shared_ptr<GgmlDecoder> & gdecoder);

    மாறிலி std::shared_ptr<GgmlDecoder> & get_model_decoder() மாறிலி;

தனிப்பட்ட:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
