#pragma once

#અન્તર્ભાવ "decoder.h"

#અન્તર્ભાવ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class આગત_Model : જાહેર ov::frontend::આગત_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

જાહેર:
    explicit આગત_Model(અચલ std::shared_ptr<GgmlDecoder> & gdecoder);

    અચલ std::shared_ptr<GgmlDecoder> & get_model_decoder() અચલ;

ખાનગી:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
