#pragma once

#ಅನ್ತರ್ಭಾವ "decoder.h"

#ಅನ್ತರ್ಭಾವ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ಒಳಹರಿವು_Model : ಸಾರ್ವಜನಿಕ ov::frontend::ಒಳಹರಿವು_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

ಸಾರ್ವಜನಿಕ:
    explicit ಒಳಹರಿವು_Model(ಧ್ರುವ std::shared_ptr<GgmlDecoder> & gdecoder);

    ಧ್ರುವ std::shared_ptr<GgmlDecoder> & get_model_decoder() ಧ್ರುವ;

ಗುಪ್ತ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
