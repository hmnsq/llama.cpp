#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class आगत_Model : सार्वजनिक ov::frontend::आगत_Model {
    मित्र class ::ov::frontend::ggml::FrontEnd;

सार्वजनिक:
    explicit आगत_Model(अचर std::shared_ptr<GgmlDecoder> & gdecoder);

    अचर std::shared_ptr<GgmlDecoder> & get_model_decoder() अचर;

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
