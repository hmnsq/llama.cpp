#pragma once

#अन्तर्भाव "decoder.h"

#अन्तर्भाव <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class आदान_Model : सार्वजनिक ov::frontend::आदान_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

सार्वजनिक:
    explicit आदान_Model(ध्रुव std::shared_ptr<GgmlDecoder> & gdecoder);

    ध्रुव std::shared_ptr<GgmlDecoder> & get_model_decoder() ध्रुव;

गुप्त:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
