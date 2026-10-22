#pragma once

#অন্তর্ভাব "decoder.h"

#অন্তর্ভাব <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class প্রবেশ_Model : সার্বজনিক ov::frontend::প্রবেশ_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

সার্বজনিক:
    explicit প্রবেশ_Model(ধ্রুব std::shared_ptr<GgmlDecoder> & gdecoder);

    ধ্রুব std::shared_ptr<GgmlDecoder> & get_model_decoder() ধ্রুব;

গুপ্ত:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
