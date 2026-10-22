#pragma once

#অন্তর্ভাৱ "decoder.h"

#অন্তর্ভাৱ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class প্ৰবেশ_Model : সার্ৱজনিক ov::frontend::প্ৰবেশ_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

সার্ৱজনিক:
    explicit প্ৰবেশ_Model(ধ্রুৱ std::shared_ptr<GgmlDecoder> & gdecoder);

    ধ্রুৱ std::shared_ptr<GgmlDecoder> & get_model_decoder() ধ্রুৱ;

গুপ্ত:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
