#pragma once

#অন্তর্ভাব "decoder.h"

#অন্তর্ভাব <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class প্রবেশ_Model : সর্বজনীন ov::frontend::প্রবেশ_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

সর্বজনীন:
    explicit প্রবেশ_Model(ধ্রুবক std::shared_ptr<GgmlDecoder> & gdecoder);

    ধ্রুবক std::shared_ptr<GgmlDecoder> & get_model_decoder() ধ্রুবক;

ব্যক্তিগত:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
