#pragma once

#ଅନ୍ତର୍ଭାଵ "decoder.h"

#ଅନ୍ତର୍ଭାଵ <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class ନିବେଶ_Model : ସର୍ବସାଧାରଣ ov::frontend::ନିବେଶ_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

ସର୍ବସାଧାରଣ:
    explicit ନିବେଶ_Model(ଅଚଳ std::shared_ptr<GgmlDecoder> & gdecoder);

    ଅଚଳ std::shared_ptr<GgmlDecoder> & get_model_decoder() ଅଚଳ;

ବ୍ୟକ୍ତିଗତ:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
