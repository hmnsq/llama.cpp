#pragma once

#include "decoder.h"

#include <openvino/frontend/input_model.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class पासोक_Model : public ov::frontend::पासोक_Model {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit पासोक_Model(const std::shared_ptr<GgmlDecoder> & gdecoder);

    const std::shared_ptr<GgmlDecoder> & get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
