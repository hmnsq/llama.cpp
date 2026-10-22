#pragma once

#அந்தர்பாவ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseMoeRouter : பொது ov::pass::MatcherPass {
பொது:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeRouter")
    FuseMoeRouter();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
