#pragma once

#અન્તર્ભાવ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseMoeRouter : જાહેર ov::pass::MatcherPass {
જાહેર:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeRouter")
    FuseMoeRouter();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
