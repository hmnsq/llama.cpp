#pragma once

#شامل <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseMoeRouter : عمومی ov::pass::MatcherPass {
عمومی:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeRouter")
    FuseMoeRouter();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
