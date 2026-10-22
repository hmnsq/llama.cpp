#pragma once

#ضمن <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseArgsortTopK : عام ov::pass::MatcherPass {
عام:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    FuseArgsortTopK();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
