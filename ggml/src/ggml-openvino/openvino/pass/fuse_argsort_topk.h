#pragma once

#અન્તર્ભાવ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_ટોચ_K : જાહેર ov::pass::MatcherPass {
જાહેર:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_ટોચ_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
