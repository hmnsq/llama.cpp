#pragma once

#ಅನ್ತರ್ಭಾವ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_ತುದಿ_K : ಸಾರ್ವಜನಿಕ ov::pass::MatcherPass {
ಸಾರ್ವಜನಿಕ:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_ತುದಿ_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
