#pragma once

#അന്തര്ഭാവ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_മുകൾ_K : പൊതു ov::pass::MatcherPass {
പൊതു:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_മുകൾ_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
