#pragma once

#شامل <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_بالاترین_K : عمومی ov::pass::MatcherPass {
عمومی:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_بالاترین_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
