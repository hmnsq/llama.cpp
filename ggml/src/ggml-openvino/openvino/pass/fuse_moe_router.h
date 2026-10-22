#pragma once

#ଅନ୍ତର୍ଭାଵ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseMoeRouter : ସର୍ବସାଧାରଣ ov::pass::MatcherPass {
ସର୍ବସାଧାରଣ:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseMoeRouter")
    FuseMoeRouter();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
