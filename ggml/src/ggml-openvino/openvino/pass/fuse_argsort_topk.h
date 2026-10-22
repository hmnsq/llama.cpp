#pragma once

#অন্তর্ভাৱ <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_শীৰ্ষ_K : সার্ৱজনিক ov::pass::MatcherPass {
সার্ৱজনিক:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_শীৰ্ষ_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
