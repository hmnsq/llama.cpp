#pragma once

#অন্তর্ভাব <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_শীর্ষ_K : সার্বজনিক ov::pass::MatcherPass {
সার্বজনিক:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_শীর্ষ_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
