#pragma once

#अन्तर्भाव <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_Argsort_शिखरम्_K : सार्वजनिक ov::pass::MatcherPass {
सार्वजनिक:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseArgsortTopK")
    Fuse_Argsort_शिखरम्_K();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
