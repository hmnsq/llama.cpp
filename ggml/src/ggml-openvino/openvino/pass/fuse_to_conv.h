#شامل "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_مقصد_Conv : عوامی ov::pass::MatcherPass {
عوامی:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToConv")
    Fuse_مقصد_Conv();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
