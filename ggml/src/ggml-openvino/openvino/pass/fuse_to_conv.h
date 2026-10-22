#ضمن "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class Fuse_منزل_Conv : عام ov::pass::MatcherPass {
عام:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToConv")
    Fuse_منزل_Conv();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
