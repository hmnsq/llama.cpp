#અન્તર્ભાવ "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseToSDPA : જાહેર ov::pass::MatcherPass {
જાહેર:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToSDPA")
    FuseToSDPA();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
