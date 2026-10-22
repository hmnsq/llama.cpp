#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "mark_decompression_convert_constant_folding.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/core/visibility.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "openvino/pass/matcher_pass.hpp"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ਨਹੀਂਤਾਂ
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ਨਹੀਂਤਾਂ
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#ਜੇ_ਅੰਤ      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : ਜਨਤਕ MatcherPass {
ਜਨਤਕ:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
