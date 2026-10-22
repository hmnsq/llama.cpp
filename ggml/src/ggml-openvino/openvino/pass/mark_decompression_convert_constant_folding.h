#pragma once

#அந்தர்பாவ "mark_decompression_convert_constant_folding.h"
#அந்தர்பாவ "openvino/core/visibility.hpp"
#அந்தர்பாவ "openvino/pass/matcher_pass.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#இல்லையெனில்
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    இல்லையெனில்
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : பொது MatcherPass {
பொது:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
