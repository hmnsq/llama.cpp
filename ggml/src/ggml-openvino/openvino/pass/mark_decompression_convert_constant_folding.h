#निर्देशः once

#अन्तर्भाव "mark_decompression_convert_constant_folding.h"
#अन्तर्भाव "openvino/core/visibility.hpp"
#अन्तर्भाव "openvino/pass/matcher_pass.hpp"

#यदि_परिभाषितम् OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#अन्यथा
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    अन्यथा
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#यदि_अन्तः      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : सार्वजनिक MatcherPass {
सार्वजनिक:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
