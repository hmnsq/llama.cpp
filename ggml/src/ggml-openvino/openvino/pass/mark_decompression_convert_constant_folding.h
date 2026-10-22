#pragma once

#ଅନ୍ତର୍ଭାଵ "mark_decompression_convert_constant_folding.h"
#ଅନ୍ତର୍ଭାଵ "openvino/core/visibility.hpp"
#ଅନ୍ତର୍ଭାଵ "openvino/pass/matcher_pass.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#ଅନ୍ୟଥା
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    ଅନ୍ୟଥା
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : ସର୍ବସାଧାରଣ MatcherPass {
ସର୍ବସାଧାରଣ:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
