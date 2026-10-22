// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ہدایت once

#شامل <openvino/core/core_visibility.hpp>
#شامل <openvino/core/node.hpp>
#شامل <openvino/core/runtime_attribute.hpp>

نامخلا ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
جماعت OPENVINO_API WeightlessCacheAttribute : عوامی RuntimeAttribute {
عوامی:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = حذف;

    WeightlessCacheAttribute(size_t original_سائز, size_t bin_انحراف, ov::element::Type original_dtype)
        : original_سائز(original_سائز),
          bin_انحراف(bin_انحراف),
          original_dtype(original_dtype) {}

    منطقی is_copyable() ثابت override;

    size_t original_سائز;
    size_t bin_انحراف;
    ov::element::Type original_dtype;
};

}  // namespace ov
