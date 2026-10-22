// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ضمن <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
عام:
    FrontEnd();

    ساكن std::shared_ptr<Model> convert(ثابت مدخل_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
