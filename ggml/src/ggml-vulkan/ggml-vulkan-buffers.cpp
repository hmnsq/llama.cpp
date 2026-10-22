#ضمن "ggml-vulkan-common.h"

#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <mutex>
#ضمن <shared_mutex>

ساكن فراغ * portable_aligned_malloc(size_t align, size_t size) {
#که_تعریف_شوی _WIN32
    أرجع _aligned_malloc(size, align);
#وإلا
    فراغ * ptr = تش_اشاره;
    إذا (posix_memalign(&ptr, align, size) != 0) {
        أرجع تش_اشاره;
    }
    أرجع ptr;
#د_که_پای
}

ساكن فراغ portable_aligned_free(فراغ * ptr) {
#که_تعریف_شوی _WIN32
    _aligned_free(ptr);
#وإلا
    std::حرر(ptr);
#د_که_پای
}

ggml_backend_بفر_type_i ggml_backend_vk_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_vk_بفر_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_vk_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_بفر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_vk_بفر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ساكن std::vector<uint32_t> ggml_vk_پیدا_کړه_حافظه_properties(ثابت vk::Physical_Device_حافظه_Properties* mem_props, vk::حافظه_Requirements* mem_req, vk::حافظه_Property_Flags flags) {
    std::vector<uint32_t> indices;

    لكل (uint32_t i = 0; i < mem_props->حافظه_Type_شمېر; ++i) {
        vk::حافظه_Type حافظه_type = mem_props->حافظه_Types[i];
        إذا ((mem_req->حافظه_Type_بټونه & ((uint64_t)1 << i)) &&
            (flags & حافظه_type.propertyFlags) == flags &&
            mem_props->حافظه_Heaps[حافظه_type.ډېرۍ_شاخص].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    أرجع indices;
}

ساكن vk_بفر ggml_vk_create_بفر(vk_device& device, size_t size, ثابت std::initializer_list<vk::حافظه_Property_Flags> & req_flags_list,
                                       فراغ *import_ptr = تش_اشاره) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    إذا (size > device->max_بفر_اندازه) {
        ارم vk::بهر_د_Device_حافظه_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_بفر buf = std::make_shared<vk_بفر_struct>();

    إذا (size == 0) {
        buf->size = 0;
        أرجع buf;
    }

    vk::بفر_Usage_Flags usage_flags = vk::بفر_Usage_Flag_بټونه::e_Storage_بفر | vk::بفر_Usage_Flag_بټونه::eTransferSrc | vk::بفر_Usage_Flag_بټونه::eTransferDst;
    vk::حافظه_Allocate_Flags mem_flags {};
    إذا (device->بفر_device_پته) {
        usage_flags |= vk::بفر_Usage_Flag_بټونه::e_Shader_Device_پته;
        mem_flags |= vk::حافظه_Allocate_Flag_بټونه::e_Device_پته;
    }

    vk::بفر_Create_Info بفر_create_info{
        vk::بفر_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_ډېرپېښ::eExclusive,
        0,
        تش_اشاره,
    };

    vk::External_حافظه_بفر_Create_Info external_حافظه_bci;
    إذا (import_ptr) {
        external_حافظه_bci.handleTypes = vk::External_حافظه_Handle_Type_Flag_بټونه::e_Host_ویش_EXT;
        بفر_create_info.سېټ_PNext(&external_حافظه_bci);
    }

    buf->بفر = device->device.create_بفر(بفر_create_info);

    vk::حافظه_Requirements mem_req = device->device.get_بفر_حافظه_Requirements(buf->بفر);

    vk::Physical_Device_حافظه_Properties mem_props = device->physical_device.get_حافظه_Properties();

    ثابت vk::حافظه_لومړیتوب_Allocate_Info_EXT mem_لومړیتوب_info { 1.0f };

    vk::حافظه_Allocate_Flags_Info mem_flags_info { mem_flags };

    إذا (device->حافظه_لومړیتوب) {
        mem_flags_info.سېټ_PNext(&mem_لومړیتوب_info);
    }

    إذا (import_ptr) {
        vk::حافظه_Host_اشاره_ګر_Properties_EXT host_اشاره_ګر_props;
        حاول {
            host_اشاره_ګر_props = device->device.get_حافظه_Host_اشاره_ګر_Properties_EXT(vk::External_حافظه_Handle_Type_Flag_بټونه::e_Host_ویش_EXT, import_ptr);
        } امسك (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }
        vk::Physical_Device_حافظه_Properties mem_props = device->physical_device.get_حافظه_Properties();

        uint32_t حافظه_type_idx;
        vk::حافظه_Property_Flags property_flags = *req_flags_list.begin();
        لكل (حافظه_type_idx = 0; حافظه_type_idx < 32; ++حافظه_type_idx) {
            إذا (!(host_اشاره_ګر_props.حافظه_Type_بټونه & (1u << حافظه_type_idx))) {
                استمر;
            }
            إذا (!(mem_req.حافظه_Type_بټونه & (1u << حافظه_type_idx))) {
                استمر;
            }

            vk::حافظه_Type حافظه_type = mem_props.حافظه_Types[حافظه_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            إذا ((حافظه_type.propertyFlags & property_flags) == property_flags) {
                property_flags = حافظه_type.propertyFlags;
                اكسر;
            }
        }
        إذا (حافظه_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_بفر(buf->بفر);
            أرجع {};
        }

        buf->حافظه_property_flags = mem_props.حافظه_Types[حافظه_type_idx].propertyFlags;
        حاول {
            vk::Import_حافظه_Host_اشاره_ګر_Info_EXT import_info;
            import_info.handleType = vk::External_حافظه_Handle_Type_Flag_بټونه::e_Host_ویش_EXT;
            import_info.p_Host_اشاره_ګر = import_ptr;
            import_info.سېټ_PNext(&mem_flags_info);
            buf->device_حافظه = device->device.allocate_حافظه({ size, حافظه_type_idx, &import_info });
        } امسك (ثابت vk::SystemError& e) {
        }
    } وإلا {
        لكل (خپلکاری it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ثابت خپلکاری & req_flags = *it;

            ثابت std::vector<uint32_t> حافظه_type_indices = ggml_vk_پیدا_کړه_حافظه_properties(&mem_props, &mem_req, req_flags);

            إذا (حافظه_type_indices.تشه()) {
                استمر;
            }

            منطقي بشپړ = خطأ;

            لكل (خپلکاری mtype_it = حافظه_type_indices.begin(); mtype_it != حافظه_type_indices.end(); mtype_it++) {
                حاول {
                    buf->device_حافظه = device->device.allocate_حافظه({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->حافظه_property_flags = mem_props.حافظه_Types[*mtype_it].propertyFlags;
                    بشپړ = ریښتیا;
                    اكسر;
                } امسك (ثابت vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    إذا (it + 1 == req_flags_list.end() && mtype_it + 1 == حافظه_type_indices.end()) {
                        device->device.destroy_بفر(buf->بفر);
                        ارم e;
                    }
                }
            }

            إذا (بشپړ) {
                اكسر;
            }
        }
    }

    إذا (!buf->device_حافظه) {
        device->device.destroy_بفر(buf->بفر);
        ارم vk::بهر_د_Device_حافظه_Error("No suitable memory type found");
    }

    buf->ptr = تش_اشاره;

    إذا (import_ptr) {
        buf->ptr = import_ptr;
    } وإلا {
        إذا (buf->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible) {
            buf->ptr = device->device.map_حافظه(buf->device_حافظه, 0, VK_ټولیز_اندازه);
        }
    }

    device->device.bind_بفر_حافظه(buf->بفر, buf->device_حافظه, 0);

    buf->device = device;
    buf->size = size;

    إذا (device->بفر_device_پته) {
        ثابت vk::بفر_Device_پته_Info پته_Info(buf->بفر);
        buf->bda_addr = device->device.get_بفر_پته(پته_Info);
    }

    device->حافظه_logger->log_ویش(buf, size);

    أرجع buf;
}

vk_بفر ggml_vk_create_بفر_check(vk_device& device, size_t size, vk::حافظه_Property_Flags req_flags, vk::حافظه_Property_Flags fallback_flags) {
    حاول {
        أرجع ggml_vk_create_بفر(device, size, {req_flags, fallback_flags});
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::کرښه_پای;
        std::cerr << "ggml_vulkan: " << e.what() << std::کرښه_پای;
        ارم e;
    }
}

vk_بفر ggml_vk_create_بفر_device(vk_device& device, size_t size) {
    vk_بفر buf;
    حاول {
        إذا (device->prefer_host_حافظه) {
            buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent,
                                                       vk::حافظه_Property_Flag_بټونه::eDeviceLocal});
        } وإلا إذا (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eDeviceLocal | vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent,
                                                       vk::حافظه_Property_Flag_بټونه::eDeviceLocal,
                                                       vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent});
        } وإلا إذا (device->disable_host_visible_vidmem) {
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eDeviceLocal,
                                                           vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eDeviceLocal});
            }
        } وإلا {
            // use rebar if available, otherwise fallback to device only visible memory
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eDeviceLocal | vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent,
                                                           vk::حافظه_Property_Flag_بټونه::eDeviceLocal,
                                                           vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_بفر(device, size, {vk::حافظه_Property_Flag_بټونه::eDeviceLocal | vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent,
                                                           vk::حافظه_Property_Flag_بټونه::eDeviceLocal});
            }
        }
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::کرښه_پای;
        std::cerr << "ggml_vulkan: " << e.what() << std::کرښه_پای;
        ارم e;
    }

    أرجع buf;
}

فراغ ggml_vk_destroy_بفر(vk_بفر& buf) {
    إذا (buf == تش_اشاره) {
        أرجع;
    }

    إذا (buf->device != تش_اشاره) {
        buf->device->حافظه_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ساكن size_t ggml_vk_host_ویش_پیدا_کړه(ثابت فراغ * ptr) {
    لكل (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ثابت uint8_t * بنسټ = (ثابت uint8_t *) vk_instance.host_allocations[i].بنسټ;
        إذا (ptr >= بنسټ && ptr < بنسټ + vk_instance.host_allocations[i].size) {
            أرجع i;
        }
    }
    أرجع SIZE_MAX;
}

فراغ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_حافظه("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    إذا (vk_instance.multi_device) {
        ثابت size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ثابت size_t alloc_اندازه = (size + align - 1) / align * align;

        فراغ * ptr = portable_aligned_malloc(align, alloc_اندازه);
        إذا (ptr == تش_اشاره) {
            أرجع تش_اشاره;
        }
        vk_host_ویش ثبت;
        ثبت.بنسټ = ptr;
        ثبت.size = alloc_اندازه;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(ثبت));
        أرجع ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_بفر buf;
    حاول {
        buf = ggml_vk_create_بفر(dev0, size,
            {vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent | vk::حافظه_Property_Flag_بټونه::eHostCached,
             vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent});
    } امسك (vk::SystemError&) {
        أرجع تش_اشاره;
    }

    إذا (!(buf->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_بفر(buf);
        أرجع تش_اشاره;
    }

    vk_host_ویش ثبت;
    ثبت.بنسټ             = buf->ptr;
    ثبت.size             = size;
    ثبت.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(ثبت));
    أرجع buf->ptr;
}

فراغ ggml_vk_host_free(فراغ * ptr) {
    إذا (ptr == تش_اشاره) {
        أرجع;
    }
    VK_LOG_حافظه("ggml_vk_host_free(" << ptr << ")");

    vk_host_ویش ثبت;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ثابت size_t شاخص = ggml_vk_host_ویش_پیدا_کړه(ptr);
        إذا (شاخص == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            أرجع;
        }
        ثبت = std::move(vk_instance.host_allocations[شاخص]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + شاخص);
    }

    لكل (خپلکاری & view : ثبت.views) {
        ggml_vk_destroy_بفر(view);
    }
    إذا (vk_instance.multi_device) {
        portable_aligned_free(ثبت.بنسټ);
    }
}

فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_بفر& buf, size_t& buf_دننه_واټن) {
    buf = تش_اشاره;
    buf_دننه_واټن = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ثابت size_t شاخص = ggml_vk_host_ویش_پیدا_کړه(ptr);
    إذا (شاخص == SIZE_MAX) {
        أرجع;
    }
    vk_host_ویش & ثبت = vk_instance.host_allocations[شاخص];

    إذا (ثبت.views[device->idx] == تش_اشاره) {
        ثابت uint32_t بټ = 1u << device->idx;
        إذا (!vk_instance.multi_device || !vk_instance.device_host_بفر_import[device->idx] || (vk_instance.host_import_failed & بټ)) {
            أرجع;
        }
        GGML_ASSERT(device->external_حافظه_host);
        // const only propagates from the const context; the pointee is mutable
        ثبت.views[device->idx] = ggml_vk_بفر_from_host_ptr(const_cast<vk_device&>(device), ثبت.بنسټ, ثبت.size);
        إذا (ثبت.views[device->idx] == تش_اشاره) {
            vk_instance.host_import_failed |= بټ;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            أرجع;
        }
    }

    buf = ثبت.views[device->idx];
    buf_دننه_واټن = (ثابت uint8_t *) ptr - (ثابت uint8_t *) ثبت.بنسټ;
}

فراغ ggml_vk_ensure_sync_staging_بفر(vk_device& device, size_t size) {
    إذا (device->sync_staging == تش_اشاره || device->sync_staging->size < size) {
        VK_LOG_حافظه("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(device->sync_staging);
        device->sync_staging = ggml_vk_create_بفر_check(device, size,
            vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent | vk::حافظه_Property_Flag_بټونه::eHostCached,
            vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent);
    }
}

فراغ ggml_vk_ensure_sync_staging_بفر(ggml_backend_vk_context * ctx, size_t size) {
    إذا (ctx->sync_staging == تش_اشاره || ctx->sync_staging->size < size) {
        VK_LOG_حافظه("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_بفر(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_بفر_check(ctx->device, size,
            vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent | vk::حافظه_Property_Flag_بټونه::eHostCached,
            vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent);
    }
}

ساكن فراغ ggml_vk_بفر_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_بفر& dst, size_t دننه_واټن, ثابت ggml_tensor * tensor, منطقي sync_staging = خطأ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    إذا(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::کرښه_پای;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_بفر buf = تش_اشاره;
    size_t buf_دننه_واټن = 0;
    ggml_vk_host_get(ctx->device, tensor->معلومات, buf, buf_دننه_واټن);

    ثابت uint64_t ne0 = tensor->ne[0];
    ثابت uint64_t ne1 = tensor->ne[1];
    ثابت uint64_t ne2 = tensor->ne[2];
    ثابت uint64_t ne3 = tensor->ne[3];
    ثابت uint64_t nb0 = tensor->nb[0];
    ثابت uint64_t nb1 = tensor->nb[1];
    ثابت uint64_t nb2 = tensor->nb[2];
    ثابت uint64_t nb3 = tensor->nb[3];
    ثابت ggml_type type = tensor->type;
    ثابت uint64_t ts = ggml_type_اندازه(type);
    ثابت uint64_t bs = ggml_blck_اندازه(type);

    ثابت uint64_t dstnb0 = ts;
    ثابت uint64_t dstnb1 = dstnb0*(ne0/bs);
    ثابت uint64_t dstnb2 = dstnb1*ne1;
    ثابت uint64_t dstnb3 = dstnb2*ne2;

    ثابت uint64_t ne = ggml_nelements(tensor);

    إذا (buf != تش_اشاره) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices;

        لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
            لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                إذا (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_دننه_واټن + i3*nb3 + i2*nb2, دننه_واټن + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } وإلا {
                    لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                        إذا (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_دننه_واټن + i3*nb3 + i2*nb2 + i1*nb1, دننه_واټن + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } وإلا {
                            ثابت uint64_t s_off = buf_دننه_واټن + i3*nb3 + i2*nb2 + i1*nb1;
                            ثابت uint64_t d_off = دننه_واټن + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع;
    }

    إذا (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_بفر& staging = ctx->device->sync_staging;
    ثابت uint64_t copy_اندازه = ts*ne/bs;
    ggml_vk_ensure_sync_staging_بفر(ctx->device, copy_اندازه);
    Vk_بفر_Copy buf_copy{ 0, دننه_واټن, copy_اندازه };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_بفر(subctx->s->بفر->buf, (Vk_بفر)staging->بفر, (Vk_بفر)dst->بفر, 1, &buf_copy);

    لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
        لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            إذا (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ثابت uint8_t *) tensor->معلومات + buf_دننه_واټن + i3*nb3 + i2*nb2, dstnb2, &subctx->دننه_memcpys);
            } وإلا {
                لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                    إذا (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ثابت uint8_t *) tensor->معلومات + buf_دننه_واټن + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->دننه_memcpys);
                    } وإلا {
                        ثابت uint64_t s_off = buf_دننه_واټن + i3*nb3 + i2*nb2 + i1*nb1;
                        ثابت uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ثابت uint8_t *) tensor->معلومات + s_off + i0*nb0, dstnb0, &subctx->دننه_memcpys);
                        }
                    }
                }
            }
        }
    }
}

منطقي ggml_vk_بفر_write_2d_async(vk_context subctx, vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << پلنوالی << ", " << لوړوالی << ")");
    // Check if src is pinned memory
    vk_بفر buf = تش_اشاره;
    size_t buf_دننه_واټن = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_دننه_واټن);

    إذا (buf != تش_اشاره) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::بفر_Copy> slices(1);
        إذا (پلنوالی == spitch && پلنوالی == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_دننه_واټن_2 = buf_دننه_واټن;
            slices[0].dst_دننه_واټن_2 = دننه_واټن;
            slices[0].size = پلنوالی * لوړوالی;
        } وإلا {
            slices.resize(لوړوالی);
            لكل (size_t i = 0; i < لوړوالی; i++) {
                slices[i].src_دننه_واټن_2 = buf_دننه_واټن + i * spitch;
                slices[i].dst_دننه_واټن_2 = دننه_واټن + i * dpitch;
                slices[i].size = پلنوالی;
            }
        }

        ggml_vk_sync_buffers(تش_اشاره, subctx);
        subctx->s->بفر->buf.copy_بفر(buf->بفر, dst->بفر, slices);
        أرجع ریښتیا;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Staging buffer required
    ثابت size_t staging_اندازه = پلنوالی * لوړوالی;
    ggml_vk_ensure_sync_staging_بفر(dst->device, staging_اندازه);

    vk_بفر& staging_بفر = dst->device->sync_staging;

    std::vector<vk::بفر_Copy> slices(1);
    إذا (پلنوالی == dpitch) {
        slices[0].src_دننه_واټن_2 = 0;
        slices[0].dst_دننه_واټن_2 = دننه_واټن;
        slices[0].size = staging_اندازه;
    } وإلا {
        slices.resize(لوړوالی);
        لكل (size_t i = 0; i < لوړوالی; i++) {
            slices[i].src_دننه_واټن_2 = i * پلنوالی;
            slices[i].dst_دننه_واټن_2 = دننه_واټن + i * dpitch;
            slices[i].size = پلنوالی;
        }
    }

    ggml_vk_sync_buffers(تش_اشاره, subctx);
    subctx->s->بفر->buf.copy_بفر(staging_بفر->بفر, dst->بفر, slices);

    إذا (پلنوالی == spitch) {
        deferred_memcpy((uint8_t *)staging_بفر->ptr, src, staging_اندازه, &subctx->دننه_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < لوړوالی; i++) {
            deferred_memcpy((uint8_t *)staging_بفر->ptr + i * پلنوالی, (ثابت uint8_t *) src + i * spitch, پلنوالی, &subctx->دننه_memcpys);
        }
    }
    أرجع ریښتیا;
}

منطقي ggml_vk_بفر_write_async(vk_context subctx, vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t size, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    أرجع ggml_vk_بفر_write_2d_async(subctx, dst, دننه_واټن, src, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_write_2d(vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << پلنوالی << ", " << لوړوالی << ")");
    // Buffer is already mapped
    إذا(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible) {
        GGML_ASSERT(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostCoherent);

        إذا (پلنوالی == spitch && پلنوالی == dpitch) {
            حافظه_نقل((uint8_t *)dst->ptr + دننه_واټن, src, پلنوالی * لوړوالی);
        } وإلا {
            لكل (size_t i = 0; i < لوړوالی; i++) {
                حافظه_نقل((uint8_t *)dst->ptr + دننه_واټن + i * dpitch, (ثابت uint8_t *) src + i * spitch, پلنوالی);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_کتار->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        منطقي ret = ggml_vk_بفر_write_2d_async(subctx, dst, دننه_واټن, src, spitch, dpitch, پلنوالی, لوړوالی, ریښتیا);
        GGML_ASSERT(ret);
        ggml_vk_ctx_پای_ټکی(subctx);

        لكل (خپلکاری& cpy : subctx->دننه_memcpys) {
            حافظه_نقل(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خپلکاری& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ریښتیا, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_کتار_command_pools_cleanup(dst->device);
    }
}

فراغ ggml_vk_بفر_write(vk_بفر& dst, size_t دننه_واټن, ثابت فراغ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_بفر_write_2d(dst, دننه_واټن, src, size, size, size, 1);
}

منطقي ggml_vk_بفر_read_2d_async(vk_context subctx, vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << دننه_واټن << ", width=" << پلنوالی << ", height=" << لوړوالی << ")");
    GGML_ASSERT(پلنوالی > 0);
    GGML_ASSERT(لوړوالی > 0);
    GGML_ASSERT(src != تش_اشاره);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_بفر buf = تش_اشاره;
    size_t buf_دننه_واټن = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_دننه_واټن);

    std::vector<vk::بفر_Copy> slices(1);
    إذا (پلنوالی == spitch && پلنوالی == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_دننه_واټن_2 = دننه_واټن;
        slices[0].dst_دننه_واټن_2 = buf_دننه_واټن;
        slices[0].size = پلنوالی * لوړوالی;
    } وإلا {
        slices.resize(لوړوالی);
        لكل (size_t i = 0; i < لوړوالی; i++) {
            slices[i].src_دننه_واټن_2 = دننه_واټن + i * spitch;
            slices[i].dst_دننه_واټن_2 = buf_دننه_واټن + i * dpitch;
            slices[i].size = پلنوالی;
        }
    }

    إذا (buf != تش_اشاره) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(تش_اشاره, subctx);
        subctx->s->بفر->buf.copy_بفر(src->بفر, buf->بفر, slices);

        أرجع ریښتیا;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Fall back to staging buffer
    ثابت size_t staging_اندازه = پلنوالی * لوړوالی;
    ggml_vk_ensure_sync_staging_بفر(src->device, staging_اندازه);

    vk_بفر& staging_بفر = src->device->sync_staging;

    std::vector<vk::بفر_Copy> staging_slices(1);
    إذا (پلنوالی == spitch) {
        staging_slices[0].src_دننه_واټن_2 = دننه_واټن;
        staging_slices[0].dst_دننه_واټن_2 = 0;
        staging_slices[0].size = staging_اندازه;
    } وإلا {
        staging_slices.resize(لوړوالی);
        لكل (size_t i = 0; i < لوړوالی; i++) {
            staging_slices[i].src_دننه_واټن_2 = دننه_واټن + i * spitch;
            staging_slices[i].dst_دننه_واټن_2 = i * پلنوالی;
            staging_slices[i].size = پلنوالی;
        }
    }

    ggml_vk_sync_buffers(تش_اشاره, subctx);
    subctx->s->بفر->buf.copy_بفر(src->بفر, staging_بفر->بفر, staging_slices);

    إذا (پلنوالی == dpitch) {
        deferred_memcpy(dst, staging_بفر->ptr, staging_اندازه, &subctx->بهر_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < لوړوالی; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) staging_بفر->ptr + i * پلنوالی, پلنوالی, &subctx->بهر_memcpys);
        }
    }
    أرجع ریښتیا;
}

ساكن منطقي ggml_vk_بفر_read_async(vk_context subctx, vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t size, منطقي sync_staging = خطأ) {
    أرجع ggml_vk_بفر_read_2d_async(subctx, src, دننه_واټن, dst, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_بفر_read_2d(vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t spitch, size_t dpitch, size_t پلنوالی, size_t لوړوالی) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->بفر << ", " << دننه_واټن << ", " << پلنوالی << ", " << لوړوالی << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ثابت منطقي slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostCached);
    إذا(src->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_کتار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->بفر->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_بټونه::eComputeShader | vk::Pipeline_Stage_Flag_بټونه::eTransfer,
            vk::Pipeline_Stage_Flag_بټونه::eHost,
            {},
            { { vk::Access_Flag_بټونه::eShaderWrite | vk::Access_Flag_بټونه::eTransferWrite,
                vk::Access_Flag_بټونه::eHostRead } },
            {}, {});
        ggml_vk_ctx_پای_ټکی(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ریښتیا, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_کتار_command_pools_cleanup(src->device);

        إذا (پلنوالی == spitch && پلنوالی == dpitch) {
            حافظه_نقل(dst, (ثابت uint8_t *) src->ptr + دننه_واټن, پلنوالی * لوړوالی);
        } وإلا {
            لكل (size_t i = 0; i < لوړوالی; i++) {
                حافظه_نقل((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) src->ptr + دننه_واټن + i * spitch, پلنوالی);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_کتار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        منطقي ret = ggml_vk_بفر_read_2d_async(subctx, src, دننه_واټن, dst, spitch, dpitch, پلنوالی, لوړوالی, ریښتیا);
        GGML_ASSERT(ret);
        ggml_vk_ctx_پای_ټکی(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ریښتیا, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_کتار_command_pools_cleanup(src->device);

        لكل (خپلکاری& cpy : subctx->بهر_memcpys) {
            حافظه_نقل(cpy.dst, cpy.src, cpy.n);
        }
    }
}

فراغ ggml_vk_بفر_read(vk_بفر& src, size_t دننه_واټن, فراغ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->بفر << ", " << دننه_واټن << ", " << size << ")");
    ggml_vk_بفر_read_2d(src, دننه_واټن, dst, size, size, size, 1);
}

فراغ ggml_vk_بفر_copy_async(vk_context& ctx, vk_بفر& dst, size_t dst_دننه_واټن, vk_بفر& src, size_t src_دننه_واټن, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_بفر_Copy bc{ src_دننه_واټن, dst_دننه_واټن, size };

    vk_Cmd_Copy_بفر(ctx->s->بفر->buf, (Vk_بفر)src->بفر, (Vk_بفر)dst->بفر, 1, &bc);
}

فراغ ggml_vk_بفر_copy(vk_بفر& dst, size_t dst_دننه_واټن, vk_بفر& src, size_t src_دننه_واټن, size_t size) {
    إذا (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_کتار->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_بفر_copy_async(subctx, dst, dst_دننه_واټن, src, src_دننه_واټن, size);
        ggml_vk_ctx_پای_ټکی(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, ریښتیا, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_کتار_command_pools_cleanup(src->device);
    } وإلا {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_بفر(src->device, size);

        // Copy to src staging buffer
        ggml_vk_بفر_copy(src->device->sync_staging, 0, src, src_دننه_واټن, size);
        // Copy to dst buffer
        ggml_vk_بفر_write(dst, dst_دننه_واټن, src->device->sync_staging->ptr, size);
    }
}

فراغ ggml_vk_بفر_memset_async(vk_context& ctx, vk_بفر& dst, size_t دننه_واټن, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << دننه_واټن << ", " << c << ", " << size << ")");

    إذا (dst->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + دننه_واټن, c, size, &ctx->memsets);
        أرجع;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->بفر->buf.fill_بفر(dst->بفر, دننه_واټن, size, c);
}

فراغ ggml_vk_بفر_memset(vk_بفر& dst, size_t دننه_واټن, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << دننه_واټن << ", " << c << ", " << size << ")");

    إذا (dst->حافظه_property_flags & vk::حافظه_Property_Flag_بټونه::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + دننه_واټن, c, size);
        أرجع;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_کتار->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->بفر->buf.fill_بفر(dst->بفر, دننه_واټن, size, c);
    ggml_vk_ctx_پای_ټکی(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, ریښتیا, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_کتار_command_pools_cleanup(dst->device);
}

ggml_backend_بفر_i ggml_backend_vk_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_vk_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_vk_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_vk_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_vk_بفر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_بفر_سېټ_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_بفر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_بفر_clear,
    /* .reset           = */ عدم,
};

vk_بفر ggml_vk_بفر_from_host_ptr(vk_device & device, فراغ * ptr, size_t size) {
    إذا (!device->external_حافظه_host) {
        أرجع {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    إذا (uptr & (device->min_imported_host_اشاره_ګر_alignment - 1)) {
        أرجع {};
    }
    إذا (size & (device->min_imported_host_اشاره_ګر_alignment - 1)) {
        أرجع {};
    }

    ثابت vk::حافظه_Property_Flags property_flags = vk::حافظه_Property_Flag_بټونه::eHostVisible | vk::حافظه_Property_Flag_بټونه::eHostCoherent | vk::حافظه_Property_Flag_بټونه::eHostCached;

    vk_بفر buf {};
    حاول {
        buf = ggml_vk_create_بفر(device, size, { property_flags }, ptr);
    } امسك (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    أرجع buf;
}

