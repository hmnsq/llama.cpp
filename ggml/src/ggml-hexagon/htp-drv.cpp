#অন্তর্ভাৱ <filesystem>
#অন্তর্ভাৱ <set>
#অন্তর্ভাৱ <sstream>
#অন্তর্ভাৱ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#অন্যথা
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "htp-drv.h"
#অন্তর্ভাৱ "libdl.h"

#অন্তর্ভাৱ <domain.h>

//
// Driver API types
//

প্রকারনির্দেশ শূন্য * (*rpcmem_alloc_pfn_t)(পূর্ণাঙ্ক heapid, uint32_t flags, পূর্ণাঙ্ক size);
প্রকারনির্দেশ শূন্য * (*rpcmem_alloc2_pfn_t)(পূর্ণাঙ্ক heapid, uint32_t flags, size_t size);
প্রকারনির্দেশ শূন্য   (*rpcmem_free_pfn_t)(শূন্য * po);
প্রকারনির্দেশ পূর্ণাঙ্ক    (*rpcmem_to_fd_pfn_t)(শূন্য * po);

প্রকারনির্দেশ AEEResult (*dspqueue_create_pfn_t)(পূর্ণাঙ্ক                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           শূন্য *              callback_context,
                                           dspqueue_t *        queue);
প্রকারনির্দেশ AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
প্রকারনির্দেশ AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
প্রকারনির্দেশ AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t অঙ্ক_buffers,
                                          সংরচনা dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          ধ্রুৱ uint8_t *message,
                                          uint32_t timeout_us);
প্রকারনির্দেশ AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *অঙ্ক_buffers,
                                         সংরচনা dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
প্রকারনির্দেশ AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *অঙ্ক_buffers,
                                                 সংরচনা dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
প্রকারনির্দেশ পূর্ণাঙ্ক (*fastrpc_mmap_pfn_t)(পূর্ণাঙ্ক domain, পূর্ণাঙ্ক fd, শূন্য *addr, পূর্ণাঙ্ক offset, size_t length, গণনা fastrpc_map_flags flags);
প্রকারনির্দেশ পূর্ণাঙ্ক (*fastrpc_munmap_pfn_t)(পূর্ণাঙ্ক domain, পূর্ণাঙ্ক fd, শূন্য *addr, size_t length);

প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_handle64_open_pfn_t)(ধ্রুৱ ৱর্ণ* name, remote_handle64 *ph);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_handle64_close_pfn_t)(remote_handle h);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_handle_control_pfn_t)(uint32_t req, শূন্য* data, uint32_t datalen);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, শূন্য* data, uint32_t datalen);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_session_control_pfn_t)(uint32_t req, শূন্য *data, uint32_t datalen);
প্রকারনির্দেশ পূর্ণাঙ্ক (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

শূন্য * rpcmem_alloc(পূর্ণাঙ্ক heapid, uint32_t flags, পূর্ণাঙ্ক size) {
    ঘূৰাই rpcmem_alloc_pfn(heapid, flags, size);
}

শূন্য * rpcmem_alloc2(পূর্ণাঙ্ক heapid, uint32_t flags, size_t size) {
    যদি (rpcmem_alloc2_pfn) {
        ঘূৰাই rpcmem_alloc2_pfn(heapid, flags, size);
    } অন্যথা {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ঘূৰাই rpcmem_alloc_pfn(heapid, flags, size);
    }
}

শূন্য rpcmem_free(শূন্য * po) {
    ঘূৰাই rpcmem_free_pfn(po);
}

পূর্ণাঙ্ক rpcmem_to_fd(শূন্য * po) {
    ঘূৰাই rpcmem_to_fd_pfn(po);
}

HTPDRV_API পূর্ণাঙ্ক fastrpc_mmap(পূর্ণাঙ্ক domain, পূর্ণাঙ্ক fd, শূন্য * addr, পূর্ণাঙ্ক offset, size_t length, গণনা fastrpc_map_flags flags) {
    ঘূৰাই fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API পূর্ণাঙ্ক fastrpc_munmap(পূর্ণাঙ্ক domain, পূর্ণাঙ্ক fd, শূন্য * addr, size_t length) {
    ঘূৰাই fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(পূর্ণাঙ্ক                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          শূন্য *              callback_context,
                          dspqueue_t *        queue) {
    ঘূৰাই dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ঘূৰাই dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ঘূৰাই dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 অঙ্ক_buffers,
                         সংরচনা dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         ধ্রুৱ uint8_t *          message,
                         uint32_t                 timeout_us) {
    ঘূৰাই dspqueue_write_pfn(queue, flags, অঙ্ক_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               অঙ্ক_buffers,
                        সংরচনা dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    যদি (timeout_us == 0) {
        ঘূৰাই dspqueue_read_noblock_pfn(queue, flags, max_buffers, অঙ্ক_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    ঘূৰাই dspqueue_read_pfn(queue, flags, max_buffers, অঙ্ক_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API পূর্ণাঙ্ক remote_handle64_open(ধ্রুৱ ৱর্ণ * name, remote_handle64 * ph) {
    ঘূৰাই remote_handle64_open_pfn(name, ph);
}

HTPDRV_API পূর্ণাঙ্ক remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ঘূৰাই remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API পূর্ণাঙ্ক remote_handle64_close(remote_handle64 h) {
    ঘূৰাই remote_handle64_close_pfn(h);
}

HTPDRV_API পূর্ণাঙ্ক remote_handle_control(uint32_t req, শূন্য * data, uint32_t datalen) {
    ঘূৰাই remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API পূর্ণাঙ্ক remote_handle64_control(remote_handle64 h, uint32_t req, শূন্য * data, uint32_t datalen) {
    ঘূৰাই remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API পূর্ণাঙ্ক remote_session_control(uint32_t req, শূন্য * data, uint32_t datalen) {
    ঘূৰাই remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API পূর্ণাঙ্ক remote_system_request(system_req_payload * req) {
    যদি (!remote_system_request_pfn) {
        ঘূৰাই AEE_EUNSUPPORTEDAPI;
    }
    ঘূৰাই remote_system_request_pfn(req);
}

#ifdef _WIN32

স্থির std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    যদি (wstr.empty()) {
        ঘূৰাই result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (পূর্ণাঙ্ক) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    যদি (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ক্ষিপ std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    পূর্ণাঙ্ক bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (পূর্ণাঙ্ক) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    যদি (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ক্ষিপ std::runtime_error("Wstring conversion failed");
    }
    ঘূৰাই result;
}

স্থির std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(রিক্ত, রিক্ত, STANDARD_RIGHTS_READ);
    যদি (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ঘূৰাই result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    যদি (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ঘূৰাই result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    যদি (!QueryServiceConfigW(schService, রিক্ত, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ঘূৰাই result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    যদি (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ঘূৰাই result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_দ্বিমিক_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ধ্রুৱ std::wstring system_মূল_Placeholder = L"\\SystemRoot";
    যদি (0 != driverPath.compare(0, system_মূল_Placeholder.length(), system_মূল_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ঘূৰাই result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ধ্রুৱ std::wstring system_মূল_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD অঙ্ক_শব্দগুলি = GetEnvironmentVariableW(system_মূল_Env.c_str(), রিক্ত, 0);
    যদি (অঙ্ক_শব্দগুলি == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ঘূৰাই result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_মূল(অঙ্ক_শব্দগুলি + 1);
    অঙ্ক_শব্দগুলি = GetEnvironmentVariableW(system_মূল_Env.c_str(), system_মূল.data(), অঙ্ক_শব্দগুলি + 1);
    যদি (অঙ্ক_শব্দগুলি == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ঘূৰাই result;
    }
    driverPath.replace(0, system_মূল_Placeholder.length(), std::wstring(system_মূল.data()));

    ঘূৰাই wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

পূর্ণাঙ্ক htpdrv_init() {
    স্থির dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    স্থির তার্কিক initialized = মিছা;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#অন্যথা
    std::string drv_path = "libcdsprpc.so";
#endif
    যদি (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ঘূৰাই AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    যদি (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ঘূৰাই AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, প্ৰতীক, ignore)                               \
    কৰা {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        যদি (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ঘূৰাই AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } যেতিয়ালৈ (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, মিছা);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, সঁচা);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, মিছা);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, মিছা);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, মিছা);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, মিছা);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, মিছা);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, মিছা);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, মিছা);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, মিছা);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, মিছা);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, মিছা);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, মিছা);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, মিছা);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, মিছা);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, মিছা);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, মিছা);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, মিছা);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, সঁচা);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = সঁচা;

    ঘূৰাই AEE_SUCCESS;
}

domain * htpdrv_get_domain(পূর্ণাঙ্ক domain_id) {
    পূর্ণাঙ্ক i    = 0;
    পূর্ণাঙ্ক size = পরিমাণ(supported_domains) / পরিমাণ(domain);

    বাবে (i = 0; i < size; i++) {
        যদি (supported_domains[i].id == domain_id) {
            ঘূৰাই &supported_domains[i];
        }
    }

    ঘূৰাই রিক্ত;
}

পূর্ণাঙ্ক htpdrv_get_arch(পূর্ণাঙ্ক domain, পূর্ণাঙ্ক * arch) {
    যদি (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ঘূৰাই AEE_EUNSUPPORTEDAPI;
    }

    সংরচনা remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    পূর্ণাঙ্ক err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, পরিমাণ(arch_ver));
    যদি ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ঘূৰাই AEE_EUNSUPPORTEDAPI;
    }

    যদি (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ঘূৰাই err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (পূর্ণাঙ্ক) ((val >> 4) * 10 + (val & 0x0f));
    ঘূৰাই 0;
}
