#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <sstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdexcept>
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>
#ਅਨ੍ਤਰ੍ਭਾਵ <thread>
#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>
#ਅਨ੍ਤਰ੍ਭਾਵ <future>
#ਅਨ੍ਤਰ੍ਭਾਵ <queue>
#ਅਨ੍ਤਰ੍ਭਾਵ <condition_variable>
#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/stat.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
    #ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>
    #ਅਨ੍ਤਰ੍ਭਾਵ <direct.h> // For _mkdir on Windows
#ਨਹੀਂਤਾਂ
    #ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
    #ਅਨ੍ਤਰ੍ਭਾਵ <sys/wait.h>
    #ਅਨ੍ਤਰ੍ਭਾਵ <fcntl.h>
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ASYNCIO_CONCURRENCY 64

std::mutex ਤਾਲਾ;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ> compile_failed{ਝੂਠ};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string ਨਿਵੇਸ਼_filepath = "";
std::string output_dir = "/tmp";
std::string ਨਿਸ਼ਾਨਾ_hpp = "";
std::string ਨਿਸ਼ਾਨਾ_cpp = "";

ਅਚਲ std::vector<std::string> type_ਨਾਂ = {
    "f32",
    "f16",
    "q1_0",
    "q2_0",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "nvfp4",
    "tq1_0",
    "tq2_0",
    "bf16",
};

ਗਿਣਤੀ MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

ਨਾਮਾਕਾਸ਼ {

ਪੂਰਨ execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ਗੁਣ sa = { ਆਕਾਰ(SECURITY_ਗੁਣ), ਨਿਰਾ, TRUE };

    ਜੇ (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !ਸੈੱਟ_Handle_Information(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        ਸੁੱਟੋ std::runtime_error("Failed to create stdout pipe");
    }

    ਜੇ (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !ਸੈੱਟ_Handle_Information(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        ਸੁੱਟੋ std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = ਆਕਾਰ(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& part : command) {
        cmd += part + " ";
    }

    ਜੇ (!CreateProcessA(ਨਿਰਾ, cmd.ਅੰਕੜੇ(), ਨਿਰਾ, ਨਿਰਾ, TRUE, 0, ਨਿਰਾ, ਨਿਰਾ, &si, &pi)) {
        ਸੁੱਟੋ std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<ਅੱਖਰ, 128> ਬਫਰ;
    DWORD ਬਾਈਟਾਂ_read;

    ਜਦੋਂ (ReadFile(stdout_read, ਬਫਰ.ਅੰਕੜੇ(), (DWORD)ਬਫਰ.size(), &ਬਾਈਟਾਂ_read, ਨਿਰਾ) && ਬਾਈਟਾਂ_read > 0) {
        stdout_str.append(ਬਫਰ.ਅੰਕੜੇ(), ਬਾਈਟਾਂ_read);
    }

    ਜਦੋਂ (ReadFile(stderr_read, ਬਫਰ.ਅੰਕੜੇ(), (DWORD)ਬਫਰ.size(), &ਬਾਈਟਾਂ_read, ਨਿਰਾ) && ਬਾਈਟਾਂ_read > 0) {
        stderr_str.append(ਬਫਰ.ਅੰਕੜੇ(), ਬਾਈਟਾਂ_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_ਕੋਡ = 1;
    Get_Exit_ਕੋਡ_Process(pi.hProcess, &exit_ਕੋਡ);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    ਵਾਪਸ (ਪੂਰਨ)exit_ਕੋਡ;
#ਨਹੀਂਤਾਂ
    ਪੂਰਨ stdout_pipe[2];
    ਪੂਰਨ stderr_pipe[2];

    ਜੇ (ਪ੍ਰਣਾਲ(stdout_pipe) != 0 || ਪ੍ਰਣਾਲ(stderr_pipe) != 0) {
        ਸੁੱਟੋ std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    ਜੇ (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        ਸੁੱਟੋ std::runtime_error("Failed to fork process");
    }

    std::vector<ਅੱਖਰ*> argv;
    ਲਈ (std::string& part : command) {
        argv.push_back(part.ਅੰਕੜੇ());
    }
    argv.push_back(ਖਾਲੀ_ਸੂਚਕ);

    ਜੇ (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.ਅੰਕੜੇ());
        _exit(EXIT_FAILURE);
    } ਨਹੀਂਤਾਂ {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<ਅੱਖਰ, 128> ਬਫਰ;
        ssize_t ਬਾਈਟਾਂ_read;

        ਜਦੋਂ ((ਬਾਈਟਾਂ_read = ਪਠ(stdout_pipe[0], ਬਫਰ.ਅੰਕੜੇ(), ਬਫਰ.size())) > 0) {
            stdout_str.append(ਬਫਰ.ਅੰਕੜੇ(), ਬਾਈਟਾਂ_read);
        }

        ਜਦੋਂ ((ਬਾਈਟਾਂ_read = ਪਠ(stderr_pipe[0], ਬਫਰ.ਅੰਕੜੇ(), ਬਫਰ.size())) > 0) {
            stderr_str.append(ਬਫਰ.ਅੰਕੜੇ(), ਬਾਈਟਾਂ_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        ਪੂਰਨ ਸਥਿਤੀ = 0;
        waitpid(pid, &ਸਥਿਤੀ, 0);
        ਵਾਪਸ WIFEXITED(ਸਥਿਤੀ) ? WEXITSTATUS(ਸਥਿਤੀ) : -1;
    }
#ਜੇ_ਅੰਤ
}

ਤਾਰਕਿਕ directory_exists(ਅਚਲ std::string& path) {
    ਬਣਤਰ stat info;
    ਜੇ (stat(path.c_str(), &info) != 0) {
        ਵਾਪਸ ਝੂਠ; // Path doesn't exist or can't be accessed
    }
    ਵਾਪਸ (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

ਤਾਰਕਿਕ create_directory(ਅਚਲ std::string& path) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#ਜੇ_ਅੰਤ
}

std::string ਮੰਜ਼ਿਲ_uppercase(ਅਚਲ std::string& ਨਿਵੇਸ਼) {
    std::string ਨਤੀਜਾ = ਨਿਵੇਸ਼;
    ਲਈ (ਅੱਖਰ& c : ਨਤੀਜਾ) {
        c = std::ਵੱਡਾ_ਅੱਖਰ(c);
    }
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਤਾਰਕਿਕ string_starts_with(ਅਚਲ std::string& str, ਅਚਲ std::string& prefix) {
    ਜੇ (prefix.size() > str.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ std::equal(prefix.begin(), prefix.end(), str.begin());
}

ਤਾਰਕਿਕ string_ਸਿਰੇ_with(ਅਚਲ std::string& str, ਅਚਲ std::string& suffix) {
    ਜੇ (suffix.size() > str.size()) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

ਤਾਰਕਿਕ is_quantized_type(ਅਚਲ std::string& type_name) {
    ਵਾਪਸ type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

ਤਾਰਕਿਕ is_legacy_quant(ਅਚਲ std::string& type_name) {
    ਵਾਪਸ type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

ਤਾਰਕਿਕ is_k_quant(ਅਚਲ std::string& type_name) {
    ਵਾਪਸ string_ਸਿਰੇ_with(type_name, "_k");
}

ਤਾਰਕਿਕ is_iq_quant(ਅਚਲ std::string& type_name) {
    ਵਾਪਸ string_starts_with(type_name, "iq");
}

ਤਾਰਕਿਕ is_lut_quant(ਅਚਲ std::string& type_name) {
    ਵਾਪਸ is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_ਲੱਦੋ_vec_a(ਅਚਲ std::string& type_name) {
    ਜੇ (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        ਵਾਪਸ "8";
    }
    ਵਾਪਸ "4";
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ ਰਸਤਾ_separator = '/';

std::string join_paths(ਅਚਲ std::string& path1, ਅਚਲ std::string& path2) {
    ਵਾਪਸ path1 + ਰਸਤਾ_separator + path2;
}

std::string basename(ਅਚਲ std::string &path) {
    ਵਾਪਸ path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    ਵਾਪਸ ss;
}

std::string read_ਦੁਆਧਾਰੀ_file(ਅਚਲ std::string& path, ਤਾਰਕਿਕ may_not_exist = ਝੂਠ) {
    FILE* f = fopen(path.c_str(), "rb");
    ਜੇ (!f) {
        ਜੇ (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        ਵਾਪਸ {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string ਅੰਕੜੇ(size, '\0');
    size_t read_ਮਾਪ = fread(ਅੰਕੜੇ.ਅੰਕੜੇ(), 1, size, f);
    fclose(f);
    ਜੇ (read_ਮਾਪ != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        ਵਾਪਸ {};
    }

    ਵਾਪਸ ਅੰਕੜੇ;
}

ਖਾਲੀ write_ਦੁਆਧਾਰੀ_file(ਅਚਲ std::string& path, ਅਚਲ std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    ਜੇ (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        ਵਾਪਸ;
    }

    size_t write_ਮਾਪ = fwrite(content.ਅੰਕੜੇ(), 1, content.size(), f);
    fclose(f);
    ਜੇ (write_ਮਾਪ != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        ਵਾਪਸ;
    }
}

ਖਾਲੀ write_file_if_ਬਦਲਿਆ(ਅਚਲ std::string& path, ਅਚਲ std::string& content) {
    std::string existing = read_ਦੁਆਧਾਰੀ_file(path, ਸੱਚ);
    ਜੇ (existing != content) {
        write_ਦੁਆਧਾਰੀ_file(path, content);
    }
}


// variables to track number of compiles in progress
ਸਥਿਰ uint32_t compile_ਗਿਣਤੀ = 0;
ਸਥਿਰ std::mutex compile_ਗਿਣਤੀ_mutex;
ਸਥਿਰ std::condition_variable compile_ਗਿਣਤੀ_cond;
ਸਥਿਰ ਤਾਰਕਿਕ generate_dep_file = ਸੱਚ;

ਖਾਲੀ decrement_compile_ਗਿਣਤੀ(uint32_t * count) {
    ਜੇ (count) {
        std::lock_guard<std::mutex> guard(compile_ਗਿਣਤੀ_mutex);
        ਪ੍ਰਤਿਜ੍ਞਾ(compile_ਗਿਣਤੀ > 0);
        compile_ਗਿਣਤੀ--;
        compile_ਗਿਣਤੀ_cond.notify_all();
    }
}

ਵਰਤੋਂ compile_ਗਿਣਤੀ_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_ਗਿਣਤੀ)>;

compile_ਗਿਣਤੀ_guard acquire_compile_ਥਾਂ() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_ਗਿਣਤੀ_mutex);
    compile_ਗਿਣਤੀ_cond.wait(guard, [N] { ਵਾਪਸ compile_ਗਿਣਤੀ < N; });
    compile_ਗਿਣਤੀ++;
    ਵਾਪਸ compile_ਗਿਣਤੀ_guard(&compile_ਗਿਣਤੀ, &decrement_compile_ਗਿਣਤੀ);
}

ਖਾਲੀ string_ਮੰਜ਼ਿਲ_spv_func(std::string name, std::string ਵਿੱਚ_ਰਸਤਾ, std::string ਬਾਹਰਲਾ_ਰਸਤਾ, std::map<std::string, std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਤਾਰਕਿਕ coopmat, ਤਾਰਕਿਕ dep_file, compile_ਗਿਣਤੀ_guard ਥਾਂ) {
    std::string ਨਿਸ਼ਾਨਾ_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ਨਿਸ਼ਾਨਾ_env, "\"" + ਵਿੱਚ_ਰਸਤਾ + "\"", "-o", "\"" + ਬਾਹਰਲਾ_ਰਸਤਾ + "\""};
    #ਨਹੀਂਤਾਂ
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ਨਿਸ਼ਾਨਾ_env, ਵਿੱਚ_ਰਸਤਾ, "-o", ਬਾਹਰਲਾ_ਰਸਤਾ};
    #ਜੇ_ਅੰਤ

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    ਜੇ (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    ਜੇ (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
        cmd.push_back("\"" + ਨਿਸ਼ਾਨਾ_cpp + ".d\"");
#ਨਹੀਂਤਾਂ
        cmd.push_back(ਨਿਸ਼ਾਨਾ_cpp + ".d");
#ਜੇ_ਅੰਤ
    }

    #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #ਜੇ_ਅੰਤ

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& define : ਨਿਰਧਾਰਤ_ਕਰਦਾ) {
        cmd.push_back("-D" + define.ਪਹਿਲਾ + "=" + define.second);
    }

    std::string command;
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    ਕੋਸ਼ਿਸ਼ {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        ਪੂਰਨ exit_ਕੋਡ = execute_command(cmd, stdout_str, stderr_str);
        ਜੇ (exit_ਕੋਡ != 0 || !stderr_str.ਖਾਲੀ_ਥਾਵਾਂ()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_ਕੋਡ << ")\n\n";
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::ਸਤਰ_ਅੰਤ;
            compile_failed = ਸੱਚ;
            ਵਾਪਸ;
        }

        ਜੇ (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_ਦੁਆਧਾਰੀ_file(ਨਿਸ਼ਾਨਾ_cpp + ".d", ਸੱਚ);
            ਜੇ (!dep.ਖਾਲੀ_ਥਾਵਾਂ()) {
                size_t pos = dep.find(ਬਾਹਰਲਾ_ਰਸਤਾ);
                ਜੇ (pos != std::string::npos) {
                    dep.replace(pos, ਬਾਹਰਲਾ_ਰਸਤਾ.ਲੰਮਾਈ(), ਨਿਸ਼ਾਨਾ_cpp);
                }
                write_ਦੁਆਧਾਰੀ_file(ਨਿਸ਼ਾਨਾ_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(ਤਾਲਾ);
        shader_fnames.push_back(std::make_pair(name, ਬਾਹਰਲਾ_ਰਸਤਾ));
    } ਫੜੋ (ਅਚਲ std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::ਸਤਰ_ਅੰਤ;
        compile_failed = ਸੱਚ;
    }
}

std::map<std::string, std::string> ਮਿਲਾਓ_maps(ਅਚਲ std::map<std::string, std::string>& a, ਅਚਲ std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> ਨਤੀਜਾ = a;
    ਨਤੀਜਾ.insert(b.begin(), b.end());
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ std::deque<std::future<ਖਾਲੀ>> compiles;
ਖਾਲੀ string_ਮੰਜ਼ਿਲ_spv(std::string name, ਅਚਲ std::string& ਸਰੋਤ, ਅਚਲ std::map<std::string, std::string>& ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਤਾਰਕਿਕ fp16 = ਸੱਚ, ਤਾਰਕਿਕ coopmat = ਝੂਠ, ਤਾਰਕਿਕ coopmat2 = ਝੂਠ, ਤਾਰਕਿਕ f16acc = ਝੂਠ, ਅਚਲ std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string ਬਾਹਰਲਾ_ਰਸਤਾ = join_paths(output_dir, name + ".spv");

    ਜੇ (ਨਿਵੇਸ਼_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, ਬਾਹਰਲਾ_ਰਸਤਾ));
        ਵਾਪਸ;
    } ਨਹੀਂਤਾਂ ਜੇ (basename(ਨਿਵੇਸ਼_filepath) != ਸਰੋਤ) {
        // Only compile shader variants matching the input filename
        ਵਾਪਸ;
    }

    compile_ਗਿਣਤੀ_guard ਥਾਂ = acquire_compile_ਥਾਂ();
    compiles.push_back(std::async(
        string_ਮੰਜ਼ਿਲ_spv_func, name, ਨਿਵੇਸ਼_filepath, ਬਾਹਰਲਾ_ਰਸਤਾ, ਨਿਰਧਾਰਤ_ਕਰਦਾ, coopmat, generate_dep_file, std::move(ਥਾਂ)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = ਝੂਠ;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    ਜਦੋਂ (!compiles.ਖਾਲੀ_ਥਾਵਾਂ() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

ਖਾਲੀ matmul_shaders(ਤਾਰਕਿਕ fp16, MatMulIdType matmul_id_type, ਤਾਰਕਿਕ coopmat, ਤਾਰਕਿਕ coopmat2, ਤਾਰਕਿਕ f16acc, ਤਾਰਕਿਕ dot2 = ਝੂਠ) {
    std::string ਲੱਦੋ_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> ਅਧਾਰ_dict;
    std::string shader_name = "matmul";

    ਜੇ (matmul_id_type == MatMulIdType::DEFAULT) {
        ਅਧਾਰ_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } ਨਹੀਂਤਾਂ ਜੇ (matmul_id_type == MatMulIdType::SUBGROUP) {
        ਅਧਾਰ_dict["MUL_MAT_ID"] = "1";
        ਅਧਾਰ_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    ਜੇ (fp16) {
        ਅਧਾਰ_dict["FLOAT16"] = "1";
    }

    ਅਧਾਰ_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    ਅਧਾਰ_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    ਅਧਾਰ_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    ਜੇ (f16acc) {
        ਅਧਾਰ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        ਅਧਾਰ_dict["ACC_F16"] = "1";
    }

    ਜੇ (coopmat) {
        ਅਧਾਰ_dict["COOPMAT"] = "1";
    }
#ਜੇ defined(GGML_VULKAN_COOPMAT2_DECODE_ਸਦਿਸ਼_GLSLC_SUPPORT)
    ਜੇ (coopmat2) {
        ਅਧਾਰ_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#ਜੇ_ਅੰਤ

    ਜੇ (dot2) {
        ਅਧਾਰ_dict["DOT2_F16"] = "1";
    }

    ਅਚਲ std::string ਸਰੋਤ_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    ਸਵੈਚਾਲਿਤ ਅਚਲ &FLOAT_TYPE = [&](ਪੂਰਨ vec, ਅਚਲ std::string &t) -> std::string {
        ਚੋਣ (vec) {
        ਹਾਲਤ 1:
            ਜੇ (t == "bf16") {
                // scalar path promotes to float
                ਜੇ (!coopmat && !coopmat2) {
                    ਵਾਪਸ "float";
                }
                ਵਾਪਸ "bfloat16_t";
            }
            ਜੇ (coopmat2 || fp16) {
                ਵਾਪਸ "float16_t";
            }
            ਵਾਪਸ "float";
        ਹਾਲਤ 2:
            ਜੇ (t == "bf16") {
                // scalar path promotes to float
                ਜੇ (!coopmat && !coopmat2) {
                    ਵਾਪਸ "vec2";
                }
                ਵਾਪਸ "bf16vec2";
            }
            ਜੇ (coopmat2 || fp16) {
                ਵਾਪਸ "f16vec2";
            }
            ਵਾਪਸ "vec2";
        ਹਾਲਤ 4:
            ਜੇ (t == "bf16") {
                // scalar path promotes to float
                ਜੇ (!coopmat && !coopmat2) {
                    ਵਾਪਸ "vec4";
                }
                ਵਾਪਸ "bf16vec4";
            }
            ਜੇ (coopmat2 || fp16) {
                ਵਾਪਸ "f16vec4";
            }
            ਵਾਪਸ "vec4";
        ਹਾਲਤ 8:
            ਜੇ (t == "bf16") {
                // scalar path promotes to float
                ਜੇ (!coopmat && !coopmat2) {
                    ਵਾਪਸ "mat2x4";
                }
                ਸੁੱਟੋ std::runtime_error("bf16 vec8 not supported");
            }
            ਜੇ (coopmat2 || fp16) {
                ਵਾਪਸ "f16mat2x4";
            }
            ਵਾਪਸ "mat2x4";
        ਮੂਲ:
            ਸੁੱਟੋ std::runtime_error("invalid vector size");
        }
    };

    ਅਚਲ std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_ਮੰਜ਼ਿਲ_spv(shader_name + "_f32_f16" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", ਲੱਦੋ_vec}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_ਮੰਜ਼ਿਲ_spv(shader_name + "_f16" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", ਲੱਦੋ_vec}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string ਲੱਦੋ_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string ਮੰਜ਼ਿਲ_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        ਅਚਲ std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#ਜੇ !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ਜੇ (!(coopmat || coopmat2))
#ਜੇ_ਅੰਤ
        {
            ਜੇ (!dot2) {
                string_ਮੰਜ਼ਿਲ_spv(shader_name + "_bf16", ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", ਮੰਜ਼ਿਲ_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", ਲੱਦੋ_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& tname : type_ਨਾਂ) {
        ਜੇ (tname == "bf16") {
            ਜਾਰੀ;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        ਜੇ (tname == "f32" || tname == "f16") {
            std::string ਅੰਕੜੇ_a_ਕੁੰਜੀ = "DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(tname);

            ਅਚਲ std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            ਜੇ (!coopmat2) {
                string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"LOAD_VEC_A", ਲੱਦੋ_vec}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            ਜਾਰੀ;
        }

        std::string ਅੰਕੜੇ_a_ਕੁੰਜੀ = "DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(tname);
        ਅਚਲ std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ਜੇ (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#ਜੇ_ਅੰਤ

        ਜੇ (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        ਜੇ (is_lut_quant(tname)) {
            std::string lva = lut_ਲੱਦੋ_vec_a(tname);

            string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            ਜੇ (!coopmat2) {
                string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            ਜੇ ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                ਜੇ (!coopmat2) {
                    string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#ਜੇ_ਅੰਤ
            ਜਾਰੀ;
        }

        // dedicated shader needed due to regression on Ampere
        ਜੇ (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_ਮੰਜ਼ਿਲ_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, float_type_dict), {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"LOAD_VEC_A", ਲੱਦੋ_vec}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        ਅਚਲ std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_ਮੰਜ਼ਿਲ_spv(shader_name + "_quant_f16" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        ਜੇ (!coopmat2) {
            string_ਮੰਜ਼ਿਲ_spv(shader_name + "_quant_f32" + dot2_sfx, ਸਰੋਤ_name, ਮਿਲਾਓ_maps(ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ਲੱਦੋ_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

ਖਾਲੀ process_shaders() {
    // matmul
    ਲਈ (ਅਚਲ MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(ਝੂਠ, matmul_id_type, ਝੂਠ, ਝੂਠ, ਝੂਠ);

        // fp16, fp32acc and fp16acc
        matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਝੂਠ, ਝੂਠ);
        matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਝੂਠ, ਸੱਚ);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਝੂਠ, ਝੂਠ, ਸੱਚ);
        matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਝੂਠ, ਸੱਚ,  ਸੱਚ);

        ਜੇ (matmul_id_type != MatMulIdType::DEFAULT) {
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(ਸੱਚ, matmul_id_type, ਸੱਚ, ਝੂਠ, ਝੂਠ);
            matmul_shaders(ਸੱਚ, matmul_id_type, ਸੱਚ, ਝੂਠ, ਸੱਚ);
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਸੱਚ, ਝੂਠ);
            matmul_shaders(ਸੱਚ, matmul_id_type, ਝੂਠ, ਸੱਚ, ਸੱਚ);
#ਜੇ_ਅੰਤ
        }
    }

    ਲਈ (ਅਚਲ ਤਾਰਕਿਕ& fp16 : {ਝੂਠ, ਸੱਚ}) {
        std::map<std::string, std::string> ਅਧਾਰ_dict;
        ਜੇ (fp16) {
            ਅਧਾਰ_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } ਨਹੀਂਤਾਂ {
            ਅਧਾਰ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        ਲਈ (ਅਚਲ ਤਾਰਕਿਕ& f16acc : {ਝੂਠ, ਸੱਚ}) {
            std::map<std::string, std::string> fa_ਅਧਾਰ_dict = ਅਧਾਰ_dict;
            fa_ਅਧਾਰ_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_ਅਧਾਰ_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_ਅਧਾਰ_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_ਅਧਾਰ_dict["DATA_A_IQ4_NL"] = "1";
            ਜੇ (fp16 && f16acc) {
                fa_ਅਧਾਰ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            ਜੇ (fp16) {
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    ਮਿਲਾਓ_maps(fa_ਅਧਾਰ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ਝੂਠ, ਸੱਚ, f16acc);
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    ਮਿਲਾਓ_maps(fa_ਅਧਾਰ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, ਸੱਚ, ਝੂਠ, f16acc);
#ਜੇ_ਅੰਤ
            }

            string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16", "flash_attn.comp",
                ਮਿਲਾਓ_maps(fa_ਅਧਾਰ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ਝੂਠ, ਝੂਠ, f16acc);

            ਜੇ (fp16) {
                string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    ਮਿਲਾਓ_maps(fa_ਅਧਾਰ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, ਝੂਠ, ਝੂਠ, f16acc);
            }

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16", "flash_attn.comp",
                ਮਿਲਾਓ_maps(fa_ਅਧਾਰ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, ਝੂਠ, ਝੂਠ, f16acc, "_int8");
#ਜੇ_ਅੰਤ
        }
    }

    ਅਚਲ std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        ਮਿਲਾਓ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        ਸੱਚ, ਸੱਚ, ਝੂਠ, ਝੂਠ);
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_ਮੰਜ਼ਿਲ_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        ਮਿਲਾਓ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        ਸੱਚ, ਝੂਠ, ਸੱਚ, ਝੂਠ);
#ਜੇ_ਅੰਤ

    std::map<std::string, std::string> ਅਧਾਰ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& tname : type_ਨਾਂ) {
        // mul mat vec
        std::string ਅੰਕੜੇ_a_ਕੁੰਜੀ = "DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(tname);
        std::string shader = (string_ਸਿਰੇ_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ਜੇ (tname == "mxfp4" || tname == "nvfp4") {
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#ਜੇ_ਅੰਤ

        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ਜੇ (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"MUL_MAT_ID", "1"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#ਜੇ_ਅੰਤ

        // Dequant shaders
        ਜੇ (tname != "f16" && tname != "bf16") {
            string_ਮੰਜ਼ਿਲ_spv("dequant_" + tname, "dequant_" + tname + ".comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        ਜੇ (tname == "q8_0") {
            string_ਮੰਜ਼ਿਲ_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        ਜੇ (tname == "f16") {
            string_ਮੰਜ਼ਿਲ_spv("get_rows_" + tname, shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } ਨਹੀਂਤਾਂ {
            string_ਮੰਜ਼ਿਲ_spv("get_rows_" + tname, shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_ਮੰਜ਼ਿਲ_spv("get_rows_" + tname + "_f32", shader, ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ਅੰਕੜੇ_a_ਕੁੰਜੀ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_ਮੰਜ਼ਿਲ_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_ਮੰਜ਼ਿਲ_spv("norm_f32", "norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("group_norm_f32", "group_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_f32", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_mul_add_f32", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_partials_f32", "rms_norm_partials.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("rms_norm_back_f32", "rms_norm_back.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("l2_norm_f32", "l2_norm.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    ਲਈ (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_ਮੰਜ਼ਿਲ_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_ਮੰਜ਼ਿਲ_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        ਲਈ (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_ਮੰਜ਼ਿਲ_spv("set_rows_" + std::string(src.ਪਹਿਲਾ) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_ਮੰਜ਼ਿਲ_spv("set_rows_" + std::string(src.ਪਹਿਲਾ) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + ਮੰਜ਼ਿਲ_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    ਸਵੈਚਾਲਿਤ get_type_str = [](ਤਾਰਕਿਕ f16) {
        ਵਾਪਸ f16 ? "float16_t" : "float";
    };
    ਸਵੈਚਾਲਿਤ get_suffix = [](ਤਾਰਕਿਕ src0_f16, ਤਾਰਕਿਕ src1_f16, ਤਾਰਕਿਕ dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        ਵਾਪਸ s;
    };
    ਲਈ (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    ਲਈ (ਸਵੈਚਾਲਿਤ src0_f16 : {ਝੂਠ, ਸੱਚ}) {
    ਲਈ (ਸਵੈਚਾਲਿਤ src1_f16 : {ਝੂਠ, ਸੱਚ}) {
    ਲਈ (ਸਵੈਚਾਲਿਤ dst_f16  : {ਝੂਠ, ਸੱਚ}) {
        ਸਵੈਚਾਲਿਤ ਸਰੋਤ = op == "add_rms" ? std::string("add") : op;
        ਸਵੈਚਾਲਿਤ name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        ਸਵੈਚਾਲਿਤ ਜੋੜੋ_rms = op == "add_rms" ? "1" : "0";
        string_ਮੰਜ਼ਿਲ_spv(name.c_str(), ਸਰੋਤ + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , ਜੋੜੋ_rms}});
    }
    }
    }
    }

    string_ਮੰਜ਼ਿਲ_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_ਮੰਜ਼ਿਲ_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_ਮੰਜ਼ਿਲ_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, ਸੱਚ, ਸੱਚ, ਝੂਠ, ਝੂਠ);
    string_ਮੰਜ਼ਿਲ_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, ਸੱਚ, ਸੱਚ, ਝੂਠ, ਝੂਠ);
#ਜੇ_ਅੰਤ

    string_ਮੰਜ਼ਿਲ_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_ਮੰਜ਼ਿਲ_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_ਮੰਜ਼ਿਲ_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_ਮੰਜ਼ਿਲ_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_ਮੰਜ਼ਿਲ_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_ਮੰਜ਼ਿਲ_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ਮੰਜ਼ਿਲ_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_ਮੰਜ਼ਿਲ_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_ਮੰਜ਼ਿਲ_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_ਮੰਜ਼ਿਲ_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_ਮੰਜ਼ਿਲ_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_ਮੰਜ਼ਿਲ_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_ਮੰਜ਼ਿਲ_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_ਮੰਜ਼ਿਲ_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_ਮੰਜ਼ਿਲ_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_ਮੰਜ਼ਿਲ_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_ਮੰਜ਼ਿਲ_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_ਮੰਜ਼ਿਲ_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_ਮੰਜ਼ਿਲ_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_ਮੰਜ਼ਿਲ_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_ਮੰਜ਼ਿਲ_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_ਮੰਜ਼ਿਲ_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_ਮੰਜ਼ਿਲ_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_ਮੰਜ਼ਿਲ_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_ਮੰਜ਼ਿਲ_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_ਮੰਜ਼ਿਲ_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_ਮੰਜ਼ਿਲ_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_ਮੰਜ਼ਿਲ_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_ਮੰਜ਼ਿਲ_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_ਮੰਜ਼ਿਲ_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_ਮੰਜ਼ਿਲ_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_ਮੰਜ਼ਿਲ_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_ਮੰਜ਼ਿਲ_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_ਮੰਜ਼ਿਲ_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_ਮੰਜ਼ਿਲ_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_ਮੰਜ਼ਿਲ_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_ਮੰਜ਼ਿਲ_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_ਮੰਜ਼ਿਲ_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_ਮੰਜ਼ਿਲ_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_ਮੰਜ਼ਿਲ_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_ਮੰਜ਼ਿਲ_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_ਮੰਜ਼ਿਲ_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_ਮੰਜ਼ਿਲ_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_ਮੰਜ਼ਿਲ_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_ਮੰਜ਼ਿਲ_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_ਮੰਜ਼ਿਲ_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_ਮੰਜ਼ਿਲ_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_ਮੰਜ਼ਿਲ_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_ਮੰਜ਼ਿਲ_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_ਮੰਜ਼ਿਲ_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_ਮੰਜ਼ਿਲ_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_ਮੰਜ਼ਿਲ_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_ਮੰਜ਼ਿਲ_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_ਮੰਜ਼ਿਲ_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_ਮੰਜ਼ਿਲ_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_ਮੰਜ਼ਿਲ_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_ਮੰਜ਼ਿਲ_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_ਮੰਜ਼ਿਲ_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_ਮੰਜ਼ਿਲ_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_ਮੰਜ਼ਿਲ_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("soft_max_f32", "soft_max.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_f32_f16", "soft_max.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_back_f32", "soft_max_back.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("soft_max_large1_f32", "soft_max_large1.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_large2_f32", "soft_max_large2.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_large3_f32", "soft_max_large3.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("argmax_f32", "argmax.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_ਮੰਜ਼ਿਲ_spv("sum_rows_f32", "sum_rows.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("fwht_f32", "fwht.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_ਮੰਜ਼ਿਲ_spv("count_equal_i32", "count_equal.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_ਮੰਜ਼ਿਲ_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_ਮੰਜ਼ਿਲ_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_ਮੰਜ਼ਿਲ_spv("cumsum_f32", "cumsum.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("count_experts", "count_experts.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_ਮੰਜ਼ਿਲ_spv("count_experts_subgroup", "count_experts.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    ਲਈ (std::string dim_str : {"", "_3d"}) {
        ਲਈ (ਤਾਰਕਿਕ bda : {ਝੂਠ, ਸੱਚ}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_ਮੰਜ਼ਿਲ_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_ਮੰਜ਼ਿਲ_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_ਮੰਜ਼ਿਲ_spv("out_prod_f32", "out_prod.comp", {});

    string_ਮੰਜ਼ਿਲ_spv("timestep_embedding_f32", "timestep_embedding.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ਮੰਜ਼ਿਲ_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_ਮੰਜ਼ਿਲ_spv("pool1d_f32", "pool1d.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("pool2d_f32", "pool2d.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("pool2d_subgroup_f32", "pool2d.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_ਮੰਜ਼ਿਲ_spv("rwkv_wkv6_f32", "wkv6.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("gated_linear_attn_f32", "gla.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_ਮੰਜ਼ਿਲ_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_ਮੰਜ਼ਿਲ_spv("rwkv_wkv7_f32", "wkv7.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("gated_delta_net_f32", "gated_delta_net.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_ਮੰਜ਼ਿਲ_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_ਮੰਜ਼ਿਲ_spv("opt_step_adamw_f32", "opt_step_adamw.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}}));
    string_ਮੰਜ਼ਿਲ_spv("opt_step_sgd_f32", "opt_step_sgd.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("solve_tri_f32", "solve_tri.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    ਲਈ (ਸਵੈਚਾਲਿਤ transpose : {ਝੂਠ, ਸੱਚ}) {
        ਲਈ (ਸਵੈਚਾਲਿਤ unroll : {ਝੂਠ, ਸੱਚ}) {
            ਲਈ (ਸਵੈਚਾਲਿਤ a_f16 : {ਝੂਠ, ਸੱਚ}) {
                std::map<std::string, std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                ਜੇ (transpose) ਨਿਰਧਾਰਤ_ਕਰਦਾ["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_ਮੰਜ਼ਿਲ_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", ਨਿਰਧਾਰਤ_ਕਰਦਾ);
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                ਜੇ (unroll) {
                    ਸਵੈਚਾਲਿਤ cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ = ਨਿਰਧਾਰਤ_ਕਰਦਾ;
                    cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ["COOPMAT2"] = "1";
                    string_ਮੰਜ਼ਿਲ_spv(name, "conv2d_mm.comp", cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਸੱਚ, ਝੂਠ, ਸੱਚ);
                }
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                ਜੇ (unroll) {
                    ਸਵੈਚਾਲਿਤ cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ = ਨਿਰਧਾਰਤ_ਕਰਦਾ;
                    cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ["COOPMAT"] = "1";
                    string_ਮੰਜ਼ਿਲ_spv(name, "conv2d_mm.comp", cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਸੱਚ, ਸੱਚ, ਝੂਠ);
                }
#ਜੇ_ਅੰਤ
            }
        }
    }

    ਲਈ (ਸਵੈਚਾਲਿਤ unroll : {ਝੂਠ, ਸੱਚ}) {
        ਲਈ (ਸਵੈਚਾਲਿਤ a_f16 : {ਝੂਠ, ਸੱਚ}) {
            std::map<std::string, std::string> ਨਿਰਧਾਰਤ_ਕਰਦਾ = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_ਮੰਜ਼ਿਲ_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", ਨਿਰਧਾਰਤ_ਕਰਦਾ);
#ਜੇ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ਜੇ (unroll) {
                ਸਵੈਚਾਲਿਤ cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ = ਨਿਰਧਾਰਤ_ਕਰਦਾ;
                cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ["COOPMAT2"] = "1";
                string_ਮੰਜ਼ਿਲ_spv(name, "conv3d_mm.comp", cm2_ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਸੱਚ, ਝੂਠ, ਸੱਚ);
            }
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            ਜੇ (unroll) {
                ਸਵੈਚਾਲਿਤ cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ = ਨਿਰਧਾਰਤ_ਕਰਦਾ;
                cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ["COOPMAT"] = "1";
                string_ਮੰਜ਼ਿਲ_spv(name, "conv3d_mm.comp", cm1_ਨਿਰਧਾਰਤ_ਕਰਦਾ, ਸੱਚ, ਸੱਚ, ਝੂਠ);
            }
#ਜੇ_ਅੰਤ
        }
    }

    string_ਮੰਜ਼ਿਲ_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_ਮੰਜ਼ਿਲ_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_ਮੰਜ਼ਿਲ_spv("roll_f32", "roll.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("add_id_f32", "add_id.comp", ਮਿਲਾਓ_maps(ਅਧਾਰ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ਮੰਜ਼ਿਲ_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_ਮੰਜ਼ਿਲ_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_ਮੰਜ਼ਿਲ_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_ਮੰਜ਼ਿਲ_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_ਮੰਜ਼ਿਲ_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_ਮੰਜ਼ਿਲ_spv("topk_moe_f32", "topk_moe.comp", {});

    ਲਈ (ਸਵੈਚਾਲਿਤ &c : compiles) {
        c.wait();
    }
}

ਖਾਲੀ write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(ਨਿਸ਼ਾਨਾ_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& pair : shader_fnames) {
        ਅਚਲ std::string& name = pair.ਪਹਿਲਾ;
        #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #ਨਹੀਂਤਾਂ
            ਅਚਲ std::string& path = pair.second;
        #ਜੇ_ਅੰਤ

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        ਜੇ (ਨਿਵੇਸ਼_filepath != "") {
            std::string ਅੰਕੜੇ = read_ਦੁਆਧਾਰੀ_file(path);
            ਜੇ (ਅੰਕੜੇ.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਜਾਰੀ;
            }

            src << "const uint64_t " << name << "_len = " << ਅੰਕੜੇ.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << ਅੰਕੜੇ.size() << "] = {\n" << std::hex;
            ਸਵੈਚਾਲਿਤ ਬਾਈਟਾਂ = reinterpret_cast<ਅਚਲ uint8_t*>(ਅੰਕੜੇ.ਅੰਕੜੇ());
            ਲਈ (size_t i = 0; i < ਅੰਕੜੇ.size(); ++i) {
                src << "0x" << static_cast<ਪੂਰਨ>(ਬਾਈਟਾਂ[i]) << ",";
                ਜੇ ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    ਲਈ (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        ਜੇ (basename(ਨਿਵੇਸ਼_filepath) != op_file) {
            ਜਾਰੀ;
        }
        std::stringstream ਅੰਕੜੇ = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        ਅੰਕੜੇ << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        ਲਈ (uint32_t t0 = 0; t0 < 2; ++t0) {
            ਜੇ (t0 == 0) {
                ਅੰਕੜੇ << "{";
                len  << "{";
            }
            ਲਈ (uint32_t t1 = 0; t1 < 2; ++t1) {
                ਜੇ (t1 == 0) {
                    ਅੰਕੜੇ << "{";
                    len  << "{";
                }
                ਲਈ (uint32_t t2 = 0; t2 < 2; ++t2) {
                    ਜੇ (t2 == 0) {
                        ਅੰਕੜੇ << "{";
                        len  << "{";
                    }
                    ਅੰਕੜੇ << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    ਅੰਕੜੇ << "_data,";
                    len  << "_len,";
                    ਜੇ (t2 == 1) {
                        ਅੰਕੜੇ << "}, ";
                        len  << "}, ";
                    }
                }
                ਜੇ (t1 == 1) {
                    ਅੰਕੜੇ << "}, ";
                    len  << "}, ";
                }
            }
            ਜੇ (t0 == 1) {
                ਅੰਕੜੇ << "};\n";
                len  << "};\n";
            }
        }
        src << ਅੰਕੜੇ.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#ਜੇ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#ਜੇ_ਅੰਤ

    ਲਈ (ਅਚਲ std::string& btype : btypes) {
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& tname : type_ਨਾਂ) {
        ਜੇ (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            ਜਾਰੀ;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        ਜੇ (basename(ਨਿਵੇਸ਼_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        ਜੇ (btype == "f16") {
            ਜਾਰੀ;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        ਜੇ (basename(ਨਿਵੇਸ਼_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#ਜੇ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ਲਈ (ਅਚਲ std::string& btype : {"f16", "f32"}) {
    ਲਈ (ਅਚਲ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        ਜੇ (basename(ਨਿਵੇਸ਼_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    ਲਈ (ਅਚਲ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        ਜੇ (basename(ਨਿਵੇਸ਼_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#ਜੇ_ਅੰਤ

    ਜੇ (ਨਿਵੇਸ਼_filepath == "") {
        write_file_if_ਬਦਲਿਆ(ਨਿਸ਼ਾਨਾ_hpp, hdr.str());
    }
    ਜੇ (ਨਿਸ਼ਾਨਾ_cpp != "") {
        write_ਦੁਆਧਾਰੀ_file(ਨਿਸ਼ਾਨਾ_cpp, src.str());
    }
}

} // namespace

ਪੂਰਨ ਮੁੱਖ(ਪੂਰਨ argc, ਅੱਖਰ** argv) {
    std::map<std::string, std::string> ਦਲੀਲਾਂ_2;
    ਲਈ (ਪੂਰਨ i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        ਜੇ (arg.rfind("--", 0) == 0) {
            ਜੇ (i + 1 < argc && argv[i + 1][0] != '-') {
                ਦਲੀਲਾਂ_2[arg] = argv[i + 1];
                ++i;
            } ਨਹੀਂਤਾਂ {
                ਦਲੀਲਾਂ_2[arg] = "";
            }
        }
    }

    ਜੇ (ਦਲੀਲਾਂ_2.find("--glslc") != ਦਲੀਲਾਂ_2.end()) {
        GLSLC = ਦਲੀਲਾਂ_2["--glslc"]; // Path to glslc
    }
    ਜੇ (ਦਲੀਲਾਂ_2.find("--source") != ਦਲੀਲਾਂ_2.end()) {
        ਨਿਵੇਸ਼_filepath = ਦਲੀਲਾਂ_2["--source"]; // The shader source file to compile
    }
    ਜੇ (ਦਲੀਲਾਂ_2.find("--output-dir") != ਦਲੀਲਾਂ_2.end()) {
        output_dir = ਦਲੀਲਾਂ_2["--output-dir"]; // Directory for containing SPIR-V output
    }
    ਜੇ (ਦਲੀਲਾਂ_2.find("--target-hpp") != ਦਲੀਲਾਂ_2.end()) {
        ਨਿਸ਼ਾਨਾ_hpp = ਦਲੀਲਾਂ_2["--target-hpp"]; // Path to generated header file
    }
    ਜੇ (ਦਲੀਲਾਂ_2.find("--target-cpp") != ਦਲੀਲਾਂ_2.end()) {
        ਨਿਸ਼ਾਨਾ_cpp = ਦਲੀਲਾਂ_2["--target-cpp"]; // Path to generated cpp file
    }

    ਜੇ (!directory_exists(output_dir)) {
        ਜੇ (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            ਵਾਪਸ EXIT_FAILURE;
        }
    }

    process_shaders();

    ਜੇ (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::ਸਤਰ_ਅੰਤ;
        ਵਾਪਸ EXIT_FAILURE;
    }

    write_output_files();

    ਵਾਪਸ EXIT_SUCCESS;
}
