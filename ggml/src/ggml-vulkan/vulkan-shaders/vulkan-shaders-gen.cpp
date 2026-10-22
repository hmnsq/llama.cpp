#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <map>
#ಅನ್ತರ್ಭಾವ <thread>
#ಅನ್ತರ್ಭಾವ <mutex>
#ಅನ್ತರ್ಭಾವ <future>
#ಅನ್ತರ್ಭಾವ <queue>
#ಅನ್ತರ್ಭಾವ <condition_variable>
#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <sys/stat.h>
#ಅನ್ತರ್ಭಾವ <sys/types.h>
#ಅನ್ತರ್ಭಾವ <filesystem>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    #ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
    #ಅನ್ತರ್ಭಾವ <windows.h>
    #ಅನ್ತರ್ಭಾವ <direct.h> // For _mkdir on Windows
#ಇಲ್ಲದಿದ್ದರೆ
    #ಅನ್ತರ್ಭಾವ <unistd.h>
    #ಅನ್ತರ್ಭಾವ <sys/wait.h>
    #ಅನ್ತರ್ಭಾವ <fcntl.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು ASYNCIO_CONCURRENCY 64

std::mutex ಬೀಗ;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ> compile_failed{ಸುಳ್ಳು};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string ಒಳಹರಿವು_filepath = "";
std::string output_dir = "/tmp";
std::string ಗುರಿ_hpp = "";
std::string ಗುರಿ_cpp = "";

ಸ್ಥಿರಾಂಕ std::vector<std::string> type_ಹೆಸರುಗಳು = {
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

ಎಣಿಕೆ MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

ನಾಮಾಕಾಶ {

ಪೂರ್ಣಾಂಕ execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ಗುಣಲಕ್ಷಣಗಳು sa = { ಗಾತ್ರ(SECURITY_ಗುಣಲಕ್ಷಣಗಳು), ಖಾಲಿ, TRUE };

    ಆದರೆ (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !ಗಣ_Handle_Information(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        ಎಸೆ std::runtime_error("Failed to create stdout pipe");
    }

    ಆದರೆ (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !ಗಣ_Handle_Information(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        ಎಸೆ std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = ಗಾತ್ರ(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& part : command) {
        cmd += part + " ";
    }

    ಆದರೆ (!CreateProcessA(ಖಾಲಿ, cmd.ದತ್ತಾಂಶ(), ಖಾಲಿ, ಖಾಲಿ, TRUE, 0, ಖಾಲಿ, ಖಾಲಿ, &si, &pi)) {
        ಎಸೆ std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<ಅಕ್ಷರ, 128> ಬಫರ್;
    DWORD ಬೈಟುಗಳು_read;

    ಆಗ (ReadFile(stdout_read, ಬಫರ್.ದತ್ತಾಂಶ(), (DWORD)ಬಫರ್.size(), &ಬೈಟುಗಳು_read, ಖಾಲಿ) && ಬೈಟುಗಳು_read > 0) {
        stdout_str.append(ಬಫರ್.ದತ್ತಾಂಶ(), ಬೈಟುಗಳು_read);
    }

    ಆಗ (ReadFile(stderr_read, ಬಫರ್.ದತ್ತಾಂಶ(), (DWORD)ಬಫರ್.size(), &ಬೈಟುಗಳು_read, ಖಾಲಿ) && ಬೈಟುಗಳು_read > 0) {
        stderr_str.append(ಬಫರ್.ದತ್ತಾಂಶ(), ಬೈಟುಗಳು_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_ಸಂಕೇತ = 1;
    Get_Exit_ಸಂಕೇತ_Process(pi.hProcess, &exit_ಸಂಕೇತ);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ)exit_ಸಂಕೇತ;
#ಇಲ್ಲದಿದ್ದರೆ
    ಪೂರ್ಣಾಂಕ stdout_pipe[2];
    ಪೂರ್ಣಾಂಕ stderr_pipe[2];

    ಆದರೆ (ಪ್ರಣಾಲ(stdout_pipe) != 0 || ಪ್ರಣಾಲ(stderr_pipe) != 0) {
        ಎಸೆ std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    ಆದರೆ (pid < 0) {
        std::cerr << strerror(errno) << "\n";
        ಎಸೆ std::runtime_error("Failed to fork process");
    }

    std::vector<ಅಕ್ಷರ*> argv;
    ಗಾಗಿ (std::string& part : command) {
        argv.push_back(part.ದತ್ತಾಂಶ());
    }
    argv.push_back(ಶೂನ್ಯಸೂಚಕ);

    ಆದರೆ (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.ದತ್ತಾಂಶ());
        _exit(EXIT_FAILURE);
    } ಇಲ್ಲದಿದ್ದರೆ {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<ಅಕ್ಷರ, 128> ಬಫರ್;
        ssize_t ಬೈಟುಗಳು_read;

        ಆಗ ((ಬೈಟುಗಳು_read = ಪಠ(stdout_pipe[0], ಬಫರ್.ದತ್ತಾಂಶ(), ಬಫರ್.size())) > 0) {
            stdout_str.append(ಬಫರ್.ದತ್ತಾಂಶ(), ಬೈಟುಗಳು_read);
        }

        ಆಗ ((ಬೈಟುಗಳು_read = ಪಠ(stderr_pipe[0], ಬಫರ್.ದತ್ತಾಂಶ(), ಬಫರ್.size())) > 0) {
            stderr_str.append(ಬಫರ್.ದತ್ತಾಂಶ(), ಬೈಟುಗಳು_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = 0;
        waitpid(pid, &ಸ್ಥಿತಿ_2, 0);
        ಹಿಂತಿರುಗಿ WIFEXITED(ಸ್ಥಿತಿ_2) ? WEXITSTATUS(ಸ್ಥಿತಿ_2) : -1;
    }
#ಆದರೆ_ಅಂತ್ಯ
}

ತಾರ್ಕಿಕ directory_exists(ಸ್ಥಿರಾಂಕ std::string& path) {
    ರಚನೆ stat info;
    ಆದರೆ (stat(path.c_str(), &info) != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; // Path doesn't exist or can't be accessed
    }
    ಹಿಂತಿರುಗಿ (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

ತಾರ್ಕಿಕ create_directory(ಸ್ಥಿರಾಂಕ std::string& path) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#ಆದರೆ_ಅಂತ್ಯ
}

std::string ಗಮ್ಯ_uppercase(ಸ್ಥಿರಾಂಕ std::string& ಒಳಹರಿವು) {
    std::string ಫಲಿತಾಂಶ = ಒಳಹರಿವು;
    ಗಾಗಿ (ಅಕ್ಷರ& c : ಫಲಿತಾಂಶ) {
        c = std::ದೊಡ್ಡ_ಅಕ್ಷರ(c);
    }
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ತಾರ್ಕಿಕ string_starts_with(ಸ್ಥಿರಾಂಕ std::string& str, ಸ್ಥಿರಾಂಕ std::string& prefix) {
    ಆದರೆ (prefix.size() > str.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ std::equal(prefix.begin(), prefix.end(), str.begin());
}

ತಾರ್ಕಿಕ string_ತುದಿಗಳು_with(ಸ್ಥಿರಾಂಕ std::string& str, ಸ್ಥಿರಾಂಕ std::string& suffix) {
    ಆದರೆ (suffix.size() > str.size()) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

ತಾರ್ಕಿಕ is_quantized_type(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಹಿಂತಿರುಗಿ type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

ತಾರ್ಕಿಕ is_legacy_quant(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಹಿಂತಿರುಗಿ type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

ತಾರ್ಕಿಕ is_k_quant(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಹಿಂತಿರುಗಿ string_ತುದಿಗಳು_with(type_name, "_k");
}

ತಾರ್ಕಿಕ is_iq_quant(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಹಿಂತಿರುಗಿ string_starts_with(type_name, "iq");
}

ತಾರ್ಕಿಕ is_lut_quant(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಹಿಂತಿರುಗಿ is_iq_quant(type_name) || type_name == "mxfp4" || type_name == "nvfp4";
}

std::string lut_ತುಂಬು_vec_a(ಸ್ಥಿರಾಂಕ std::string& type_name) {
    ಆದರೆ (type_name == "iq1_s" || type_name == "iq1_m" || type_name == "iq2_xxs" || type_name == "iq2_xs" || type_name == "iq2_s" || type_name == "iq4_xs") {
        ಹಿಂತಿರುಗಿ "8";
    }
    ಹಿಂತಿರುಗಿ "4";
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ ಮಾರ್ಗ_separator = '/';

std::string join_paths(ಸ್ಥಿರಾಂಕ std::string& path1, ಸ್ಥಿರಾಂಕ std::string& path2) {
    ಹಿಂತಿರುಗಿ path1 + ಮಾರ್ಗ_separator + path2;
}

std::string basename(ಸ್ಥಿರಾಂಕ std::string &path) {
    ಹಿಂತಿರುಗಿ path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    ಹಿಂತಿರುಗಿ ss;
}

std::string read_ದ್ವಿಮಾನ_file(ಸ್ಥಿರಾಂಕ std::string& path, ತಾರ್ಕಿಕ may_not_exist = ಸುಳ್ಳು) {
    FILE* f = fopen(path.c_str(), "rb");
    ಆದರೆ (!f) {
        ಆದರೆ (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        ಹಿಂತಿರುಗಿ {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string ದತ್ತಾಂಶ(size, '\0');
    size_t read_ಗಾತ್ರ = fread(ದತ್ತಾಂಶ.ದತ್ತಾಂಶ(), 1, size, f);
    fclose(f);
    ಆದರೆ (read_ಗಾತ್ರ != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        ಹಿಂತಿರುಗಿ {};
    }

    ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ;
}

ಶೂನ್ಯ write_ದ್ವಿಮಾನ_file(ಸ್ಥಿರಾಂಕ std::string& path, ಸ್ಥಿರಾಂಕ std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    ಆದರೆ (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        ಹಿಂತಿರುಗಿ;
    }

    size_t write_ಗಾತ್ರ = fwrite(content.ದತ್ತಾಂಶ(), 1, content.size(), f);
    fclose(f);
    ಆದರೆ (write_ಗಾತ್ರ != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        ಹಿಂತಿರುಗಿ;
    }
}

ಶೂನ್ಯ write_file_if_ಬದಲಾಯಿತು(ಸ್ಥಿರಾಂಕ std::string& path, ಸ್ಥಿರಾಂಕ std::string& content) {
    std::string existing = read_ದ್ವಿಮಾನ_file(path, ನಿಜ);
    ಆದರೆ (existing != content) {
        write_ದ್ವಿಮಾನ_file(path, content);
    }
}


// variables to track number of compiles in progress
ಸ್ಥಿರ uint32_t compile_ಎಣಿಕೆ = 0;
ಸ್ಥಿರ std::mutex compile_ಎಣಿಕೆ_mutex;
ಸ್ಥಿರ std::condition_variable compile_ಎಣಿಕೆ_cond;
ಸ್ಥಿರ ತಾರ್ಕಿಕ generate_dep_file = ನಿಜ;

ಶೂನ್ಯ decrement_compile_ಎಣಿಕೆ(uint32_t * count) {
    ಆದರೆ (count) {
        std::lock_guard<std::mutex> guard(compile_ಎಣಿಕೆ_mutex);
        ಪ್ರತಿಜ್ಞಾ(compile_ಎಣಿಕೆ > 0);
        compile_ಎಣಿಕೆ--;
        compile_ಎಣಿಕೆ_cond.notify_all();
    }
}

ಬಳಸಿ compile_ಎಣಿಕೆ_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_ಎಣಿಕೆ)>;

compile_ಎಣಿಕೆ_guard acquire_compile_ಸ್ಥಾನ() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_ಎಣಿಕೆ_mutex);
    compile_ಎಣಿಕೆ_cond.wait(guard, [N] { ಹಿಂತಿರುಗಿ compile_ಎಣಿಕೆ < N; });
    compile_ಎಣಿಕೆ++;
    ಹಿಂತಿರುಗಿ compile_ಎಣಿಕೆ_guard(&compile_ಎಣಿಕೆ, &decrement_compile_ಎಣಿಕೆ);
}

ಶೂನ್ಯ string_ಗಮ್ಯ_spv_func(std::string name, std::string ಒಳ_ಮಾರ್ಗ, std::string ಹೊರಹೋಗು_ಮಾರ್ಗ, std::map<std::string, std::string> ನಿರೂಪಿಸುತ್ತದೆ, ತಾರ್ಕಿಕ coopmat, ತಾರ್ಕಿಕ dep_file, compile_ಎಣಿಕೆ_guard ಸ್ಥಾನ) {
    std::string ಗುರಿ_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ಗುರಿ_env, "\"" + ಒಳ_ಮಾರ್ಗ + "\"", "-o", "\"" + ಹೊರಹೋಗು_ಮಾರ್ಗ + "\""};
    #ಇಲ್ಲದಿದ್ದರೆ
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", ಗುರಿ_env, ಒಳ_ಮಾರ್ಗ, "-o", ಹೊರಹೋಗು_ಮಾರ್ಗ};
    #ಆದರೆ_ಅಂತ್ಯ

    // disable spirv-opt for coopmat shaders for https://github.com/ggml-org/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
    ಆದರೆ (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos && name.find("_dot2") == std::string::npos) {
        cmd.push_back("-O");
    }

    ಆದರೆ (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
        cmd.push_back("\"" + ಗುರಿ_cpp + ".d\"");
#ಇಲ್ಲದಿದ್ದರೆ
        cmd.push_back(ಗುರಿ_cpp + ".d");
#ಆದರೆ_ಅಂತ್ಯ
    }

    #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& define : ನಿರೂಪಿಸುತ್ತದೆ) {
        cmd.push_back("-D" + define.ಮೊದಲ + "=" + define.second);
    }

    std::string command;
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    ಪ್ರಯತ್ನಿಸು {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        ಪೂರ್ಣಾಂಕ exit_ಸಂಕೇತ = execute_command(cmd, stdout_str, stderr_str);
        ಆದರೆ (exit_ಸಂಕೇತ != 0 || !stderr_str.ಖಾಲಿಗಳು()) {
            std::cerr << "cannot compile " << name << " (exit code " << exit_ಸಂಕೇತ << ")\n\n";
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::ಸಾಲು_ಅಂತ್ಯ;
            compile_failed = ನಿಜ;
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_ದ್ವಿಮಾನ_file(ಗುರಿ_cpp + ".d", ನಿಜ);
            ಆದರೆ (!dep.ಖಾಲಿಗಳು()) {
                size_t pos = dep.find(ಹೊರಹೋಗು_ಮಾರ್ಗ);
                ಆದರೆ (pos != std::string::npos) {
                    dep.replace(pos, ಹೊರಹೋಗು_ಮಾರ್ಗ.ಉದ್ದಳತೆ(), ಗುರಿ_cpp);
                }
                write_ದ್ವಿಮಾನ_file(ಗುರಿ_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(ಬೀಗ);
        shader_fnames.push_back(std::make_pair(name, ಹೊರಹೋಗು_ಮಾರ್ಗ));
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::ಸಾಲು_ಅಂತ್ಯ;
        compile_failed = ನಿಜ;
    }
}

std::map<std::string, std::string> ವಿಲೀನ_maps(ಸ್ಥಿರಾಂಕ std::map<std::string, std::string>& a, ಸ್ಥಿರಾಂಕ std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> ಫಲಿತಾಂಶ = a;
    ಫಲಿತಾಂಶ.insert(b.begin(), b.end());
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ std::deque<std::future<ಶೂನ್ಯ>> compiles;
ಶೂನ್ಯ string_ಗಮ್ಯ_spv(std::string name, ಸ್ಥಿರಾಂಕ std::string& ಮೂಲ, ಸ್ಥಿರಾಂಕ std::map<std::string, std::string>& ನಿರೂಪಿಸುತ್ತದೆ, ತಾರ್ಕಿಕ fp16 = ನಿಜ, ತಾರ್ಕಿಕ coopmat = ಸುಳ್ಳು, ತಾರ್ಕಿಕ coopmat2 = ಸುಳ್ಳು, ತಾರ್ಕಿಕ f16acc = ಸುಳ್ಳು, ಸ್ಥಿರಾಂಕ std::string& suffix = "") {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32")) + suffix;
    std::string ಹೊರಹೋಗು_ಮಾರ್ಗ = join_paths(output_dir, name + ".spv");

    ಆದರೆ (ಒಳಹರಿವು_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, ಹೊರಹೋಗು_ಮಾರ್ಗ));
        ಹಿಂತಿರುಗಿ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (basename(ಒಳಹರಿವು_filepath) != ಮೂಲ) {
        // Only compile shader variants matching the input filename
        ಹಿಂತಿರುಗಿ;
    }

    compile_ಎಣಿಕೆ_guard ಸ್ಥಾನ = acquire_compile_ಸ್ಥಾನ();
    compiles.push_back(std::async(
        string_ಗಮ್ಯ_spv_func, name, ಒಳಹರಿವು_filepath, ಹೊರಹೋಗು_ಮಾರ್ಗ, ನಿರೂಪಿಸುತ್ತದೆ, coopmat, generate_dep_file, std::move(ಸ್ಥಾನ)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = ಸುಳ್ಳು;

    // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
    ಆಗ (!compiles.ಖಾಲಿಗಳು() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        compiles.pop_front();
    }
}

ಶೂನ್ಯ matmul_shaders(ತಾರ್ಕಿಕ fp16, MatMulIdType matmul_id_type, ತಾರ್ಕಿಕ coopmat, ತಾರ್ಕಿಕ coopmat2, ತಾರ್ಕಿಕ f16acc, ತಾರ್ಕಿಕ dot2 = ಸುಳ್ಳು) {
    std::string ತುಂಬು_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";
    std::string dot2_sfx = dot2 ? "_dot2" : "";

    std::map<std::string, std::string> ತಳ_dict;
    std::string shader_name = "matmul";

    ಆದರೆ (matmul_id_type == MatMulIdType::DEFAULT) {
        ತಳ_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (matmul_id_type == MatMulIdType::SUBGROUP) {
        ತಳ_dict["MUL_MAT_ID"] = "1";
        ತಳ_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    ಆದರೆ (fp16) {
        ತಳ_dict["FLOAT16"] = "1";
    }

    ತಳ_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    ತಳ_dict["ACC_TYPEV2"   ] = f16acc ? "f16vec2"   : "vec2";
    ತಳ_dict["ACC_TYPE_VEC4"] = f16acc ? "f16vec4"   : "vec4";
    ಆದರೆ (f16acc) {
        ತಳ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        ತಳ_dict["ACC_F16"] = "1";
    }

    ಆದರೆ (coopmat) {
        ತಳ_dict["COOPMAT"] = "1";
    }
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_DECODE_ಸದಿಶ_GLSLC_SUPPORT)
    ಆದರೆ (coopmat2) {
        ತಳ_dict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (dot2) {
        ತಳ_dict["DOT2_F16"] = "1";
    }

    ಸ್ಥಿರಾಂಕ std::string ಮೂಲ_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಿರಾಂಕ &FLOAT_TYPE = [&](ಪೂರ್ಣಾಂಕ vec, ಸ್ಥಿರಾಂಕ std::string &t) -> std::string {
        ಆಯ್ಕೆ (vec) {
        ಸ್ಥಿತಿ 1:
            ಆದರೆ (t == "bf16") {
                // scalar path promotes to float
                ಆದರೆ (!coopmat && !coopmat2) {
                    ಹಿಂತಿರುಗಿ "float";
                }
                ಹಿಂತಿರುಗಿ "bfloat16_t";
            }
            ಆದರೆ (coopmat2 || fp16) {
                ಹಿಂತಿರುಗಿ "float16_t";
            }
            ಹಿಂತಿರುಗಿ "float";
        ಸ್ಥಿತಿ 2:
            ಆದರೆ (t == "bf16") {
                // scalar path promotes to float
                ಆದರೆ (!coopmat && !coopmat2) {
                    ಹಿಂತಿರುಗಿ "vec2";
                }
                ಹಿಂತಿರುಗಿ "bf16vec2";
            }
            ಆದರೆ (coopmat2 || fp16) {
                ಹಿಂತಿರುಗಿ "f16vec2";
            }
            ಹಿಂತಿರುಗಿ "vec2";
        ಸ್ಥಿತಿ 4:
            ಆದರೆ (t == "bf16") {
                // scalar path promotes to float
                ಆದರೆ (!coopmat && !coopmat2) {
                    ಹಿಂತಿರುಗಿ "vec4";
                }
                ಹಿಂತಿರುಗಿ "bf16vec4";
            }
            ಆದರೆ (coopmat2 || fp16) {
                ಹಿಂತಿರುಗಿ "f16vec4";
            }
            ಹಿಂತಿರುಗಿ "vec4";
        ಸ್ಥಿತಿ 8:
            ಆದರೆ (t == "bf16") {
                // scalar path promotes to float
                ಆದರೆ (!coopmat && !coopmat2) {
                    ಹಿಂತಿರುಗಿ "mat2x4";
                }
                ಎಸೆ std::runtime_error("bf16 vec8 not supported");
            }
            ಆದರೆ (coopmat2 || fp16) {
                ಹಿಂತಿರುಗಿ "f16mat2x4";
            }
            ಹಿಂತಿರುಗಿ "mat2x4";
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("invalid vector size");
        }
    };

    ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",   FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPEV2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPEV4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPEV8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_ಗಮ್ಯ_spv(shader_name + "_f32_f16" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", ತುಂಬು_vec}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    string_ಗಮ್ಯ_spv(shader_name + "_f16" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", ತುಂಬು_vec}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string ತುಂಬು_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string ಗಮ್ಯ_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#ಆದರೆ !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        ಆದರೆ (!(coopmat || coopmat2))
#ಆದರೆ_ಅಂತ್ಯ
        {
            ಆದರೆ (!dot2) {
                string_ಗಮ್ಯ_spv(shader_name + "_bf16", ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", ಗಮ್ಯ_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", ತುಂಬು_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"}, {"B_TYPE_SCALAR", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"B_TYPEV4", "bf16vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}), fp16, coopmat, coopmat2, f16acc);
            }
        }
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& tname : type_ಹೆಸರುಗಳು) {
        ಆದರೆ (tname == "bf16") {
            ಮುಂದುವರಿ;
        }

        // Float types keep per-type compilation (different accumulation loop structure)
        ಆದರೆ (tname == "f32" || tname == "f16") {
            std::string ದತ್ತಾಂಶ_a_ಕೀಲಿ = "DATA_A_" + ಗಮ್ಯ_uppercase(tname);

            ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> float_type_dict = {
                {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
                {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
                {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
                {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
            };

            ಆದರೆ (!coopmat2) {
                string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"LOAD_VEC_A", ತುಂಬು_vec}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }
            ಮುಂದುವರಿ;
        }

        std::string ದತ್ತಾಂಶ_a_ಕೀಲಿ = "DATA_A_" + ಗಮ್ಯ_uppercase(tname);
        ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, tname)},
        };

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ಆದರೆ (!f16acc && !coopmat && !coopmat2 && !dot2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4" || tname == "iq3_s" || tname == "iq4_xs")) {
            string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#ಆದರೆ_ಅಂತ್ಯ

        ಆದರೆ (!f16acc && coopmat && (tname == "q4_0" || tname == "q4_1" || tname == "q5_0" || tname == "q5_1" || tname == "q8_0" || tname == "iq4_nl" || tname == "iq4_xs" || tname == "mxfp4"
                     || tname == "q3_k" || tname == "q4_k" || tname == "q5_k" || tname == "q6_k" || tname == "nvfp4")) {
            string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq_cm1.comp", ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"D_TYPE_VEC4", "vec4"}}), fp16, coopmat, coopmat2, f16acc);
        }

        ಆದರೆ (is_lut_quant(tname)) {
            std::string lva = lut_ತುಂಬು_vec_a(tname);

            string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

            ಆದರೆ (!coopmat2) {
                string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f32" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            }

#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
            ಆದರೆ ((tname == "mxfp4" || tname == "nvfp4") && (coopmat || coopmat2)) {
                string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f16_ocp" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                ಆದರೆ (!coopmat2) {
                    string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f32_ocp" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"LOAD_VEC_A", lva}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
                }
            }
#ಆದರೆ_ಅಂತ್ಯ
            ಮುಂದುವರಿ;
        }

        // dedicated shader needed due to regression on Ampere
        ಆದರೆ (coopmat2 && (tname == "q4_k" || tname == "q5_k")) {
            string_ಗಮ್ಯ_spv(shader_name + "_" + tname + "_f16" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, float_type_dict), {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"LOAD_VEC_A", ತುಂಬು_vec}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    // Quant shader: one SPIR-V for all quant types, selected via MmTypeA spec constant
    {
        ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> quant_float_type_dict = {
            {"FLOAT_TYPE",   FLOAT_TYPE(1, "q4_0")},
            {"FLOAT_TYPEV2", FLOAT_TYPE(2, "q4_0")},
            {"FLOAT_TYPEV4", FLOAT_TYPE(4, "q4_0")},
            {"FLOAT_TYPEV8", FLOAT_TYPE(8, "q4_0")},
        };

        string_ಗಮ್ಯ_spv(shader_name + "_quant_f16" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE_SCALAR", "float16_t"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

        ಆದರೆ (!coopmat2) {
            string_ಗಮ್ಯ_spv(shader_name + "_quant_f32" + dot2_sfx, ಮೂಲ_name, ವಿಲೀನ_maps(ವಿಲೀನ_maps(ತಳ_dict, quant_float_type_dict), {{"MULMAT_QUANT", "1"}, {"LOAD_VEC_B", ತುಂಬು_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE_SCALAR", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }
}

ಶೂನ್ಯ process_shaders() {
    // matmul
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(ಸುಳ್ಳು, matmul_id_type, ಸುಳ್ಳು, ಸುಳ್ಳು, ಸುಳ್ಳು);

        // fp16, fp32acc and fp16acc
        matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ಸುಳ್ಳು, ಸುಳ್ಳು);
        matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ಸುಳ್ಳು, ನಿಜ);

        // dot2 variants (scalar fp16 only)
        matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ಸುಳ್ಳು, ಸುಳ್ಳು, ನಿಜ);
        matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ಸುಳ್ಳು, ನಿಜ,  ನಿಜ);

        ಆದರೆ (matmul_id_type != MatMulIdType::DEFAULT) {
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(ನಿಜ, matmul_id_type, ನಿಜ, ಸುಳ್ಳು, ಸುಳ್ಳು);
            matmul_shaders(ನಿಜ, matmul_id_type, ನಿಜ, ಸುಳ್ಳು, ನಿಜ);
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ನಿಜ, ಸುಳ್ಳು);
            matmul_shaders(ನಿಜ, matmul_id_type, ಸುಳ್ಳು, ನಿಜ, ನಿಜ);
#ಆದರೆ_ಅಂತ್ಯ
        }
    }

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ& fp16 : {ಸುಳ್ಳು, ನಿಜ}) {
        std::map<std::string, std::string> ತಳ_dict;
        ಆದರೆ (fp16) {
            ತಳ_dict = {{"FLOAT_TYPE", "float16_t"}, {"FLOAT_TYPEV2", "f16vec2"}, {"FLOAT_TYPEV4", "f16vec4"}, {"FLOAT16", "1"}, {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
        } ಇಲ್ಲದಿದ್ದರೆ {
            ತಳ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
        }

        // flash attention
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ& f16acc : {ಸುಳ್ಳು, ನಿಜ}) {
            std::map<std::string, std::string> fa_ತಳ_dict = ತಳ_dict;
            fa_ತಳ_dict["ACC_TYPE"] = fp16 && f16acc ? "float16_t" : "float";
            fa_ತಳ_dict["ACC_TYPEV2"] = fp16 && f16acc ? "f16vec2" : "vec2";
            fa_ತಳ_dict["ACC_TYPEV4"] = fp16 && f16acc ? "f16vec4" : "vec4";
            // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
            fa_ತಳ_dict["DATA_A_IQ4_NL"] = "1";
            ಆದರೆ (fp16 && f16acc) {
                fa_ತಳ_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
            }

            ಆದರೆ (fp16) {
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                string_ಗಮ್ಯ_spv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                    ವಿಲೀನ_maps(fa_ತಳ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ಸುಳ್ಳು, ನಿಜ, f16acc);
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                string_ಗಮ್ಯ_spv("flash_attn_f32_f16", "flash_attn_cm1.comp",
                    ವಿಲೀನ_maps(fa_ತಳ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), fp16, ನಿಜ, ಸುಳ್ಳು, f16acc);
#ಆದರೆ_ಅಂತ್ಯ
            }

            string_ಗಮ್ಯ_spv("flash_attn_f32_f16", "flash_attn.comp",
                ವಿಲೀನ_maps(fa_ತಳ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), fp16, ಸುಳ್ಳು, ಸುಳ್ಳು, f16acc);

            ಆದರೆ (fp16) {
                string_ಗಮ್ಯ_spv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                    ವಿಲೀನ_maps(fa_ತಳ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}), fp16, ಸುಳ್ಳು, ಸುಳ್ಳು, f16acc);
            }

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            string_ಗಮ್ಯ_spv("flash_attn_f32_f16", "flash_attn.comp",
                ವಿಲೀನ_maps(fa_ತಳ_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}), fp16, ಸುಳ್ಳು, ಸುಳ್ಳು, f16acc, "_int8");
#ಆದರೆ_ಅಂತ್ಯ
        }
    }

    ಸ್ಥಿರಾಂಕ std::map<std::string, std::string> fa_bf16_dict = {
        {"FLOAT_TYPE",   "bfloat16_t"},
        {"FLOAT_TYPEV2", "bf16vec2"},
        {"FLOAT_TYPEV4", "bf16vec4"},
        {"ACC_TYPE",     "float"},
        {"ACC_TYPEV2",   "vec2"},
        {"ACC_TYPEV4",   "vec4"},
        {"BFLOAT16",     "1"},
    };

#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_ಗಮ್ಯ_spv("flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
        ವಿಲೀನ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
        ನಿಜ, ನಿಜ, ಸುಳ್ಳು, ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_ಗಮ್ಯ_spv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
        ವಿಲೀನ_maps(fa_bf16_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}),
        ನಿಜ, ಸುಳ್ಳು, ನಿಜ, ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

    std::map<std::string, std::string> ತಳ_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& tname : type_ಹೆಸರುಗಳು) {
        // mul mat vec
        std::string ದತ್ತಾಂಶ_a_ಕೀಲಿ = "DATA_A_" + ಗಮ್ಯ_uppercase(tname);
        std::string shader = (string_ತುದಿಗಳು_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_") || tname == "iq4_xs" || tname == "tq2_0" || tname == "tq1_0") ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
        ಆದರೆ (tname == "mxfp4" || tname == "nvfp4") {
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_f16_f32_ocp_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float16_t"}, {"B_TYPEV2", "f16vec2"}, {"B_TYPEV4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32_ocp_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"USE_OCP_FP4", "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#ಆದರೆ_ಅಂತ್ಯ

        string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
        string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        ಆದರೆ (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m" || tname == "iq4_xs") {
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_ಗಮ್ಯ_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"MUL_MAT_ID", "1"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#ಆದರೆ_ಅಂತ್ಯ

        // Dequant shaders
        ಆದರೆ (tname != "f16" && tname != "bf16") {
            string_ಗಮ್ಯ_spv("dequant_" + tname, "dequant_" + tname + ".comp", ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float16_t"}}));
        }
        // Fused dequant+transpose variant for FA quant-KV (per-head-contiguous f16 scratch).
        ಆದರೆ (tname == "q8_0") {
            string_ಗಮ್ಯ_spv("dequant_" + tname + "_transpose", "dequant_" + tname + ".comp", ವಿಲೀನ_maps(ತಳ_dict, {{ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"D_TYPE", "float16_t"}, {"DEQUANT_TRANSPOSE", "1"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        ಆದರೆ (tname == "f16") {
            string_ಗಮ್ಯ_spv("get_rows_" + tname, shader, ವಿಲೀನ_maps(ತಳ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } ಇಲ್ಲದಿದ್ದರೆ {
            string_ಗಮ್ಯ_spv("get_rows_" + tname, shader, ವಿಲೀನ_maps(ತಳ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_ಗಮ್ಯ_spv("get_rows_" + tname + "_f32", shader, ವಿಲೀನ_maps(ತಳ_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {ದತ್ತಾಂಶ_a_ಕೀಲಿ, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_ಗಮ್ಯ_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_ಗಮ್ಯ_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_ಗಮ್ಯ_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPEV4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_ಗಮ್ಯ_spv("norm_f32", "norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("group_norm_f32", "group_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("rms_norm_f32", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("rms_norm_mul_add_f32", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_mul_add_partials_f32", "rms_norm_partials.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_ADD_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_set_rows_f32_f32", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_set_rows_f32_f16", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RMS_NORM_SET_ROWS_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_partials_f32", "rms_norm_partials.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_mul_rope_f32_f16", "rms_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_ಗಮ್ಯ_spv("rms_norm_back_f32", "rms_norm_back.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("l2_norm_f32", "l2_norm.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("cpy_32", "copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ಗಮ್ಯ_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("cpy_16", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ಗಮ್ಯ_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_ಗಮ್ಯ_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_ಗಮ್ಯ_spv("cpy_bf16_f32","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_ಗಮ್ಯ_spv("cpy_bf16_f16","copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_ಗಮ್ಯ_spv("contig_cpy_32", "contig_copy.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ಗಮ್ಯ_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_ಗಮ್ಯ_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("contig_cpy_16", "contig_copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ಗಮ್ಯ_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_ಗಮ್ಯ_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_ಗಮ್ಯ_spv("contig_cpy_bf16_f32","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
    string_ಗಮ್ಯ_spv("contig_cpy_bf16_f16","contig_copy.comp",{{"A_TYPE", "uint16_t"}, {"D_TYPE", "float16_t"}, {"DATA_A_BF16", "1"}});
    string_ಗಮ್ಯ_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_ಗಮ್ಯ_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ಗಮ್ಯ_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ಗಮ್ಯ_spv("cpy_transpose_02_16", "copy_transpose_02.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ಗಮ್ಯ_spv("cpy_transpose_02_32", "copy_transpose_02.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    ಗಾಗಿ (std::string t : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_ಗಮ್ಯ_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + ಗಮ್ಯ_uppercase(t), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_ಗಮ್ಯ_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + ಗಮ್ಯ_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ src : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
        ಗಾಗಿ (std::string dst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
            string_ಗಮ್ಯ_spv("set_rows_" + std::string(src.ಮೊದಲ) + "_" + dst + "_i32", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + ಗಮ್ಯ_uppercase(dst), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
            string_ಗಮ್ಯ_spv("set_rows_" + std::string(src.ಮೊದಲ) + "_" + dst + "_i64", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + ಗಮ್ಯ_uppercase(dst), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"S_TYPE", src.second}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        }
    }

    ಸ್ವಯಂಚಾಲಿತ get_type_str = [](ತಾರ್ಕಿಕ f16) {
        ಹಿಂತಿರುಗಿ f16 ? "float16_t" : "float";
    };
    ಸ್ವಯಂಚಾಲಿತ get_suffix = [](ತಾರ್ಕಿಕ src0_f16, ತಾರ್ಕಿಕ src1_f16, ತಾರ್ಕಿಕ dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        ಹಿಂತಿರುಗಿ s;
    };
    ಗಾಗಿ (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ src0_f16 : {ಸುಳ್ಳು, ನಿಜ}) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ src1_f16 : {ಸುಳ್ಳು, ನಿಜ}) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ dst_f16  : {ಸುಳ್ಳು, ನಿಜ}) {
        ಸ್ವಯಂಚಾಲಿತ ಮೂಲ = op == "add_rms" ? std::string("add") : op;
        ಸ್ವಯಂಚಾಲಿತ name = op + get_suffix(src0_f16, src1_f16, dst_f16);
        ಸ್ವಯಂಚಾಲಿತ ಕೂಡಿಸು_rms = op == "add_rms" ? "1" : "0";
        string_ಗಮ್ಯ_spv(name.c_str(), ಮೂಲ + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , ಕೂಡಿಸು_rms}});
    }
    }
    }
    }

    string_ಗಮ್ಯ_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_ಗಮ್ಯ_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_ಗಮ್ಯ_spv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    string_ಗಮ್ಯ_spv("fa_decode_ph1", "flash_attn_decode_phase_1.comp", {}, ನಿಜ, ನಿಜ, ಸುಳ್ಳು, ಸುಳ್ಳು);
    string_ಗಮ್ಯ_spv("fa_decode_ph2", "flash_attn_decode_phase_2.comp", {}, ನಿಜ, ನಿಜ, ಸುಳ್ಳು, ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

    string_ಗಮ್ಯ_spv("fa_sparse_compact", "flash_attn_sparse_compact.comp", {});
    string_ಗಮ್ಯ_spv("fa_sparse_compact_subgroup", "flash_attn_sparse_compact.comp", {{"USE_SUBGROUPS", "1"}});

    string_ಗಮ್ಯ_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_ಗಮ್ಯ_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_ಗಮ್ಯ_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_ಗಮ್ಯ_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_ಗಮ್ಯ_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
    string_ಗಮ್ಯ_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("get_rows_back_f32", "get_rows_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

    string_ಗಮ್ಯ_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("pad_reflect_1d_f32", "pad_reflect_1d.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
    string_ಗಮ್ಯ_spv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_ಗಮ್ಯ_spv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
    string_ಗಮ್ಯ_spv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

    string_ಗಮ್ಯ_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("exp_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
    string_ಗಮ್ಯ_spv("exp_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_exp"}});
    string_ಗಮ್ಯ_spv("expm1_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
    string_ಗಮ್ಯ_spv("expm1_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_expm1"}});

    string_ಗಮ್ಯ_spv("log_f16",        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("log_f32",        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("gelu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
    string_ಗಮ್ಯ_spv("gelu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu"}});
    string_ಗಮ್ಯ_spv("gelu_erf_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
    string_ಗಮ್ಯ_spv("gelu_erf_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_erf"}});
    string_ಗಮ್ಯ_spv("gelu_quick_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
    string_ಗಮ್ಯ_spv("gelu_quick_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_gelu_quick"}});
    string_ಗಮ್ಯ_spv("silu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
    string_ಗಮ್ಯ_spv("silu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_silu"}});
    string_ಗಮ್ಯ_spv("relu_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
    string_ಗಮ್ಯ_spv("relu_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_relu"}});
    string_ಗಮ್ಯ_spv("sqr_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
    string_ಗಮ್ಯ_spv("sqr_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqr"}});
    string_ಗಮ್ಯ_spv("sqrt_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
    string_ಗಮ್ಯ_spv("sqrt_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sqrt"}});
    string_ಗಮ್ಯ_spv("sin_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
    string_ಗಮ್ಯ_spv("sin_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sin"}});
    string_ಗಮ್ಯ_spv("cos_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
    string_ಗಮ್ಯ_spv("cos_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_cos"}});
    string_ಗಮ್ಯ_spv("clamp_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
    string_ಗಮ್ಯ_spv("clamp_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_clamp"}});
    string_ಗಮ್ಯ_spv("leaky_relu_f16", "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
    string_ಗಮ್ಯ_spv("leaky_relu_f32", "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_leaky_relu"}});
    string_ಗಮ್ಯ_spv("neg_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
    string_ಗಮ್ಯ_spv("neg_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_neg"}});
    string_ಗಮ್ಯ_spv("tanh_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
    string_ಗಮ್ಯ_spv("tanh_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_tanh"}});
    string_ಗಮ್ಯ_spv("sigmoid_f16",    "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
    string_ಗಮ್ಯ_spv("sigmoid_f32",    "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sigmoid"}});
    string_ಗಮ್ಯ_spv("hardsigmoid_f16","unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
    string_ಗಮ್ಯ_spv("hardsigmoid_f32","unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardsigmoid"}});
    string_ಗಮ್ಯ_spv("hardswish_f16",  "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
    string_ಗಮ್ಯ_spv("hardswish_f32",  "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_hardswish"}});
    string_ಗಮ್ಯ_spv("abs_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
    string_ಗಮ್ಯ_spv("abs_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_abs"}});
    string_ಗಮ್ಯ_spv("elu_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
    string_ಗಮ್ಯ_spv("elu_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_elu"}});
    string_ಗಮ್ಯ_spv("xielu_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
    string_ಗಮ್ಯ_spv("xielu_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_xielu"}});
    string_ಗಮ್ಯ_spv("sgn_f16",        "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
    string_ಗಮ್ಯ_spv("sgn_f32",        "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_sgn"}});

    string_ಗಮ್ಯ_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("softplus_f16",   "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
    string_ಗಮ್ಯ_spv("softplus_f32",   "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_softplus"}});

    string_ಗಮ್ಯ_spv("gelu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("gelu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_gelu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("sigmoid_mul_f32", "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("sigmoid_mul_f16", "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_sigmoid"},  {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("silu_mul_f32",    "unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("silu_mul_f16",    "unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_silu"},     {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("softplus_mul_f32","unary.comp",      {{"A_TYPE", "float"},       {"B_TYPE", "float"},     {"D_TYPE", "float"},     {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});
    string_ಗಮ್ಯ_spv("softplus_mul_f16","unary.comp",      {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}, {"OP", "op_softplus"}, {"UNARY_MUL_FUSION", "1"}});

    string_ಗಮ್ಯ_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("fill_f16",       "fill.comp",        {{"D_TYPE", "float16_t"},   {"FLOAT_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("step_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
    string_ಗಮ್ಯ_spv("step_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_step"}});
    string_ಗಮ್ಯ_spv("round_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
    string_ಗಮ್ಯ_spv("round_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_round"}});
    string_ಗಮ್ಯ_spv("ceil_f16",       "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
    string_ಗಮ್ಯ_spv("ceil_f32",       "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_ceil"}});
    string_ಗಮ್ಯ_spv("floor_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
    string_ಗಮ್ಯ_spv("floor_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_floor"}});
    string_ಗಮ್ಯ_spv("trunc_f16",      "unary.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
    string_ಗಮ್ಯ_spv("trunc_f32",      "unary.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},     {"OP", "op_trunc"}});

    string_ಗಮ್ಯ_spv("geglu_f16",      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("geglu_f32",      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("reglu_f16",      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("reglu_f32",      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("swiglu_f16",     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("swiglu_f32",     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("swiglu_oai_f16", "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("swiglu_oai_f32", "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("geglu_erf_f16",  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("geglu_erf_f32",  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("geglu_quick_f16","geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("geglu_quick_f32","geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("soft_max_f32", "soft_max.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_f32_f16", "soft_max.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_back_f32", "soft_max_back.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("soft_max_large1_f32", "soft_max_large1.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_large2_f32", "soft_max_large2.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_large3_f32", "soft_max_large3.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ಗಮ್ಯ_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ಗಮ್ಯ_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ಗಮ್ಯ_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

    string_ಗಮ್ಯ_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("topk_radix_select_f32", "topk_radix_select.comp", {{"A_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("argmax_f32", "argmax.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_ಗಮ್ಯ_spv("sum_rows_f32", "sum_rows.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("cross_entropy_loss_f32", "cross_entropy_loss.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("cross_entropy_loss_back_f32", "cross_entropy_loss_back.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("fwht_f32", "fwht.comp", {});
    string_ಗಮ್ಯ_spv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
    string_ಗಮ್ಯ_spv("count_equal_i32", "count_equal.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_ಗಮ್ಯ_spv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
    string_ಗಮ್ಯ_spv("dsv4_hc_pre_f32",  "dsv4_hc_pre.comp",  {});
    string_ಗಮ್ಯ_spv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
    string_ಗಮ್ಯ_spv("cumsum_f32", "cumsum.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("count_experts", "count_experts.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));
    string_ಗಮ್ಯ_spv("count_experts_subgroup", "count_experts.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}, {"USE_SUBGROUPS", "1"}}));

    ಗಾಗಿ (std::string dim_str : {"", "_3d"}) {
        ಗಾಗಿ (ತಾರ್ಕಿಕ bda : {ಸುಳ್ಳು, ನಿಜ}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_ಗಮ್ಯ_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_ಗಮ್ಯ_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
        }
    }

    string_ಗಮ್ಯ_spv("out_prod_f32", "out_prod.comp", {});

    string_ಗಮ್ಯ_spv("timestep_embedding_f32", "timestep_embedding.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("col2im_1d_f32",  "col2im_1d.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("col2im_1d_f16",  "col2im_1d.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("col2im_1d_bf16", "col2im_1d.comp", {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_ಗಮ್ಯ_spv("snake_f32",  "snake.comp", {{"DATA_A_F32", "1"},  {"A_TYPE", "float"},     {"D_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("snake_f16",  "snake.comp", {{"DATA_A_F16", "1"},  {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_ಗಮ್ಯ_spv("snake_bf16", "snake.comp", {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"},  {"D_TYPE", "uint16_t"}});

    string_ಗಮ್ಯ_spv("pool1d_f32", "pool1d.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("pool2d_f32", "pool2d.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("pool2d_subgroup_f32", "pool2d.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP", "1"}}));

    string_ಗಮ್ಯ_spv("rwkv_wkv6_f32", "wkv6.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("gated_linear_attn_f32", "gla.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}}));

    // Compile IQ4_NL support in so its shared LUT is available when K uses it.
    // K quant type is selected at runtime via the FaTypeK spec constant.
    std::map<std::string, std::string> li_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
    string_ಗಮ್ಯ_spv("lightning_indexer_f32", "lightning_indexer.comp", li_dict);

    string_ಗಮ್ಯ_spv("rwkv_wkv7_f32", "wkv7.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("gated_delta_net_f32", "gated_delta_net.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
    string_ಗಮ್ಯ_spv("gated_delta_net_f32_nocluster", "gated_delta_net.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
    string_ಗಮ್ಯ_spv("gated_delta_net_f32_shmem", "gated_delta_net.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

    string_ಗಮ್ಯ_spv("opt_step_adamw_f32", "opt_step_adamw.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}}));
    string_ಗಮ್ಯ_spv("opt_step_sgd_f32", "opt_step_sgd.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("solve_tri_f32", "solve_tri.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ transpose : {ಸುಳ್ಳು, ನಿಜ}) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ unroll : {ಸುಳ್ಳು, ನಿಜ}) {
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ a_f16 : {ಸುಳ್ಳು, ನಿಜ}) {
                std::map<std::string, std::string> ನಿರೂಪಿಸುತ್ತದೆ = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                ಆದರೆ (transpose) ನಿರೂಪಿಸುತ್ತದೆ["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_ಗಮ್ಯ_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", ನಿರೂಪಿಸುತ್ತದೆ);
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                ಆದರೆ (unroll) {
                    ಸ್ವಯಂಚಾಲಿತ cm2_ನಿರೂಪಿಸುತ್ತದೆ = ನಿರೂಪಿಸುತ್ತದೆ;
                    cm2_ನಿರೂಪಿಸುತ್ತದೆ["COOPMAT2"] = "1";
                    string_ಗಮ್ಯ_spv(name, "conv2d_mm.comp", cm2_ನಿರೂಪಿಸುತ್ತದೆ, ನಿಜ, ಸುಳ್ಳು, ನಿಜ);
                }
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
                ಆದರೆ (unroll) {
                    ಸ್ವಯಂಚಾಲಿತ cm1_ನಿರೂಪಿಸುತ್ತದೆ = ನಿರೂಪಿಸುತ್ತದೆ;
                    cm1_ನಿರೂಪಿಸುತ್ತದೆ["COOPMAT"] = "1";
                    string_ಗಮ್ಯ_spv(name, "conv2d_mm.comp", cm1_ನಿರೂಪಿಸುತ್ತದೆ, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
                }
#ಆದರೆ_ಅಂತ್ಯ
            }
        }
    }

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ unroll : {ಸುಳ್ಳು, ನಿಜ}) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ a_f16 : {ಸುಳ್ಳು, ನಿಜ}) {
            std::map<std::string, std::string> ನಿರೂಪಿಸುತ್ತದೆ = {
                {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                {"UNROLL", unroll ? "[[unroll]]" : ""},
            };
            std::string name = std::string("conv3d") + (a_f16 ? "_f16" : "") + "_f32";
            string_ಗಮ್ಯ_spv(name + (unroll ? "_unroll" : ""), "conv3d_mm.comp", ನಿರೂಪಿಸುತ್ತದೆ);
#ಆದರೆ defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            ಆದರೆ (unroll) {
                ಸ್ವಯಂಚಾಲಿತ cm2_ನಿರೂಪಿಸುತ್ತದೆ = ನಿರೂಪಿಸುತ್ತದೆ;
                cm2_ನಿರೂಪಿಸುತ್ತದೆ["COOPMAT2"] = "1";
                string_ಗಮ್ಯ_spv(name, "conv3d_mm.comp", cm2_ನಿರೂಪಿಸುತ್ತದೆ, ನಿಜ, ಸುಳ್ಳು, ನಿಜ);
            }
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            ಆದರೆ (unroll) {
                ಸ್ವಯಂಚಾಲಿತ cm1_ನಿರೂಪಿಸುತ್ತದೆ = ನಿರೂಪಿಸುತ್ತದೆ;
                cm1_ನಿರೂಪಿಸುತ್ತದೆ["COOPMAT"] = "1";
                string_ಗಮ್ಯ_spv(name, "conv3d_mm.comp", cm1_ನಿರೂಪಿಸುತ್ತದೆ, ನಿಜ, ನಿಜ, ಸುಳ್ಳು);
            }
#ಆದರೆ_ಅಂತ್ಯ
        }
    }

    string_ಗಮ್ಯ_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_ಗಮ್ಯ_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_ಗಮ್ಯ_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_ಗಮ್ಯ_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_ಗಮ್ಯ_spv("roll_f32", "roll.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("add_id_f32", "add_id.comp", ವಿಲೀನ_maps(ತಳ_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_ಗಮ್ಯ_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "0"}});
    string_ಗಮ್ಯ_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS" , "1"}});

    string_ಗಮ್ಯ_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_ಗಮ್ಯ_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_ಗಮ್ಯ_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_ಗಮ್ಯ_spv("topk_moe_f32", "topk_moe.comp", {});

    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &c : compiles) {
        c.wait();
    }
}

ಶೂನ್ಯ write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(ಗುರಿ_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& pair : shader_fnames) {
        ಸ್ಥಿರಾಂಕ std::string& name = pair.ಮೊದಲ;
        #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #ಇಲ್ಲದಿದ್ದರೆ
            ಸ್ಥಿರಾಂಕ std::string& path = pair.second;
        #ಆದರೆ_ಅಂತ್ಯ

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        ಆದರೆ (ಒಳಹರಿವು_filepath != "") {
            std::string ದತ್ತಾಂಶ = read_ದ್ವಿಮಾನ_file(path);
            ಆದರೆ (ದತ್ತಾಂಶ.ಖಾಲಿಗಳು()) {
                ಮುಂದುವರಿ;
            }

            src << "const uint64_t " << name << "_len = " << ದತ್ತಾಂಶ.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << ದತ್ತಾಂಶ.size() << "] = {\n" << std::hex;
            ಸ್ವಯಂಚಾಲಿತ ಬೈಟುಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t*>(ದತ್ತಾಂಶ.ದತ್ತಾಂಶ());
            ಗಾಗಿ (size_t i = 0; i < ದತ್ತಾಂಶ.size(); ++i) {
                src << "0x" << static_cast<ಪೂರ್ಣಾಂಕ>(ಬೈಟುಗಳು[i]) << ",";
                ಆದರೆ ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    ಗಾಗಿ (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        ಆದರೆ (basename(ಒಳಹರಿವು_filepath) != op_file) {
            ಮುಂದುವರಿ;
        }
        std::stringstream ದತ್ತಾಂಶ = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        ದತ್ತಾಂಶ << "const void * " << op << "_data[2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2] = ";
        ಗಾಗಿ (uint32_t t0 = 0; t0 < 2; ++t0) {
            ಆದರೆ (t0 == 0) {
                ದತ್ತಾಂಶ << "{";
                len  << "{";
            }
            ಗಾಗಿ (uint32_t t1 = 0; t1 < 2; ++t1) {
                ಆದರೆ (t1 == 0) {
                    ದತ್ತಾಂಶ << "{";
                    len  << "{";
                }
                ಗಾಗಿ (uint32_t t2 = 0; t2 < 2; ++t2) {
                    ಆದರೆ (t2 == 0) {
                        ದತ್ತಾಂಶ << "{";
                        len  << "{";
                    }
                    ದತ್ತಾಂಶ << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2];
                    ದತ್ತಾಂಶ << "_data,";
                    len  << "_len,";
                    ಆದರೆ (t2 == 1) {
                        ದತ್ತಾಂಶ << "}, ";
                        len  << "}, ";
                    }
                }
                ಆದರೆ (t1 == 1) {
                    ದತ್ತಾಂಶ << "}, ";
                    len  << "}, ";
                }
            }
            ಆದರೆ (t0 == 1) {
                ದತ್ತಾಂಶ << "};\n";
                len  << "};\n";
            }
        }
        src << ದತ್ತಾಂಶ.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#ಆದರೆ defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#ಆದರೆ_ಅಂತ್ಯ

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string& btype : btypes) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& tname : type_ಹೆಸರುಗಳು) {
        ಆದರೆ (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m" && tname != "iq4_xs") {
            ಮುಂದುವರಿ;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        ಆದರೆ (basename(ಒಳಹರಿವು_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        ಆದರೆ (btype == "f16") {
            ಮುಂದುವರಿ;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        ಆದರೆ (basename(ಒಳಹರಿವು_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

#ಆದರೆ defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string& btype : {"f16", "f32"}) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3];\n";
        ಆದರೆ (basename(ಒಳಹರಿವು_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_ocp_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_ocp_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_ocp_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ std::string& tname : {"mxfp4", "nvfp4"}) {
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3];\n";
        ಆದರೆ (basename(ಒಳಹರಿವು_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << tname << "_f32_f32_ocp_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << tname << "_f32_f32_ocp_len,  mul_mat_vec_id_" << tname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_"  << tname << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (ಒಳಹರಿವು_filepath == "") {
        write_file_if_ಬದಲಾಯಿತು(ಗುರಿ_hpp, hdr.str());
    }
    ಆದರೆ (ಗುರಿ_cpp != "") {
        write_ದ್ವಿಮಾನ_file(ಗುರಿ_cpp, src.str());
    }
}

} // namespace

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ(ಪೂರ್ಣಾಂಕ argc, ಅಕ್ಷರ** argv) {
    std::map<std::string, std::string> ವಾದಗಳು_2;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        ಆದರೆ (arg.rfind("--", 0) == 0) {
            ಆದರೆ (i + 1 < argc && argv[i + 1][0] != '-') {
                ವಾದಗಳು_2[arg] = argv[i + 1];
                ++i;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ವಾದಗಳು_2[arg] = "";
            }
        }
    }

    ಆದರೆ (ವಾದಗಳು_2.find("--glslc") != ವಾದಗಳು_2.end()) {
        GLSLC = ವಾದಗಳು_2["--glslc"]; // Path to glslc
    }
    ಆದರೆ (ವಾದಗಳು_2.find("--source") != ವಾದಗಳು_2.end()) {
        ಒಳಹರಿವು_filepath = ವಾದಗಳು_2["--source"]; // The shader source file to compile
    }
    ಆದರೆ (ವಾದಗಳು_2.find("--output-dir") != ವಾದಗಳು_2.end()) {
        output_dir = ವಾದಗಳು_2["--output-dir"]; // Directory for containing SPIR-V output
    }
    ಆದರೆ (ವಾದಗಳು_2.find("--target-hpp") != ವಾದಗಳು_2.end()) {
        ಗುರಿ_hpp = ವಾದಗಳು_2["--target-hpp"]; // Path to generated header file
    }
    ಆದರೆ (ವಾದಗಳು_2.find("--target-cpp") != ವಾದಗಳು_2.end()) {
        ಗುರಿ_cpp = ವಾದಗಳು_2["--target-cpp"]; // Path to generated cpp file
    }

    ಆದರೆ (!directory_exists(output_dir)) {
        ಆದರೆ (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            ಹಿಂತಿರುಗಿ EXIT_FAILURE;
        }
    }

    process_shaders();

    ಆದರೆ (compile_failed) {
        std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::ಸಾಲು_ಅಂತ್ಯ;
        ಹಿಂತಿರುಗಿ EXIT_FAILURE;
    }

    write_output_files();

    ಹಿಂತಿರುಗಿ EXIT_SUCCESS;
}
