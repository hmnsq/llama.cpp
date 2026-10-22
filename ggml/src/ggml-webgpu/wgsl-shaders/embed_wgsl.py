ಆಮದು os
ಆಮದು re
ಆಮದು argparse


ವ್ಯಾಖ್ಯಾನ expand_includes(shader, ಒಳಹರಿವು_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_ನಮೂನೆ = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    ವ್ಯಾಖ್ಯಾನ replacer(ಹೊಂದಿಕೆ):
        fname = ಹೊಂದಿಕೆ.ಗುಂಪು(1)
        file_ಮಾರ್ಗ = os.ಮಾರ್ಗ.join(ಒಳಹರಿವು_dir, fname)
        ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(file_ಮಾರ್ಗ):
            ಎತ್ತು File_Not_ಸಿಕ್ಕಿತು_Error(f"Included file not found: {file_path}")
        ಜೊತೆ open(file_ಮಾರ್ಗ, "r", encoding="utf-8") ಇವ f:
            included_ಸಂಕೇತ = f.ಪಠ()
        # Recursively expand includes inside the included file
        ಹಿಂತಿರುಗಿ expand_includes(included_ಸಂಕೇತ, ಒಳಹರಿವು_dir)

    ಹಿಂತಿರುಗಿ include_ನಮೂನೆ.sub(replacer, shader)


ವ್ಯಾಖ್ಯಾನ chunk_shader(shader_ಸಂಕೇತ, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ಹಿಂತಿರುಗಿ [shader_ಸಂಕೇತ[i : i + max_chunk_len] ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(0, ಉದ್ದ(shader_ಸಂಕೇತ), max_chunk_len)]


ವ್ಯಾಖ್ಯಾನ raw_delim(shader_ಸಂಕೇತ):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    ಆಗ f"){delim}\"" ಒಳಗೆ shader_ಸಂಕೇತ:
        delim += "_x"
    ಹಿಂತಿರುಗಿ delim


ವ್ಯಾಖ್ಯಾನ write_shader(shader_name, shader_ಸಂಕೇತ, output_dir, outfile, ಒಳಹರಿವು_dir):
    shader_ಸಂಕೇತ = expand_includes(shader_ಸಂಕೇತ, ಒಳಹರಿವು_dir)

    ಆದರೆ output_dir:
        wgsl_filename = os.ಮಾರ್ಗ.join(output_dir, f"{shader_name}.wgsl")
        ಜೊತೆ open(wgsl_filename, "w", encoding="utf-8") ಇವ f_ಹೊರಹೋಗು:
            f_ಹೊರಹೋಗು.ಲಿಖ(shader_ಸಂಕೇತ)

    delim = raw_delim(shader_ಸಂಕೇತ)
    chunks = chunk_shader(shader_ಸಂಕೇತ)

    ಆದರೆ ಉದ್ದ(chunks) == 1:
        outfile.ಲಿಖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    ಇಲ್ಲದಿದ್ದರೆ:
        ಗಾಗಿ idx, chunk ಒಳಗೆ enumerate(chunks):
            outfile.ಲಿಖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.ಲಿಖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.ಲಿಖ('    static const std::string s = []{\n')
        outfile.ಲಿಖ('        std::string tmp;\n')
        outfile.ಲಿಖ(f'        tmp.reserve({len(shader_code)});\n')
        ಗಾಗಿ idx ಒಳಗೆ ಶ್ರೇಣಿ(ಉದ್ದ(chunks)):
            outfile.ಲಿಖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.ಲಿಖ('        return tmp;\n')
        outfile.ಲಿಖ('    }();\n')
        outfile.ಲಿಖ('    return s;\n')
        outfile.ಲಿಖ('}\n')
        outfile.ಲಿಖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


ವ್ಯಾಖ್ಯಾನ ಮುಖ್ಯ():
    parser = argparse.ವಾದ_Parser()
    parser.ಕೂಡಿಸು_ವಾದ("--input_dir", required=ನಿಜ)
    parser.ಕೂಡಿಸು_ವಾದ("--output_file", required=ನಿಜ)
    ವಾದಗಳು_2 = parser.parse_ವಾದಗಳು()

    ಜೊತೆ open(ವಾದಗಳು_2.output_file, "w", encoding="utf-8") ಇವ ಹೊರಹೋಗು:
        ಹೊರಹೋಗು.ಲಿಖ("// Auto-generated shader embedding\n")
        ಹೊರಹೋಗು.ಲಿಖ("#include <string>\n\n")
        ಗಾಗಿ fname ಒಳಗೆ sorted(os.listdir(ವಾದಗಳು_2.ಒಳಹರಿವು_dir)):
            ಆದರೆ fname.endswith(".wgsl"):
                shader_ಮಾರ್ಗ = os.ಮಾರ್ಗ.join(ವಾದಗಳು_2.ಒಳಹರಿವು_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                ಜೊತೆ open(shader_ಮಾರ್ಗ, "r", encoding="utf-8") ಇವ f:
                    shader_ಸಂಕೇತ = f.ಪಠ()

                write_shader(shader_name, shader_ಸಂಕೇತ, ಏನೂಇಲ್ಲ, ಹೊರಹೋಗು, ವಾದಗಳು_2.ಒಳಹರಿವು_dir)


ಆದರೆ __name__ == "__main__":
    ಮುಖ್ಯ()
