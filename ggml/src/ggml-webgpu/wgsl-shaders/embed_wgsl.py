ಆನಯನ os
ಆನಯನ re
ಆನಯನ argparse


def expand_includes(shader, ಒಳಹರಿವು_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(ಒಳಹರಿವು_dir, fname)
        ಯದಿ ನ os.path.exists(file_path):
            ಉತ್ಥಾಪಯ FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ಇವ f:
            included_code = f.ಪಠ()
        # Recursively expand includes inside the included file
        ಪ್ರತ್ಯಾಗಮ expand_includes(included_code, ಒಳಹರಿವು_dir)

    ಪ್ರತ್ಯಾಗಮ include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ಪ್ರತ್ಯಾಗಮ [shader_code[i : i + max_chunk_len] ಪ್ರತಿ i ಅನ್ತಃ range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    ಯಾವತ್ f"){delim}\"" ಅನ್ತಃ shader_code:
        delim += "_x"
    ಪ್ರತ್ಯಾಗಮ delim


def write_shader(shader_name, shader_code, output_dir, outfile, ಒಳಹರಿವು_dir):
    shader_code = expand_includes(shader_code, ಒಳಹರಿವು_dir)

    ಯದಿ output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ಇವ f_out:
            f_out.ಲಿಖ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    ಯದಿ len(chunks) == 1:
        outfile.ಲಿಖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    ಅನ್ಯಥಾ:
        ಪ್ರತಿ idx, chunk ಅನ್ತಃ enumerate(chunks):
            outfile.ಲಿಖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.ಲಿಖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.ಲಿಖ('    static const std::string s = []{\n')
        outfile.ಲಿಖ('        std::string tmp;\n')
        outfile.ಲಿಖ(f'        tmp.reserve({len(shader_code)});\n')
        ಪ್ರತಿ idx ಅನ್ತಃ range(len(chunks)):
            outfile.ಲಿಖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.ಲಿಖ('        return tmp;\n')
        outfile.ಲಿಖ('    }();\n')
        outfile.ಲಿಖ('    return s;\n')
        outfile.ಲಿಖ('}\n')
        outfile.ಲಿಖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=ಸತ್ಯ)
    parser.add_argument("--output_file", required=ಸತ್ಯ)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ಇವ out:
        out.ಲಿಖ("// Auto-generated shader embedding\n")
        out.ಲಿಖ("#include <string>\n\n")
        ಪ್ರತಿ fname ಅನ್ತಃ sorted(os.listdir(args.ಒಳಹರಿವು_dir)):
            ಯದಿ fname.endswith(".wgsl"):
                shader_path = os.path.join(args.ಒಳಹರಿವು_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ಇವ f:
                    shader_code = f.ಪಠ()

                write_shader(shader_name, shader_code, None, out, args.ಒಳಹರಿವು_dir)


ಯದಿ __name__ == "__main__":
    main()
