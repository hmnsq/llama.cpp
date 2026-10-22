ഇറക്കുമതി os
ഇറക്കുമതി re
ഇറക്കുമതി argparse


def expand_includes(shader, നിവേശം_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(നിവേശം_dir, fname)
        എങ്കിൽ അല്ല os.path.exists(file_path):
            ഉയർത്തുക FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ഇവ f:
            included_code = f.പഠ()
        # Recursively expand includes inside the included file
        തിരികെ expand_includes(included_code, നിവേശം_dir)

    തിരികെ include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    തിരികെ [shader_code[i : i + max_chunk_len] വേണ്ടി i ഇൽ range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    ആകുമ്പോൾ f"){delim}\"" ഇൽ shader_code:
        delim += "_x"
    തിരികെ delim


def write_shader(shader_name, shader_code, output_dir, outfile, നിവേശം_dir):
    shader_code = expand_includes(shader_code, നിവേശം_dir)

    എങ്കിൽ output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ഇവ f_out:
            f_out.ലിഖ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    എങ്കിൽ len(chunks) == 1:
        outfile.ലിഖ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    അല്ലെങ്കിൽ:
        വേണ്ടി idx, chunk ഇൽ enumerate(chunks):
            outfile.ലിഖ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.ലിഖ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.ലിഖ('    static const std::string s = []{\n')
        outfile.ലിഖ('        std::string tmp;\n')
        outfile.ലിഖ(f'        tmp.reserve({len(shader_code)});\n')
        വേണ്ടി idx ഇൽ range(len(chunks)):
            outfile.ലിഖ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.ലിഖ('        return tmp;\n')
        outfile.ലിഖ('    }();\n')
        outfile.ലിഖ('    return s;\n')
        outfile.ലിഖ('}\n')
        outfile.ലിഖ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def പ്രധാന():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=സത്യം)
    parser.add_argument("--output_file", required=സത്യം)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ഇവ out:
        out.ലിഖ("// Auto-generated shader embedding\n")
        out.ലിഖ("#include <string>\n\n")
        വേണ്ടി fname ഇൽ sorted(os.listdir(args.നിവേശം_dir)):
            എങ്കിൽ fname.endswith(".wgsl"):
                shader_path = os.path.join(args.നിവേശം_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ഇവ f:
                    shader_code = f.പഠ()

                write_shader(shader_name, shader_code, None, out, args.നിവേശം_dir)


എങ്കിൽ __name__ == "__main__":
    പ്രധാന()
