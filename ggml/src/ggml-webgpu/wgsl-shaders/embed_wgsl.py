आयात os
आयात re
आयात argparse


def expand_includes(shader, निवेश_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.मेल(निवेश_dir, fname)
        अगर not os.path.exists(file_path):
            raise FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            included_code = f.म_पढ़ो()
        # Recursively expand includes inside the included file
        वापस expand_includes(included_code, निवेश_dir)

    वापस include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    वापस [shader_code[i : i + max_chunk_len] क्रम i in range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    जबतक f"){delim}\"" in shader_code:
        delim += "_x"
    वापस delim


def write_shader(shader_name, shader_code, output_dir, outfile, निवेश_dir):
    shader_code = expand_includes(shader_code, निवेश_dir)

    अगर output_dir:
        wgsl_filename = os.path.मेल(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") as f_out:
            f_out.म_लिखो(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    अगर len(chunks) == 1:
        outfile.म_लिखो(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    अन्यथा:
        क्रम idx, chunk in enumerate(chunks):
            outfile.म_लिखो(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.म_लिखो(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.म_लिखो('    static const std::string s = []{\n')
        outfile.म_लिखो('        std::string tmp;\n')
        outfile.म_लिखो(f'        tmp.reserve({len(shader_code)});\n')
        क्रम idx in range(len(chunks)):
            outfile.म_लिखो(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.म_लिखो('        return tmp;\n')
        outfile.म_लिखो('    }();\n')
        outfile.म_लिखो('    return s;\n')
        outfile.म_लिखो('}\n')
        outfile.म_लिखो(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def मुख्य():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    तर्क = parser.parse_args()

    with open(तर्क.output_file, "w", encoding="utf-8") as बाहर:
        बाहर.म_लिखो("// Auto-generated shader embedding\n")
        बाहर.म_लिखो("#include <string>\n\n")
        क्रम fname in sorted(os.listdir(तर्क.निवेश_dir)):
            अगर fname.endswith(".wgsl"):
                shader_path = os.path.मेल(तर्क.निवेश_dir, fname)
                shader_name = fname.प्रतिस्थापन(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") as f:
                    shader_code = f.म_पढ़ो()

                write_shader(shader_name, shader_code, None, बाहर, तर्क.निवेश_dir)


अगर __name__ == "__main__":
    मुख्य()
