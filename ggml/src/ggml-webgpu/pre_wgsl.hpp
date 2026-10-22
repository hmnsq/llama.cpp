#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pre_wgsl {

//==============================================================
// Options
//==============================================================
sa_nracanaa Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
static std::string trim(const std::string & s) {
    size_t a = 0;
    སྐབས (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    སྐབས (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ཕྱིར s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ཕྱིར trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ཕྱིར std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    སྐབས (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ཕྱིར i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    སྐབས (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    གལ་ཏེ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    གལ་ཏེ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    གལ་ཏེ (it == macros.end()) {
        visiting.erase(name);
        ཕྱིར name;
    }

    const std::string & value = it->second;
    གལ་ཏེ (value.empty()) {
        visiting.erase(name);
        ཕྱིར "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ཕྱིར expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    སྐབས (i < line.size()) {
        གལ་ཏེ (isIdentChar(line[i])) {
            size_t start = i;
            སྐབས (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            གལ་ཏེ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } དེ་མིན {
                result += token;
            }
        } དེ་མིན {
            result += line[i];
            i++;
        }
    }

    ཕྱིར result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ཕྱིར expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
class ExprLexer {
  public:
    ga_nanaa Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    sa_nracanaa Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        གལ་ཏེ (pos >= src.size()) {
            ཕྱིར { END, "" };
        }

        var_na c = src[pos];

        // number
        གལ་ཏེ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            སྐབས (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ཕྱིར { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        གལ་ཏེ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            སྐབས (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ཕྱིར { IDENT, std::string(src.substr(start, pos - start)) };
        }

        གལ་ཏེ (c == '(') {
            pos++;
            ཕྱིར { LPAREN, "(" };
        }
        གལ་ཏེ (c == ')') {
            pos++;
            ཕྱིར { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ཆེད་དུ (auto op : two_ops) {
            གལ་ཏེ (src.substr(pos, 2) == op) {
                pos += 2;
                ཕྱིར { OP, std::string(op) };
            }
        }

        // single-char operators
        གལ་ཏེ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ཕྱིར { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ཕྱིར { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        སྐབས (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
class ExprParser {
  public:
    ExprParser(std::string_view                                     expr,
               const std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    ཧྲིལ parse() { ཕྱིར parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        གལ་ཏེ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ཕྱིར true;
        }
        ཕྱིར false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        གལ་ཏེ (tok.kind == k) {
            advance();
            ཕྱིར true;
        }
        ཕྱིར false;
    }

    ཧྲིལ parseLogicalOr() {
        ཧྲིལ v = parseLogicalAnd();
        སྐབས (acceptOp("||")) {
            ཧྲིལ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ཕྱིར v;
    }

    ཧྲིལ parseLogicalAnd() {
        ཧྲིལ v = parseEquality();
        སྐབས (acceptOp("&&")) {
            ཧྲིལ rhs = parseEquality();
            v       = (v && rhs);
        }
        ཕྱིར v;
    }

    ཧྲིལ parseEquality() {
        ཧྲིལ v = parseRelational();
        ཆེད་དུ (;;) {
            གལ་ཏེ (acceptOp("==")) {
                ཧྲིལ rhs = parseRelational();
                v       = (v == rhs);
            } དེ་མིན གལ་ཏེ (acceptOp("!=")) {
                ཧྲིལ rhs = parseRelational();
                v       = (v != rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ཕྱིར v;
    }

    ཧྲིལ parseRelational() {
        ཧྲིལ v = parseShift();
        ཆེད་དུ (;;) {
            གལ་ཏེ (acceptOp("<")) {
                ཧྲིལ rhs = parseShift();
                v       = (v < rhs);
            } དེ་མིན གལ་ཏེ (acceptOp(">")) {
                ཧྲིལ rhs = parseShift();
                v       = (v > rhs);
            } དེ་མིན གལ་ཏེ (acceptOp("<=")) {
                ཧྲིལ rhs = parseShift();
                v       = (v <= rhs);
            } དེ་མིན གལ་ཏེ (acceptOp(">=")) {
                ཧྲིལ rhs = parseShift();
                v       = (v >= rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ཕྱིར v;
    }

    ཧྲིལ parseShift() {
        ཧྲིལ v = parseAdd();
        ཆེད་དུ (;;) {
            གལ་ཏེ (acceptOp("<<")) {
                ཧྲིལ rhs = parseAdd();
                v       = (v << rhs);
            } དེ་མིན གལ་ཏེ (acceptOp(">>")) {
                ཧྲིལ rhs = parseAdd();
                v       = (v >> rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ཕྱིར v;
    }

    ཧྲིལ parseAdd() {
        ཧྲིལ v = parseMult();
        ཆེད་དུ (;;) {
            གལ་ཏེ (acceptOp("+")) {
                ཧྲིལ rhs = parseMult();
                v       = (v + rhs);
            } དེ་མིན གལ་ཏེ (acceptOp("-")) {
                ཧྲིལ rhs = parseMult();
                v       = (v - rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ཕྱིར v;
    }

    ཧྲིལ parseMult() {
        ཧྲིལ v = parseUnary();
        ཆེད་དུ (;;) {
            གལ་ཏེ (acceptOp("*")) {
                ཧྲིལ rhs = parseUnary();
                v       = (v * rhs);
            } དེ་མིན གལ་ཏེ (acceptOp("/")) {
                ཧྲིལ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } དེ་མིན གལ་ཏེ (acceptOp("%")) {
                ཧྲིལ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ཕྱིར v;
    }

    ཧྲིལ parseUnary() {
        གལ་ཏེ (acceptOp("!")) {
            ཕྱིར !parseUnary();
        }
        གལ་ཏེ (acceptOp("-")) {
            ཕྱིར -parseUnary();
        }
        གལ་ཏེ (acceptOp("+")) {
            ཕྱིར +parseUnary();
        }
        ཕྱིར parsePrimary();
    }

    ཧྲིལ parsePrimary() {
        // '(' expr ')'
        གལ་ཏེ (acceptKind(ExprLexer::LPAREN)) {
            ཧྲིལ v = parse();
            གལ་ཏེ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ཕྱིར v;
        }

        // number
        གལ་ཏེ (tok.kind == ExprLexer::NUMBER) {
            ཧྲིལ v = std::stoi(tok.text);
            advance();
            ཕྱིར v;
        }

        // defined(identifier)
        གལ་ཏེ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            གལ་ཏེ (acceptKind(ExprLexer::LPAREN)) {
                གལ་ཏེ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                གལ་ཏེ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ཕྱིར macros.count(name) ? 1 : 0;
            } དེ་མིན {
                // defined NAME
                གལ་ཏེ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ཕྱིར macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        གལ་ཏེ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            གལ་ཏེ (it == macros.end()) {
                ཕྱིར 0;
            }
            གལ་ཏེ (it->second.empty()) {
                ཕྱིར 1;
            }
            ཕྱིར evalMacroExpression(name, it->second);
        }

        // unexpected
        ཕྱིར 0;
    }

    ཧྲིལ evalMacroExpression(const std::string & name, const std::string & value) {
        གལ་ཏེ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        ཧྲིལ        v = ep.parse();
        visiting.erase(name);
        ཕྱིར v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        གལ་ཏེ (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(const std::string & filename, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        ཕྱིར result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ཕྱིར result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ཕྱིར result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ཕྱིར result;
    }

  private:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ga_nanaa class DirectiveMode { All, IncludesOnly };

    sa_nracanaa Cond {
        bool parent_active;
        bool active;
        bool taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    void parseMacroDefinitions(const std::vector<std::string> & macro_defs) {
        ཆེད་དུ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            གལ་ཏེ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } དེ་མིན {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    void buildMacros(const std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        ཆེད་དུ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ཆེད་དུ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            གལ་ཏེ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } དེ་མིན {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(const std::string & fname) {
        std::ifstream f(fname);
        གལ་ཏེ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ཕྱིར ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        གལ་ཏེ (cond.empty()) {
            ཕྱིར true;
        }
        ཕྱིར cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        གལ་ཏེ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ཕྱིར out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ཕྱིར processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(const std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              const std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        སྐབས (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            གལ་ཏེ (!t.empty() && t[0] == '#') {
                སྐབས (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    གལ་ཏེ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            གལ་ཏེ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                གལ་ཏེ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } དེ་མིན {
                གལ་ཏེ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } དེ་མིན གལ་ཏེ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        གལ་ཏེ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ཕྱིར out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    bool handleDirective(const std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         const std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        གལ་ཏེ (cmd == "include") {
            གལ་ཏེ (mode == DirectiveMode::All && !condActive(cond)) {
                ཕྱིར true;
            }
            std::string file;
            iss >> file;
            གལ་ཏེ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ཕྱིར true;
        }

        གལ་ཏེ (mode == DirectiveMode::IncludesOnly) {
            ཕྱིར false;
        }

        གལ་ཏེ (cmd == "define") {
            གལ་ཏེ (!condActive(cond)) {
                ཕྱིར true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            གལ་ཏེ (predefined_macros.count(name)) {
                ཕྱིར true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "undef") {
            གལ་ཏེ (!condActive(cond)) {
                ཕྱིར true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            གལ་ཏེ (predefined_macros.count(name)) {
                ཕྱིར true;
            }
            macros.erase(name);
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            གལ་ཏེ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "elif") {
            std::string expr = trim_value(iss);

            གལ་ཏེ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            གལ་ཏེ (!c.parent_active) {
                c.active = false;
                ཕྱིར true;
            }

            གལ་ཏེ (c.taken) {
                c.active = false;
                ཕྱིར true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            གལ་ཏེ (v) {
                c.taken = true;
            }
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "else") {
            གལ་ཏེ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            གལ་ཏེ (!c.parent_active) {
                c.active = false;
                ཕྱིར true;
            }
            གལ་ཏེ (c.taken) {
                c.active = false;
            } དེ་མིན {
                c.active = true;
                c.taken  = true;
            }
            ཕྱིར true;
        }

        གལ་ཏེ (cmd == "endif") {
            གལ་ཏེ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ཕྱིར true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
