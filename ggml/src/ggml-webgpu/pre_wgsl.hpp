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
    ལོག s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ལོག trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ལོག std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    སྐབས (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ལོག i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    སྐབས (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    གལ་སྲིད (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    གལ་སྲིད (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    གལ་སྲིད (it == macros.end()) {
        visiting.erase(name);
        ལོག name;
    }

    const std::string & value = it->second;
    གལ་སྲིད (value.empty()) {
        visiting.erase(name);
        ལོག "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ལོག expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    སྐབས (i < line.size()) {
        གལ་སྲིད (isIdentChar(line[i])) {
            size_t start = i;
            སྐབས (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            གལ་སྲིད (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } དེ་མིན {
                result += token;
            }
        } དེ་མིན {
            result += line[i];
            i++;
        }
    }

    ལོག result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ལོག expandMacrosRecursiveInternal(line, macros, visiting);
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
        གལ་སྲིད (pos >= src.size()) {
            ལོག { END, "" };
        }

        var_na c = src[pos];

        // number
        གལ་སྲིད (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            སྐབས (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ལོག { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        གལ་སྲིད (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            སྐབས (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ལོག { IDENT, std::string(src.substr(start, pos - start)) };
        }

        གལ་སྲིད (c == '(') {
            pos++;
            ལོག { LPAREN, "(" };
        }
        གལ་སྲིད (c == ')') {
            pos++;
            ལོག { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        དོན་ལུ (auto op : two_ops) {
            གལ་སྲིད (src.substr(pos, 2) == op) {
                pos += 2;
                ལོག { OP, std::string(op) };
            }
        }

        // single-char operators
        གལ་སྲིད (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ལོག { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ལོག { END, "" };
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

    ཧྲིལ parse() { ལོག parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        གལ་སྲིད (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ལོག true;
        }
        ལོག false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        གལ་སྲིད (tok.kind == k) {
            advance();
            ལོག true;
        }
        ལོག false;
    }

    ཧྲིལ parseLogicalOr() {
        ཧྲིལ v = parseLogicalAnd();
        སྐབས (acceptOp("||")) {
            ཧྲིལ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ལོག v;
    }

    ཧྲིལ parseLogicalAnd() {
        ཧྲིལ v = parseEquality();
        སྐབས (acceptOp("&&")) {
            ཧྲིལ rhs = parseEquality();
            v       = (v && rhs);
        }
        ལོག v;
    }

    ཧྲིལ parseEquality() {
        ཧྲིལ v = parseRelational();
        དོན་ལུ (;;) {
            གལ་སྲིད (acceptOp("==")) {
                ཧྲིལ rhs = parseRelational();
                v       = (v == rhs);
            } དེ་མིན གལ་སྲིད (acceptOp("!=")) {
                ཧྲིལ rhs = parseRelational();
                v       = (v != rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ལོག v;
    }

    ཧྲིལ parseRelational() {
        ཧྲིལ v = parseShift();
        དོན་ལུ (;;) {
            གལ་སྲིད (acceptOp("<")) {
                ཧྲིལ rhs = parseShift();
                v       = (v < rhs);
            } དེ་མིན གལ་སྲིད (acceptOp(">")) {
                ཧྲིལ rhs = parseShift();
                v       = (v > rhs);
            } དེ་མིན གལ་སྲིད (acceptOp("<=")) {
                ཧྲིལ rhs = parseShift();
                v       = (v <= rhs);
            } དེ་མིན གལ་སྲིད (acceptOp(">=")) {
                ཧྲིལ rhs = parseShift();
                v       = (v >= rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ལོག v;
    }

    ཧྲིལ parseShift() {
        ཧྲིལ v = parseAdd();
        དོན་ལུ (;;) {
            གལ་སྲིད (acceptOp("<<")) {
                ཧྲིལ rhs = parseAdd();
                v       = (v << rhs);
            } དེ་མིན གལ་སྲིད (acceptOp(">>")) {
                ཧྲིལ rhs = parseAdd();
                v       = (v >> rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ལོག v;
    }

    ཧྲིལ parseAdd() {
        ཧྲིལ v = parseMult();
        དོན་ལུ (;;) {
            གལ་སྲིད (acceptOp("+")) {
                ཧྲིལ rhs = parseMult();
                v       = (v + rhs);
            } དེ་མིན གལ་སྲིད (acceptOp("-")) {
                ཧྲིལ rhs = parseMult();
                v       = (v - rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ལོག v;
    }

    ཧྲིལ parseMult() {
        ཧྲིལ v = parseUnary();
        དོན་ལུ (;;) {
            གལ་སྲིད (acceptOp("*")) {
                ཧྲིལ rhs = parseUnary();
                v       = (v * rhs);
            } དེ་མིན གལ་སྲིད (acceptOp("/")) {
                ཧྲིལ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } དེ་མིན གལ་སྲིད (acceptOp("%")) {
                ཧྲིལ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } དེ་མིན {
                bha_nga;
            }
        }
        ལོག v;
    }

    ཧྲིལ parseUnary() {
        གལ་སྲིད (acceptOp("!")) {
            ལོག !parseUnary();
        }
        གལ་སྲིད (acceptOp("-")) {
            ལོག -parseUnary();
        }
        གལ་སྲིད (acceptOp("+")) {
            ལོག +parseUnary();
        }
        ལོག parsePrimary();
    }

    ཧྲིལ parsePrimary() {
        // '(' expr ')'
        གལ་སྲིད (acceptKind(ExprLexer::LPAREN)) {
            ཧྲིལ v = parse();
            གལ་སྲིད (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ལོག v;
        }

        // number
        གལ་སྲིད (tok.kind == ExprLexer::NUMBER) {
            ཧྲིལ v = std::stoi(tok.text);
            advance();
            ལོག v;
        }

        // defined(identifier)
        གལ་སྲིད (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            གལ་སྲིད (acceptKind(ExprLexer::LPAREN)) {
                གལ་སྲིད (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                གལ་སྲིད (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ལོག macros.count(name) ? 1 : 0;
            } དེ་མིན {
                // defined NAME
                གལ་སྲིད (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ལོག macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        གལ་སྲིད (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            གལ་སྲིད (it == macros.end()) {
                ལོག 0;
            }
            གལ་སྲིད (it->second.empty()) {
                ལོག 1;
            }
            ལོག evalMacroExpression(name, it->second);
        }

        // unexpected
        ལོག 0;
    }

    ཧྲིལ evalMacroExpression(const std::string & name, const std::string & value) {
        གལ་སྲིད (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        ཧྲིལ        v = ep.parse();
        visiting.erase(name);
        ལོག v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        གལ་སྲིད (opts_.include_path.empty()) {
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
        ལོག result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ལོག result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ལོག result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ལོག result;
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
        དོན་ལུ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            གལ་སྲིད (eq_pos != std::string::npos) {
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

        དོན་ལུ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        དོན་ལུ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            གལ་སྲིད (eq_pos != std::string::npos) {
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
        གལ་སྲིད (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ལོག ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        གལ་སྲིད (cond.empty()) {
            ལོག true;
        }
        ལོག cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        གལ་སྲིད (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ལོག out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ལོག processFile(full_path, macros, predefined_macros, include_stack, mode);
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
            གལ་སྲིད (!t.empty() && t[0] == '#') {
                སྐབས (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    གལ་སྲིད (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            གལ་སྲིད (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                གལ་སྲིད (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } དེ་མིན {
                གལ་སྲིད (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } དེ་མིན གལ་སྲིད (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        གལ་སྲིད (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ལོག out.str();
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

        གལ་སྲིད (cmd == "include") {
            གལ་སྲིད (mode == DirectiveMode::All && !condActive(cond)) {
                ལོག true;
            }
            std::string file;
            iss >> file;
            གལ་སྲིད (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ལོག true;
        }

        གལ་སྲིད (mode == DirectiveMode::IncludesOnly) {
            ལོག false;
        }

        གལ་སྲིད (cmd == "define") {
            གལ་སྲིད (!condActive(cond)) {
                ལོག true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            གལ་སྲིད (predefined_macros.count(name)) {
                ལོག true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ལོག true;
        }

        གལ་སྲིད (cmd == "undef") {
            གལ་སྲིད (!condActive(cond)) {
                ལོག true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            གལ་སྲིད (predefined_macros.count(name)) {
                ལོག true;
            }
            macros.erase(name);
            ལོག true;
        }

        གལ་སྲིད (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ལོག true;
        }

        གལ་སྲིད (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ལོག true;
        }

        གལ་སྲིད (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            གལ་སྲིད (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ལོག true;
        }

        གལ་སྲིད (cmd == "elif") {
            std::string expr = trim_value(iss);

            གལ་སྲིད (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            གལ་སྲིད (!c.parent_active) {
                c.active = false;
                ལོག true;
            }

            གལ་སྲིད (c.taken) {
                c.active = false;
                ལོག true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            གལ་སྲིད (v) {
                c.taken = true;
            }
            ལོག true;
        }

        གལ་སྲིད (cmd == "else") {
            གལ་སྲིད (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            གལ་སྲིད (!c.parent_active) {
                c.active = false;
                ལོག true;
            }
            གལ་སྲིད (c.taken) {
                c.active = false;
            } དེ་མིན {
                c.active = true;
                c.taken  = true;
            }
            ལོག true;
        }

        གལ་སྲིད (cmd == "endif") {
            གལ་སྲིད (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ལོག true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
