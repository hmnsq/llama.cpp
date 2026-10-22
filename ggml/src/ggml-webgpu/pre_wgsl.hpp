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
    နေစဉ် (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    နေစဉ် (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ပြန်ပေး s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ပြန်ပေး trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ပြန်ပေး std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    နေစဉ် (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ပြန်ပေး i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    နေစဉ် (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    အကယ်၍ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    အကယ်၍ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    အကယ်၍ (it == macros.end()) {
        visiting.erase(name);
        ပြန်ပေး name;
    }

    const std::string & value = it->second;
    အကယ်၍ (value.empty()) {
        visiting.erase(name);
        ပြန်ပေး "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ပြန်ပေး expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    နေစဉ် (i < line.size()) {
        အကယ်၍ (isIdentChar(line[i])) {
            size_t start = i;
            နေစဉ် (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            အကယ်၍ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } မဟုတ်ပါက {
                result += token;
            }
        } မဟုတ်ပါက {
            result += line[i];
            i++;
        }
    }

    ပြန်ပေး result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ပြန်ပေး expandMacrosRecursiveInternal(line, macros, visiting);
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
        အကယ်၍ (pos >= src.size()) {
            ပြန်ပေး { END, "" };
        }

        var_na c = src[pos];

        // number
        အကယ်၍ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            နေစဉ် (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ပြန်ပေး { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        အကယ်၍ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            နေစဉ် (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ပြန်ပေး { IDENT, std::string(src.substr(start, pos - start)) };
        }

        အကယ်၍ (c == '(') {
            pos++;
            ပြန်ပေး { LPAREN, "(" };
        }
        အကယ်၍ (c == ')') {
            pos++;
            ပြန်ပေး { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        အတွက် (auto op : two_ops) {
            အကယ်၍ (src.substr(pos, 2) == op) {
                pos += 2;
                ပြန်ပေး { OP, std::string(op) };
            }
        }

        // single-char operators
        အကယ်၍ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ပြန်ပေး { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ပြန်ပေး { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        နေစဉ် (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    ကိန်းပြည့် parse() { ပြန်ပေး parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        အကယ်၍ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ပြန်ပေး true;
        }
        ပြန်ပေး false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        အကယ်၍ (tok.kind == k) {
            advance();
            ပြန်ပေး true;
        }
        ပြန်ပေး false;
    }

    ကိန်းပြည့် parseLogicalOr() {
        ကိန်းပြည့် v = parseLogicalAnd();
        နေစဉ် (acceptOp("||")) {
            ကိန်းပြည့် rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseLogicalAnd() {
        ကိန်းပြည့် v = parseEquality();
        နေစဉ် (acceptOp("&&")) {
            ကိန်းပြည့် rhs = parseEquality();
            v       = (v && rhs);
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseEquality() {
        ကိန်းပြည့် v = parseRelational();
        အတွက် (;;) {
            အကယ်၍ (acceptOp("==")) {
                ကိန်းပြည့် rhs = parseRelational();
                v       = (v == rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp("!=")) {
                ကိန်းပြည့် rhs = parseRelational();
                v       = (v != rhs);
            } မဟုတ်ပါက {
                bha_nga;
            }
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseRelational() {
        ကိန်းပြည့် v = parseShift();
        အတွက် (;;) {
            အကယ်၍ (acceptOp("<")) {
                ကိန်းပြည့် rhs = parseShift();
                v       = (v < rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp(">")) {
                ကိန်းပြည့် rhs = parseShift();
                v       = (v > rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp("<=")) {
                ကိန်းပြည့် rhs = parseShift();
                v       = (v <= rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp(">=")) {
                ကိန်းပြည့် rhs = parseShift();
                v       = (v >= rhs);
            } မဟုတ်ပါက {
                bha_nga;
            }
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseShift() {
        ကိန်းပြည့် v = parseAdd();
        အတွက် (;;) {
            အကယ်၍ (acceptOp("<<")) {
                ကိန်းပြည့် rhs = parseAdd();
                v       = (v << rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp(">>")) {
                ကိန်းပြည့် rhs = parseAdd();
                v       = (v >> rhs);
            } မဟုတ်ပါက {
                bha_nga;
            }
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseAdd() {
        ကိန်းပြည့် v = parseMult();
        အတွက် (;;) {
            အကယ်၍ (acceptOp("+")) {
                ကိန်းပြည့် rhs = parseMult();
                v       = (v + rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp("-")) {
                ကိန်းပြည့် rhs = parseMult();
                v       = (v - rhs);
            } မဟုတ်ပါက {
                bha_nga;
            }
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseMult() {
        ကိန်းပြည့် v = parseUnary();
        အတွက် (;;) {
            အကယ်၍ (acceptOp("*")) {
                ကိန်းပြည့် rhs = parseUnary();
                v       = (v * rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp("/")) {
                ကိန်းပြည့် rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } မဟုတ်ပါက အကယ်၍ (acceptOp("%")) {
                ကိန်းပြည့် rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } မဟုတ်ပါက {
                bha_nga;
            }
        }
        ပြန်ပေး v;
    }

    ကိန်းပြည့် parseUnary() {
        အကယ်၍ (acceptOp("!")) {
            ပြန်ပေး !parseUnary();
        }
        အကယ်၍ (acceptOp("-")) {
            ပြန်ပေး -parseUnary();
        }
        အကယ်၍ (acceptOp("+")) {
            ပြန်ပေး +parseUnary();
        }
        ပြန်ပေး parsePrimary();
    }

    ကိန်းပြည့် parsePrimary() {
        // '(' expr ')'
        အကယ်၍ (acceptKind(ExprLexer::LPAREN)) {
            ကိန်းပြည့် v = parse();
            အကယ်၍ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ပြန်ပေး v;
        }

        // number
        အကယ်၍ (tok.kind == ExprLexer::NUMBER) {
            ကိန်းပြည့် v = std::stoi(tok.text);
            advance();
            ပြန်ပေး v;
        }

        // defined(identifier)
        အကယ်၍ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            အကယ်၍ (acceptKind(ExprLexer::LPAREN)) {
                အကယ်၍ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                အကယ်၍ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ပြန်ပေး macros.count(name) ? 1 : 0;
            } မဟုတ်ပါက {
                // defined NAME
                အကယ်၍ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ပြန်ပေး macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        အကယ်၍ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            အကယ်၍ (it == macros.end()) {
                ပြန်ပေး 0;
            }
            အကယ်၍ (it->second.empty()) {
                ပြန်ပေး 1;
            }
            ပြန်ပေး evalMacroExpression(name, it->second);
        }

        // unexpected
        ပြန်ပေး 0;
    }

    ကိန်းပြည့် evalMacroExpression(const std::string & name, const std::string & value) {
        အကယ်၍ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        ကိန်းပြည့်        v = ep.parse();
        visiting.erase(name);
        ပြန်ပေး v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        အကယ်၍ (opts_.include_path.empty()) {
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
        ပြန်ပေး result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ပြန်ပေး result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ပြန်ပေး result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ပြန်ပေး result;
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
        အတွက် (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            အကယ်၍ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } မဟုတ်ပါက {
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

        အတွက် (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        အတွက် (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            အကယ်၍ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } မဟုတ်ပါက {
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
        အကယ်၍ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ပြန်ပေး ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        အကယ်၍ (cond.empty()) {
            ပြန်ပေး true;
        }
        ပြန်ပေး cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        အကယ်၍ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ပြန်ပေး out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ပြန်ပေး processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        နေစဉ် (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            အကယ်၍ (!t.empty() && t[0] == '#') {
                နေစဉ် (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    အကယ်၍ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            အကယ်၍ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                အကယ်၍ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } မဟုတ်ပါက {
                အကယ်၍ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } မဟုတ်ပါက အကယ်၍ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        အကယ်၍ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ပြန်ပေး out.str();
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

        အကယ်၍ (cmd == "include") {
            အကယ်၍ (mode == DirectiveMode::All && !condActive(cond)) {
                ပြန်ပေး true;
            }
            std::string file;
            iss >> file;
            အကယ်၍ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ပြန်ပေး true;
        }

        အကယ်၍ (mode == DirectiveMode::IncludesOnly) {
            ပြန်ပေး false;
        }

        အကယ်၍ (cmd == "define") {
            အကယ်၍ (!condActive(cond)) {
                ပြန်ပေး true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            အကယ်၍ (predefined_macros.count(name)) {
                ပြန်ပေး true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "undef") {
            အကယ်၍ (!condActive(cond)) {
                ပြန်ပေး true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            အကယ်၍ (predefined_macros.count(name)) {
                ပြန်ပေး true;
            }
            macros.erase(name);
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            အကယ်၍ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "elif") {
            std::string expr = trim_value(iss);

            အကယ်၍ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            အကယ်၍ (!c.parent_active) {
                c.active = false;
                ပြန်ပေး true;
            }

            အကယ်၍ (c.taken) {
                c.active = false;
                ပြန်ပေး true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            အကယ်၍ (v) {
                c.taken = true;
            }
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "else") {
            အကယ်၍ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            အကယ်၍ (!c.parent_active) {
                c.active = false;
                ပြန်ပေး true;
            }
            အကယ်၍ (c.taken) {
                c.active = false;
            } မဟုတ်ပါက {
                c.active = true;
                c.taken  = true;
            }
            ပြန်ပေး true;
        }

        အကယ်၍ (cmd == "endif") {
            အကယ်၍ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ပြန်ပေး true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
