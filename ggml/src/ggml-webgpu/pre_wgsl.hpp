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
    ខណៈពេល (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ខណៈពេល (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ត្រឡប់ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ត្រឡប់ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ត្រឡប់ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ខណៈពេល (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ត្រឡប់ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ខណៈពេល (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ប្រសិនបើ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ប្រសិនបើ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ប្រសិនបើ (it == macros.end()) {
        visiting.erase(name);
        ត្រឡប់ name;
    }

    const std::string & value = it->second;
    ប្រសិនបើ (value.empty()) {
        visiting.erase(name);
        ត្រឡប់ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ត្រឡប់ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ខណៈពេល (i < line.size()) {
        ប្រសិនបើ (isIdentChar(line[i])) {
            size_t start = i;
            ខណៈពេល (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ប្រសិនបើ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } បើមិនដូច្នេះ {
                result += token;
            }
        } បើមិនដូច្នេះ {
            result += line[i];
            i++;
        }
    }

    ត្រឡប់ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ត្រឡប់ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ប្រសិនបើ (pos >= src.size()) {
            ត្រឡប់ { END, "" };
        }

        var_na c = src[pos];

        // number
        ប្រសិនបើ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ខណៈពេល (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ត្រឡប់ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ប្រសិនបើ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ខណៈពេល (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ត្រឡប់ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ប្រសិនបើ (c == '(') {
            pos++;
            ត្រឡប់ { LPAREN, "(" };
        }
        ប្រសិនបើ (c == ')') {
            pos++;
            ត្រឡប់ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        សម្រាប់ (auto op : two_ops) {
            ប្រសិនបើ (src.substr(pos, 2) == op) {
                pos += 2;
                ត្រឡប់ { OP, std::string(op) };
            }
        }

        // single-char operators
        ប្រសិនបើ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ត្រឡប់ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ត្រឡប់ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ខណៈពេល (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    ចំនួនគត់ parse() { ត្រឡប់ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ប្រសិនបើ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ត្រឡប់ true;
        }
        ត្រឡប់ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ប្រសិនបើ (tok.kind == k) {
            advance();
            ត្រឡប់ true;
        }
        ត្រឡប់ false;
    }

    ចំនួនគត់ parseLogicalOr() {
        ចំនួនគត់ v = parseLogicalAnd();
        ខណៈពេល (acceptOp("||")) {
            ចំនួនគត់ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseLogicalAnd() {
        ចំនួនគត់ v = parseEquality();
        ខណៈពេល (acceptOp("&&")) {
            ចំនួនគត់ rhs = parseEquality();
            v       = (v && rhs);
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseEquality() {
        ចំនួនគត់ v = parseRelational();
        សម្រាប់ (;;) {
            ប្រសិនបើ (acceptOp("==")) {
                ចំនួនគត់ rhs = parseRelational();
                v       = (v == rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp("!=")) {
                ចំនួនគត់ rhs = parseRelational();
                v       = (v != rhs);
            } បើមិនដូច្នេះ {
                bha_nga;
            }
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseRelational() {
        ចំនួនគត់ v = parseShift();
        សម្រាប់ (;;) {
            ប្រសិនបើ (acceptOp("<")) {
                ចំនួនគត់ rhs = parseShift();
                v       = (v < rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp(">")) {
                ចំនួនគត់ rhs = parseShift();
                v       = (v > rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp("<=")) {
                ចំនួនគត់ rhs = parseShift();
                v       = (v <= rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp(">=")) {
                ចំនួនគត់ rhs = parseShift();
                v       = (v >= rhs);
            } បើមិនដូច្នេះ {
                bha_nga;
            }
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseShift() {
        ចំនួនគត់ v = parseAdd();
        សម្រាប់ (;;) {
            ប្រសិនបើ (acceptOp("<<")) {
                ចំនួនគត់ rhs = parseAdd();
                v       = (v << rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp(">>")) {
                ចំនួនគត់ rhs = parseAdd();
                v       = (v >> rhs);
            } បើមិនដូច្នេះ {
                bha_nga;
            }
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseAdd() {
        ចំនួនគត់ v = parseMult();
        សម្រាប់ (;;) {
            ប្រសិនបើ (acceptOp("+")) {
                ចំនួនគត់ rhs = parseMult();
                v       = (v + rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp("-")) {
                ចំនួនគត់ rhs = parseMult();
                v       = (v - rhs);
            } បើមិនដូច្នេះ {
                bha_nga;
            }
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseMult() {
        ចំនួនគត់ v = parseUnary();
        សម្រាប់ (;;) {
            ប្រសិនបើ (acceptOp("*")) {
                ចំនួនគត់ rhs = parseUnary();
                v       = (v * rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp("/")) {
                ចំនួនគត់ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } បើមិនដូច្នេះ ប្រសិនបើ (acceptOp("%")) {
                ចំនួនគត់ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } បើមិនដូច្នេះ {
                bha_nga;
            }
        }
        ត្រឡប់ v;
    }

    ចំនួនគត់ parseUnary() {
        ប្រសិនបើ (acceptOp("!")) {
            ត្រឡប់ !parseUnary();
        }
        ប្រសិនបើ (acceptOp("-")) {
            ត្រឡប់ -parseUnary();
        }
        ប្រសិនបើ (acceptOp("+")) {
            ត្រឡប់ +parseUnary();
        }
        ត្រឡប់ parsePrimary();
    }

    ចំនួនគត់ parsePrimary() {
        // '(' expr ')'
        ប្រសិនបើ (acceptKind(ExprLexer::LPAREN)) {
            ចំនួនគត់ v = parse();
            ប្រសិនបើ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ត្រឡប់ v;
        }

        // number
        ប្រសិនបើ (tok.kind == ExprLexer::NUMBER) {
            ចំនួនគត់ v = std::stoi(tok.text);
            advance();
            ត្រឡប់ v;
        }

        // defined(identifier)
        ប្រសិនបើ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ប្រសិនបើ (acceptKind(ExprLexer::LPAREN)) {
                ប្រសិនបើ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ប្រសិនបើ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ត្រឡប់ macros.count(name) ? 1 : 0;
            } បើមិនដូច្នេះ {
                // defined NAME
                ប្រសិនបើ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ត្រឡប់ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ប្រសិនបើ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ប្រសិនបើ (it == macros.end()) {
                ត្រឡប់ 0;
            }
            ប្រសិនបើ (it->second.empty()) {
                ត្រឡប់ 1;
            }
            ត្រឡប់ evalMacroExpression(name, it->second);
        }

        // unexpected
        ត្រឡប់ 0;
    }

    ចំនួនគត់ evalMacroExpression(const std::string & name, const std::string & value) {
        ប្រសិនបើ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        ចំនួនគត់        v = ep.parse();
        visiting.erase(name);
        ត្រឡប់ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ប្រសិនបើ (opts_.include_path.empty()) {
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
        ត្រឡប់ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ត្រឡប់ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ត្រឡប់ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ត្រឡប់ result;
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
        សម្រាប់ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ប្រសិនបើ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } បើមិនដូច្នេះ {
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

        សម្រាប់ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        សម្រាប់ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ប្រសិនបើ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } បើមិនដូច្នេះ {
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
        ប្រសិនបើ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ត្រឡប់ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ប្រសិនបើ (cond.empty()) {
            ត្រឡប់ true;
        }
        ត្រឡប់ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ប្រសិនបើ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ត្រឡប់ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ត្រឡប់ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ខណៈពេល (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ប្រសិនបើ (!t.empty() && t[0] == '#') {
                ខណៈពេល (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ប្រសិនបើ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ប្រសិនបើ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ប្រសិនបើ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } បើមិនដូច្នេះ {
                ប្រសិនបើ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } បើមិនដូច្នេះ ប្រសិនបើ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ប្រសិនបើ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ត្រឡប់ out.str();
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

        ប្រសិនបើ (cmd == "include") {
            ប្រសិនបើ (mode == DirectiveMode::All && !condActive(cond)) {
                ត្រឡប់ true;
            }
            std::string file;
            iss >> file;
            ប្រសិនបើ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ត្រឡប់ true;
        }

        ប្រសិនបើ (mode == DirectiveMode::IncludesOnly) {
            ត្រឡប់ false;
        }

        ប្រសិនបើ (cmd == "define") {
            ប្រសិនបើ (!condActive(cond)) {
                ត្រឡប់ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ប្រសិនបើ (predefined_macros.count(name)) {
                ត្រឡប់ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "undef") {
            ប្រសិនបើ (!condActive(cond)) {
                ត្រឡប់ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ប្រសិនបើ (predefined_macros.count(name)) {
                ត្រឡប់ true;
            }
            macros.erase(name);
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ប្រសិនបើ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ប្រសិនបើ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ប្រសិនបើ (!c.parent_active) {
                c.active = false;
                ត្រឡប់ true;
            }

            ប្រសិនបើ (c.taken) {
                c.active = false;
                ត្រឡប់ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ប្រសិនបើ (v) {
                c.taken = true;
            }
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "else") {
            ប្រសិនបើ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ប្រសិនបើ (!c.parent_active) {
                c.active = false;
                ត្រឡប់ true;
            }
            ប្រសិនបើ (c.taken) {
                c.active = false;
            } បើមិនដូច្នេះ {
                c.active = true;
                c.taken  = true;
            }
            ត្រឡប់ true;
        }

        ប្រសិនបើ (cmd == "endif") {
            ប្រសិនបើ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ត្រឡប់ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
