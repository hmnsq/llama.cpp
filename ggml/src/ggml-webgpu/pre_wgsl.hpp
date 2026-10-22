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
    ၶၢဝ်း (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ၶၢဝ်း (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ပွၵ်ႈ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ပွၵ်ႈ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ပွၵ်ႈ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ၶၢဝ်း (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ပွၵ်ႈ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ၶၢဝ်း (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ပေႃး (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ပေႃး (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ပေႃး (it == macros.end()) {
        visiting.erase(name);
        ပွၵ်ႈ name;
    }

    const std::string & value = it->second;
    ပေႃး (value.empty()) {
        visiting.erase(name);
        ပွၵ်ႈ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ပွၵ်ႈ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ၶၢဝ်း (i < line.size()) {
        ပေႃး (isIdentChar(line[i])) {
            size_t start = i;
            ၶၢဝ်း (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ပေႃး (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ပေႃးဢမ်ႇ {
                result += token;
            }
        } ပေႃးဢမ်ႇ {
            result += line[i];
            i++;
        }
    }

    ပွၵ်ႈ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ပွၵ်ႈ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ပေႃး (pos >= src.size()) {
            ပွၵ်ႈ { END, "" };
        }

        var_na c = src[pos];

        // number
        ပေႃး (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ၶၢဝ်း (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ပွၵ်ႈ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ပေႃး (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ၶၢဝ်း (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ပွၵ်ႈ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ပေႃး (c == '(') {
            pos++;
            ပွၵ်ႈ { LPAREN, "(" };
        }
        ပေႃး (c == ')') {
            pos++;
            ပွၵ်ႈ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            ပေႃး (src.substr(pos, 2) == op) {
                pos += 2;
                ပွၵ်ႈ { OP, std::string(op) };
            }
        }

        // single-char operators
        ပေႃး (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ပွၵ်ႈ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ပွၵ်ႈ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ၶၢဝ်း (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ပွၵ်ႈ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ပေႃး (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ပွၵ်ႈ true;
        }
        ပွၵ်ႈ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ပေႃး (tok.kind == k) {
            advance();
            ပွၵ်ႈ true;
        }
        ပွၵ်ႈ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ၶၢဝ်း (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ၶၢဝ်း (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        for (;;) {
            ပေႃး (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ပေႃးဢမ်ႇ {
                bha_nga;
            }
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        for (;;) {
            ပေႃး (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ပေႃးဢမ်ႇ {
                bha_nga;
            }
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        for (;;) {
            ပေႃး (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ပေႃးဢမ်ႇ {
                bha_nga;
            }
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        for (;;) {
            ပေႃး (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ပေႃးဢမ်ႇ {
                bha_nga;
            }
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        for (;;) {
            ပေႃး (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ပေႃးဢမ်ႇ ပေႃး (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ပေႃးဢမ်ႇ {
                bha_nga;
            }
        }
        ပွၵ်ႈ v;
    }

    puur_naa_nka parseUnary() {
        ပေႃး (acceptOp("!")) {
            ပွၵ်ႈ !parseUnary();
        }
        ပေႃး (acceptOp("-")) {
            ပွၵ်ႈ -parseUnary();
        }
        ပေႃး (acceptOp("+")) {
            ပွၵ်ႈ +parseUnary();
        }
        ပွၵ်ႈ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ပေႃး (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ပေႃး (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ပွၵ်ႈ v;
        }

        // number
        ပေႃး (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ပွၵ်ႈ v;
        }

        // defined(identifier)
        ပေႃး (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ပေႃး (acceptKind(ExprLexer::LPAREN)) {
                ပေႃး (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ပေႃး (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ပွၵ်ႈ macros.count(name) ? 1 : 0;
            } ပေႃးဢမ်ႇ {
                // defined NAME
                ပေႃး (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ပွၵ်ႈ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ပေႃး (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ပေႃး (it == macros.end()) {
                ပွၵ်ႈ 0;
            }
            ပေႃး (it->second.empty()) {
                ပွၵ်ႈ 1;
            }
            ပွၵ်ႈ evalMacroExpression(name, it->second);
        }

        // unexpected
        ပွၵ်ႈ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ပေႃး (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ပွၵ်ႈ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ပေႃး (opts_.include_path.empty()) {
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
        ပွၵ်ႈ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ပွၵ်ႈ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ပွၵ်ႈ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ပွၵ်ႈ result;
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
        for (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ပေႃး (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ပေႃးဢမ်ႇ {
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

        for (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        for (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ပေႃး (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ပေႃးဢမ်ႇ {
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
        ပေႃး (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ပွၵ်ႈ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ပေႃး (cond.empty()) {
            ပွၵ်ႈ true;
        }
        ပွၵ်ႈ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ပေႃး (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ပွၵ်ႈ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ပွၵ်ႈ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ၶၢဝ်း (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ပေႃး (!t.empty() && t[0] == '#') {
                ၶၢဝ်း (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ပေႃး (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ပေႃး (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ပေႃး (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ပေႃးဢမ်ႇ {
                ပေႃး (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ပေႃးဢမ်ႇ ပေႃး (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ပေႃး (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ပွၵ်ႈ out.str();
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

        ပေႃး (cmd == "include") {
            ပေႃး (mode == DirectiveMode::All && !condActive(cond)) {
                ပွၵ်ႈ true;
            }
            std::string file;
            iss >> file;
            ပေႃး (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ပွၵ်ႈ true;
        }

        ပေႃး (mode == DirectiveMode::IncludesOnly) {
            ပွၵ်ႈ false;
        }

        ပေႃး (cmd == "define") {
            ပေႃး (!condActive(cond)) {
                ပွၵ်ႈ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ပေႃး (predefined_macros.count(name)) {
                ပွၵ်ႈ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "undef") {
            ပေႃး (!condActive(cond)) {
                ပွၵ်ႈ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ပေႃး (predefined_macros.count(name)) {
                ပွၵ်ႈ true;
            }
            macros.erase(name);
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ပေႃး (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "elif") {
            std::string expr = trim_value(iss);

            ပေႃး (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ပေႃး (!c.parent_active) {
                c.active = false;
                ပွၵ်ႈ true;
            }

            ပေႃး (c.taken) {
                c.active = false;
                ပွၵ်ႈ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ပေႃး (v) {
                c.taken = true;
            }
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "else") {
            ပေႃး (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ပေႃး (!c.parent_active) {
                c.active = false;
                ပွၵ်ႈ true;
            }
            ပေႃး (c.taken) {
                c.active = false;
            } ပေႃးဢမ်ႇ {
                c.active = true;
                c.taken  = true;
            }
            ပွၵ်ႈ true;
        }

        ပေႃး (cmd == "endif") {
            ပေႃး (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ပွၵ်ႈ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
