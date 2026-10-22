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
    ᯀᯬᯝ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ᯀᯬᯝ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ᯀᯮᯞ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ᯀᯮᯞ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ᯀᯮᯞ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ᯀᯬᯝ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᯀᯮᯞ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ᯀᯬᯝ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᯀᯮᯝ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ᯀᯮᯝ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ᯀᯮᯝ (it == macros.end()) {
        visiting.erase(name);
        ᯀᯮᯞ name;
    }

    const std::string & value = it->second;
    ᯀᯮᯝ (value.empty()) {
        visiting.erase(name);
        ᯀᯮᯞ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ᯀᯮᯞ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ᯀᯬᯝ (i < line.size()) {
        ᯀᯮᯝ (isIdentChar(line[i])) {
            size_t start = i;
            ᯀᯬᯝ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ᯀᯮᯝ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ᯀᯮᯝᯂᯪ {
                result += token;
            }
        } ᯀᯮᯝᯂᯪ {
            result += line[i];
            i++;
        }
    }

    ᯀᯮᯞ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ᯀᯮᯞ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ᯀᯮᯝ (pos >= src.size()) {
            ᯀᯮᯞ { END, "" };
        }

        var_na c = src[pos];

        // number
        ᯀᯮᯝ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ᯀᯬᯝ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ᯀᯮᯞ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ᯀᯮᯝ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ᯀᯬᯝ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ᯀᯮᯞ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ᯀᯮᯝ (c == '(') {
            pos++;
            ᯀᯮᯞ { LPAREN, "(" };
        }
        ᯀᯮᯝ (c == ')') {
            pos++;
            ᯀᯮᯞ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            ᯀᯮᯝ (src.substr(pos, 2) == op) {
                pos += 2;
                ᯀᯮᯞ { OP, std::string(op) };
            }
        }

        // single-char operators
        ᯀᯮᯝ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ᯀᯮᯞ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ᯀᯮᯞ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ᯀᯬᯝ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ᯀᯮᯞ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ᯀᯮᯝ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ᯀᯮᯞ true;
        }
        ᯀᯮᯞ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ᯀᯮᯝ (tok.kind == k) {
            advance();
            ᯀᯮᯞ true;
        }
        ᯀᯮᯞ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ᯀᯬᯝ (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ᯀᯬᯝ (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        for (;;) {
            ᯀᯮᯝ (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ᯀᯮᯝᯂᯪ {
                bha_nga;
            }
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        for (;;) {
            ᯀᯮᯝ (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ᯀᯮᯝᯂᯪ {
                bha_nga;
            }
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        for (;;) {
            ᯀᯮᯝ (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ᯀᯮᯝᯂᯪ {
                bha_nga;
            }
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        for (;;) {
            ᯀᯮᯝ (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ᯀᯮᯝᯂᯪ {
                bha_nga;
            }
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        for (;;) {
            ᯀᯮᯝ (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ᯀᯮᯝᯂᯪ {
                bha_nga;
            }
        }
        ᯀᯮᯞ v;
    }

    puur_naa_nka parseUnary() {
        ᯀᯮᯝ (acceptOp("!")) {
            ᯀᯮᯞ !parseUnary();
        }
        ᯀᯮᯝ (acceptOp("-")) {
            ᯀᯮᯞ -parseUnary();
        }
        ᯀᯮᯝ (acceptOp("+")) {
            ᯀᯮᯞ +parseUnary();
        }
        ᯀᯮᯞ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ᯀᯮᯝ (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ᯀᯮᯝ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ᯀᯮᯞ v;
        }

        // number
        ᯀᯮᯝ (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ᯀᯮᯞ v;
        }

        // defined(identifier)
        ᯀᯮᯝ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ᯀᯮᯝ (acceptKind(ExprLexer::LPAREN)) {
                ᯀᯮᯝ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ᯀᯮᯝ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ᯀᯮᯞ macros.count(name) ? 1 : 0;
            } ᯀᯮᯝᯂᯪ {
                // defined NAME
                ᯀᯮᯝ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ᯀᯮᯞ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ᯀᯮᯝ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ᯀᯮᯝ (it == macros.end()) {
                ᯀᯮᯞ 0;
            }
            ᯀᯮᯝ (it->second.empty()) {
                ᯀᯮᯞ 1;
            }
            ᯀᯮᯞ evalMacroExpression(name, it->second);
        }

        // unexpected
        ᯀᯮᯞ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ᯀᯮᯝ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ᯀᯮᯞ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ᯀᯮᯝ (opts_.include_path.empty()) {
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
        ᯀᯮᯞ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ᯀᯮᯞ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᯀᯮᯞ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᯀᯮᯞ result;
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
            ᯀᯮᯝ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ᯀᯮᯝᯂᯪ {
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
            ᯀᯮᯝ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ᯀᯮᯝᯂᯪ {
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
        ᯀᯮᯝ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ᯀᯮᯞ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ᯀᯮᯝ (cond.empty()) {
            ᯀᯮᯞ true;
        }
        ᯀᯮᯞ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ᯀᯮᯝ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ᯀᯮᯞ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ᯀᯮᯞ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ᯀᯬᯝ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ᯀᯮᯝ (!t.empty() && t[0] == '#') {
                ᯀᯬᯝ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ᯀᯮᯝ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ᯀᯮᯝ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ᯀᯮᯝ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ᯀᯮᯝᯂᯪ {
                ᯀᯮᯝ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ᯀᯮᯝᯂᯪ ᯀᯮᯝ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ᯀᯮᯝ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ᯀᯮᯞ out.str();
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

        ᯀᯮᯝ (cmd == "include") {
            ᯀᯮᯝ (mode == DirectiveMode::All && !condActive(cond)) {
                ᯀᯮᯞ true;
            }
            std::string file;
            iss >> file;
            ᯀᯮᯝ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (mode == DirectiveMode::IncludesOnly) {
            ᯀᯮᯞ false;
        }

        ᯀᯮᯝ (cmd == "define") {
            ᯀᯮᯝ (!condActive(cond)) {
                ᯀᯮᯞ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ᯀᯮᯝ (predefined_macros.count(name)) {
                ᯀᯮᯞ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "undef") {
            ᯀᯮᯝ (!condActive(cond)) {
                ᯀᯮᯞ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ᯀᯮᯝ (predefined_macros.count(name)) {
                ᯀᯮᯞ true;
            }
            macros.erase(name);
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ᯀᯮᯝ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ᯀᯮᯝ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ᯀᯮᯝ (!c.parent_active) {
                c.active = false;
                ᯀᯮᯞ true;
            }

            ᯀᯮᯝ (c.taken) {
                c.active = false;
                ᯀᯮᯞ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ᯀᯮᯝ (v) {
                c.taken = true;
            }
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "else") {
            ᯀᯮᯝ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ᯀᯮᯝ (!c.parent_active) {
                c.active = false;
                ᯀᯮᯞ true;
            }
            ᯀᯮᯝ (c.taken) {
                c.active = false;
            } ᯀᯮᯝᯂᯪ {
                c.active = true;
                c.taken  = true;
            }
            ᯀᯮᯞ true;
        }

        ᯀᯮᯝ (cmd == "endif") {
            ᯀᯮᯝ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ᯀᯮᯞ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
