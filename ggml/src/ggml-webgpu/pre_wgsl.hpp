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
    ᮞᮨᮓᮨᮜ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ᮞᮨᮓᮨᮜ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ᮝᮜᮤ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ᮝᮜᮤ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ᮝᮜᮤ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ᮞᮨᮓᮨᮜ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᮝᮜᮤ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ᮞᮨᮓᮨᮜ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᮚᮦᮔ᮪ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ᮚᮦᮔ᮪ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ᮚᮦᮔ᮪ (it == macros.end()) {
        visiting.erase(name);
        ᮝᮜᮤ name;
    }

    const std::string & value = it->second;
    ᮚᮦᮔ᮪ (value.empty()) {
        visiting.erase(name);
        ᮝᮜᮤ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ᮝᮜᮤ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ᮞᮨᮓᮨᮜ (i < line.size()) {
        ᮚᮦᮔ᮪ (isIdentChar(line[i])) {
            size_t start = i;
            ᮞᮨᮓᮨᮜ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ᮚᮦᮔ᮪ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ᮜᮤᮚ {
                result += token;
            }
        } ᮜᮤᮚ {
            result += line[i];
            i++;
        }
    }

    ᮝᮜᮤ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ᮝᮜᮤ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ᮚᮦᮔ᮪ (pos >= src.size()) {
            ᮝᮜᮤ { END, "" };
        }

        var_na c = src[pos];

        // number
        ᮚᮦᮔ᮪ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ᮞᮨᮓᮨᮜ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ᮝᮜᮤ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ᮚᮦᮔ᮪ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ᮞᮨᮓᮨᮜ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ᮝᮜᮤ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ᮚᮦᮔ᮪ (c == '(') {
            pos++;
            ᮝᮜᮤ { LPAREN, "(" };
        }
        ᮚᮦᮔ᮪ (c == ')') {
            pos++;
            ᮝᮜᮤ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            ᮚᮦᮔ᮪ (src.substr(pos, 2) == op) {
                pos += 2;
                ᮝᮜᮤ { OP, std::string(op) };
            }
        }

        // single-char operators
        ᮚᮦᮔ᮪ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ᮝᮜᮤ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ᮝᮜᮤ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ᮞᮨᮓᮨᮜ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ᮝᮜᮤ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ᮚᮦᮔ᮪ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ᮝᮜᮤ true;
        }
        ᮝᮜᮤ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ᮚᮦᮔ᮪ (tok.kind == k) {
            advance();
            ᮝᮜᮤ true;
        }
        ᮝᮜᮤ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ᮞᮨᮓᮨᮜ (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ᮞᮨᮓᮨᮜ (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        for (;;) {
            ᮚᮦᮔ᮪ (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ᮜᮤᮚ {
                bha_nga;
            }
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        for (;;) {
            ᮚᮦᮔ᮪ (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ᮜᮤᮚ {
                bha_nga;
            }
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        for (;;) {
            ᮚᮦᮔ᮪ (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ᮜᮤᮚ {
                bha_nga;
            }
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        for (;;) {
            ᮚᮦᮔ᮪ (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ᮜᮤᮚ {
                bha_nga;
            }
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        for (;;) {
            ᮚᮦᮔ᮪ (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ᮜᮤᮚ ᮚᮦᮔ᮪ (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ᮜᮤᮚ {
                bha_nga;
            }
        }
        ᮝᮜᮤ v;
    }

    puur_naa_nka parseUnary() {
        ᮚᮦᮔ᮪ (acceptOp("!")) {
            ᮝᮜᮤ !parseUnary();
        }
        ᮚᮦᮔ᮪ (acceptOp("-")) {
            ᮝᮜᮤ -parseUnary();
        }
        ᮚᮦᮔ᮪ (acceptOp("+")) {
            ᮝᮜᮤ +parseUnary();
        }
        ᮝᮜᮤ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ᮚᮦᮔ᮪ (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ᮚᮦᮔ᮪ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ᮝᮜᮤ v;
        }

        // number
        ᮚᮦᮔ᮪ (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ᮝᮜᮤ v;
        }

        // defined(identifier)
        ᮚᮦᮔ᮪ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ᮚᮦᮔ᮪ (acceptKind(ExprLexer::LPAREN)) {
                ᮚᮦᮔ᮪ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ᮚᮦᮔ᮪ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ᮝᮜᮤ macros.count(name) ? 1 : 0;
            } ᮜᮤᮚ {
                // defined NAME
                ᮚᮦᮔ᮪ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ᮝᮜᮤ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ᮚᮦᮔ᮪ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ᮚᮦᮔ᮪ (it == macros.end()) {
                ᮝᮜᮤ 0;
            }
            ᮚᮦᮔ᮪ (it->second.empty()) {
                ᮝᮜᮤ 1;
            }
            ᮝᮜᮤ evalMacroExpression(name, it->second);
        }

        // unexpected
        ᮝᮜᮤ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ᮚᮦᮔ᮪ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ᮝᮜᮤ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ᮚᮦᮔ᮪ (opts_.include_path.empty()) {
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
        ᮝᮜᮤ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ᮝᮜᮤ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᮝᮜᮤ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᮝᮜᮤ result;
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
            ᮚᮦᮔ᮪ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ᮜᮤᮚ {
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
            ᮚᮦᮔ᮪ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ᮜᮤᮚ {
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
        ᮚᮦᮔ᮪ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ᮝᮜᮤ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ᮚᮦᮔ᮪ (cond.empty()) {
            ᮝᮜᮤ true;
        }
        ᮝᮜᮤ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ᮚᮦᮔ᮪ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ᮝᮜᮤ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ᮝᮜᮤ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ᮞᮨᮓᮨᮜ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ᮚᮦᮔ᮪ (!t.empty() && t[0] == '#') {
                ᮞᮨᮓᮨᮜ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ᮚᮦᮔ᮪ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ᮚᮦᮔ᮪ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ᮚᮦᮔ᮪ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ᮜᮤᮚ {
                ᮚᮦᮔ᮪ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ᮜᮤᮚ ᮚᮦᮔ᮪ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ᮚᮦᮔ᮪ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ᮝᮜᮤ out.str();
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

        ᮚᮦᮔ᮪ (cmd == "include") {
            ᮚᮦᮔ᮪ (mode == DirectiveMode::All && !condActive(cond)) {
                ᮝᮜᮤ true;
            }
            std::string file;
            iss >> file;
            ᮚᮦᮔ᮪ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (mode == DirectiveMode::IncludesOnly) {
            ᮝᮜᮤ false;
        }

        ᮚᮦᮔ᮪ (cmd == "define") {
            ᮚᮦᮔ᮪ (!condActive(cond)) {
                ᮝᮜᮤ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ᮚᮦᮔ᮪ (predefined_macros.count(name)) {
                ᮝᮜᮤ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "undef") {
            ᮚᮦᮔ᮪ (!condActive(cond)) {
                ᮝᮜᮤ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ᮚᮦᮔ᮪ (predefined_macros.count(name)) {
                ᮝᮜᮤ true;
            }
            macros.erase(name);
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ᮚᮦᮔ᮪ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ᮚᮦᮔ᮪ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ᮚᮦᮔ᮪ (!c.parent_active) {
                c.active = false;
                ᮝᮜᮤ true;
            }

            ᮚᮦᮔ᮪ (c.taken) {
                c.active = false;
                ᮝᮜᮤ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ᮚᮦᮔ᮪ (v) {
                c.taken = true;
            }
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "else") {
            ᮚᮦᮔ᮪ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ᮚᮦᮔ᮪ (!c.parent_active) {
                c.active = false;
                ᮝᮜᮤ true;
            }
            ᮚᮦᮔ᮪ (c.taken) {
                c.active = false;
            } ᮜᮤᮚ {
                c.active = true;
                c.taken  = true;
            }
            ᮝᮜᮤ true;
        }

        ᮚᮦᮔ᮪ (cmd == "endif") {
            ᮚᮦᮔ᮪ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ᮝᮜᮤ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
