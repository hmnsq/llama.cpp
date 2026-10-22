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
    ᤀᤠ᤺ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ᤀᤠ᤺ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ᤀᤡᤴ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ᤀᤡᤴ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ᤀᤡᤴ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ᤀᤠ᤺ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᤀᤡᤴ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ᤀᤠ᤺ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᤀᤡ᤺ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ᤀᤡ᤺ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ᤀᤡ᤺ (it == macros.end()) {
        visiting.erase(name);
        ᤀᤡᤴ name;
    }

    const std::string & value = it->second;
    ᤀᤡ᤺ (value.empty()) {
        visiting.erase(name);
        ᤀᤡᤴ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ᤀᤡᤴ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ᤀᤠ᤺ (i < line.size()) {
        ᤀᤡ᤺ (isIdentChar(line[i])) {
            size_t start = i;
            ᤀᤠ᤺ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ᤀᤡ᤺ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ᤀᤡ᤺ᤆᤡ {
                result += token;
            }
        } ᤀᤡ᤺ᤆᤡ {
            result += line[i];
            i++;
        }
    }

    ᤀᤡᤴ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ᤀᤡᤴ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ᤀᤡ᤺ (pos >= src.size()) {
            ᤀᤡᤴ { END, "" };
        }

        var_na c = src[pos];

        // number
        ᤀᤡ᤺ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ᤀᤠ᤺ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ᤀᤡᤴ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ᤀᤡ᤺ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ᤀᤠ᤺ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ᤀᤡᤴ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ᤀᤡ᤺ (c == '(') {
            pos++;
            ᤀᤡᤴ { LPAREN, "(" };
        }
        ᤀᤡ᤺ (c == ')') {
            pos++;
            ᤀᤡᤴ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            ᤀᤡ᤺ (src.substr(pos, 2) == op) {
                pos += 2;
                ᤀᤡᤴ { OP, std::string(op) };
            }
        }

        // single-char operators
        ᤀᤡ᤺ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ᤀᤡᤴ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ᤀᤡᤴ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ᤀᤠ᤺ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ᤀᤡᤴ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ᤀᤡ᤺ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ᤀᤡᤴ true;
        }
        ᤀᤡᤴ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ᤀᤡ᤺ (tok.kind == k) {
            advance();
            ᤀᤡᤴ true;
        }
        ᤀᤡᤴ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ᤀᤠ᤺ (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ᤀᤠ᤺ (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        for (;;) {
            ᤀᤡ᤺ (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ᤀᤡ᤺ᤆᤡ {
                bha_nga;
            }
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        for (;;) {
            ᤀᤡ᤺ (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ᤀᤡ᤺ᤆᤡ {
                bha_nga;
            }
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        for (;;) {
            ᤀᤡ᤺ (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ᤀᤡ᤺ᤆᤡ {
                bha_nga;
            }
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        for (;;) {
            ᤀᤡ᤺ (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ᤀᤡ᤺ᤆᤡ {
                bha_nga;
            }
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        for (;;) {
            ᤀᤡ᤺ (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ᤀᤡ᤺ᤆᤡ {
                bha_nga;
            }
        }
        ᤀᤡᤴ v;
    }

    puur_naa_nka parseUnary() {
        ᤀᤡ᤺ (acceptOp("!")) {
            ᤀᤡᤴ !parseUnary();
        }
        ᤀᤡ᤺ (acceptOp("-")) {
            ᤀᤡᤴ -parseUnary();
        }
        ᤀᤡ᤺ (acceptOp("+")) {
            ᤀᤡᤴ +parseUnary();
        }
        ᤀᤡᤴ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ᤀᤡ᤺ (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ᤀᤡ᤺ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ᤀᤡᤴ v;
        }

        // number
        ᤀᤡ᤺ (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ᤀᤡᤴ v;
        }

        // defined(identifier)
        ᤀᤡ᤺ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ᤀᤡ᤺ (acceptKind(ExprLexer::LPAREN)) {
                ᤀᤡ᤺ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ᤀᤡ᤺ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ᤀᤡᤴ macros.count(name) ? 1 : 0;
            } ᤀᤡ᤺ᤆᤡ {
                // defined NAME
                ᤀᤡ᤺ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ᤀᤡᤴ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ᤀᤡ᤺ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ᤀᤡ᤺ (it == macros.end()) {
                ᤀᤡᤴ 0;
            }
            ᤀᤡ᤺ (it->second.empty()) {
                ᤀᤡᤴ 1;
            }
            ᤀᤡᤴ evalMacroExpression(name, it->second);
        }

        // unexpected
        ᤀᤡᤴ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ᤀᤡ᤺ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ᤀᤡᤴ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ᤀᤡ᤺ (opts_.include_path.empty()) {
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
        ᤀᤡᤴ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ᤀᤡᤴ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᤀᤡᤴ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᤀᤡᤴ result;
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
            ᤀᤡ᤺ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ᤀᤡ᤺ᤆᤡ {
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
            ᤀᤡ᤺ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ᤀᤡ᤺ᤆᤡ {
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
        ᤀᤡ᤺ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ᤀᤡᤴ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ᤀᤡ᤺ (cond.empty()) {
            ᤀᤡᤴ true;
        }
        ᤀᤡᤴ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ᤀᤡ᤺ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ᤀᤡᤴ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ᤀᤡᤴ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ᤀᤠ᤺ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ᤀᤡ᤺ (!t.empty() && t[0] == '#') {
                ᤀᤠ᤺ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ᤀᤡ᤺ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ᤀᤡ᤺ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ᤀᤡ᤺ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ᤀᤡ᤺ᤆᤡ {
                ᤀᤡ᤺ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ᤀᤡ᤺ᤆᤡ ᤀᤡ᤺ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ᤀᤡ᤺ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ᤀᤡᤴ out.str();
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

        ᤀᤡ᤺ (cmd == "include") {
            ᤀᤡ᤺ (mode == DirectiveMode::All && !condActive(cond)) {
                ᤀᤡᤴ true;
            }
            std::string file;
            iss >> file;
            ᤀᤡ᤺ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (mode == DirectiveMode::IncludesOnly) {
            ᤀᤡᤴ false;
        }

        ᤀᤡ᤺ (cmd == "define") {
            ᤀᤡ᤺ (!condActive(cond)) {
                ᤀᤡᤴ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ᤀᤡ᤺ (predefined_macros.count(name)) {
                ᤀᤡᤴ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "undef") {
            ᤀᤡ᤺ (!condActive(cond)) {
                ᤀᤡᤴ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ᤀᤡ᤺ (predefined_macros.count(name)) {
                ᤀᤡᤴ true;
            }
            macros.erase(name);
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ᤀᤡ᤺ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ᤀᤡ᤺ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ᤀᤡ᤺ (!c.parent_active) {
                c.active = false;
                ᤀᤡᤴ true;
            }

            ᤀᤡ᤺ (c.taken) {
                c.active = false;
                ᤀᤡᤴ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ᤀᤡ᤺ (v) {
                c.taken = true;
            }
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "else") {
            ᤀᤡ᤺ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ᤀᤡ᤺ (!c.parent_active) {
                c.active = false;
                ᤀᤡᤴ true;
            }
            ᤀᤡ᤺ (c.taken) {
                c.active = false;
            } ᤀᤡ᤺ᤆᤡ {
                c.active = true;
                c.taken  = true;
            }
            ᤀᤡᤴ true;
        }

        ᤀᤡ᤺ (cmd == "endif") {
            ᤀᤡ᤺ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ᤀᤡᤴ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
