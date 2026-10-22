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
    ໃນຂະນະທີ່ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ໃນຂະນະທີ່ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ສົ່ງຄືນ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ສົ່ງຄືນ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ສົ່ງຄືນ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ໃນຂະນະທີ່ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ສົ່ງຄືນ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ໃນຂະນະທີ່ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ຖ້າ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ຖ້າ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ຖ້າ (it == macros.end()) {
        visiting.erase(name);
        ສົ່ງຄືນ name;
    }

    const std::string & value = it->second;
    ຖ້າ (value.empty()) {
        visiting.erase(name);
        ສົ່ງຄືນ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ສົ່ງຄືນ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ໃນຂະນະທີ່ (i < line.size()) {
        ຖ້າ (isIdentChar(line[i])) {
            size_t start = i;
            ໃນຂະນະທີ່ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ຖ້າ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ບໍ່ດັ່ງນັ້ນ {
                result += token;
            }
        } ບໍ່ດັ່ງນັ້ນ {
            result += line[i];
            i++;
        }
    }

    ສົ່ງຄືນ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ສົ່ງຄືນ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ຖ້າ (pos >= src.size()) {
            ສົ່ງຄືນ { END, "" };
        }

        var_na c = src[pos];

        // number
        ຖ້າ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ໃນຂະນະທີ່ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ສົ່ງຄືນ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ຖ້າ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ໃນຂະນະທີ່ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ສົ່ງຄືນ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ຖ້າ (c == '(') {
            pos++;
            ສົ່ງຄືນ { LPAREN, "(" };
        }
        ຖ້າ (c == ')') {
            pos++;
            ສົ່ງຄືນ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ສຳລັບ (auto op : two_ops) {
            ຖ້າ (src.substr(pos, 2) == op) {
                pos += 2;
                ສົ່ງຄືນ { OP, std::string(op) };
            }
        }

        // single-char operators
        ຖ້າ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ສົ່ງຄືນ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ສົ່ງຄືນ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ໃນຂະນະທີ່ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    ຈຳນວນເຕັມ parse() { ສົ່ງຄືນ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ຖ້າ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ສົ່ງຄືນ true;
        }
        ສົ່ງຄືນ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ຖ້າ (tok.kind == k) {
            advance();
            ສົ່ງຄືນ true;
        }
        ສົ່ງຄືນ false;
    }

    ຈຳນວນເຕັມ parseLogicalOr() {
        ຈຳນວນເຕັມ v = parseLogicalAnd();
        ໃນຂະນະທີ່ (acceptOp("||")) {
            ຈຳນວນເຕັມ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseLogicalAnd() {
        ຈຳນວນເຕັມ v = parseEquality();
        ໃນຂະນະທີ່ (acceptOp("&&")) {
            ຈຳນວນເຕັມ rhs = parseEquality();
            v       = (v && rhs);
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseEquality() {
        ຈຳນວນເຕັມ v = parseRelational();
        ສຳລັບ (;;) {
            ຖ້າ (acceptOp("==")) {
                ຈຳນວນເຕັມ rhs = parseRelational();
                v       = (v == rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp("!=")) {
                ຈຳນວນເຕັມ rhs = parseRelational();
                v       = (v != rhs);
            } ບໍ່ດັ່ງນັ້ນ {
                bha_nga;
            }
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseRelational() {
        ຈຳນວນເຕັມ v = parseShift();
        ສຳລັບ (;;) {
            ຖ້າ (acceptOp("<")) {
                ຈຳນວນເຕັມ rhs = parseShift();
                v       = (v < rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp(">")) {
                ຈຳນວນເຕັມ rhs = parseShift();
                v       = (v > rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp("<=")) {
                ຈຳນວນເຕັມ rhs = parseShift();
                v       = (v <= rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp(">=")) {
                ຈຳນວນເຕັມ rhs = parseShift();
                v       = (v >= rhs);
            } ບໍ່ດັ່ງນັ້ນ {
                bha_nga;
            }
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseShift() {
        ຈຳນວນເຕັມ v = parseAdd();
        ສຳລັບ (;;) {
            ຖ້າ (acceptOp("<<")) {
                ຈຳນວນເຕັມ rhs = parseAdd();
                v       = (v << rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp(">>")) {
                ຈຳນວນເຕັມ rhs = parseAdd();
                v       = (v >> rhs);
            } ບໍ່ດັ່ງນັ້ນ {
                bha_nga;
            }
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseAdd() {
        ຈຳນວນເຕັມ v = parseMult();
        ສຳລັບ (;;) {
            ຖ້າ (acceptOp("+")) {
                ຈຳນວນເຕັມ rhs = parseMult();
                v       = (v + rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp("-")) {
                ຈຳນວນເຕັມ rhs = parseMult();
                v       = (v - rhs);
            } ບໍ່ດັ່ງນັ້ນ {
                bha_nga;
            }
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseMult() {
        ຈຳນວນເຕັມ v = parseUnary();
        ສຳລັບ (;;) {
            ຖ້າ (acceptOp("*")) {
                ຈຳນວນເຕັມ rhs = parseUnary();
                v       = (v * rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp("/")) {
                ຈຳນວນເຕັມ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ບໍ່ດັ່ງນັ້ນ ຖ້າ (acceptOp("%")) {
                ຈຳນວນເຕັມ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ບໍ່ດັ່ງນັ້ນ {
                bha_nga;
            }
        }
        ສົ່ງຄືນ v;
    }

    ຈຳນວນເຕັມ parseUnary() {
        ຖ້າ (acceptOp("!")) {
            ສົ່ງຄືນ !parseUnary();
        }
        ຖ້າ (acceptOp("-")) {
            ສົ່ງຄືນ -parseUnary();
        }
        ຖ້າ (acceptOp("+")) {
            ສົ່ງຄືນ +parseUnary();
        }
        ສົ່ງຄືນ parsePrimary();
    }

    ຈຳນວນເຕັມ parsePrimary() {
        // '(' expr ')'
        ຖ້າ (acceptKind(ExprLexer::LPAREN)) {
            ຈຳນວນເຕັມ v = parse();
            ຖ້າ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ສົ່ງຄືນ v;
        }

        // number
        ຖ້າ (tok.kind == ExprLexer::NUMBER) {
            ຈຳນວນເຕັມ v = std::stoi(tok.text);
            advance();
            ສົ່ງຄືນ v;
        }

        // defined(identifier)
        ຖ້າ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ຖ້າ (acceptKind(ExprLexer::LPAREN)) {
                ຖ້າ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ຖ້າ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ສົ່ງຄືນ macros.count(name) ? 1 : 0;
            } ບໍ່ດັ່ງນັ້ນ {
                // defined NAME
                ຖ້າ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ສົ່ງຄືນ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ຖ້າ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ຖ້າ (it == macros.end()) {
                ສົ່ງຄືນ 0;
            }
            ຖ້າ (it->second.empty()) {
                ສົ່ງຄືນ 1;
            }
            ສົ່ງຄືນ evalMacroExpression(name, it->second);
        }

        // unexpected
        ສົ່ງຄືນ 0;
    }

    ຈຳນວນເຕັມ evalMacroExpression(const std::string & name, const std::string & value) {
        ຖ້າ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        ຈຳນວນເຕັມ        v = ep.parse();
        visiting.erase(name);
        ສົ່ງຄືນ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ຖ້າ (opts_.include_path.empty()) {
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
        ສົ່ງຄືນ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ສົ່ງຄືນ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ສົ່ງຄືນ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ສົ່ງຄືນ result;
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
        ສຳລັບ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ຖ້າ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ບໍ່ດັ່ງນັ້ນ {
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

        ສຳລັບ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ສຳລັບ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ຖ້າ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ບໍ່ດັ່ງນັ້ນ {
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
        ຖ້າ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ສົ່ງຄືນ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ຖ້າ (cond.empty()) {
            ສົ່ງຄືນ true;
        }
        ສົ່ງຄືນ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ຖ້າ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ສົ່ງຄືນ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ສົ່ງຄືນ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ໃນຂະນະທີ່ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ຖ້າ (!t.empty() && t[0] == '#') {
                ໃນຂະນະທີ່ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ຖ້າ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ຖ້າ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ຖ້າ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ບໍ່ດັ່ງນັ້ນ {
                ຖ້າ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ບໍ່ດັ່ງນັ້ນ ຖ້າ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ຖ້າ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ສົ່ງຄືນ out.str();
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

        ຖ້າ (cmd == "include") {
            ຖ້າ (mode == DirectiveMode::All && !condActive(cond)) {
                ສົ່ງຄືນ true;
            }
            std::string file;
            iss >> file;
            ຖ້າ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ສົ່ງຄືນ true;
        }

        ຖ້າ (mode == DirectiveMode::IncludesOnly) {
            ສົ່ງຄືນ false;
        }

        ຖ້າ (cmd == "define") {
            ຖ້າ (!condActive(cond)) {
                ສົ່ງຄືນ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ຖ້າ (predefined_macros.count(name)) {
                ສົ່ງຄືນ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "undef") {
            ຖ້າ (!condActive(cond)) {
                ສົ່ງຄືນ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ຖ້າ (predefined_macros.count(name)) {
                ສົ່ງຄືນ true;
            }
            macros.erase(name);
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ຖ້າ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ຖ້າ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ຖ້າ (!c.parent_active) {
                c.active = false;
                ສົ່ງຄືນ true;
            }

            ຖ້າ (c.taken) {
                c.active = false;
                ສົ່ງຄືນ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ຖ້າ (v) {
                c.taken = true;
            }
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "else") {
            ຖ້າ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ຖ້າ (!c.parent_active) {
                c.active = false;
                ສົ່ງຄືນ true;
            }
            ຖ້າ (c.taken) {
                c.active = false;
            } ບໍ່ດັ່ງນັ້ນ {
                c.active = true;
                c.taken  = true;
            }
            ສົ່ງຄືນ true;
        }

        ຖ້າ (cmd == "endif") {
            ຖ້າ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ສົ່ງຄືນ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
