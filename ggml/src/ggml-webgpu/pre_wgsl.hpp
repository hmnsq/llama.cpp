#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <string_view>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ pre_wgsl {

//==============================================================
// Options
//==============================================================
ಸಂರಚನಾ Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ಸ್ಥಿರ std::string trim(ಧ್ರುವ std::string & s) {
    size_t a = 0;
    ಯಾವತ್ (a < s.size() && std::isspace((unsigned ವರ್ಣ) s[a])) {
        a++;
    }
    size_t b = s.size();
    ಯಾವತ್ (b > a && std::isspace((unsigned ವರ್ಣ) s[b - 1])) {
        b--;
    }
    ಪ್ರತ್ಯಾಗಮ s.substr(a, b - a);
}

ಸ್ಥಿರ std::string trim_ಮೌಲ್ಯ(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ಪ್ರತ್ಯಾಗಮ trim(ss.str());
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ isIdentChar(ವರ್ಣ c) {
    ಪ್ರತ್ಯಾಗಮ std::isalnum(static_cast<unsigned ವರ್ಣ>(c)) || c == '_';
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ endsWithContinuation(ಧ್ರುವ std::string & ಸಾಲು) {
    size_t i = ಸಾಲು.size();
    ಯಾವತ್ (i > 0 && std::isspace((unsigned ವರ್ಣ) ಸಾಲು[i - 1])) {
        i--;
    }
    ಪ್ರತ್ಯಾಗಮ i > 0 && ಸಾಲು[i - 1] == '\\';
}

ಸ್ಥಿರ ಶೂನ್ಯ stripContinuation(std::string & ಸಾಲು) {
    size_t i = ಸಾಲು.size();
    ಯಾವತ್ (i > 0 && std::isspace((unsigned ವರ್ಣ) ಸಾಲು[i - 1])) {
        i--;
    }
    ಯದಿ (i > 0 && ಸಾಲು[i - 1] == '\\') {
        ಸಾಲು.erase(i - 1);
    }
}

ಸ್ಥಿರ std::string expandMacrosRecursiveInternal(ಧ್ರುವ std::string &                                  ಸಾಲು,
                                                 ಧ್ರುವ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ಸ್ಥಿರ std::string expand_Macro_ಮೌಲ್ಯ(ಧ್ರುವ std::string &                                  name,
                                    ಧ್ರುವ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ಯದಿ (visiting.count(name)) {
        ಕ್ಷಿಪ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ಯದಿ (it == macros.end()) {
        visiting.erase(name);
        ಪ್ರತ್ಯಾಗಮ name;
    }

    ಧ್ರುವ std::string & value = it->second;
    ಯದಿ (value.empty()) {
        visiting.erase(name);
        ಪ್ರತ್ಯಾಗಮ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ಪ್ರತ್ಯಾಗಮ expanded;
}

ಸ್ಥಿರ std::string expandMacrosRecursiveInternal(ಧ್ರುವ std::string &                                  ಸಾಲು,
                                                 ಧ್ರುವ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(ಸಾಲು.size());

    size_t i = 0;
    ಯಾವತ್ (i < ಸಾಲು.size()) {
        ಯದಿ (isIdentChar(ಸಾಲು[i])) {
            size_t start = i;
            ಯಾವತ್ (i < ಸಾಲು.size() && isIdentChar(ಸಾಲು[i])) {
                i++;
            }
            std::string token = ಸಾಲು.substr(start, i - start);

            auto it = macros.find(token);
            ಯದಿ (it != macros.end()) {
                result += expand_Macro_ಮೌಲ್ಯ(token, macros, visiting);
            } ಅನ್ಯಥಾ {
                result += token;
            }
        } ಅನ್ಯಥಾ {
            result += ಸಾಲು[i];
            i++;
        }
    }

    ಪ್ರತ್ಯಾಗಮ result;
}

ಸ್ಥಿರ std::string expandMacrosRecursive(ಧ್ರುವ std::string &                                  ಸಾಲು,
                                         ಧ್ರುವ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ಪ್ರತ್ಯಾಗಮ expandMacrosRecursiveInternal(ಸಾಲು, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
ವರ್ಗ ಅಭಿವ್ಯಕ್ತಿ_Lexer {
  ಸಾರ್ವಜನಿಕ:
    ಗಣನಾ Kind { END, IDENT, ಸಂಖ್ಯೆ_2, OP, LPAREN, RPAREN };

    ಸಂರಚನಾ Tok {
        Kind        kind;
        std::string text;
    };

    explicit ಅಭಿವ್ಯಕ್ತಿ_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        ಯದಿ (pos >= src.size()) {
            ಪ್ರತ್ಯಾಗಮ { END, "" };
        }

        ವರ್ಣ c = src[pos];

        //ಸಂಖ್ಯೆ
        ಯದಿ (std::isdigit((unsigned ವರ್ಣ) c)) {
            size_t start = pos;
            ಯಾವತ್ (pos < src.size() && std::isdigit((unsigned ವರ್ಣ) src[pos])) {
                pos++;
            }
            ಪ್ರತ್ಯಾಗಮ { ಸಂಖ್ಯೆ_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ಯದಿ (std::isalpha((unsigned ವರ್ಣ) c) || c == '_') {
            size_t start = pos;
            ಯಾವತ್ (pos < src.size() && (std::isalnum((unsigned ವರ್ಣ) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ಪ್ರತ್ಯಾಗಮ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ಯದಿ (c == '(') {
            pos++;
            ಪ್ರತ್ಯಾಗಮ { LPAREN, "(" };
        }
        ಯದಿ (c == ')') {
            pos++;
            ಪ್ರತ್ಯಾಗಮ { RPAREN, ")" };
        }

        // multi-char operators
        ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ಪ್ರತಿ (auto op : two_ops) {
            ಯದಿ (src.substr(pos, 2) == op) {
                pos += 2;
                ಪ್ರತ್ಯಾಗಮ { OP, std::string(op) };
            }
        }

        // single-char operators
        ಯದಿ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ಪ್ರತ್ಯಾಗಮ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ಪ್ರತ್ಯಾಗಮ { END, "" };
    }

  ಗುಪ್ತ:
    std::string_view src;
    size_t           pos;

    ಶೂನ್ಯ skipWS() {
        ಯಾವತ್ (pos < src.size() && std::isspace((unsigned ವರ್ಣ) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
ವರ್ಗ ಅಭಿವ್ಯಕ್ತಿ_Parser {
  ಸಾರ್ವಜನಿಕ:
    ಅಭಿವ್ಯಕ್ತಿ_Parser(std::string_view                                     ಅಭಿವ್ಯಕ್ತಿ,
               ಧ್ರುವ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(ಅಭಿವ್ಯಕ್ತಿ),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    ಪೂರ್ಣಾಙ್ಕ parse() { ಪ್ರತ್ಯಾಗಮ parseLogicalOr(); }

  ಗುಪ್ತ:
    ಅಭಿವ್ಯಕ್ತಿ_Lexer                                            lex;
    ಅಭಿವ್ಯಕ್ತಿ_Lexer::Tok                                       tok;
    ಧ್ರುವ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    ಶೂನ್ಯ advance() { tok = lex.next(); }

    ತಾರ್ಕಿಕ acceptOp(ಧ್ರುವ std::string & s) {
        ಯದಿ (tok.kind == ಅಭಿವ್ಯಕ್ತಿ_Lexer::OP && tok.text == s) {
            advance();
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ತಾರ್ಕಿಕ acceptKind(ಅಭಿವ್ಯಕ್ತಿ_Lexer::Kind k) {
        ಯದಿ (tok.kind == k) {
            advance();
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪೂರ್ಣಾಙ್ಕ parseLogicalOr() {
        ಪೂರ್ಣಾಙ್ಕ v = parseLogicalAnd();
        ಯಾವತ್ (acceptOp("||")) {
            ಪೂರ್ಣಾಙ್ಕ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseLogicalAnd() {
        ಪೂರ್ಣಾಙ್ಕ v = parseEquality();
        ಯಾವತ್ (acceptOp("&&")) {
            ಪೂರ್ಣಾಙ್ಕ rhs = parseEquality();
            v       = (v && rhs);
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseEquality() {
        ಪೂರ್ಣಾಙ್ಕ v = parseRelational();
        ಪ್ರತಿ (;;) {
            ಯದಿ (acceptOp("==")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseRelational();
                v       = (v == rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp("!=")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseRelational();
                v       = (v != rhs);
            } ಅನ್ಯಥಾ {
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseRelational() {
        ಪೂರ್ಣಾಙ್ಕ v = parseShift();
        ಪ್ರತಿ (;;) {
            ಯದಿ (acceptOp("<")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseShift();
                v       = (v < rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp(">")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseShift();
                v       = (v > rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp("<=")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseShift();
                v       = (v <= rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp(">=")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseShift();
                v       = (v >= rhs);
            } ಅನ್ಯಥಾ {
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseShift() {
        ಪೂರ್ಣಾಙ್ಕ v = parseAdd();
        ಪ್ರತಿ (;;) {
            ಯದಿ (acceptOp("<<")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseAdd();
                v       = (v << rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp(">>")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseAdd();
                v       = (v >> rhs);
            } ಅನ್ಯಥಾ {
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseAdd() {
        ಪೂರ್ಣಾಙ್ಕ v = parseMult();
        ಪ್ರತಿ (;;) {
            ಯದಿ (acceptOp("+")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseMult();
                v       = (v + rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp("-")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseMult();
                v       = (v - rhs);
            } ಅನ್ಯಥಾ {
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseMult() {
        ಪೂರ್ಣಾಙ್ಕ v = parseUnary();
        ಪ್ರತಿ (;;) {
            ಯದಿ (acceptOp("*")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseUnary();
                v       = (v * rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp("/")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ಅನ್ಯಥಾ ಯದಿ (acceptOp("%")) {
                ಪೂರ್ಣಾಙ್ಕ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ಅನ್ಯಥಾ {
                ಭಙ್ಗ;
            }
        }
        ಪ್ರತ್ಯಾಗಮ v;
    }

    ಪೂರ್ಣಾಙ್ಕ parseUnary() {
        ಯದಿ (acceptOp("!")) {
            ಪ್ರತ್ಯಾಗಮ !parseUnary();
        }
        ಯದಿ (acceptOp("-")) {
            ಪ್ರತ್ಯಾಗಮ -parseUnary();
        }
        ಯದಿ (acceptOp("+")) {
            ಪ್ರತ್ಯಾಗಮ +parseUnary();
        }
        ಪ್ರತ್ಯಾಗಮ parsePrimary();
    }

    ಪೂರ್ಣಾಙ್ಕ parsePrimary() {
        // '(' expr ')'
        ಯದಿ (acceptKind(ಅಭಿವ್ಯಕ್ತಿ_Lexer::LPAREN)) {
            ಪೂರ್ಣಾಙ್ಕ v = parse();
            ಯದಿ (!acceptKind(ಅಭಿವ್ಯಕ್ತಿ_Lexer::RPAREN)) {
                ಕ್ಷಿಪ std::runtime_error("missing ')'");
            }
            ಪ್ರತ್ಯಾಗಮ v;
        }

        //ಸಂಖ್ಯೆ
        ಯದಿ (tok.kind == ಅಭಿವ್ಯಕ್ತಿ_Lexer::ಸಂಖ್ಯೆ_2) {
            ಪೂರ್ಣಾಙ್ಕ v = std::stoi(tok.text);
            advance();
            ಪ್ರತ್ಯಾಗಮ v;
        }

        // defined(identifier)
        ಯದಿ (tok.kind == ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT && tok.text == "defined") {
            advance();
            ಯದಿ (acceptKind(ಅಭಿವ್ಯಕ್ತಿ_Lexer::LPAREN)) {
                ಯದಿ (tok.kind != ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
                    ಕ್ಷಿಪ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ಯದಿ (!acceptKind(ಅಭಿವ್ಯಕ್ತಿ_Lexer::RPAREN)) {
                    ಕ್ಷಿಪ std::runtime_error("missing ) in defined()");
                }
                ಪ್ರತ್ಯಾಗಮ macros.count(name) ? 1 : 0;
            } ಅನ್ಯಥಾ {
                // defined NAME
                ಯದಿ (tok.kind != ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
                    ಕ್ಷಿಪ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ಪ್ರತ್ಯಾಗಮ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ಯದಿ (tok.kind == ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ಯದಿ (it == macros.end()) {
                ಪ್ರತ್ಯಾಗಮ 0;
            }
            ಯದಿ (it->second.empty()) {
                ಪ್ರತ್ಯಾಗಮ 1;
            }
            ಪ್ರತ್ಯಾಗಮ evalMacroExpression(name, it->second);
        }

        // unexpected
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಪೂರ್ಣಾಙ್ಕ evalMacroExpression(ಧ್ರುವ std::string & name, ಧ್ರುವ std::string & value) {
        ಯದಿ (visiting.count(name)) {
            ಕ್ಷಿಪ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ಅಭಿವ್ಯಕ್ತಿ_Parser ep(value, macros, visiting);
        ಪೂರ್ಣಾಙ್ಕ        v = ep.parse();
        visiting.erase(name);
        ಪ್ರತ್ಯಾಗಮ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
ವರ್ಗ Preprocessor {
  ಸಾರ್ವಜನಿಕ:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ಯದಿ (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ಧ್ರುವ std::string & filename, ಧ್ರುವ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_ರಾಶಿ, DirectiveMode::All);
        ಪ್ರತ್ಯಾಗಮ result;
    }

    std::string preprocess(ಧ್ರುವ std::string & contents, ಧ್ರುವ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_ರಾಶಿ, DirectiveMode::All);
        ಪ್ರತ್ಯಾಗಮ result;
    }

    std::string preprocess_includes_file(ಧ್ರುವ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        std::string result = processFile(filename, macros, predefined, include_ರಾಶಿ, DirectiveMode::IncludesOnly);
        ಪ್ರತ್ಯಾಗಮ result;
    }

    std::string preprocess_includes(ಧ್ರುವ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        std::string result = processString(contents, macros, predefined, include_ರಾಶಿ, DirectiveMode::IncludesOnly);
        ಪ್ರತ್ಯಾಗಮ result;
    }

  ಗುಪ್ತ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ಗಣನಾ ವರ್ಗ DirectiveMode { All, IncludesOnly };

    ಸಂರಚನಾ Cond {
        ತಾರ್ಕಿಕ parent_active;
        ತಾರ್ಕಿಕ active;
        ತಾರ್ಕಿಕ taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    ಶೂನ್ಯ parseMacroDefinitions(ಧ್ರುವ std::vector<std::string> & macro_defs) {
        ಪ್ರತಿ (ಧ್ರುವ auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ಯದಿ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ಅನ್ಯಥಾ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    ಶೂನ್ಯ buildMacros(ಧ್ರುವ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        ಪ್ರತಿ (ಧ್ರುವ auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ಪ್ರತಿ (ಧ್ರುವ auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ಯದಿ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ಅನ್ಯಥಾ {
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
    std::string loadFile(ಧ್ರುವ std::string & fname) {
        std::ifstream f(fname);
        ಯದಿ (!f.is_open()) {
            ಕ್ಷಿಪ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ಪ್ರತ್ಯಾಗಮ ss.str();
    }

    ತಾರ್ಕಿಕ condActive(ಧ್ರುವ std::vector<Cond> & cond) ಧ್ರುವ {
        ಯದಿ (cond.empty()) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ಧ್ರುವ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ಧ್ರುವ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ರಾಶಿ,
                            DirectiveMode                                  mode) {
        ಯದಿ (include_ರಾಶಿ.count(name)) {
            ಕ್ಷಿಪ std::runtime_error("Recursive include: " + name);
        }

        include_ರಾಶಿ.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_ರಾಶಿ, mode);
        include_ರಾಶಿ.erase(name);
        ಪ್ರತ್ಯಾಗಮ out;
    }

    std::string processIncludeFile(ಧ್ರುವ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ಧ್ರುವ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ರಾಶಿ,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ಪ್ರತ್ಯಾಗಮ processFile(full_path, macros, predefined_macros, include_ರಾಶಿ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ಧ್ರುವ std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              ಧ್ರುವ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ರಾಶಿ,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        ಸಾಲು;

        ಯಾವತ್ (std::getline(in, ಸಾಲು)) {
            std::string logical = ಸಾಲು;
            std::string t       = trim(logical);
            ಯದಿ (!t.empty() && t[0] == '#') {
                ಯಾವತ್ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ಯದಿ (!std::getline(in, ಸಾಲು)) {
                        ಭಙ್ಗ;
                    }
                    logical += "\n";
                    logical += ಸಾಲು;
                }
                t = trim(logical);
            }

            ಯದಿ (!t.empty() && t[0] == '#') {
                ತಾರ್ಕಿಕ handled = handleDirective(t, out, macros, predefined_macros, cond, include_ರಾಶಿ, mode);
                ಯದಿ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ಅನ್ಯಥಾ ಯದಿ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ಯದಿ (mode == DirectiveMode::All && !cond.empty()) {
            ಕ್ಷಿಪ std::runtime_error("Unclosed #if directive");
        }

        ಪ್ರತ್ಯಾಗಮ out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    ತಾರ್ಕಿಕ handleDirective(ಧ್ರುವ std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         ಧ್ರುವ std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ರಾಶಿ,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ಯದಿ (cmd == "include") {
            ಯದಿ (mode == DirectiveMode::All && !condActive(cond)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            std::string file;
            iss >> file;
            ಯದಿ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_ರಾಶಿ, mode);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (mode == DirectiveMode::IncludesOnly) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (cmd == "define") {
            ಯದಿ (!condActive(cond)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ಯದಿ (predefined_macros.count(name)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            std::string value = trim_ಮೌಲ್ಯ(iss);
            macros[name]      = value;
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "undef") {
            ಯದಿ (!condActive(cond)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ಯದಿ (predefined_macros.count(name)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            macros.erase(name);
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            ತಾರ್ಕಿಕ p = condActive(cond);
            ತಾರ್ಕಿಕ v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            ತಾರ್ಕಿಕ p = condActive(cond);
            ತಾರ್ಕಿಕ v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "if") {
            std::string ಅಭಿವ್ಯಕ್ತಿ = trim_ಮೌಲ್ಯ(iss);
            ತಾರ್ಕಿಕ        p    = condActive(cond);
            ತಾರ್ಕಿಕ        v    = ಅಸತ್ಯ;
            ಯದಿ (p) {
                std::unordered_set<std::string> visiting;
                ಅಭಿವ್ಯಕ್ತಿ_Parser                      ep(ಅಭಿವ್ಯಕ್ತಿ, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "elif") {
            std::string ಅಭಿವ್ಯಕ್ತಿ = trim_ಮೌಲ್ಯ(iss);

            ಯದಿ (cond.empty()) {
                ಕ್ಷಿಪ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ಯದಿ (!c.parent_active) {
                c.active = ಅಸತ್ಯ;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }

            ಯದಿ (c.taken) {
                c.active = ಅಸತ್ಯ;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }

            std::unordered_set<std::string> visiting;
            ಅಭಿವ್ಯಕ್ತಿ_Parser                      ep(ಅಭಿವ್ಯಕ್ತಿ, macros, visiting);
            ತಾರ್ಕಿಕ                            v = ep.parse() != 0;
            c.active                          = v;
            ಯದಿ (v) {
                c.taken = ಸತ್ಯ;
            }
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "else") {
            ಯದಿ (cond.empty()) {
                ಕ್ಷಿಪ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ಯದಿ (!c.parent_active) {
                c.active = ಅಸತ್ಯ;
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
            ಯದಿ (c.taken) {
                c.active = ಅಸತ್ಯ;
            } ಅನ್ಯಥಾ {
                c.active = ಸತ್ಯ;
                c.taken  = ಸತ್ಯ;
            }
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        ಯದಿ (cmd == "endif") {
            ಯದಿ (cond.empty()) {
                ಕ್ಷಿಪ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }

        // Unknown directive
        ಕ್ಷಿಪ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
