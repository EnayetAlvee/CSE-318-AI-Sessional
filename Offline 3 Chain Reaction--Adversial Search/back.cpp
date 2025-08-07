#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <climits>
#include <chrono>
#include <thread>
using namespace std;

struct Cell {
    int orbs;
    char color; // 'R', 'B', or '\0' for empty
};

class ChainReaction {
private:
    int m, n;
    std::vector<std::vector<Cell>> board;
    const std::string game_state_file = "gamestate.txt";
    const int DEPTH_LIMIT = 3;
    bool is_ai_vs_ai = false;

public:
    ChainReaction() : m(0), n(0) {
        // Board will be initialized after reading dimensions from file
    }

    void run() {
        while (true) {
            if (read_game_state()) {
                is_ai_vs_ai = is_ai_vs_ai_mode();
                if (is_ai_vs_ai) {
                    run_ai_vs_ai();
                } else {
                    auto [best_move_i, best_move_j] = minimax_decision();
                    if (best_move_i == -1 && best_move_j == -1) {
                        std::cerr << "No valid moves for AI (Blue), ending turn" << std::endl;
                        write_game_state("AI Move:", "Human");
                    } else {
                        make_move(best_move_i, best_move_j, 'B');
                        process_explosions();
                        write_game_state("AI Move:", "Human");
                    }
                    if (check_winner() != '\0') {
                        std::cout << "Game ended with winner: " << check_winner() << std::endl;
                        break;
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

private:
    bool is_ai_vs_ai_mode() {
        std::ifstream file(game_state_file);
        if (!file.is_open()) return false;
        std::string line;
        std::getline(file, line); // Skip Board Size
        std::getline(file, line); // Read header
        file.close();
        return line == "AI vs AI Move:";
    }

    string board_to_string() {
        stringstream ss;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].orbs == 0) ss << "0 ";
                else ss << board[i][j].orbs << board[i][j].color << " ";
            }
            ss << "\n";
        }
        return ss.str();
    }

    void run_ai_vs_ai() {
        char current_player = 'R'; // Red AI starts
        string prev_board = "";
        int no_progress_count = 0;
        while (true) {
            if (!read_game_state()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            if (!is_ai_vs_ai) break; // Exit if mode changes
            string curr_board = board_to_string();
            if (curr_board == prev_board) {
                no_progress_count++;
                if (no_progress_count >= 2) {
                    std::cerr << "No progress made after 2 attempts, ending AI vs AI game" << std::endl;
                    break;
                }
            } else {
                no_progress_count = 0;
            }
            prev_board = curr_board;
            auto [best_move_i, best_move_j] = minimax_decision(current_player);
            if (best_move_i == -1 && best_move_j == -1) {
                std::cerr << "No valid moves for AI player " << current_player << std::endl;
                write_game_state("AI vs AI Move:", current_player == 'R' ? "AI Blue" : "AI Red");
                if (no_progress_count >= 2) break; // Both players have no moves
                current_player = (current_player == 'R') ? 'B' : 'R';
                continue;
            }
            make_move(best_move_i, best_move_j, current_player);
            process_explosions();
            std::cout << "AI " << current_player << " move made at (" << best_move_i << "," << best_move_j << "), board:\n" << board_to_string();
            write_game_state("AI vs AI Move:", current_player == 'R' ? "AI Blue" : "AI Red");
            if (check_winner() != '\0') {
                std::cout << "Game ended with winner: " << check_winner() << std::endl;
                break;
            }
            current_player = (current_player == 'R') ? 'B' : 'R';
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    }

    std::pair<int, int> minimax_decision(char player = 'B') {
        int best_value = (player == 'B') ? INT_MIN : INT_MAX;
        std::pair<int, int> best_move = {-1, -1};
        bool valid_move_found = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (is_valid_move(i, j, player)) {
                    valid_move_found = true;
                    auto board_copy = board;
                    make_move(i, j, player);
                    process_explosions();
                    int value = minimax(0, player == 'B' ? false : true, INT_MIN, INT_MAX, player);
                    board = board_copy;
                    if (player == 'B' && value > best_value) {
                        best_value = value;
                        best_move = {i, j};
                    } else if (player == 'R' && value < best_value) {
                        best_value = value;
                        best_move = {i, j};
                    }
                }
            }
        }
        if (valid_move_found) {
            std::cout << "AI player " << player << " selected move: (" << best_move.first << "," << best_move.second << ")" << std::endl;
        }
        return best_move;
    }

    int minimax(int depth, bool is_maximizing, int alpha, int beta, char player = 'B') {
        char winner = check_winner();
        if (winner == 'B') return (player == 'B') ? 1000 : -1000;
        if (winner == 'R') return (player == 'R') ? 1000 : -1000;
        if (depth >= DEPTH_LIMIT) {
            return is_ai_vs_ai ? evaluate_board_control() : evaluate_critical_cells();
        }

        char current_player = is_maximizing ? player : (player == 'B' ? 'R' : 'B');
        if (is_maximizing) {
            int best_value = INT_MIN;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (is_valid_move(i, j, current_player)) {
                        auto board_copy = board;
                        make_move(i, j, current_player);
                        process_explosions();
                        best_value = std::max(best_value, minimax(depth + 1, false, alpha, beta, player));
                        board = board_copy;
                        alpha = std::max(alpha, best_value);
                        if (beta <= alpha) break;
                    }
                }
            }
            return best_value;
        } else {
            int best_value = INT_MAX;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (is_valid_move(i, j, current_player)) {
                        auto board_copy = board;
                        make_move(i, j, current_player);
                        process_explosions();
                        best_value = std::min(best_value, minimax(depth + 1, true, alpha, beta, player));
                        board = board_copy;
                        beta = std::min(beta, best_value);
                        if (beta <= alpha) break;
                    }
                }
            }
            return best_value;
        }
    }

    bool read_game_state() {
        std::ifstream file(game_state_file);
        if (!file.is_open()) {
            std::cerr << "Could not open " << game_state_file << ", retrying..." << std::endl;
            return false;
        }
        std::string line;
        // Read board size
        if (!std::getline(file, line) || line.find("Board Size:") != 0) {
            std::cerr << "Invalid or missing Board Size line: " << line << std::endl;
            file.close();
            return false;
        }
        std::istringstream size_ss(line.substr(11)); // Skip "Board Size: "
        int new_m, new_n;
        if (!(size_ss >> new_m >> new_n) || new_m <= 0 || new_n <= 0) {
            std::cerr << "Invalid board dimensions in " << line << std::endl;
            file.close();
            return false;
        }
        // Initialize or resize board if dimensions changed
        if (new_m != m || new_n != n) {
            m = new_m;
            n = new_n;
            board = std::vector<std::vector<Cell>>(m, std::vector<Cell>(n, {0, '\0'}));
            std::cout << "Initialized backend with board size " << m << "x" << n << std::endl;
        }
        // Read header
        if (!std::getline(file, line) || (line != "Human Move:" && line != "AI vs AI Move:")) {
            std::cerr << "Invalid or missing header: " << line << std::endl;
            file.close();
            return false;
        }
        // Read next move
        if (!std::getline(file, line) || (line != "Next Move: AI" && line != "Next Move: AI Red" && line != "Next Move: AI Blue")) {
            std::cerr << "Invalid or missing next move: " << line << std::endl;
            file.close();
            return false;
        }
        // Read board
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            if (!line.empty()) lines.push_back(line);
        }
        file.close();
        // Validate number of rows
        if (lines.size() != m) {
            std::cerr << "Error: Expected " << m << " rows, got " << lines.size() << " in " << game_state_file << std::endl;
            return false;
        }
        for (int i = 0; i < m; ++i) {
            std::istringstream iss(lines[i]);
            std::vector<std::string> cells;
            std::string cell;
            while (iss >> cell) cells.push_back(cell);
            // Validate number of columns
            if (cells.size() != n) {
                std::cerr << "Error: Row " << i << " has " << cells.size() << " columns, expected " << n << std::endl;
                return false;
            }
            for (int j = 0; j < n; ++j) {
                if (cells[j] == "0") {
                    board[i][j] = {0, '\0'};
                } else {
                    try {
                        if (cells[j].size() < 2 || (cells[j].back() != 'R' && cells[j].back() != 'B')) {
                            std::cerr << "Error: Invalid cell format at (" << i << "," << j << "): " << cells[j] << std::endl;
                            return false;
                        }
                        int orbs = std::stoi(cells[j].substr(0, cells[j].size() - 1));
                        if (orbs <= 0) {
                            std::cerr << "Error: Invalid orb count at (" << i << "," << j << "): " << cells[j] << std::endl;
                            return false;
                        }
                        board[i][j] = {orbs, cells[j].back()};
                    } catch (const std::exception& e) {
                        std::cerr << "Error: Failed to parse cell at (" << i << "," << j << "): " << cells[j] << " (" << e.what() << ")" << std::endl;
                        return false;
                    }
                }
            }
        }
        std::cout << "Successfully read board state from " << game_state_file << std::endl;
        return true;
    }

    void write_game_state(const std::string& header, const std::string& next_move) {
        std::ofstream file(game_state_file);
        file << "Board Size: " << m << " " << n << "\n";
        file << header << "\n";
        file << "Next Move: " << next_move << "\n";
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].orbs == 0) {
                    file << "0";
                } else {
                    file << board[i][j].orbs << board[i][j].color;
                }
                if (j < n - 1) file << " ";
            }
            file << "\n";
        }
        file.close();
        std::cout << "Wrote to " << game_state_file << ": header=" << header << ", next_move=" << next_move << "\n";
    }

    // Heuristic 1: Evaluation function 1
    int evaluate() {
        int blue_orbs = 0, red_orbs = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].color == 'B') blue_orbs += board[i][j].orbs;
                if (board[i][j].color == 'R') red_orbs += board[i][j].orbs;
            }
        }
        return blue_orbs - red_orbs;
    }

    // Heuristic 2: Control of critical cells (corners and edges)
    int evaluate_critical_cells() {
        int blue_critical = 0, red_critical = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].color == 'B') {
                    if (get_critical_mass(i, j) == 2) blue_critical += 5; // Corners
                    else if (get_critical_mass(i, j) == 3) blue_critical += 3; // Edges
                } else if (board[i][j].color == 'R') {
                    if (get_critical_mass(i, j) == 2) red_critical += 5; // Corners
                    else if (get_critical_mass(i, j) == 3) red_critical += 3; // Edges
                }
            }
        }
        return blue_critical - red_critical;
    }

    // Heuristic 3: Explosion potential (proximity to critical mass)
    int evaluate_explosion_potential() {
        int blue_explosion = 0, red_explosion = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].orbs > 0) {
                    int critical = get_critical_mass(i, j);
                    int closeness = critical - board[i][j].orbs;
                    if (closeness <= 1 && board[i][j].color == 'B') blue_explosion += 4;
                    else if (closeness <= 1 && board[i][j].color == 'R') red_explosion += 4;
                }
            }
        }
        return blue_explosion - red_explosion;
    }

    // Heuristic 4: Board control (number of cells occupied) - Used for AI vs AI
    int evaluate_board_control() {
        int blue_cells = 0, red_cells = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].color == 'B') blue_cells++;
                if (board[i][j].color == 'R') red_cells++;
            }
        }
        return blue_cells - red_cells;
    }

    // Heuristic 5: Chain reaction length
    int evaluate_chain_length() {
        int blue_chain = 0, red_chain = 0;
        auto board_copy = board;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].orbs > 0 && board[i][j].orbs >= get_critical_mass(i, j)) {
                    int chain_length = simulate_chain(i, j, board[i][j].color);
                    if (board[i][j].color == 'B') blue_chain += chain_length;
                    else if (board[i][j].color == 'R') red_chain += chain_length;
                }
            }
        }
        board = board_copy; // Restore original board
        return blue_chain - red_chain;
    }

    int simulate_chain(int i, int j, char player) {
        int chain_length = 0;
        auto temp_board = board;
        std::vector<std::pair<int, int>> to_explode = {{i, j}};
        while (!to_explode.empty()) {
            std::vector<std::pair<int, int>> next_explode;
            for (auto [x, y] : to_explode) {
                chain_length++;
                int critical = get_critical_mass(x, y);
                temp_board[x][y].orbs -= critical;
                if (temp_board[x][y].orbs == 0) temp_board[x][y].color = '\0';
                std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                for (auto [di, dj] : directions) {
                    int ni = x + di, nj = y + dj;
                    if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                        temp_board[ni][nj].orbs += 1;
                        temp_board[ni][nj].color = player;
                        if (temp_board[ni][nj].orbs >= get_critical_mass(ni, nj)) {
                            next_explode.emplace_back(ni, nj);
                        }
                    }
                }
            }
            to_explode = next_explode;
        }
        return chain_length;
    }

    bool is_valid_move(int i, int j, char player) {
        // cout << board[i][j].orbs << " " << board[i][j].color << endl;
        return board[i][j].color == '\0' || board[i][j].color == player;
    }

    void make_move(int i, int j, char player) {
        board[i][j].orbs += 1;
        board[i][j].color = player;
    }

    void process_explosions() {
        while (true) {
            std::vector<std::pair<int, int>> explosions;
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (board[i][j].orbs >= get_critical_mass(i, j)) {
                        explosions.emplace_back(i, j);
                    }
                }
            }
            if (explosions.empty()) break;
            for (auto [i, j] : explosions) {
                explode_cell(i, j);
            }
        }
    }

    int get_critical_mass(int i, int j) {
        if ((i == 0 || i == m - 1) && (j == 0 || j == n - 1)) return 2;
        if (i == 0 || i == m - 1 || j == 0 || j == n - 1) return 3;
        return 4;
    }

    void explode_cell(int i, int j) {
        int critical_mass = get_critical_mass(i, j);
        char player = board[i][j].color;
        board[i][j].orbs -= critical_mass;
        if (board[i][j].orbs == 0) board[i][j].color = '\0';
        std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (auto [di, dj] : directions) {
            int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                board[ni][nj].orbs += 1;
                board[ni][nj].color = player;
                // std::cout << "Exploded (" << i << "," << j << ") -> Adjacent (" << ni << "," << nj << ") set to " << player << " with " << board[ni][nj].orbs << " orbs" << std::endl;
            }
        }
    }

    char check_winner() {
        int red_count = 0, blue_count = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j].color == 'R' && board[i][j].orbs > 0) red_count += board[i][j].orbs;
                if (board[i][j].color == 'B' && board[i][j].orbs > 0) blue_count += board[i][j].orbs;
            }
        }
        int total_moves = red_count + blue_count;
        if (total_moves < 2) return '\0';
        if (red_count == 0 && blue_count > 0) return 'B';
        if (blue_count == 0 && red_count > 0) return 'R';
        return '\0';
    }
};

int main() {
    ChainReaction game;
    game.run();
    return 0;
}