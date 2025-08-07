import tkinter as tk
from tkinter import messagebox, simpledialog
import os
import time

class ChainReactionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chain Reaction Game")
        self.game_mode = None
        self.board = None
        self.m = 0
        self.n = 0
        self.current_player = "R"  # Red starts
        self.buttons = []
        self.game_state_file = "gamestate.txt"
        self.setup_main_menu()

    def setup_main_menu(self):
        self.clear_window()
        tk.Label(self.root, text="Chain Reaction", font=("Arial", 24)).pack(pady=20)
        tk.Button(self.root, text="AI vs Human", command=lambda: self.select_mode("AI vs Human")).pack(pady=10)
        tk.Button(self.root, text="Human vs Human", command=lambda: self.select_mode("Human vs Human")).pack(pady=10)
        tk.Button(self.root, text="AI vs AI", command=lambda: self.select_mode("AI vs AI")).pack(pady=10)

    def select_mode(self, mode):
        self.game_mode = mode
        self.m = simpledialog.askinteger("Input", "Enter number of rows (m):", minvalue=1, maxvalue=20, parent=self.root)
        self.n = simpledialog.askinteger("Input", "Enter number of columns (n):", minvalue=1, maxvalue=20, parent=self.root)
        if self.m and self.n:
            print(f"Frontend initialized with board size {self.m}x{self.n}")
            self.initialize_board()
            self.setup_game_board()
            if self.game_mode == "AI vs Human":
                self.write_gamestate_file("Human Move:", "AI")
            elif self.game_mode == "AI vs AI":
                self.start_ai_vs_ai()

    def initialize_board(self):
        self.board = [[{"orbs": 0, "color": None} for _ in range(self.n)] for _ in range(self.m)]

    def setup_game_board(self):
        self.clear_window()
        tk.Label(self.root, text=f"Game Mode: {self.game_mode}", font=("Arial", 16)).pack(pady=10)
        self.player_label = tk.Label(self.root, text="Human (Red) Move" if self.game_mode == "AI vs Human" else "Red's Move" if self.game_mode == "Human vs Human" else "AI Red Move", font=("Arial", 14))
        self.player_label.pack(pady=10)
        board_frame = tk.Frame(self.root)
        board_frame.pack(pady=10)
        for i in range(self.m):
            row = []
            for j in range(self.n):
                btn = tk.Button(board_frame, text="0", width=4, height=2, font=("Arial", 12), bg="white", command=lambda x=i, y=j: self.handle_click(x, y))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.buttons.append(row)
        self.update_board_display()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def handle_click(self, i, j):
        if self.game_mode == "AI vs Human" and self.current_player == "R":
            if self.is_valid_move(i, j, self.current_player):
                self.make_move(i, j, self.current_player)
                print(f"Human (Red) made move at ({i},{j})")
                self.update_board_display()
                winner = self.check_winner()
                if winner:
                    self.end_game(winner)
                    return
                print(f"Writing human move to {self.game_state_file}: ({i},{j})")
                self.write_gamestate_file("Human Move:", "AI")
                self.current_player = "B"
                self.player_label.config(text="AI (Blue) Move")
                self.root.after(100, self.wait_for_ai_move)
        elif self.game_mode == "Human vs Human":
            if self.is_valid_move(i, j, self.current_player):
                self.make_move(i, j, self.current_player)
                print(f"Player {self.current_player} made move at ({i},{j})")
                self.update_board_display()
                winner = self.check_winner()
                if winner:
                    self.end_game(winner)
                    return
                self.current_player = "B" if self.current_player == "R" else "R"
                self.player_label.config(text=f"{'Red' if self.current_player == 'R' else 'Blue'}'s Move")

    def is_valid_move(self, i, j, player):
        return self.board[i][j]["color"] is None or self.board[i][j]["color"] == player

    def make_move(self, i, j, player):
        self.board[i][j]["orbs"] += 1
        self.board[i][j]["color"] = player
        self.process_explosions()

    def process_explosions(self):
        while True:
            explosions = []
            for i in range(self.m):
                for j in range(self.n):
                    critical_mass = self.get_critical_mass(i, j)
                    if self.board[i][j]["orbs"] >= critical_mass:
                        explosions.append((i, j))
            if not explosions:
                break
            for i, j in explosions:
                self.explode_cell(i, j)

    def get_critical_mass(self, i, j):
        if (i == 0 or i == self.m - 1) and (j == 0 or j == self.n - 1):
            return 2
        if i == 0 or i == self.m - 1 or j == 0 or j == self.n - 1:
            return 3
        return 4

    def explode_cell(self, i, j):
        critical_mass = self.get_critical_mass(i, j)
        player = self.board[i][j]["color"]
        self.board[i][j]["orbs"] -= critical_mass
        if self.board[i][j]["orbs"] == 0:
            self.board[i][j]["color"] = None
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.m and 0 <= nj < self.n:
                self.board[ni][nj]["orbs"] += 1
                self.board[ni][nj]["color"] = player
                print(f"Exploded ({i},{j}) -> Adjacent ({ni},{nj}) set to {player} with {self.board[ni][nj]['orbs']} orbs")

    def write_gamestate_file(self, header, next_move):
        with open(self.game_state_file, "w") as f:
            f.write(f"Board Size: {self.m} {self.n}\n")
            f.write(f"{header}\n")
            f.write(f"Next Move: {next_move}\n")
            for i in range(self.m):
                row = []
                for j in range(self.n):
                    cell = self.board[i][j]
                    if cell["orbs"] == 0:
                        row.append("0")
                    else:
                        row.append(f"{cell['orbs']}{cell['color']}")
                f.write(" ".join(row) + "\n")
        print(f"Wrote to {self.game_state_file}: {self.m} rows, {self.n} columns")

    def start_ai_vs_ai(self):
        self.current_player = "R"
        self.write_gamestate_file("AI vs AI Move:", "AI Red")
        self.root.after(300, self.poll_ai_vs_ai)

    def poll_ai_vs_ai(self):
        if self.read_gamestate_file():
            self.update_board_display()
            winner = self.check_winner()
            if winner:
                self.end_game(winner)
                return
            next_player = "B" if self.current_player == "R" else "R"
            self.current_player = next_player
            self.player_label.config(text=f"AI {'Red' if next_player == 'R' else 'Blue'} Move")
            self.write_gamestate_file("AI vs AI Move:", f"AI {'Red' if next_player == 'R' else 'Blue'}")
        self.root.after(300, self.poll_ai_vs_ai)

    def wait_for_ai_move(self):
        if self.read_gamestate_file():
            self.update_board_display()
            winner = self.check_winner()
            if winner:
                self.end_game(winner)
            else:
                self.current_player = "R"
                self.player_label.config(text="Human (Red) Move")
        else:
            self.root.after(100, self.wait_for_ai_move)

    def read_gamestate_file(self):
        if os.path.exists(self.game_state_file):
            try:
                with open(self.game_state_file, "r") as f:
                    lines = f.readlines()
                if len(lines) < 3 or not lines[0].startswith("Board Size:"):
                    print(f"Error: Invalid {self.game_state_file} format: {lines[:3]}")
                    return False
                header = lines[1].strip()
                next_move = lines[2].strip()
                print(f"Read {self.game_state_file}: header={header}, next_move={next_move}, board={lines[3:]}")
                if self.game_mode == "AI vs Human":
                    if header != "AI Move:" or next_move != "Next Move: Human":
                        print(f"Error: Expected 'AI Move:' and 'Next Move: Human', got {header}, {next_move}")
                        return False
                elif self.game_mode == "AI vs AI":
                    if header != "AI vs AI Move:" or next_move not in ["Next Move: AI Red", "Next Move: AI Blue"]:
                        print(f"Error: Expected 'AI vs AI Move:' and 'Next Move: AI Red/Blue', got {header}, {next_move}")
                        return False
                board_lines = lines[3:]
                if len(board_lines) != self.m:
                    print(f"Error: Expected {self.m} rows in {self.game_state_file}, got {len(board_lines)}")
                    return False
                self.parse_board(board_lines)
                print(f"Successfully read move from {self.game_state_file}: {self.m} rows, {self.n} columns")
                return True
            except Exception as e:
                print(f"Error reading {self.game_state_file}: {e}")
                return False
        print(f"Error: {self.game_state_file} does not exist")
        return False

    def parse_board(self, lines):
        for i in range(self.m):
            row = lines[i].strip().split()
            if len(row) != self.n:
                print(f"Error: Row {i} has {len(row)} columns, expected {self.n}")
                return
            for j in range(self.n):
                cell = row[j]
                if cell == "0":
                    self.board[i][j] = {"orbs": 0, "color": None}
                else:
                    try:
                        orbs = int(cell[:-1])
                        color = cell[-1]
                        if color not in ['R', 'B'] or orbs <= 0:
                            print(f"Error: Invalid cell at ({i},{j}): {cell}")
                            return
                        self.board[i][j] = {"orbs": orbs, "color": color}
                    except ValueError:
                        print(f"Error: Failed to parse cell at ({i},{j}): {cell}")
                        return

    def update_board_display(self):
        for i in range(self.m):
            for j in range(self.n):
                cell = self.board[i][j]
                text = str(cell["orbs"]) if cell["orbs"] > 0 else "0"
                bg = "red" if cell["color"] == "R" else "blue" if cell["color"] == "B" else "white"
                self.buttons[i][j].config(text=text, bg=bg)

    def check_winner(self):
        red_count = sum(1 for row in self.board for cell in row if cell["color"] == "R" and cell["orbs"] > 0)
        blue_count = sum(1 for row in self.board for cell in row if cell["color"] == "B" and cell["orbs"] > 0)
        total_moves = red_count + blue_count
        if total_moves < 2:
            return None
        if red_count == 0 and blue_count > 0:
            return "Blue"
        if blue_count == 0 and red_count > 0:
            return "Red"
        return None

    def end_game(self, winner):
        messagebox.showinfo("Game Over", f"{winner} wins!")
        self.setup_main_menu()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChainReactionUI(root)
    root.mainloop()