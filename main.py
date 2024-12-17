import random
import pickle

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True

    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True

    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

def is_draw(board):
    return all(cell != " " for row in board for cell in row)

class QLearningAI:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def state_to_string(self, board):
        return ''.join(cell if cell != " " else "_" for row in board for cell in row)

    def get_possible_moves(self, board):
        return [(row, col) for row in range(3) for col in range(3) if board[row][col] == " "]

    def choose_action(self, board):
        state = self.state_to_string(board)
        possible_moves = self.get_possible_moves(board)
        if not possible_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(possible_moves)
        else:
            q_values = {move: self.q_table.get((state, move), 0) for move in possible_moves}
            max_q_value = max(q_values.values())
            best_moves = [move for move, q in q_values.items() if q == max_q_value]
            return random.choice(best_moves)

    def update_q_table(self, old_board, action, reward, new_board):
        old_state = self.state_to_string(old_board)
        new_state = self.state_to_string(new_board)
        max_future_q = max(
            [self.q_table.get((new_state, move), 0) for move in self.get_possible_moves(new_board)],
            default=0
        )
        current_q = self.q_table.get((old_state, action), 0)
        self.q_table[(old_state, action)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            self.q_table = {}

def train_ai(games_to_play=10000, save_interval=1000):
    ai = QLearningAI(alpha=0.1, gamma=0.9, epsilon=0.2)
    ai.load_q_table()

    for game in range(1, games_to_play + 1):
        board = [[" " for _ in range(3)] for _ in range(3)]
        players = ["X", "O"]
        current_player = 0
        move_history = []

        while True:
            player_symbol = players[current_player]
            old_board = [row[:] for row in board]
            action = ai.choose_action(board)

            if action is None:
                break

            row, col = action
            board[row][col] = player_symbol
            move_history.append((old_board, action))

            if check_winner(board, player_symbol):
                reward = 1 if player_symbol == "X" else -1
                for i, (old_b, act) in enumerate(reversed(move_history)):
                    ai.update_q_table(old_b, act, reward, board)
                    reward = -reward
                break

            if is_draw(board):
                for old_b, act in move_history:
                    ai.update_q_table(old_b, act, 0.5, board)
                break

            current_player = 1 - current_player

        if game % save_interval == 0:
            print(f"Game {game}/{games_to_play} completed. Saving Q-table...")
            ai.save_q_table()

    print("Training complete. Saving final Q-table...")
    ai.save_q_table()

def play_with_ai():
    ai = QLearningAI()
    ai.load_q_table()
    board = [[" " for _ in range(3)] for _ in range(3)]
    players = ["Human", "AI"]
    current_player = 0

    while True:
        print_board(board)

        if players[current_player] == "Human":
            try:
                row = int(input("Enter row (0, 1, 2): "))
                col = int(input("Enter column (0, 1, 2): "))
                if board[row][col] != " ":
                    print("Invalid move! Cell already taken.")
                    continue
            except (ValueError, IndexError):
                print("Invalid input! Enter numbers between 0 and 2.")
                continue
        else:
            print("AI is making a move...")
            action = ai.choose_action(board)
            if action is None:
                break
            row, col = action

        board[row][col] = "X" if players[current_player] == "Human" else "O"

        if check_winner(board, "X" if players[current_player] == "Human" else "O"):
            print_board(board)
            print(f"{players[current_player]} wins!")
            break

        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = 1 - current_player

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

if __name__ == "__main__":
    print("Training the AI...")
    train_ai(games_to_play=50000)
    print("Training complete! Now you can play against the AI.")
    play_with_ai()
