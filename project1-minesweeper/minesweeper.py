import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) > 0 and self.count == len(self.cells):
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)

        # 2) mark the cell as safe
        self.mark_safe(cell)

        # 3) add a new sentence based on the cell's neighbors and the count
        i, j = cell
        neighbors = set()
        mines_already_known = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if (ni, nj) == cell:
                    continue
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    ncell = (ni, nj)
                    if ncell in self.mines:
                        mines_already_known += 1
                    elif ncell not in self.safes and ncell not in self.moves_made:
                        neighbors.add(ncell)

        # Adjust count by subtracting already-known mines among neighbors
        adjusted_count = max(0, count - mines_already_known)

        if len(neighbors) > 0:
            self.knowledge.append(Sentence(neighbors, adjusted_count))

        # 4) Iterate over knowledge to mark any additional safe/mine cells
        changed = True
        while changed:
            changed = False

            # Collect new safes and mines from current sentences
            safes_to_mark = set()
            mines_to_mark = set()
            for sentence in self.knowledge:
                safes_to_mark |= sentence.known_safes()
                mines_to_mark |= sentence.known_mines()

            # Apply markings
            for s in safes_to_mark:
                if s not in self.safes:
                    self.mark_safe(s)
                    changed = True
            for m in mines_to_mark:
                if m not in self.mines:
                    self.mark_mine(m)
                    changed = True

            # Remove empty sentences
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

            # 5) Infer new sentences via subset relationships
            new_sentences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 is s2:
                        continue
                    if len(s1.cells) == 0 or len(s2.cells) == 0:
                        continue
                    if s1.cells.issubset(s2.cells):
                        diff_cells = s2.cells - s1.cells
                        diff_count = s2.count - s1.count
                        candidate = Sentence(diff_cells, diff_count)
                        if len(diff_cells) > 0 and candidate not in self.knowledge and candidate not in new_sentences:
                            new_sentences.append(candidate)
            if new_sentences:
                self.knowledge.extend(new_sentences)
                changed = True

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        candidates = self.safes - self.moves_made
        if not candidates:
            return None
        return next(iter(candidates))

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # All possible cells
        all_cells = set((i, j) for i in range(self.height) for j in range(self.width))
        # Exclude moves already made and cells known to be mines
        candidates = list(all_cells - self.moves_made - self.mines)
        if not candidates:
            return None
        return random.choice(candidates)
