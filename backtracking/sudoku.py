class Solution:
    """https://leetcode.com/problems/sudoku-solver/"""
    def block_off(self,n):
        """gives offset value, that helps loops to run only inside 3x3 cells."""
        if 0<=n<3:
            return 0
        if 3<=n<6:
            return 3
        return 6

    def is_block_valid(self,num,r,c,board):
        r_off = self.block_off(r)
        c_off = self.block_off(c)
        num_set = set()
        for i in range(3):
            for j in range(3):
                v = board[i+r_off][j+c_off]
                if r == i+r_off and c == j+c_off:
                    v = num
                elif v == ".":
                    continue
                v = int(v)
                if v in num_set:
                    return False
                num_set.add(v)
        return True

    def is_col_valid(self,num,r,c,board):
        col = set()
        for i in range(9):
            v = board[i][c]
            if i == r:
                v = num
            elif v == ".":
                continue
            v = int(v)

            if v in col:
                return False
            col.add(v)
        return True

    def is_row_valid(self,num,r,c,board):
        row = set()
        for i in range(9):
            v = board[r][i]
            if i == c:
                v = num
            elif v == ".":
                continue
            v = int(v)
            if v in row:
                return False
            row.add(v)
        return True

    def is_valid(self,num,r,c,board):
        valid_row = self.is_row_valid(num,r,c,board)
        valid_col = self.is_col_valid(num,r,c,board)
        valid_block = self.is_block_valid(num,r,c,board)

        return valid_row and valid_col and valid_block

    def rec(self,board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".": # find empty cell
                    for num in range(1,10):  # values 1 to 9, will be tried one by one if valid. try new if 
                        if self.is_valid(num,r,c,board):
                            board[r][c] = str(num)
                            if self.rec(board): 
                                return True # will be executed only when one of the child rec calls was able to fill last cell on sudoku
                            else:
                                board[r][c] = "."
                    return False
        return True # This will be executed only when last cell is filled

    def solveSudoku(self, board) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.rec(board)
        print(board)

board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
Solution().solveSudoku(board)