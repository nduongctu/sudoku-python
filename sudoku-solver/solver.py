#ham nhan ma tran 9x9 va tim tat ca vi tri trong trong bang
##0 co nghi la o trong
def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None
#ham valid kiem tra bang co giai duoc hay khong
##lay mot so trong bang de kiem tra co ton` tai trung nhau hay k
def valid(board, num, pos):
    # kiem tra hang`
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # kiem tra cot
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # kiem tra luoi con
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True

#giai quyet tung phan tu rong va khong co gia tri nao
##xac thuc dieu kien  va bat cu noi nao no tim thay khoang trong no se dat so da duoc xac thuc dieu kien trong khoang 1-9
###vong lap se tiep tuc cho den khi lap day cac o trong bang cac so da xac thuc dieu kien
###neu bang dc giai quyet se tra ve gia tri true va nguoc lai
def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve(board):
                return True
            board[row][col] = 0
    return False

#ham get_board goi ham solve neu bang dc giai se tra ve gia tri bang da giai nguoc lai se tra ve gia tri ValueError
def get_board(bo):
    if solve(bo):
        return bo
    else:
        raise ValueError


