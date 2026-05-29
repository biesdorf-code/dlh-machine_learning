#!/usr/bin/env python3
"""inverse of a matrix"""


def determinant(matrix):
    """calculates the determinant of a matrix"""

    n = len(matrix)

    # 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # 2x2 matrix
    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return a * d - b * c

    # 3x3 or more
    det = 0
    for j in range(n):
        sign = (-1) ** j
        element = matrix[0][j]

        # delete row 0 and column j
        sub = []
        for i in range(1, n):
            row = matrix[i][:j] + matrix[i][j + 1:]
            sub.append(row)

        cofactor = sign * determinant(sub)
        det += element * cofactor

    return det


def cofactor(matrix):
    """cofactor matrix of a matrix"""

    n = len(matrix)

    # 1x1 matrix
    if n == 1:
        return [[1]]

    # build the cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):

            # delete row i and column j
            sub = []
            for r in range(n):
                if r == i:
                    continue
                row = matrix[r][:j] + matrix[r][j + 1:]
                sub.append(row)

            # minor × sign = cofactor
            m_ij = determinant(sub)
            sign = (-1) ** (i + j)
            c_ij = sign * m_ij
            cofactor_row.append(c_ij)

        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """adjugate matrix of a matrix"""

    n = len(matrix)

    # step 1: get the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # step 2: transpose it
    adjugate_matrix = []
    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix


def inverse(matrix):
    """inverse matrix of a matrix"""

    # **key thinking** The inverse is the matrix that
    # undoes what the original matrix did.

    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # non-empty
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    # no inverse
    if det == 0:
        return None

    adj = adjugate(matrix)

    # step 4: divide every element by the determinant
    # Famous formula: A^-1 = (1 / det(A)) * adj(A)
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
