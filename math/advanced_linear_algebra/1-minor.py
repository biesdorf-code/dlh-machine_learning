#!/usr/bin/env python3
"""module for the determinant and minor functions"""


def determinant(matrix):
    """determinant of matrix, raises exceptions"""

    # check if the parameter is valid

    # check if it is a list of lists
    if not isinstance(matrix, list) or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # handle 0x0 matrix: [[]] is 1

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # check if the matrix is square
    # (every row must have the same length as the number of rows)
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # base case: 1x1
    if n == 1:
        return matrix[0][0]

    # if 2x2

    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = a * d - b * c
        return det

    # if 3x3 or more, cofactor expansion
    det = 0
    for j in range(n):

        # the sign alternates: +, -, +, -, ...
        sign = (-1) ** j

        # the element from the first row we're expanding on
        element = matrix[0][j]

        # build the submatrix (delete row 0 and column j)
        # go through every row below row 0
        # and skip the column at index j
        sub = []
        for i in range(1, n):
            row = matrix[i][:j] + matrix[i][j + 1:]
            sub.append(row)

        # the cofactor is: sign * determinant of the submatrix
        cofactor = sign * determinant(sub)

        # add this term to the total
        det += element * cofactor

    return det


def minor(matrix):
    """minor of that matrix, uses the determinant()"""

    # checks if it is a list of lists
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # checks if the matrix is non-empty and square
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    #  1x1 matrix is 1
    if n == 1:
        return [[1]]

    # build the minor matrix

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):

            # build the submatrix (delete row i and column j)
            sub = []
            for r in range(n):
                if r == i:
                    continue  # skip row i
                row = matrix[r][:j] + matrix[r][j + 1:]
                sub.append(row)

            # the minor minore_in_the_round = determinant of the submatrix
            minor_e_in_the_round = determinant(sub)
            minor_row.append(minor_e_in_the_round)

        minor_matrix.append(minor_row)

    return minor_matrix
