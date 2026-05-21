#!/usr/bin/env python3
"""module for the determinant function"""


def determinant(matrix):
    """determinant of matrix, raises exceptions"""

    # check if the parameter is valid

    # check if it is a list of lists
    if not isinstance(matrix, list) or \
            not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # handle 0x0 matrix: [[]] is a valid 0x0 matrix
    # by convention, its determinant is 1
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # check if the matrix is square
    # (every row must have the same length as the number of rows)
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # base case: 1x1 matrix like [[5]]
    # the determinant is just the single element
    if n == 1:
        return matrix[0][0]

    # if 2x2, calculate the determinant and return
    # using the formula: det = ad - bc
    # [[ a, b ],
    #  [ c, d ]]
    if n == 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        det = a * d - b * c
        return det

    # if 3x3 or more, calculate the determinant using
    # cofactor expansion along the first row and return
    #
    # for each element in the first row:
    #   1. determine the sign using (-1)^column
    #   2. build the submatrix by removing row 0 and current column
    #   3. multiply: sign * element * determinant(submatrix)
    #   4. add to running total
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
