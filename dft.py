# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
import math as mt

class Dft:
    def __init__(self):
        pass

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        row,column=np.shape(matrix)
        matrix_new=np.zeros((row,column),dtype=complex)
        for u in range(row):
            for v in range(column):
                sum_1= []
                for i in range(row):
                    for j in range(column):
                        e = (np.exp(1J * -2 * mt.pi * (((u * i) / row) + ((v * j) / column))))
                        sum_1.append(matrix[i, j] * e)
                matrix_new[u, v] = sum(sum_1)

        return matrix_new
    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        You can implement the inverse transform formula with or without the normalizing factor.
        Both formulas are accepted.
        takes as input:
        matrix: a 2d matrix (DFT) usually complex
        returns a complex matrix representing the inverse fourier transform"""
        row,column=np.shape(matrix)
        matrix_new=np.zeros((row,column),dtype=complex)
        for u in range(row):
            for v in range(column):
                sum_1= []
                for i in range(row):
                    for j in range(column):
                        e = (np.exp(1J * -2 * mt.pi * (((u * i) / row) + ((v * j) / column))))
                        sum_1.append(matrix[i, j] * e)
                matrix_new[u, v] = sum(sum_1)

        return matrix_new

    def magnitude(self, matrix):
        """Computes the magnitude of the input matrix (iDFT)
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the complex matrix"""
        row,column=np.shape(matrix)
        matrix_new=np.zeros((row,column))
        for i in range(row):
            for j in range(column):
                matrix_new[i,j]=mt.sqrt((matrix[i,j].real**2)+(matrix_new[i,j].imag**2))
        return matrix_new
