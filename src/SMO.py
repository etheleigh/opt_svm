import numpy as np
from typing import Tuple, Optional

class SVM_classifier:
    def __init__(self, X, y, kernel: str = 'linear', C: float = 1, epsilon: float = 1e-8, tol: float = 0.001,
                 max_iter: int = 500):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.kernel_func = self.select_kernel(self.kernel)
        self.C = C
        self.epsilon = epsilon  # error margin
        self.tol = tol  # tolerance for KKT
        self.max_iter = max_iter
        self.m, self.n = np.shape(self.X)  # m is number of samples, n number of features

        self.alphas = np.zeros(self.m)
        self.Error_cache = np.zeros(self.m)  # - y

        # If the kernel is linear we can store a single weight vector and use the alternative implemented in SVM

        self.w = np.zeros(self.n)
        self.b = 0  # intercept

    def select_kernel(self, kernel: str):

        ''' We have to choose a kernel based on the kernel type argument
        here we can use only linear or the gaussion, no other kernels are available'''

        if kernel == 'linear':
            return self.linear_kernel
        elif kernel == 'rbf':
            return self.rbf_kernel
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 @ x2.T

    def rbf_kernel(self, x1, x2):

        """
        RBF kernel implementation, i.e. K(u,v) = exp(-gamma_rbf*|u-v|^2).
        gamma_rbf is a hyper-parameter of the model.

        """
        # we use the default parameter gamma=1
        gamma = 1

        # In case u, v are vectors, convert to row vector
        if np.ndim(x1) == 1:
            x1 = x1[np.newaxis, :]

        if np.ndim(x2) == 1:
            x2 = x2[np.newaxis, :]

        dist_squared = np.linalg.norm(x1[:, :, np.newaxis] - x2.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)

        return np.exp(-gamma * dist_squared)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Predicts the labels for the instance
            and the respective score."""

        if self.kernel != 'linear':
            scores = (self.alphas * self.y) @ self.kernel_func(self.X, x) - self.b

        else:
            scores = self.w @ x.T - self.b

        pred = np.sign(scores)

        return pred, scores

    def get_error(self, i: int) -> float:
        """
        Computes the error for each instance
        :param i: i-th instance
        :return: the difference between scores
        """
        return self.predict(self.X[i, :])[1] - self.y[i]

    def get_params(self) -> dict:
        """
        Returns the parameter dictionary for model parameters

        """
        if self.kernel != 'linear':
            self.w = ((self.alphas * self.y) @ self.X)
        return {'w': self.w, 'b': self.b}

    def take_step(self, i1: int = None, i2: int = None) -> int:
        """
        takes one step of the SMO algorithm
        :param i1: i1-th training instance
        :param i2: i2-th training instance
        :return: 1 if success else 0
        """
        # print("I an in take_step.")
        if i1 == i2:
            # print("i1==i2\n0 returned.")
            return 0

        # Set all required parameters
        a1 = self.alphas[i1]
        a2 = self.alphas[i2]

        x1 = self.X[i1, :]
        x2 = self.X[i2, :]

        y1 = self.y[i1]
        y2 = self.y[i2]

        E1 = self.get_error(i1)
        E2 = self.get_error(i2)

        # Define parameter s
        s = y1 * y2

        # Compute L, H via equations (13) and (14) from Platt
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)

        if L == H:
            # print("L==H\n0 returned.")
            return 0

        k11 = self.kernel_func(x1, x1)
        k22 = self.kernel_func(x2, x2)
        k12 = self.kernel_func(x1, x2)

        # Compute the second derivative of the objective function along the diagonal line
        eta = k11 + k22 - 2.0 * k12

        if eta > 0:
            # Normal circumstances, using Equations (16)-(18) to compute a1 and a2
            a2_new = a2 + y2 * (E1 - E2) / eta

            if a2_new >= H:
                a2_new = H
            if a2_new <= L:
                a2_new = L
        else:
            # Strange case, we use Equations (19)
            f1 = y1 * (E1 + self.b) - a1 * k11 - s * a2 * k12
            f2 = y2 * (E2 + self.b) - s * a1 * k12 - a2 * k22
            L1 = a1 + s * (a2 - L)
            H1 = a1 + s * (a2 - H)
            psi_L = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k11 + 0.5 * L * L * k22 + s * L * L1 * k12
            psi_H = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k11 + 0.5 * H * H * k22 + s * H * H1 * k12

            if psi_L < (psi_H - self.epsilon):
                a2_new = L
            elif psi_L > (psi_H + self.epsilon):
                a2_new = H
            else:
                a2_new = a2

        if a2 < self.epsilon:
            a2 = 0.0
        elif a2 > self.C - self.epsilon:
            a2 = self.C

        if np.abs(a2_new - a2) < (self.epsilon * (a2_new + a2 + self.epsilon)):
            # print("off numerical tolerance\n0 returned.")
            return 0

        # Calculate a1_new
        a1_new = a1 + s * (a2 - a2_new)

        # Push alphas to boundaries
        if a1_new < self.epsilon:
            a1_new = 0
        if a1_new > (self.C - self.epsilon):
            a1_new = self.C

        # Update threshold b
        b1 = self.b + E1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12
        b2 = self.b + E2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22

        if 0 < a1_new < self.C:
            b_new = b1
        elif 0 < a2_new < self.C:
            b_new = b2
        else:
            b_new = 0.5 * (b1 + b2)

        # Update weight's vector if Linear kernel
        if self.kernel == 'linear':
            self.w = self.w + y1 * (a1_new - a1) * x1 + y2 * (a2_new - a2) * x2

        # Update Error_cache using alphas (see reference)

        # if a1 & a2 are not at bounds, the error will be 0
        self.Error_cache[i1] = 0
        self.Error_cache[i2] = 0

        # Update error for non boundary elements
        inner_indices = [idx for idx, a in enumerate(self.alphas) if 0 < a < self.C]
        for i in inner_indices:
            self.Error_cache[i] += \
                y1 * (a1_new - a1) * self.kernel_func(x1, self.X[i, :]) \
                + y2 * (a2_new - a2) * self.kernel_func(x2, self.X[i, :]) \
                + (self.b - b_new)

        # Update alphas
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        # Update b
        self.b = b_new

        # print("successfull pass")
        return 1  # sucessfull pass

    def examine_example(self, i2: int = None):
        """
        Examine the i2-th example in the algorithm to determine
        if eligible for usage in optimization pair
        :param i2: example to examine
        :return: 1 if successful, 0 if not
        """

        y2 = self.y[i2]
        a2 = self.alphas[i2]
        E2 = self.get_error(i2)
        r2 = E2 * y2
        # print("I am in examine_example()")
        # Check if error is within tolerance
        if (r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
            # print("Error within tolerance.")
            # If there are more than one non-bound elements use the second heuristic
            if np.count_nonzero((0 < self.alphas) & (self.alphas < self.C)) > 1:

                # use section 2.2 to select i1
                if E2 > 0:
                    i1 = np.argmin(self.Error_cache)
                else:
                    i1 = np.argmax(self.Error_cache)

                if self.take_step(i1, i2):
                    return 1

            # Loop over all non-zero and non-C alpha, starting at a random point

            # Get indices where 0 < alpha < self.C
            i1_array = np.where((0 < self.alphas) & (self.alphas < self.C))[0]

            # Roll the array by a random number of positions to ensure that we will pass all
            random_shift = np.random.choice(np.arange(self.m))
            i1_list = np.roll(i1_array, random_shift)

            # Loop over all non-boundary elements
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

            # Loop over all possible alpha elements, starting at a random point
            i1_list = np.roll(np.arange(self.m), np.random.choice(np.arange(self.m)))
            for i1 in i1_list:
                if self.take_step(i1, i2):
                    return 1

        return 0

    def fit(self) -> None:
        """This is the equivalent of the main routine in the original SMO paper.
            We use it for training the algorithm."""
        iteration_number = 0  # We count the number of iterations and bounded below max_iter
        numbers_changed = 0
        examine_all = True

        while numbers_changed > 0 or examine_all:

            if iteration_number >= self.max_iter:
                break

            numbers_changed = 0
            if examine_all:
                # Loop over all training examples
                for i in range(self.m):
                    numbers_changed += self.examine_example(i)

            else:
                # Loop i over examples where alpha is not 0 & not C
                i_array = np.where((0 < self.alphas) & (self.alphas < self.C))[0]
                for i in i_array:
                    numbers_changed += self.examine_example(i)

            if examine_all:
                examine_all = False
            if numbers_changed == 0:
                examine_all = True

            iteration_number += 1

            # print(f"Iteration:{iteration_number} ended.")
            # return self.b, self.w


