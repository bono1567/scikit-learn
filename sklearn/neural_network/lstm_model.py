import numpy as np
from ..preprocessing import LabelEncoder,OneHotEncoder
from ._base import ACTIVATIONS, DERIVATIVES
from ._lstm_util import ERROR,CONVERSION


sigmoid,tanh,softmax = ACTIVATIONS['logistic'],ACTIVATIONS['tanh'],ACTIVATIONS['softmax']
derivative_sigmoid,derivative_tanh = DERIVATIVES['logistic'],DERIVATIVES['tanh']
cross_entropy = ERROR['cross_entropy']
convert_to_one_hot_encoding = CONVERSION['convert_to_one_hot_encoding']

class LSTM_Classifier():
    def __init__(self, batch_size, number_of_lstm_neurons, number_of_features, number_of_cells, cell_each_label=True,
                 solver='adam', alpha=0.001, max_iter=100):
        self.B = batch_size
        self.H = number_of_lstm_neurons
        self.D = number_of_features
        self.n_of_cells = number_of_cells
        self.all_labels = cell_each_label
        self.optimizer = solver
        self.learning_rate = alpha
        self.epoch = max_iter

    def DeclareVariables(self):
        h = self.H
        d_h = self.H + self.D

        self.Wf = np.random.randn(self.n_of_cells, d_h, h) / np.sqrt(
            d_h / 2)  # np.sqrt(..) is to prevent vanishing gradient in tanh activation function
        self.Wi = np.random.randn(self.n_of_cells, d_h, h) / np.sqrt(d_h / 2)
        self.Wc = np.random.randn(self.n_of_cells, d_h, h) / np.sqrt(d_h / 2)
        self.Wo = np.random.randn(self.n_of_cells, d_h, h) / np.sqrt(d_h / 2)
        self.Wn = np.random.randn(self.n_of_cells, h, self.number_of_labels) / np.sqrt(d_h / 2)
        self.bf = np.zeros((self.n_of_cells, 1, h))
        self.bi = np.zeros((self.n_of_cells, 1, h))
        self.bc = np.zeros((self.n_of_cells, 1, h))
        self.bo = np.zeros((self.n_of_cells, 1, h))
        self.bn = np.zeros((self.n_of_cells, 1, self.number_of_labels))

        zero = np.zeros((self.n_of_cells, self.B, h))
        zero_b = np.zeros((self.n_of_cells, 1, h))
        self.f = zero
        self.i = zero
        self.c = zero
        self.o = zero
        self.C_t = zero
        self.H_t = zero
        self.input_softmax = np.zeros((self.n_of_cells, self.B, self.number_of_labels))
        self.y_pred = np.zeros((self.n_of_cells, self.B, self.number_of_labels))

        if (self.optimizer == 'adam'):
            self.Vdwn = np.zeros((self.n_of_cells, h, self.number_of_labels))
            self.Vdwf = zero
            self.Vdwi = zero
            self.Vdwc = zero
            self.Vdwo = zero

            self.Vdbn = np.zeros((self.n_of_cells, 1, self.number_of_labels))
            self.Vdbf = zero_b
            self.Vdbi = zero_b
            self.Vdbc = zero_b
            self.Vdbo = zero_b

            self.Sdwn = np.zeros((self.n_of_cells, h, self.number_of_labels))
            self.Sdwf = zero
            self.Sdwi = zero
            self.Sdwc = zero
            self.Sdwo = zero

            self.Sdbn = np.zeros((self.n_of_cells, 1, self.number_of_labels))
            self.Sdbf = zero_b
            self.Sdbi = zero_b
            self.dbc = zero_b
            self.Sdbo = zero_b

    def Formulate_feedforward_cell(self, C_prev, X, cell_no):
        Wf = self.Wf[cell_no]
        bf = self.bf[cell_no]
        Wi = self.Wi[cell_no]
        bi = self.bi[cell_no]
        Wc = self.Wc[cell_no]
        bc = self.bc[cell_no]
        Wo = self.Wo[cell_no]
        bo = self.bo[cell_no]
        Wn = self.Wn[cell_no]
        bn = self.bn[cell_no]

        # Forget gate
        self.f[cell_no] = sigmoid(np.matmul(X, Wf) + bf)

        # C for the present cell
        self.i[cell_no] = sigmoid(np.matmul(X, Wi) + bi)
        self.c[cell_no] = tanh(np.matmul(X, Wc) + bc)
        self.C_t[cell_no] = C_prev * self.f[cell_no] + self.i[cell_no] * self.c[cell_no]
        self.o[cell_no] = sigmoid(np.matmul(X, Wo) + bo)
        self.H_t[cell_no] = self.o[cell_no] * tanh(self.C_t[cell_no])

        # We add a dense layer then a softmax layer
        self.input_softmax[cell_no] = np.matmul(self.H_t[cell_no], Wn) + bn
        self.y_pred[cell_no] = softmax(self.input_softmax[cell_no])

        return self.C_t[cell_no], self.H_t[cell_no]

    def Formulate_backpropagation_cell(self, cell_no, X, C_next, H_next, Y_true):
        f = self.f[cell_no]
        i = self.i[cell_no]
        c = self.c[cell_no]
        o = self.o[cell_no]
        C_t = self.C_t[cell_no]
        H_t = self.H_t[cell_no]
        y_pred = self.y_pred[cell_no]

        if (self.all_labels == True):
            E = cross_entropy(y_pred, Y_true)
            dE = y_pred - Y_true              # (B,number_of_categories) is the dimension of y_pred and y_true
            dWn = np.matmul(H_t.T, dE)

        if (self.all_labels == False):
            dH_t = H_next
        else:
            dH_t = np.matmul(dE, self.Wn[cell_no].T) + H_next  # BxH

        # O derivative
        do = derivative_sigmoid(o) * tanh(C_t) * dH_t  # BxH

        # C_t derivative
        if (self.all_labels == False):
            dC_t = C_next
        else:
            dC_t = o * dH_t * derivative_tanh(C_t) + + C_next  # BxH

        # forget gate derivative
        df = np.matmul(X.T, derivative_sigmoid(f) * dC_t * C_prev)  # BxH

        # i gate derivative
        di = np.matmul(X.T, derivative_sigmoid(i) * i * dC_t)  # BxH

        # c gate derivative
        dc = np.matmul(X.T, derivative_tanh(c) * c * dC_t)

        dXf = np.matmul(X, self.Wf[cell_no])
        dXi = np.matmul(X, self.Wi[cell_no])
        dXc = np.matmul(X, self.Wc[cell_no])
        dXo = np.matmul(X, self.Wo[cell_no])

        H_next = dXf + dXi + dXc + dXo
        C_next = dH_t * dC_t

        if (optimiser == 'adam'):
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            self.Vdwn[cell_no] = beta1 * self.Vdwn[cell_no] + (1 - beta1) * dWn
            self.Vdwf[cell_no] = beta1 * self.Vdwf[cell_no] + (1 - beta1) * dWf
            self.Vdwi[cell_no] = beta1 * self.Vdwi[cell_no] + (1 - beta1) * dWi
            self.Vdwc[cell_no] = beta1 * self.Vdwc[cell_no] + (1 - beta1) * dWc
            self.Vdwo[cell_no] = beta1 * self.Vdwo[cell_no] + (1 - beta1) * dWo

            self.Vdbn[cell_no] = beta1 * self.Vdbn[cell_no] + (1 - beta1) * np.reshape(
                np.sum(softmax_derivative(y_pred), axis=0), (1, self.number_of_labels))
            self.Vdbf[cell_no] = beta1 * self.Vdbf[cell_no] + (1 - beta1) * np.reshape(np.sum(df, axis=0), (1, self.H))
            self.Vdbi[cell_no] = beta1 * self.Vdbi[cell_no] + (1 - beta1) * np.reshape(np.sum(di, axis=0), (1, self.H))
            self.Vdbc[cell_no] = beta1 * self.Vdbc[cell_no] + (1 - beta1) * np.reshape(np.sum(dc, axis=0), (1, self.H))
            self.Vdbo[cell_no] = beta1 * self.Vdbo[cell_no] + (1 - beta1) * np.reshape(np.sum(do, axis=0), (1, self.H))

            self.Sdwn[cell_no] = beta2 * self.Sdwn[cell_no] + (1 - beta2) * dWn ** 2
            self.Sdwf[cell_no] = beta2 * self.Sdwf[cell_no] + (1 - beta2) * dWf ** 2
            self.Sdwi[cell_no] = beta2 * self.Sdwi[cell_no] + (1 - beta2) * dWi ** 2
            self.Sdwc[cell_no] = beta2 * self.Sdwc[cell_no] + (1 - beta2) * dWc ** 2
            self.Sdwo[cell_no] = beta2 * self.Sdwo[cell_no] + (1 - beta2) * dWo ** 2

            self.Sdbn[cell_no] = beta2 * self.Sdbn[cell_no] + (1 - beta2) * np.reshape(
                np.sum(softmax_derivative(y_pred), axis=0), (1, self.number_of_labels)) ** 2
            self.Sdbf[cell_no] = beta2 * self.Sdbf[cell_no] + (1 - beta2) * np.reshape(np.sum(df, axis=0),
                                                                                       (1, self.H)) ** 2
            self.Sdbi[cell_no] = beta2 * self.Sdbi[cell_no] + (1 - beta2) * np.reshape(np.sum(di, axis=0),
                                                                                       (1, self.H)) ** 2
            self.Sdbc[cell_no] = beta2 * self.Sdbc[cell_no] + (1 - beta2) * np.reshape(np.sum(dc, axis=0),
                                                                                       (1, self.H)) ** 2
            self.Sdbo[cell_no] = beta2 * self.Sdbo[cell_no] + (1 - beta2) * np.reshape(np.sum(do, axis=0),
                                                                                       (1, self.H)) ** 2

            dWn = self.Vdwn[cell_no] / np.sqrt(self.Sdwn[cell_no] + eps)
            df = self.Vdwf[cell_no] / np.sqrt(self.Sdwf[cell_no] + eps)
            dc = self.Vdwc[cell_no] / np.sqrt(self.Sdwc[cell_no] + eps)
            di = self.Vdwi[cell_no] / np.sqrt(self.Sdwi[cell_no] + eps)
            do = self.Vdwo[cell_no] / np.sqrt(self.Sdwo[cell_no] + eps)

            dbn = self.Vdbn[cell_no] / np.sqrt(self.Sdbn[cell_no] + eps)
            dbf = self.Vdbf[cell_no] / np.sqrt(self.Sdbf[cell_no] + eps)
            dbc = self.Vdbc[cell_no] / np.sqrt(self.Sdbc[cell_no] + eps)
            dbi = self.Vdbi[cell_no] / np.sqrt(self.Sdbi[cell_no] + eps)
            dbo = self.Vdbo[cell_no] / np.sqrt(self.Sdbo[cell_no] + eps)

        self.Wn[cell_no] = self.Wn[cell_no] - self.learning_rate * dWn
        self.Wf[cell_no] = self.Wf[cell_no] - self.learning_rate * df
        self.Wi[cell_no] = self.Wi[cell_no] - self.learning_rate * di
        self.Wc[cell_no] = self.Wc[cell_no] - self.learning_rate * dc
        self.Wo[cell_no] = self.Wo[cell_no] - self.learning_rate * do

        self.bn[cell_no] = self.bn[cell_no] - self.learning_rate * dbn
        self.bf[cell_no] = self.bf[cell_no] - self.learning_rate * dbf
        self.bi[cell_no] = self.bi[cell_no] - self.learning_rate * dbi
        self.bc[cell_no] = self.bc[cell_no] - self.learning_rate * dbc
        self.bo[cell_no] = self.bo[cell_no] - self.learning_rate * dbo

        return E, H_next, C_next

    def build_feed_forward_network(self, X):
        for i in range(self.n_of_cells):
            if i == 0:
                prev_C = np.random.randn(self.B, self.H)
                prev_H = np.random.randn(self.B, self.H)
            X_temp = np.concatenate([X[i], prev_H], axis=1)
            prev_C, prev_H = self.Formulate_feedforward_cell(prev_C, X_temp, i)
        return

    def build_backpropagation_network(self, Y_true, X, get_cell_labels):
        for i in range(self.n_of_cells, -1):
            Xtemp = np.concatenate([X[i], self.H_t[i]], axis=1)
            if i == self.n_of_cells:
                C_next = np.zeros((self.B, self.H))
                H_next = np.zeros((self.B, self.H))
                E, H_next, C_next = self.Formulate_backpropagation_cell(i, Xtemp, C_next, H_next, Y_true[i])
                if (get_cell_labels == False):
                    E, H_next, C_next = self.Formulate_backpropagation_cell(i, Xtemp, C_next, H_next, Y_true)
            if (get_cell_labels == True):
                E, H_next, C_next = self.Formulate_backpropagation_cell(i, Xtemp, C_next, H_next, Y_true[i])
            else:
                E, H_next, C_next = self.Formulate_backpropagation_cell(i, Xtemp, C_next, H_next, 0)
        return

    def preprocess_label(self, Y):
        y_copy = Y.copy()
        y_copy = np.reshape(y_copy, (-1, 1))

        label_encoder = LabelEncoder()
        one_hot_encoder = OneHotEncoder(sparse=False)

        temp = np.reshape(label_encoder.fit_transform(y_copy), (-1, 1))
        temp = one_hot_encoder.fit_transform(temp)

        if (self.all_labels == True):
            temp = np.reshape(temp, (self.n_of_cells, self.B, self.number_of_labels))
        return temp, label_encoder, one_hot_encoder

    def label_finishing(self, model_label):
        new = []
        for cell in self.y_pred:
            new_cell = []
            for sample in cell:
                new_cell.append(np.argmax(sample))
            new.append(new_cell)

        return np.reshape(model_label.inverse_transform(new), (self.n_of_cells, self.B, 1))

    def fit(self, X, Ytrue):
        self.number_of_labels = np.size(np.unique(Ytrue))
        Y, l_emodel, o_hemodel = self.preprocess_label(Ytrue)

        self.DeclareVariables()
        for i in range(self.epoch):
            self.build_feed_forward_network(X)
            self.build_backpropagation_network(Y, X, get_cell_labels=False)

        self.y_pred = convert_to_one_hot_encoding(self.y_pred)
        self.y_pred = self.label_finishing(l_emodel)
        if (self.all_labels == False):
            return self.y_pred[self.n_of_cells - 1]

        return self.y_pred