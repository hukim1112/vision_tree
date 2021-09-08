import tensorflow as tf

# Calculate spectral Norm of ||W.T*W - I||
@tf.keras.utils.register_keras_serializable(package='Custom', name='l2_orth')
class L2_Orthogonal(tf.keras.regularizers.Regularizer):
    def __init__(self, d_rate=0.01, w_rate=1e-4):
        self.d_rate = d_rate
        self.w_rate = w_rate
    def __call__(self, weights):
        w = weights
        if len(w.shape) == 4:
            H,W,C,N = w.shape
            row_dims = H*W*C
            col_dims = N
        else:
            D,N = w.shape
            row_dims = D
            col_dims = N
        w = tf.reshape(w, (row_dims, col_dims))
        wT = tf.transpose(w)
        ident = tf.eye(col_dims)

        gram = tf.matmul(wT,w)
        obj = gram - ident #[col_dim, col_dim]

        col_weights = tf.random.uniform((col_dims,1)) #[col_dim,1]
        v1 = tf.matmul(obj, col_weights) # [col_dim, 1]
        norm = tf.reduce_sum(tf.square(v1))**0.5
        normalized_v1 = tf.divide(v1,norm) # [col_dim, 1]
        v2 = tf.matmul(obj, normalized_v1) # [col_dim, 1]
        #https://github.com/VITA-Group/Orthogonality-in-CNNs/issues/3
        return self.d_rate*(tf.reduce_sum(tf.square(v2))**0.5) + self.w_rate*(tf.reduce_sum(tf.square(w))**0.5)
    def get_config(self):
        return {'d_rate': float(self.d_rate), 'w_rate' : float(self.w_rate)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='Inverse_l1_reg')
class Inverse_l1_reg(tf.keras.regularizers.Regularizer):
    def __init__(self, lamb=0.01):
        self.lamb = lamb
    def __call__(self, w):
        return -self.lamb * tf.math.reduce_sum(tf.math.abs(w))
    def get_config(self):
        return {'lamb': float(self.lamb)}
