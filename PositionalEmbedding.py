import numpy as np
import tensorflow as tf

# Parameters
embedding_dim = 128  # Dimension of word embeddings
max_length = 50      # Maximum sequence length
vocab_size = 140000   # Vocabulary size (adjust this based on your dataset)

# Generate random word embeddings for demonstration (replace this with your trained embeddings)
word_embeddings = np.random.rand(vocab_size, embedding_dim).astype(np.float32)
print(word_embeddings)
# def get_positional_embeddings(max_length, embedding_dim):
#     positional_embeddings = np.zeros((max_length, embedding_dim))
#     for pos in range(max_length):
#         for i in range(embedding_dim):
#             if i % 2 == 0:
#                 positional_embeddings[pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
#             else:
#                 positional_embeddings[pos, i] = np.cos(pos / (10000 ** ((i - 1) / embedding_dim)))
#     return positional_embeddings

# # Generate the positional embeddings
# positional_embeddings = get_positional_embeddings(max_length, embedding_dim)

# class EmbeddingModel(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, max_length):
#         super(EmbeddingModel, self).__init__()
#         # Create the word embedding layer using pre-trained embeddings
#         self.word_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[word_embeddings], trainable=False)
        
#         # Create a trainable variable for positional embeddings
#         self.positional_embedding = tf.Variable(initial_value=positional_embeddings, trainable=False, dtype=tf.float32)

#     def call(self, inputs):
#         # inputs shape: (batch_size, sequence_length)
#         word_embeddings = self.word_embedding_layer(inputs)  # (batch_size, sequence_length, embedding_dim)
        
#         # Generate positions based on the input sequence length
#         positions = tf.range(tf.shape(inputs)[1])  # (sequence_length,)
#         positions = tf.expand_dims(positions, axis=0)  # (1, sequence_length)
        
#         # Gather positional embeddings for the input positions
#         position_embeddings = tf.gather(self.positional_embedding, positions)  # (batch_size, sequence_length, embedding_dim)

#         # Add word embeddings and positional embeddings
#         final_embeddings = word_embeddings + position_embeddings
#         return final_embeddings

# model = EmbeddingModel(vocab_size, embedding_dim, max_length)

# input_sequence = tf.constant([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]])  # Replace with your input data

# output = model(input_sequence)

# print("Output shape:", output.shape)  # Should print (batch_size, sequence_length, embedding_dim)
# print("Output embeddings:", output.numpy())  # Check the output embeddings
