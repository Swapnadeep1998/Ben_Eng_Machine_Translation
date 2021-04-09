from Utils.DataPipeline.inputData import LoadDataset
from Utils import config
import tensorflow as tf
from Utils.Models.encoder import Encoder
from Utils.Models.decoder import AttentionDecoder
import time

data_path = config.DATASET_PATH
batch_size = config.BATCH_SIZE
units = config.UNITS
embedding_dim = config.EMBEDDING_DIM
checkpoint_prefix = config.CHECKPOINT_PREFIX
epochs = config.EPOCHS

dataLoader = LoadDataset(data_path) 


inp_lang = dataLoader.input_lang_tokenizer
targ_lang = dataLoader.target_lang_tokenizer

vocab_inp_len = len(inp_lang.word_index)+1
vocab_targ_len = len(targ_lang.word_index)+1

steps_per_epoch = len(dataLoader.input_tensor_train)//batch_size

train_dataset = dataLoader.load_train_dataset(batch_size)
val_dataset = dataLoader.load_val_dataset(batch_size)


encoder = Encoder(vocab_inp_len, embedding_dim, units, batch_size)
decoder = AttentionDecoder(vocab_targ_len, embedding_dim, units, batch_size)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):            
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)            
            dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


if __name__=="__main__":    

    for epoch in range(epochs):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
