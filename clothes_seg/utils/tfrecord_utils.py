import tensorflow as tf

def image_example(X, Y):
    '''
    Creates an image example.
    X: numpy ndarray: the input image data
    Y: numpy ndarray: corresponding label information, can be an ndarray, integer, float, etc

    Returns: tf.train.Example with the following features:
        dim0, dim1, dim2, ..., dimN, X, Y, X_dtype, Y_dtype

    '''
    feature = {}
    feature['X'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[X.tobytes()]))
    feature['Y'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[Y.tobytes()]))

    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_image(record, instance_size):
    features = {'X': tf.io.FixedLenFeature([], tf.string),
                'Y': tf.io.FixedLenFeature([], tf.string),
                }

    image_features = tf.io.parse_single_example(record, features=features)

    x = tf.io.decode_raw(image_features.get('X'), tf.uint8)
    x = tf.reshape(x, (*instance_size, 3))
    x = tf.cast(x, tf.float32)

    y = tf.io.decode_raw(image_features.get('Y'), tf.uint8)
    y = tf.reshape(y, (*instance_size, 1))
    y = tf.cast(y, tf.float32)

    return x, y
