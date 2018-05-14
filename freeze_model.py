import tensorflow as tf

tf.flags.DEFINE_string('CHECKPOINT_DIR', './', 'checkpoint directory')
tf.flags.DEFINE_string('OUTPUT_NODE_NAMES', './', 'output node names')
tf.flags.DEFINE_string('OUTPUT_GRAPH_PATH', './', 'output graph directory')


FLAGS = tf.flags.FLAGS


def freeze_model(checkpoint_dir, output_node_names, output_graph_path='./frozen_model.pb'):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    with tf.Graph().as_default() as graph:
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file), clear_devices=True)
        print('Import model {} successfully'.format(checkpoint_file))
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=config) as sess:
            saver.restore(sess, checkpoint_file)

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_node_names.split(",")  # The output node names are used to select the usefull nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph_path + '/frozen_model.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

            return output_graph_def


def main(argv=None):
    freeze_model(FLAGS.CHECKPOINT_DIR, FLAGS.OUTPUT_NODE_NAMES, FLAGS.OUTPUT_GRAPH_PATH)


if __name__ == '__main__':
    tf.app.run()
