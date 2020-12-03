import tensorflow as tf
import setGPU
import numpy as np
import resource
import argparse
import os

import pofah.path_constants.sample_dict_file_parts_input as sdi
import sarewt.data_reader as dare
import util.data_generator as dage



def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return "%s: usertime=%.3f systime=%.3f mem=%.3f mb" % (point, usage[0], usage[1], usage[2]/1024.0 )


def track_usage(last_usage, msg=''):
    start_t_user, start_t_sys, start_m, *_ = last_usage 
    curr_usage = resource.getrusage(resource.RUSAGE_SELF)
    end_t_user, end_t_sys, end_m, *_ = curr_usage
    print('{}: t_user {:.2f}, t_sys {:.2f}, mem {:.2f} mb, delta-mem {:.2f} mb'.format(msg, end_t_user-start_t_user, end_t_sys-start_t_sys, end_m/1024.0, (end_m-start_m)/1024.0))
    return curr_usage


# 2 step generator (file-chunks, sample-chunks)
def rand_array_generator(sample_n=1e6):
    while True:
        # yield sample_n samples "per file"
        s = np.random.random(size=(int(sample_n),3))
        yield s, s


def sample_generator(file_max_n=2, samples_per_file_n=1e5):
    file_n = 0
    for const, feat in rand_array_generator(sample_n=samples_per_file_n):
        file_n += 1
        indices = list(range(len(const)))
        while indices:
            idx = indices.pop()
            yield const[idx], feat[idx]
        if file_n >= file_max_n:
            break


def sample_generator_batched(file_max_n=2, samples_per_file_n=1e5, batch_sz=512):
    file_n = 0
    for const, feat in rand_array_generator(sample_n=samples_per_file_n):
        file_n += 1
        curr_idx = 0
        while curr_idx < len(const):
            yield const[curr_idx:curr_idx+batch_sz], feat[curr_idx:curr_idx+batch_sz]
            curr_idx += batch_sz
        if file_n >= file_max_n:
            break


def sample_from_file_generator(max_loops=10, samples_in_parts_n=1e4):
    path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir']['qcdSide'])
    data_reader = dare.DataReader(path)
    
    for loop_n, (constituents, features) in enumerate(data_reader.generate_event_parts_from_dir(parts_n=samples_in_parts_n)):
        indices = list(range(len(constituents)))
        while indices:
            index = indices.pop(0)
            next_sample = constituents[index] #.copy() 
            yield next_sample, next_sample  # x == y in autoencoder
        if loop_n >= max_loops:
            break


def sample_from_train_generator(samples_in_parts_n=1e4, samples_max_n=1e6):
    ''' samples from original data generator as used in training '''
    path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir']['qcdSide'])
    return dage.DataGenerator(path=path, samples_in_parts_n=samples_in_parts_n, samples_max_n=samples_max_n) # generate samples_max_n jet samples


def loop(generator, txt='', print_stop=1e3):
    print( '\n {} \t {} \t {}\n'.format('*'*10, txt, '*'*10))
    total_read_n = 0
    curr_usage = resource.getrusage(resource.RUSAGE_SELF)
    for i, (const, const) in enumerate(generator):
        try:
            m = np.max(const) # do sth with data
        except BaseException:
            from traceback import print_stack; print_stack()
            from IPython import embed; embed()
        l = len(const)
        total_read_n += l
        # print elapsed time and usage every now and then
        if not i%int(print_stop):
            curr_usage = track_usage(curr_usage, 'iter {} (n={}, max={:.2f})'.format(i,l,m))
    print('total number samples read: {}'.format(total_read_n))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='profile generator memory usage')
    parser.add_argument('-s', dest='std', action='store_true', help='standard loop')
    parser.add_argument('-t1', dest='tfds1', action='store_true', help='dataset from generator simple')
    parser.add_argument('-t2', dest='tfds2', action='store_true', help='dataset from generator take')
    parser.add_argument('-t3', dest='tfds3', action='store_true', help='dataset from generator batch')
    parser.add_argument('-t4', dest='tfds4', action='store_true', help='dataset from generator take & batch')
    parser.add_argument('-t5', dest='tfds5', action='store_true', help='dataset from batched generator')
    parser.add_argument('-f1', dest='fileds1', action='store_true', help='simple file part generator')
    parser.add_argument('-f2', dest='fileds2', action='store_true', help='dataset from file part generator')
    parser.add_argument('-f3', dest='fileds3', action='store_true', help='dataset from file part generator batch')
    parser.add_argument('-f4', dest='fileds4', action='store_true', help='dataset from file part generator take & batch')
    parser.add_argument('-f5', dest='fileds5', action='store_true', help='dataset from train generator take & batch')

    args = parser.parse_args()

    file_max_n = 2
    samples_per_file_n = 1e4
    print_stop = 1e3
    batch_sz = 512

    if args.std:

        # standard generator loop
        gen = sample_generator(file_max_n=file_max_n, samples_per_file_n=samples_per_file_n)
        loop(gen, 'standard loop', print_stop=1000)      

    if args.tfds1:
    
        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_generator, args=[file_max_n, samples_per_file_n], output_types=(tf.float32, tf.float32))
        loop(tfds, 'tf dataset simple', print_stop=1000)

    if args.tfds2:
    
        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_generator, args=[file_max_n, samples_per_file_n], output_types=(tf.float32, tf.float32)).take(int(2e5))
        loop(tfds, 'tf dataset take')

    if args.tfds3:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_generator, args=[3, int(1e5)], output_types=(tf.float32, tf.float32)).batch(batch_sz)
        loop(tfds, 'tf dataset batch', print_stop=10)

    if args.tfds4:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_generator, args=[file_max_n, int(1e5)], output_types=(tf.float32, tf.float32)).take(int(3e5)).batch(batch_sz)
        loop(tfds, 'tf dataset take & batch', print_stop=10)
        
    if args.tfds5:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_generator_batched, args=[file_max_n, samples_per_file_n], output_types=(tf.float32, tf.float32))
        loop(tfds, 'tf dataset from batched generator', print_stop=10)
        
    if args.fileds1:
        # simple file generator
        gen = sample_from_file_generator(max_loops=4, samples_in_parts_n=int(5e5)) # 1.5M samples
        loop(gen, 'f1: standard file part generator', print_stop=int(1e5))

    if args.fileds2:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_from_file_generator, args=[file_max_n, int(1e4)], output_types=(tf.float32, tf.float32))
        loop(tfds, 'f2: tf dataset from file parts generator', print_stop=1000)

    if args.fileds3:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_from_file_generator, args=[4, int(5e5)], output_types=(tf.float32, tf.float32)).batch(batch_sz)
        loop(tfds, 'f3: tf dataset from file parts generator batched', print_stop=100)

    if args.fileds4:

        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(sample_from_file_generator, args=[4, int(5e5)], output_types=(tf.float32, tf.float32)).take(int(2e6)).batch(batch_sz)
        loop(tfds, 'f4: tf dataset from file parts generator take & batch', print_stop=100)

    if args.fileds5:

        path = os.path.join(sdi.path_dict['base_dir'], sdi.path_dict['sample_dir']['qcdSide'])
        gen = dage.DataGenerator(path=path, samples_in_parts_n=int(1e4), samples_max_n=int(1e6)) # generate samples_max_n jet samples
        # tf dataset from generator
        tfds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32)).batch(256, drop_remainder=True)
        for epoch in range(20):
            loop(tfds, 'f5: tf dataset from training generator batch epoch '+str(epoch), print_stop=100)
