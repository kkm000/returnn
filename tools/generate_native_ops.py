#!/usr/bin/env python3

"""
This explicitly compiles some of the native ops, and will tell you the so-filenames.
Normally all native ops (e.g. NativeLstm2 etc) are compiled on-the-fly within RETURNN.
When you export the computation graph (e.g. via ``compile_tf_graph.py``),
you explicitly must load these native ops.
"""

from __future__ import print_function

import os
import sys
import typing

import _setup_returnn_env  # noqa
from returnn import __main__ as rnn
from returnn.log import log
import returnn.util.basic as util


config = None  # type: typing.Optional["returnn.config.Config"]


def init(log_verbosity):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  rnn.init_config(config_filename=None, command_line_options=[])
  global config
  config = rnn.config
  config.set("log", None)
  config.set("log_verbosity", log_verbosity)
  config.set("use_tensorflow", True)
  rnn.init_log()
  print("Returnn compile-native-op starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  if 'network' in config.typed_dict:
    print("Loading network")
    from returnn.tf.network import TFNetwork
    network = TFNetwork(
      name="root",
      config=config,
      rnd_seed=1,
      train_flag=False,
      eval_flag=True,
      search_flag=False)
    network.construct_from_dict(config.typed_dict["network"])


def main():
  """
  Main entry.
  """

  def parse_args():
    'Scope argument parsing under carpet.'
    import argparse as ap
    p = ap.ArgumentParser(description='Generate kernel sources for native ops.',
                          formatter_class=ap.ArgumentDefaultsHelpFormatter)
    a = p.add_argument
    a('native_ops', nargs='+', metavar='OP_NAME',
      help='op name, e. g., "LstmGenericBase"')
    a('--output_dir', metavar='DIR', default='.', type=os.path.realpath,
      help='directory to output files, must exist')
    a("--verbosity", default=4, type=int, help='0 to 5')
    return p.parse_args()

  args = parse_args()
  print(f'{args}')
  init(log_verbosity=args.verbosity)

  from returnn.tf.util.basic import CudaEnv, NativeCodeCompiler
  CudaEnv.verbose_find_cuda = True
  NativeCodeCompiler.CollectedCompilers = []

  import returnn.native_op as native_op
  from returnn.tf.native_op import make_op, OpMaker
  for op_name in args.native_ops:
    print(f"Loading native op {op_name}")
    op_gen = getattr(native_op, op_name)
    assert issubclass(op_gen, native_op.NativeOpGenBase)
    op = make_op(op_gen, compiler_opts={"verbose": True},
                 search_for_numpy_blas=False)
    while op:
      # The three separate files do not work, because the included .cpp
      # defines duplicate symbols not matked inline. Try saving as one.
      # for ext, src in zip(['.h', '_cpu.cc', '_gpu.cu'], op.op_sources):
      #   if not src.strip(): continue
      #   fname = os.path.join(args.output_dir, op.op_name + ext)
      #   print(f'Writing out {fname}')
      #   with open(fname, "w") as fh:
      #     if ext != '.h': fh.write(f'#include "{op.op_name}.h"\n')
      #     fh.write(src)
      src = op.op_sources  # Intended .h, .cc, .cu, in order.
      ext = '_gpu.cu' if src[2].strip() else '_cpu.cc'
      fname = os.path.join(args.output_dir, op.op_name + ext)
      print(f'Writing out {fname}')
      with open(fname, "wt") as fh:
        fh.writelines(src)

      op = getattr(op, 'grad_op', None)

  return 0

  libs = []
  if OpMaker.with_cuda and OpMaker.tf_blas_gemm_workaround:
    print('CUDA BLAS lib:', OpMaker.cuda_blas_gemm_so_filename())
    libs.append(OpMaker.cuda_blas_gemm_so_filename())
  elif OpMaker.with_cuda is False:
    print('No CUDA.')

  for compiler in NativeCodeCompiler.CollectedCompilers:
    assert isinstance(compiler, NativeCodeCompiler)
    print(compiler)
    # noinspection PyProtectedMember
    libs.append(compiler._so_filename)

  if libs:
    print("libs:")
    for fn in libs:
      print(fn)
  else:
    print("no libs compiled. use --native_op or --config")

  if args.output_file:
    with open(args.output_file, "w") as f:
      for fn in libs:
        f.write(fn + '\n')
    print("Wrote lib list to file:", args.output_file)


if __name__ == '__main__':
  sys.exit(main())
