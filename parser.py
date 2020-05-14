"""Conditional ArgumentParser class enabling easy loading and saving of human-readable
configuration files, dictionary arguments editing through command-line, and conditional
list of arguments.
"""

import argparse
import json
from pathlib import Path
import re

class Parser:
    def __init__(self, **kwargs):

        cli_parser = argparse.ArgumentParser(add_help=False)

        cli = cli_parser.add_argument_group('command line arguments')

        ## FIRST ORDER OPTIONS
        cli.add_argument('-c', '--config', type=Path, metavar='PATH', default='configs/defaults.json',
            help='configuration file')
        cli.add_argument('-d', '--debug', action='store_true',
            help='debug mode')
        cli.add_argument('-l', '--load_model', type=Path, metavar='PATH',
            help='model to load')
        cli.add_argument('-t', '--test', action='store_true',
            help='test variations')

        parser = argparse.ArgumentParser(
            parents=[cli_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=True,
            **kwargs
        )

        # GROUP OF ARGUMENTS
        dataset = parser.add_argument_group('dataset-related options')
        network = parser.add_argument_group('architecture of the model')
        hparams = parser.add_argument_group('hyperparameters')
        diverse = parser.add_argument_group('miscellaneous')

        # MAIN ARGUMENTS
        dataset.add_argument('--batch_size', type=int, default=256,
            help='how many samples per batch to load')
        dataset.add_argument('--dataset_name', type=str, default='bach_chorales_beats',
            help='name of the dataset used')
        dataset.add_argument('--dataset_type', type=str, choices=['bach'], default='bach',
            help='type of dataset used')
        dataset.add_argument('--metadatas', type=str, nargs='*', default=[],
            help='metadata potentially used as features')
        dataset.add_argument('--sequences_size', type=int, default=2,
            help='size of generated sequences in number of beats')

        network.add_argument('--features', type=str, nargs='+', default='num_notes',
            help='type of features used')
        network.add_argument('--architecture', type=str, choices=['rnn', 'transformer'], default='rnn',
            help='architecture used for encoder and decoder')
        network.add_argument('--vq_size', type=int, default=0,
            help='size of discrete latent space for VQ (0 for no VQ)')

        hparams.add_argument('--encoder', type=dict, default={},
            help='keyword arguments of the decoder')
        hparams.add_argument('--decoder', type=dict, default={},
            help='keyword arguments of the decoder')
        hparams.add_argument('--discriminator', type=dict, default={},
            help='keyword arguments of the decoder')
        hparams.add_argument('--adam', type=dict, default={},
            help='hyperparameters of the main optimizer')
        hparams.add_argument('--discopt', type=dict, default={},
            help='hyperparameters of the optimizer of the discriminator')
        hparams.add_argument('--featopt', type=dict, default={},
            help='hyperparameters of the optimizer of the features')
        hparams.add_argument('--style_token_dim', type=int, default=16,
            help='dimension of the features space')
        hparams.add_argument('--z_dim', type=int, default=16,
            help='dimension of the latent space')
        hparams.add_argument('--note_embedding_dim', type=int, default=256,
            help='dimension of the intermediate note embeddings in encoder/decoder')
        hparams.add_argument('--weights', type=dict, default={},
            help='ponderation of the different loss terms')

        diverse.add_argument('--device', '--gpu', type=int, default=0,
            help='which GPU to use (-1 for CPU)')
        diverse.add_argument('--num_workers', type=int, default=0,
            help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
        # diverse.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
        #     help='Show this help message and exit.')

        args, _ = cli_parser.parse_known_args()

        args.train = not args.test

        # CONDITIONAL ARGUMENTS
        if args.train:
            train = parser.add_argument_group('training options')

            train.add_argument('--epochs', type=int, default=50,
                help='number of epochs of training')
            train.add_argument('--save_freq', type=int, default=5,
                help='frequency checkpoints and results are saved')

        else:
            test = parser.add_argument_group('options for test variations')

            test.add_argument('experiments', type=str, nargs='*', default='*',
                help='experiments to perform')
            test.add_argument('--n_examples', type=int, default=4,
                help='how many sequences tests are performed on')
            test.add_argument('--n_samples', type=int, default=4,
                help='how many identical experiments to run')
            test.add_argument('--list', action='store_true',
                help='only list available tests')


        if args.load_model and (args.load_model / 'config.json').exists():
            args.config = args.load_model / 'config.json'

        if args.config:
            # loading configuration file
            with open(args.config, 'r') as cfg_file:
                parser.set_defaults(**json.load(cfg_file))

        self.parser = parser

        # arguments from those groups should be saved in the config file of the experiments
        self.groups_to_save = [dataset, network, hparams]

    def __repr__(self):
        return repr(self.parser)

    def parse_args(self, *args, **kwargs):
        r"""Parse the arguments of the program
        """
        args, unknown = self.parser.parse_known_args(*args, **kwargs)
        args.train = not args.test

        dict, key = None, None
        for arg in unknown:
            match = re.match('-+(.+?)_(.+?)\Z', arg)    # detect options of kind --sth_sth
            if match is not None:
                dict, key = match.group(1), match.group(2)
            else:
                try:
                    getattr(args, dict)[key] = self.infer_type(arg)
                except AttributeError:
                    self.parser.print_usage()
                    print(f"{self.parser.prog}: error: unrecognized arguments: --{dict}_{key}")
                    exit(2)
                except TypeError:
                    self.parser.print_usage()
                    print(f"{self.parser.prog}: error: unrecognized arguments:", arg)
                    exit(2)
        return args

    def save(self, args, path):
        r"""Save the parsed options
        """
        path = Path(path)
        path.mkdir(exist_ok=True)
        args_dict = {}

        for group in self.groups_to_save:
            args_dict.update({arg.dest : getattr(args, arg.dest) for arg in group._group_actions})

        with open(path / 'config.json', 'w') as fp:
            json.dump(
                args_dict,
                fp,
                indent=4,
                default=lambda arg: str(arg),
                sort_keys=True
            )

    @staticmethod
    def infer_type(s):
        r"""Converts a string into the most probable type:
            - int if it is composed of digits only
            - float if it can be converted to float
            - str otherwise
        """
        if s.isdigit():
            return int(s)
        try:
            return float(s)
        except ValueError:
            return s
