# Python parser for efficient and simple handling of program arguments

This project consists in an efficient and easy-to-use implementation of a command-line parser for Python programs. It is based on [argparse](https://docs.python.org/3/library/argparse.html) and thus is very similar to it. Some features of this parser are:

- Clean and unified handling of default arguments, command-line options and configuration files
- Easy and configurable options saving and loading, e.g. to restore options from previous experiments
- Handling of dictionary-like options
- Conditional arguments
- Usage very similar to `argparse.ArgumentParser`

File `parser.py` is given with an example set of options originally used for a deep learning project. It thus has to be adapted to each one's project.

## Installation

This project is a single Python file, whose dependencies all come from the standard library of Python. Therefore the easiest way to use it is to directly copy-paste the file `parser.py` in your project and to adapt the set of options to the one you need.

## Usage

A parser can be used basically like `argparse.ArgumentParser`:

```python
from parser import Parser

parser = Parser()
args = parser.parse_args()
```

Everything that works with classical argparse parsers also work with that one.

Different options should be specified in the `parser.py` file, with almost the same syntax as in argparse. Only the `__init__` method of `Parser` class should be adapted to the project. Other methods should not be modified.

### Configuration files

Options can be specified by three different ways to a program: using default arguments in the argument definition, using a `json` configuration file and using directly command-line arguments. Argparse already support default arguments and command-line arguments, however some programs might have many different options and several sets of options. In that case, configuration files are very convenient.

To load arguments from a configuration file, the syntax is:

```sh
python my_program.py -c path/to/config.json
```

Note that you can mix argparse default arguments, configuration files and command-line options. In case of conflicts, the behavior is the following:

- Options from configuration files override default arguments
- Command-line options override options from configuration files

*Note that you can specify a default configuration file by setting the default value of `-c` option.*

### Loading and saving options from experiments

#### Saving

Sometimes one realize a lot of experiments and would like to save the set of options used for each experiment. The `save` method of parser enables to do that:

```python	
parser = Parser()
args = parser.parse_args()

exp_path = 'experiments/exp1'
parser.save(args, exp_path)
```

In this example, a file called `config.json` will be created in folder `experiments/exp1`.

#### Loading

Then, assuming that one wants to restore the set of options used in an experiment, the way to do it is to use `-l/--load_model` option:

```sh
python my_program.py -l experiments/exp1
```

If `experiments/exp1/config.json` is a valid path, it will override any default configuration file provided. Otherwise, it does not change anything wrt to the arguments.

The following example shows how `-l` option can be simply used in the context of deep learning.

```python
from datetime import datetime

import torch

from model import Model
from parser import Parser

parser = Parser()
args = parser.parse_args()

model = Model(args.option1, args.option2, ...)

if args.load_model:
    exp_path = args.load_model
    model.load_state_dict(exp_path + '/checkpoint.pth')
else:
    exp_path = 'experiments/' + datetime.now().strftime('%Y-%m-%d_%H:%M')
    parser.save(args, exp_path)
    
...

torch.save(model.state_dict(), exp_path + '/checkpoint.pth')
```

When a model is loaded, the options of the associated experiment are loaded too. Otherwise, a new configuration file is created in the folder containing the checkpoints.

#### Saving only specific options

In general, some options change the behavior of an experiment whereas some don't. Storing some options can even in some cases be annoying.

In order to handle that case, the proposed solution is to use groups of options, through the `add_argument_group` method of argparse parsers. In `parser.py`, one can split the arguments of the program into groups of options. Then, groups that contain  options that should be stored by method `save` should be indicated in attribute `groups_to_save` of the `Parser` class.

### Conditional arguments

Some arguments should be provided only in specific cases which depend on other arguments. For instance, in machine learning, there is usually two phases: training a model and evaluating it. It thus could be convenient to have specific options only for the training phase (number of epochs, learning rate scheduler, etc.) and others for the test phase. This example is implemented in `parser.py`, and can easily be adapted to other contexts.

### Dictionary-like options

One can provide options that are dictionaries, i.e. for storing all the keyword-arguments of a specific function into one single argument. Such dictionary-like options are as convenient to use as numbers or strings in configuration files, but modifying them on the command-line could be painful, especially if one wants to modifiy only one key of a dictionary which has a lot of keys.

If one wants to set a specific key of some dictionary option, it can use the syntax `--<option>_<key>`, which will only override the corresponding key.

Another example from the deep learning world. Assume that you want to train a neural network with optimizer [Adam](https://arxiv.org/abs/1412.6980). This optimizer has several hyperparameters. With `argparse.ArgumentParser`, a way to handle those options would be to do something like that:

```python	
parser.add_argument('--adam_lr', type=float, default=0.001)
parser.add_argument('--adam_eps', type=float, default=1e-8)
parser.add_argument('--adam_amsgrad', action='store_true')
...
```

Then, when you define your optimizer, you should do something like this:

```python
args = parser.parse_args()
...
optimizer = torch.optim.Adam(
    params,
    lr=args.adam_lr,
    eps=args.adam_eps,
    amsgrad=args.adam_amsgrad,
    ...
)
```



With this parser, a nice way to do it is to use a dictionary option

```python
parser.add_argument('adam', type=dict, default={})
```

...and to eventually provide default values in a configuration file.

Then you can just define your optimizer like this:

```python
args = parser.parse_args()
...
optimizer = torch.optim.Adam(params, **args.adam)
```

Now assume that you want to realize an experiment where you want to change only the learning rate of the optimizer. It would be boring to create a new configuration file only for that, and even more boring (and potentially dangerous) to provide the new set of keyword arguments on the command-line, like this:

```sh
python my_program.py --adam {'lr': 0.01, 'eps':1e-8, 'amsgrad': False, ...}
```

With this parser, what you can do is just to use the following syntax:

```sh
python my_program.py --adam_lr 0.01
```

All keys from `args.adam` will have their default value except `args.adam['lr']` which will be equal to 0.01 instead of its default value.

**Warning 1: Type inference for dictionary keys** *Since Python dictionaries can contain values from different types, one can have type errors using this feature when using types more complicated than strings or numbers for dictionary keys. The policy for type inference is to convert to int if the option is only composed of digits, to float if possible and to string otherwise.*

**Warning 2: No underscore in dictionary options** *Since one can use `--<dict_option>_<key>` syntax on the command-line, it would be dangerous to include `_` in the name of the dictionary option. In fact, if you create a dictionary option called `my_dict`, then typing `--my_dict_key sth` on the command-line will fail since the parser will try to set the key `dict_key`  of  dictionary option `my` to value `sth`.*



No rights reserved. Feel free to copy-paste the code without citing for any purpose.