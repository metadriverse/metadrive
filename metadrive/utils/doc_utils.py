from inspect import getsource
from pathlib import Path
from textwrap import dedent

import pygame
from PIL import Image
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from pygments.token import Keyword, Name, Comment, String, Error, \
    Number, Operator, Generic, Token, Whitespace, Punctuation


def generate_gif(frames, gif_name="demo.gif", is_pygame_surface=False, duration=30):
    """

    Args:
        frames: a list of images or pygame surfaces
        gif_name: name of the file
        is_pygame_surface: convert pygame surface to PIL.image
        duration: controlling the duration of each frame, unit: ms

    Returns:

    """
    assert gif_name.endswith("gif"), "File name should end with .gif"
    imgs = [pygame.surfarray.array3d(frame) if is_pygame_surface else frame for frame in frames]
    imgs = [Image.fromarray(img) for img in imgs]
    imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


CONFIG = {
    Token: ('', ''),
    Whitespace: ('gray', 'brightblack'),
    Comment: ('black', 'green'),
    Keyword: ('blue', 'brightblue'),
    Keyword.Type: ('cyan', 'brightcyan'),
    Operator.Word: ('magenta', 'brightmagenta'),
    Name: ('red', 'brightcyan'),
    Name.Attribute: ('magenta', 'brightcyan'),
    Name.Builtin: ('magenta', 'brightcyan'),
    Name.Builtin.Pseudo: ('magenta', 'brightcyan'),
    Name.Class: ('cyan', 'brightcyan'),
    Name.Constant: ('magenta', 'brightcyan'),
    Name.Decorator: ('magenta', 'brightcyan'),
    Name.Entity: ('magenta', 'brightcyan'),
    Name.Exception: ('magenta', 'brightcyan'),
    Name.Function: ('magenta', 'brightcyan'),
    Name.Function.Magic: ('magenta', 'brightcyan'),
    Name.Property: ('magenta', 'brightcyan'),
    Name.Label: ('magenta', 'brightcyan'),
    Name.Namespace: ('magenta', 'brightcyan'),
    Name.Other: ('green', 'brightcyan'),
    Name.Tag: ('magenta', 'brightcyan'),
    Name.Variable: ('red', 'brightcyan'),
    Name.Variable.Class: ('red', 'brightcyan'),
    Name.Variable.Global: ('red', 'brightcyan'),
    Name.Variable.Instance: ('red', 'brightcyan'),
    Name.Variable.Magic: ('red', 'brightcyan'),
    String: ('yellow', 'yellow'),
    Number: ('blue', 'blue'),
    Number.Float: ('green', 'blue'),
    Punctuation: ('magenta', 'blue'),
    Generic.Deleted: ('brightred', 'brightred'),
    Generic.Inserted: ('green', 'brightgreen'),
    Generic.Heading: ('**', '**'),
    Generic.Subheading: ('*magenta*', '*brightmagenta*'),
    Generic.Prompt: ('**', '**'),
    Generic.Error: ('brightred', 'brightred'),
    Error: ('_brightred_', '_brightred_'),
}

FUNC = {
    Token: ('', ''),
    Whitespace: ('gray', 'brightblack'),
    Comment: ('green', 'green'),
    Keyword: ('blue', 'brightblue'),
    Keyword.Type: ('cyan', 'brightcyan'),
    Operator.Word: ('magenta', 'brightmagenta'),
    Name: ('black', 'brightcyan'),
    Name.Attribute: ('magenta', 'brightcyan'),
    Name.Builtin: ('magenta', 'brightcyan'),
    Name.Builtin.Pseudo: ('magenta', 'brightcyan'),
    Name.Class: ('cyan', 'brightcyan'),
    Name.Constant: ('magenta', 'brightcyan'),
    Name.Decorator: ('magenta', 'brightcyan'),
    Name.Entity: ('magenta', 'brightcyan'),
    Name.Exception: ('magenta', 'brightcyan'),
    Name.Function: ('magenta', 'brightcyan'),
    Name.Function.Magic: ('magenta', 'brightcyan'),
    Name.Property: ('magenta', 'brightcyan'),
    Name.Label: ('magenta', 'brightcyan'),
    Name.Namespace: ('magenta', 'brightcyan'),
    Name.Other: ('green', 'brightcyan'),
    Name.Tag: ('magenta', 'brightcyan'),
    Name.Variable: ('black', 'brightcyan'),
    String: ('yellow', 'yellow'),
    Number: ('blue', 'blue'),
    Number.Float: ('green', 'blue'),
    Punctuation: ('magenta', 'blue'),
    Generic.Deleted: ('brightred', 'brightred'),
    Generic.Inserted: ('green', 'brightgreen'),
    Generic.Heading: ('**', '**'),
    Generic.Subheading: ('*magenta*', '*brightmagenta*'),
    Generic.Prompt: ('**', '**'),
    Generic.Error: ('brightred', 'brightred'),
    Error: ('_brightred_', '_brightred_'),
}

FUNC_2 = {
    Token: ('', ''),
    Whitespace: ('gray', 'brightblack'),
    Comment: ('black', 'green'),
    Keyword: ('blue', 'brightblue'),
    Keyword.Type: ('cyan', 'brightcyan'),
    Operator.Word: ('magenta', 'brightmagenta'),
    Name: ('red', 'brightcyan'),
    Name.Attribute: ('magenta', 'brightcyan'),
    Name.Builtin: ('magenta', 'brightcyan'),
    Name.Builtin.Pseudo: ('magenta', 'brightcyan'),
    Name.Class: ('cyan', 'brightcyan'),
    Name.Constant: ('magenta', 'brightcyan'),
    Name.Decorator: ('magenta', 'brightcyan'),
    Name.Entity: ('magenta', 'brightcyan'),
    Name.Exception: ('magenta', 'brightcyan'),
    Name.Function: ('magenta', 'brightcyan'),
    Name.Function.Magic: ('magenta', 'brightcyan'),
    Name.Property: ('magenta', 'brightcyan'),
    Name.Label: ('magenta', 'brightcyan'),
    Name.Namespace: ('magenta', 'brightcyan'),
    Name.Other: ('green', 'brightcyan'),
    Name.Tag: ('magenta', 'brightcyan'),
    Name.Variable: ('red', 'brightcyan'),
    Name.Variable.Class: ('red', 'brightcyan'),
    Name.Variable.Global: ('red', 'brightcyan'),
    Name.Variable.Instance: ('red', 'brightcyan'),
    Name.Variable.Magic: ('red', 'brightcyan'),
    String: ('black', 'black'),
    Number: ('blue', 'blue'),
    Number.Float: ('green', 'blue'),
    Punctuation: ('magenta', 'blue'),
    Generic.Deleted: ('brightred', 'brightred'),
    Generic.Inserted: ('green', 'brightgreen'),
    Generic.Heading: ('**', '**'),
    Generic.Subheading: ('*magenta*', '*brightmagenta*'),
    Generic.Prompt: ('**', '**'),
    Generic.Error: ('brightred', 'brightred'),
    Error: ('_brightred_', '_brightred_'),
}


def print_source(x, start_end=None, colorscheme=FUNC, **kwargs):
    """
    Print the source code of module x
    Args:
        colorscheme: color scheme of the output
        x: python module
        start_end: a tuple consists of start line content and end line content
    Returns:

    """
    code = get_source(x, start_end)

    print(highlight(code, PythonLexer(), TerminalFormatter(colorscheme=colorscheme, **kwargs)))


def get_source(x, start_end=None):
    """
    Print the source code of module x
    Args:
        x: python module
        start_end: a tuple consists of start line content and end line content
    Returns:

    """
    code = getsource(x)
    if start_end:
        dict_start = code.find(start_end[0])
        dict_end = code.find(start_end[1])
        code = code[dict_start:dict_end + 1]
    return dedent(code)


def list_files(path, prefix='', depth=2, current_depth=0):
    """
    List files given a path
    """
    path = Path(path)
    if path.is_dir() and current_depth < depth:
        print(prefix + path.name + '/')
        for child in path.iterdir():
            list_files(child, prefix + '    ', depth, current_depth=current_depth + 1)
    else:
        print(prefix + path.name)
